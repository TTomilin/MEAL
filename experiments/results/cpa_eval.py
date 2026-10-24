"""
python -m experiments.results.cpa_eval --run FT=<dir> --run EWC-multi=<dir> --run Joint=<dir> --out <figures dir>
python -m experiments.results.cpa_eval --prefix phase2 --out experiments/results/phase2_figures \\
    --run ft/seed0=<dir> --run ewc1e11/seed0=<dir> --run ewc1e9/seed0=<dir> --run erace/seed0=<dir> \\
    --run joint/seed0=<dir>
"""
import argparse
import csv
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

KINDS = ("brdiv", "planner")


@dataclass
class Run:
    name: str
    path: Path
    schedule: str
    labels: List[str]
    kinds: List[str]
    max_soup: float
    R: np.ndarray # (events, partners) normalised soups, nan where nothing completed
    ego: np.ndarray # (events, partners) summed ego soups
    partner: np.ndarray # (events, partners) summed partner soups
    event_steps: List[int]
    train: List[dict] = field(default_factory=list)

    @property
    def P(self) -> int:
        return len(self.labels)


def read_csv(path: Path) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def resolve_run_dir(path: Path) -> Path:
    path = Path(path)
    if (path / "eval_metrics.csv").exists():
        return path
    found = sorted(p.parent for p in path.rglob("eval_metrics.csv"))
    if len(found) != 1:
        raise SystemExit(f"{path}: expected one run directory, found {len(found)}")
    return found[0]


def load_run(name: str, path) -> Run:
    path = resolve_run_dir(path)
    identities = json.loads((path / "partner_bank.json").read_text())
    config = json.loads((path / "config.json").read_text())
    max_soup = float(json.loads((path / "run.json").read_text())["normalization"]["max_soup"])
    scale = max_soup if max_soup > 0 else 1.0
    labels = [i["label"] for i in identities]
    kinds = [i.get("kind", "brdiv") for i in identities]
    P = len(labels)

    events: Dict[int, Dict[int, list]] = {}
    for r in read_csv(path / "eval_metrics.csv"):
        if r["scope"] != "full" or not int(float(r["completed"])):
            continue
        slot = events.setdefault(int(r["env_steps"]), {}).setdefault(int(r["eval_partner_id"]), [])
        slot.append((float(r["soups"]), float(r.get("ego_soups") or "nan"), float(r.get("partner_soups") or "nan")))
    steps = sorted(events)
    R = np.full((len(steps), P), np.nan)
    ego, partner = np.full_like(R, np.nan), np.full_like(R, np.nan)
    for e, s in enumerate(steps):
        for i, rows in events[s].items():
            a = np.asarray(rows)
            R[e, i] = a[:, 0].mean() / scale
            ego[e, i], partner[e, i] = a[:, 1].sum(), a[:, 2].sum()
    run = Run(name, path, config.get("schedule", "sequential"), labels, kinds, max_soup, R, ego, partner, steps,
              read_csv(path / "train_metrics.csv") if (path / "train_metrics.csv").exists() else [])
    if run.schedule == "sequential" and len(steps) != P + 1:
        print(f"warning: {name}: {len(steps)} full evaluations, expected {P + 1} (incomplete run?)")
    return run


def ego_share(ego, partner):
    total = ego + partner
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, ego / total, np.nan)


def partner_metrics(run: Run) -> Dict[str, np.ndarray]:
    """Partner metrics (nan where undefined). 
    Sequential runs: see module docstring. 
    Joint run: acq = peak = max over cycles, final = last cycle"""
    R, P = run.R, run.P
    out = {k: np.full(P, np.nan) for k in ("solo", "acq", "peak", "final", "min", "forgetting", "nfm_term", "ego_share_final", "ego_share_acq")}
    out["solo"] = R[0].copy()
    out["final"] = R[-1].copy()
    out["min"] = np.nanmin(R, axis=0)
    out["ego_share_final"] = ego_share(run.ego[-1], run.partner[-1])
    if run.schedule != "sequential":
        out["peak"] = out["acq"] = np.nanmax(R[1:], axis=0) if len(R) > 1 else R[0]
        return out
    for i in range(P):
        if i + 1 >= len(R):
            continue
        out["acq"][i] = R[i + 1, i]
        out["ego_share_acq"][i] = ego_share(run.ego[i + 1, i], run.partner[i + 1, i])
        out["peak"][i] = np.nanmax(R[i + 1:, i])
        if i < P - 1 and len(R) == P + 1:
            out["forgetting"][i] = out["peak"][i] - out["final"][i]
            denom = 1.0 - out["min"][i]
            out["nfm_term"][i] = out["forgetting"][i] / denom if denom > 1e-9 else np.nan
    return out


def summarise(run: Run, m: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Means over all partners and per kind."""
    groups = {"all": np.ones(run.P, bool), **{k: np.asarray(run.kinds) == k for k in KINDS}}
    res = {}
    for g, mask in groups.items():
        def mean(key):
            v = m[key][mask]
            v = v[~np.isnan(v)]
            return float(v.mean()) if v.size else float("nan")
        res[g] = {"n": int(mask.sum()), "solo": mean("solo"), "acq": mean("acq"), "final": mean("final"),
                  "final_minus_solo": mean("final") - mean("solo"), "forgetting": mean("forgetting"),
                  "nfm": mean("nfm_term"), "ego_share_acq": mean("ego_share_acq"),
                  "ego_share_final": mean("ego_share_final")}
    return res


def stage_train_stats(run: Run) -> List[dict]:
    """Each training stage (partner, sequential order): first/last training soups (normalised) and cl_penalty."""
    scale = run.max_soup if run.max_soup > 0 else 1.0
    stats = []
    for k in range(run.P):
        rows = [r for r in run.train if int(r["stage"]) == k]
        soups = [float(r["soups"]) / scale for r in rows if r.get("soups") not in ("", None)]
        pen = [float(r["cl_penalty"]) for r in rows if r.get("cl_penalty") not in ("", None)]
        n = max(1, len(soups) // 10)
        stats.append({"stage": k, "updates": len(rows),
                      "train_soups_first": float(np.mean(soups[:n])) if soups else float("nan"),
                      "train_soups_last": float(np.mean(soups[-n:])) if soups else float("nan"),
                      "cl_penalty_mean": float(np.mean(pen)) if pen else float("nan"),
                      "cl_penalty_last": pen[-1] if pen else float("nan")})
    return stats


def fmt(x, digits=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def short(label):
    return label.split("/", 1)[-1]


def fmt_pm(mean, std, digits=3) -> str:
    if mean is None or np.isnan(mean):
        return "-"
    return fmt(float(mean), digits) + ("" if std is None or np.isnan(std) else f" +- {float(std):.{digits}f}")


def group_by_method(runs: List[Run]) -> Dict[str, List[Run]]:
    """Runs each method (`ft/seed1` -> `ft`) in first-seen order; all seeds of a method must share one schedule."""
    by: Dict[str, List[Run]] = {}
    for r in runs:
        by.setdefault(method_of(r.name), []).append(r)
    for m, rs in by.items():
        if len({r.schedule for r in rs}) > 1:
            raise SystemExit(f"{m}: its seeds were run with different schedules")
    return by


def stack_seeds(arrays) -> np.ndarray:
    """(seeds, ...) stack of per-seed arrays; axis 0 of each array is cut to the shortest seed."""
    arrays = [np.asarray(a, float) for a in arrays]
    if arrays[0].ndim == 0:
        return np.stack(arrays)
    n = min(len(a) for a in arrays)
    return np.stack([a[:n] for a in arrays])


def seed_mean_std(arrays):
    """Element-wise mean and sample std over seeds, ignoring nan; std is nan where fewer than two seeds have a value."""
    s = stack_seeds(arrays)
    count = (~np.isnan(s)).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(s, axis=0)
        std = np.nanstd(s, axis=0, ddof=1)
    return mean, np.where(count > 1, std, np.nan)


def method_partner_metrics(by_method: Dict[str, List[Run]], metrics) -> Dict[str, Dict[str, tuple]]:
    """(mean, std) arrays over seeds, one entry each partner."""
    return {m: {k: seed_mean_std([metrics[r.name][k] for r in rs]) for k in metrics[rs[0].name]}
            for m, rs in by_method.items()}


TRAIN_KEYS = ("train_soups_first", "train_soups_last", "cl_penalty_mean")


def method_train_stats(by_method: Dict[str, List[Run]], train_stats) -> Dict[str, Dict[str, tuple]]:
    """(mean, std) arrays over seeds, one entry each stage."""
    return {m: {k: seed_mean_std([[s[k] for s in train_stats[r.name]] for r in rs]) for k in TRAIN_KEYS}
            for m, rs in by_method.items()}


AGG_METRICS = ("solo", "acq", "final", "final_minus_solo", "forgetting", "nfm", "ego_share_acq", "ego_share_final")


def method_of(name: str) -> str:
    """ft/seed1 -> ft (a name without a seed suffix is its own method)"""
    return name.split("/", 1)[0]


def aggregate_seeds(runs: List[Run], summaries) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    by_method: Dict[str, List[str]] = {}
    for r in runs:
        by_method.setdefault(method_of(r.name), []).append(r.name)
    agg = {}
    for method, names in by_method.items():
        agg[method] = {}
        for g in summaries[names[0]]:
            agg[method][g] = {}
            for key in AGG_METRICS:
                v = np.asarray([summaries[n][g][key] for n in names], float)
                v = v[~np.isnan(v)]
                agg[method][g][key] = {
                    "mean": float(v.mean()) if v.size else float("nan"),
                    "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                    "n_seeds": int(v.size)}
    return agg


def fmt_ms(cell) -> str:
    return fmt_pm(cell["mean"], cell["std"])


def figures(runs: List[Run], metrics, train_stats, out: Path, formats=("png",), prefix: str = "condc"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        for f in formats:
            fig.savefig(out / f"{prefix}_{name}.{f}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    by = group_by_method(runs)
    pm = method_partner_metrics(by, metrics)
    ts = method_train_stats(by, train_stats)
    seq = [m for m, rs in by.items() if rs[0].schedule == "sequential"]
    joint = [m for m in by if m not in seq]
    R_mean = {m: seed_mean_std([r.R for r in rs])[0] for m, rs in by.items()}

    def title(m):
        return f"{m} (mean of {len(by[m])} seeds)" if len(by[m]) > 1 else m

    ref = runs[0]
    ticks = [f"{i}:{short(l)}" for i, l in enumerate(ref.labels)]
    vmax = max(np.nanmax(R) for R in R_mean.values() if np.isfinite(R).any()) or 1.0
    fig, axes = plt.subplots(1, len(by), figsize=(5.5 * len(by), 5.5), squeeze=False)
    for ax, m in zip(axes[0], by):
        R = R_mean[m]
        im = ax.imshow(R, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{title(m)}, {by[m][0].schedule}", fontsize=9)
        ax.set_xticks(range(ref.P), ticks, rotation=90, fontsize=7)
        ax.set_xlabel("evaluated partner")
        ax.set_ylabel("after stage (row 0 = init)" if m in seq else "after cycle (row 0 = init)")
        if m in seq:
            ax.plot(range(ref.P), range(1, ref.P + 1), "w.", ms=4)  # the diagonal: just trained
    fig.colorbar(im, ax=axes[0].tolist(), label="normalised soups (seed mean)")
    save(fig, "retention")

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * ref.P), 4.5))
    x = np.arange(ref.P)
    series = [("solo (untrained ego)", seed_mean_std([metrics[r.name]["solo"] for r in runs]), "lightgray")]
    for m in seq:
        series += [(f"{m} acq", pm[m]["acq"], None), (f"{m} final", pm[m]["final"], None)]
    series += [(f"{m} final (UB)", pm[m]["final"], "black") for m in joint]
    w = 0.8 / len(series)
    for j, (lab, (mean, std), c) in enumerate(series):
        ax.bar(x + (j - len(series) / 2 + 0.5) * w, np.nan_to_num(mean), w, yerr=np.nan_to_num(std), label=lab,
               color=c, error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(x, ticks, rotation=90, fontsize=7)
    ax.set_ylabel("normalised soups (mean +- std over seeds)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "per_partner")

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * ref.P), 4))
    w = 0.8 / len(by)
    for j, m in enumerate(by):
        mean, std = pm[m]["ego_share_final"]
        ax.bar(x + (j - len(by) / 2 + 0.5) * w, np.nan_to_num(mean), w, yerr=np.nan_to_num(std), label=m,
               error_kw=dict(lw=0.6, capsize=1.5))
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x, ticks, rotation=90, fontsize=7)
    ax.set_ylabel("ego soups / team soups (final, seed mean)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    save(fig, "ego_share")

    if seq:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for m in seq:
            curves = []
            for r in by[m]:
                scale = r.max_soup if r.max_soup > 0 else 1.0
                curves.append([float(t["soups"]) / scale if t.get("soups") not in ("", None) else np.nan
                               for t in r.train])
            if not any(curves):
                continue
            mean, std = seed_mean_std(curves)
            steps = np.asarray([int(t["env_steps"]) for t in by[m][0].train][:len(mean)])
            line, = axes[0].plot(steps, mean, lw=0.7, label=title(m))
            axes[0].fill_between(steps, mean - np.nan_to_num(std), mean + np.nan_to_num(std),
                                 color=line.get_color(), alpha=0.15, lw=0)
            pen_mean, pen_std = ts[m]["cl_penalty_mean"]
            if np.nanmax(np.nan_to_num(pen_mean)) > 0:
                axes[1].errorbar(range(len(pen_mean)), pen_mean, yerr=np.nan_to_num(pen_std), fmt="o-", capsize=2,
                                 label=title(m))
        for boundary in by[seq[0]][0].event_steps[1:-1]:  # stage ends
            axes[0].axvline(boundary, color="gray", lw=0.3)
        axes[0].set_xlabel("joint env steps")
        axes[0].set_ylabel("training soups (normalised)")
        axes[0].set_title("acquisition (training episodes, mean +- std over seeds)")
        axes[0].legend(fontsize=7)
        axes[1].set_xlabel("stage")
        axes[1].set_ylabel("mean cl_penalty")
        axes[1].set_title("EWC/MAS penalty per stage")
        if axes[1].lines:
            axes[1].legend(fontsize=7)
        save(fig, "acquisition_penalty")

    if joint:
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in joint:
            for kind in KINDS:
                mask = np.asarray(ref.kinds) == kind
                if mask.any():
                    mean, std = seed_mean_std([np.nanmean(r.R[:, mask], axis=1) for r in by[m]])
                    line, = ax.plot(range(len(mean)), mean, label=f"{title(m)} {kind}")
                    ax.fill_between(range(len(mean)), mean - np.nan_to_num(std), mean + np.nan_to_num(std),
                                    color=line.get_color(), alpha=0.15, lw=0)
        for m in seq:
            ax.axhline(float(np.nanmean(pm[m]["final"][0])), ls="--", lw=0.8, label=f"{m} final (all)")
        ax.set_xlabel("cycle (0 = init)")
        ax.set_ylabel("mean normalised soups")
        ax.legend(fontsize=7)
        save(fig, "joint_curve")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR", help="a run directory (or its seed directory) under a display name; repeat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--prefix", default="condc", help="file name prefix of the summary and figures")
    args = ap.parse_args()
    named = {}
    for item in args.run:
        name, _, d = item.partition("=")
        if not d:
            ap.error(f"--run expects NAME=DIR, got {item!r}")
        if name in named:
            ap.error(f"--run name {name!r} given twice")
        named[name] = d


if __name__ == "__main__":
    main()
