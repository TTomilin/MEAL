"""
python -m experiments.results.cpa_figures --runs experiments/results/condc_runs --out experiments/results/condc_figures
python -m experiments.results.cpa_figures --prefix phase2 --seeds 0 1 2 \\
    --runs $MEAL_CPA_ROOT/runs/coord_ring/phase2 --out experiments/results/phase2_figures \\
    --method FT=ft --method "EWC 1e11=ewc1e11" --method "EWC 1e9=ewc1e9" --method ER-ACE=erace \\
    --method Joint=joint
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.results.cpa_condition_c_analysis import KINDS, analyse, method_of, seed_mean_std, short

PHASE1_METHODS = ("FT=ft_sequential", "EWC-multi=ewc_multi_sequential", "Joint=ft_interleaved")


def per_partner_train(run, key):
    vals = np.full(run.P, np.nan)
    for i in range(run.P):
        v = [float(r[key]) for r in run.train if int(r["partner_id"]) == i and r.get(key) not in ("", None)]
        if v:
            vals[i] = np.mean(v)
    return vals


def parse_methods(items):
    methods = []
    for item in items:
        label, _, sub = item.partition("=")
        if not label or not sub or "/" in label:
            raise SystemExit(f"--method expects LABEL=SUBDIR (no '/' in LABEL), got {item!r}")
        methods.append((label, sub))
    if len({m for m, _ in methods}) != len(methods):
        raise SystemExit("--method labels must be distinct")
    return methods


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="experiments/results/condc_runs")
    ap.add_argument("--out", default="experiments/results/condc_figures")
    ap.add_argument("--method", action="append", metavar="LABEL=SUBDIR",
                    help="a method under a display label; repeat (default: the Phase 1 FT / EWC-multi / Joint)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="ego seeds: run directories are <runs>/<SUBDIR>/seed<S>; omit for <runs>/<SUBDIR>")
    ap.add_argument("--prefix", default="condc", help="file name prefix of the figures and numbers")
    args = ap.parse_args()
    d, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = parse_methods(args.method or PHASE1_METHODS)
    named = {}
    for label, sub in methods:
        if args.seeds:
            for seed in args.seeds:
                named[f"{label}/seed{seed}"] = d / sub / f"seed{seed}"
        else:
            named[label] = d / sub
    runs, metrics, summaries, _ = analyse(named, None, False)
    by_method = {label: [r for r in runs if method_of(r.name) == label] for label, _ in methods}
    seq_labels = [m for m, rs in by_method.items() if rs[0].schedule == "sequential"]
    joint_labels = [m for m, rs in by_method.items() if rs[0].schedule != "sequential"]
    if not seq_labels:
        raise SystemExit("at least one sequential method is needed")
    ref_label = seq_labels[0]
    ft = by_method[ref_label][0]
    kinds = np.asarray(ft.kinds)
    brdiv, planner = np.where(kinds == "brdiv")[0], np.where(kinds == "planner")[0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.savefig(out / f"{args.prefix}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def over_seeds(label, kind, key):
        v = np.asarray([summaries[r.name][kind][key] for r in by_method[label]], float)
        v = v[~np.isnan(v)]
        if not v.size:
            return np.nan, np.nan
        return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0

    numbers = {"methods": {m: [r.name for r in rs] for m, rs in by_method.items()}}

    # Fig 1: group summary (error bars = std over seeds).
    labels = [m for m, _ in methods]
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 2.2 * len(labels)), 3.8), sharey=True)
    for ax, kind in zip(axes, KINDS):
        acq = [over_seeds(m, kind, "acq") if m in seq_labels else (np.nan, np.nan) for m in labels]
        fin = [over_seeds(m, kind, "final") for m in labels]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, np.nan_to_num([a for a, _ in acq]), 0.4, yerr=np.nan_to_num([s for _, s in acq]),
               capsize=3, label="acquisition (right after training)", color="#9ecae1")
        ax.bar(x + 0.2, [f for f, _ in fin], 0.4, yerr=np.nan_to_num([s for _, s in fin]), capsize=3,
               label="final (end of run)", color="#08519c")
        for xi, ((a, _), (f, _)) in enumerate(zip(acq, fin)):
            if not np.isnan(a):
                ax.text(xi - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
            if not np.isnan(f):
                ax.text(xi + 0.2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
        ax.axhline(over_seeds(ref_label, kind, "solo")[0], color="gray", ls=":", lw=1, label="solo (untrained ego)")
        ax.set_xticks(x, [f"{m}\n({'interleaved' if m in joint_labels else 'sequential'})" for m in labels],
                      fontsize=8)
        ax.set_title(f"{kind} partners (n={len(brdiv) if kind == 'brdiv' else len(planner)})")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("normalised team soups")
    axes[0].legend(fontsize=7, loc="upper left")
    save(fig, "fig1_group_summary")
    numbers["group"] = {m: {k: {key: over_seeds(m, k, key) for key in ("acq", "final", "nfm")} for k in KINDS}
                        for m in labels}

    def mean_std(arrays):
        mean, std = seed_mean_std(arrays)
        return mean, np.nan_to_num(std)

    def band(ax, x, mean, std, color, **kw):
        line, = ax.plot(x, mean, color=color, **kw)
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.12, lw=0)

    ft_runs = by_method[ref_label]
    n_ft = len(ft_runs)

    # Fig 2: interference in the first sequential method (mean +- std over its seeds).
    fig, ax = plt.subplots(figsize=(8, 3.8))
    R_mean, R_std = mean_std([r.R for r in ft_runs])
    stages = np.arange(len(R_mean))
    colors = plt.cm.Reds(np.linspace(0.45, 0.95, max(1, len(brdiv)))) if len(brdiv) > 3 else \
        ["#e41a1c", "#ff7f00", "#984ea3"]
    for c, i in zip(colors, brdiv):
        band(ax, stages, R_mean[:, i], R_std[:, i], c, marker="o", ms=3, label=f"BRDiv {short(ft.labels[i])}")
        ax.axvline(i + 1, color=c, lw=0.6, ls=":")
    if len(planner):
        pm_mean, pm_std = mean_std([np.nanmean(r.R[:, planner], axis=1) for r in ft_runs])
        band(ax, stages[:len(pm_mean)], pm_mean, pm_std, "#08519c", marker="s", ms=3,
             label=f"planner mean (all {len(planner)})")
        zero_shot = [float(np.mean([r.R[i, i] for i in planner[1:]])) for r in ft_runs] if len(planner) > 1 else []
        numbers["ft_planner_zero_shot_after_first_planner"] = float(np.mean(zero_shot)) if zero_shot else None
        numbers["ft_planner_acq"] = float(np.mean([np.nanmean(metrics[r.name]["acq"][planner]) for r in ft_runs]))
        ax.axvspan(planner[0] + 0.5, planner[0] + 1.5, color="#08519c", alpha=0.08)
    ax.set_xticks(stages, ["init"] + [f"{k}:{short(l)}" for k, l in enumerate(ft.labels)][:len(stages) - 1],
                  rotation=90, fontsize=7)
    ax.set_xlabel("after training stage")
    ax.set_ylabel("normalised team soups")
    ax.set_title(f"{ref_label} (mean ± std over {n_ft} seeds): retention of each BRDiv convention per stage "
                 f"(shaded: first planner)", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    save(fig, "fig2_interference")

    if not joint_labels:
        (out / f"{args.prefix}_paper_numbers.json").write_text(json.dumps(numbers, indent=2))
        print(json.dumps(numbers, indent=2))
        return
    joint_label = joint_labels[0]
    joint_runs = by_method[joint_label]
    joint = joint_runs[0]
    n_joint = len(joint_runs)

    # Fig 3: joint per partner over cycles (mean +- std over seeds).
    fig, ax = plt.subplots(figsize=(8, 3.8))
    J_mean, J_std = mean_std([r.R for r in joint_runs])
    cycles = np.arange(len(J_mean))
    ft_acq = seed_mean_std([metrics[r.name]["acq"] for r in ft_runs])[0]
    for c, i in zip(colors, brdiv):
        band(ax, cycles, J_mean[:, i], J_std[:, i], c, lw=1.2, label=f"Joint, BRDiv {short(joint.labels[i])}")
        ax.axhline(ft_acq[i], color=c, ls="--", lw=0.8)
    jp_mean, jp_std = mean_std([np.nanmean(r.R[:, planner], axis=1) for r in joint_runs])
    band(ax, cycles[:len(jp_mean)], jp_mean, jp_std, "#08519c", lw=1.5, label="Joint, planner mean")
    ax.set_xlabel("cycle (every partner once per cycle; 0 = untrained)")
    ax.set_ylabel("normalised team soups")
    ax.set_title(f"{joint_label} (mean ± std over {n_joint} seeds): per-partner score over cycles "
                 f"({ref_label} acquisition dashed)", fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "fig3_joint_partners")
    per_seed = {"second_half": [], "last": [], "slope": [], "cycle1": []}
    for r in joint_runs:
        late = r.R[len(r.R) // 2:]
        per_seed["second_half"].append(float(late[:, brdiv].mean()))
        per_seed["last"].append(float(r.R[-1, brdiv].mean()))
        per_seed["slope"].append(float(np.polyfit(np.arange(len(late)), late[:, brdiv].mean(1), 1)[0]))
        per_seed["cycle1"].append(float(r.R[1, planner].mean()))
    numbers["joint_brdiv_mean_second_half"] = float(np.mean(per_seed["second_half"]))
    numbers["joint_brdiv_mean_last"] = float(np.mean(per_seed["last"]))
    numbers["joint_brdiv_slope_second_half_per_cycle"] = float(np.mean(per_seed["slope"]))
    numbers["joint_planner_cycle1"] = float(np.mean(per_seed["cycle1"]))

    # Fig 4: partner behaviour (mean +- std over seeds).
    share_mean, share_std = mean_std([
        per_partner_train(r, "partner_raw_shaped_return")
        / (per_partner_train(r, "partner_raw_shaped_return") + per_partner_train(r, "ego_raw_shaped_return"))
        for r in joint_runs])
    solo_mean, solo_std = mean_std([metrics[r.name]["solo"] for r in ft_runs])
    ent_mean, ent_std = mean_std([per_partner_train(r, "entropy") for r in joint_runs])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    x = np.arange(joint.P)
    ticks = [f"{i}:{short(l)}" for i, l in enumerate(joint.labels)]
    bar_colors = ["#e41a1c" if k == "brdiv" else "#08519c" for k in kinds]
    err = dict(capsize=2, error_kw=dict(lw=0.7))
    axes[0].bar(x, share_mean, color=bar_colors, yerr=share_std, **err)
    axes[0].set_ylabel("partner share of shaped sub-task reward")
    axes[0].set_title(f"(a) partner's share of sub-task work ({joint_label})")
    axes[1].bar(x, solo_mean, color=bar_colors, yerr=solo_std, **err)
    axes[1].set_ylabel("normalised team soups")
    axes[1].set_title("(b) solo score: partner + untrained ego")
    axes[2].bar(x, ent_mean, color=bar_colors, yerr=ent_std, **err)
    axes[2].set_ylabel("ego policy entropy (nats)")
    axes[2].set_title(f"(c) ego action entropy with each partner ({joint_label})")
    for ax in axes:
        ax.set_xticks(x, ticks, rotation=90, fontsize=7)
    fig.suptitle(f"mean ± std over seeds ({joint_label}: {n_joint}, {ref_label}: {n_ft})", fontsize=9)
    save(fig, "fig4_partner_behaviour")
    numbers["joint_partner_shaped_share"] = {short(l): float(s) for l, s in zip(joint.labels, share_mean)}
    numbers["joint_entropy"] = {"brdiv": float(ent_mean[brdiv].mean()), "planner": float(ent_mean[planner].mean())}
    joint_final = seed_mean_std([metrics[r.name]["final"] for r in joint_runs])[0]
    numbers["joint_over_ft_acq"] = {k: float(np.nanmean(joint_final[idx]) / np.nanmean(ft_acq[idx]))
                                    for k, idx in (("brdiv", brdiv), ("planner", planner))}
    (out / f"{args.prefix}_paper_numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
