#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

from experiments.results.plotting.utils import collect_env_curves  # (env_names, [array_per_env])

# ---------- helpers copied to match plotting semantics ----------

def _forward_fill_1d(a: np.ndarray) -> np.ndarray:
    a = a.astype(float, copy=False)
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = np.where(mask, 0, np.arange(a.size))
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    if mask[0]:
        first_valid = np.argmax(~mask)
        if first_valid > 0:
            out[:first_valid] = np.nan
    return out

def _calc_curve_based_forgetting(task_curve: np.ndarray, training_end_idx: int | None) -> float:
    if task_curve.size <= 1:
        return 0.0
    if training_end_idx is None:
        training_end_idx = task_curve.size - 1
    training_end_idx = int(np.clip(training_end_idx, 0, task_curve.size - 1))
    p_eot = float(task_curve[training_end_idx])
    if training_end_idx >= task_curve.size - 1:
        return 0.0
    post = task_curve[training_end_idx + 1:]
    if post.size == 0:
        return 0.0
    eps = 1e-8
    denom = max(abs(p_eot), eps)
    drop = np.maximum(p_eot - post, 0.0) / denom
    lam = 2.0
    t = np.arange(post.size)
    w = np.exp(-lam * (t / (post.size - 1))) if post.size > 1 else np.ones_like(post)
    num = float(np.sum(drop * w))
    den = float(np.sum(w))
    F = num / den if den > 0 else 0.0
    return float(np.clip(F, 0.0, 1.0))

def _load_training_chunks(
        base: Path, algo: str, method: str, strat: str, seq_len: int, seeds: List[int], level: int, agents: int
) -> Dict[int, Tuple[int, int]]:
    chunks = {}
    folder = base / algo / method / f"level_{level}" / f"agents_{agents}" / f"{strat}_{seq_len}"
    for seed in seeds:
        fp = folder / f"seed_{seed}" / "training_soup.json"
        if not fp.exists():
            continue
        try:
            import json as _json
            arr = np.array(_json.loads(fp.read_text()), dtype=float)
            n_train = int(arr.size)
            if n_train <= 0:
                continue
            chunks[seed] = (n_train, max(1, n_train // max(1, seq_len)))
        except Exception:
            continue
    return chunks

def _mean_ci(vals: np.ndarray, confidence: float) -> Tuple[float, float]:
    """NaN-safe mean ± z*sd/sqrt(n) across seeds for a single x."""
    mu = float(np.nanmean(vals))
    n = int(np.sum(~np.isnan(vals)))
    if n <= 1:
        return mu, 0.0
    sd = float(np.nanstd(vals, ddof=1))
    z = 1.96 if abs(confidence - 0.95) < 1e-6 else {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    ci = z * sd / np.sqrt(n)
    return mu, float(ci)

# ---------- same semantics as your plotting script ----------

def build_series_over_tasks(
        base: Path,
        algo: str,
        method: str,
        strat: str,
        seq_len: int,
        seeds: List[int],
        metric: str,
        level: int,
        y_mode: str,
        agents: int,
        ap_window: int,
        confidence: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (xs=1..N, mu(x), ci(x)) across seeds.
      performance: per-task, per-seed final (mean last K) → per-seed prefix mean across tasks 1..x
      forgetting: per-task, per-seed curve-based forgetting → per-seed cumulative sum across tasks 1..x
    """
    _, curves = collect_env_curves(base, algo, method, strat, seq_len, seeds, metric=metric, level=level)
    n_envs = min(len(curves), seq_len)
    curves = curves[:n_envs]

    processed, max_T = [], 0
    for C in curves:
        S, T = C.shape
        max_T = max(max_T, T)
    for C in curves:
        S, T = C.shape
        Cf = np.vstack([_forward_fill_1d(C[s]) for s in range(S)])
        if T < max_T:
            pad = np.full((S, max_T - T), np.nan, dtype=float)
            Cf = np.hstack([Cf, pad])
        processed.append(Cf)

    xs = np.arange(1, n_envs + 1, dtype=int)

    if y_mode == "performance":
        K = max(1, ap_window)
        finals = np.full((n_envs, len(seeds)), np.nan, dtype=float)  # [env, seed]
        for env_idx, Cf in enumerate(processed):
            for s_i in range(Cf.shape[0]):
                row = Cf[s_i]
                if np.all(np.isnan(row)):
                    continue
                finite = row[~np.isnan(row)]
                if finite.size == 0:
                    continue
                tail = finite[-K:] if finite.size >= K else finite
                finals[env_idx, s_i] = float(np.mean(tail))

        # per-seed prefix mean over tasks 1..x → vals_seeds[seed, x]
        vals_seeds = np.full((len(seeds), n_envs), np.nan, dtype=float)
        for x in range(1, n_envs + 1):
            vals_seeds[:, x - 1] = np.nanmean(finals[:x, :], axis=0)

    elif y_mode == "forgetting":
        seed_chunks = _load_training_chunks(base, algo, method, strat, seq_len, seeds, level, agents)
        F = np.full((n_envs, len(seeds)), np.nan, dtype=float)
        for env_idx, Cf in enumerate(processed):
            S, T = Cf.shape
            for s_i, seed in enumerate(seeds):
                if seed not in seed_chunks:
                    continue
                n_train, chunk = seed_chunks[seed]
                train_end_step = (env_idx + 1) * chunk
                frac = min(1.0, train_end_step / max(1, n_train))
                t_eot = int(round(frac * (T - 1)))
                curve = _forward_fill_1d(Cf[s_i])
                if np.all(np.isnan(curve)):
                    continue
                F[env_idx, s_i] = _calc_curve_based_forgetting(curve, t_eot)

        vals_seeds = np.full((len(seeds), n_envs), np.nan, dtype=float)
        for s_i in range(len(seeds)):
            per_task = F[:, s_i]
            if np.all(np.isnan(per_task)):
                continue
            cs = np.cumsum(np.nan_to_num(per_task, nan=0.0))
            vals_seeds[s_i, :] = cs
    else:
        raise ValueError(f"Unknown y_mode: {y_mode}")

    # mean/CI per x across seeds
    mu = np.empty(n_envs, dtype=float)
    ci = np.empty(n_envs, dtype=float)
    for x in range(n_envs):
        mu[x], ci[x] = _mean_ci(vals_seeds[:, x], confidence)
    return xs, mu, ci

# ---------- latex formatting ----------

def _fmt_cell(mean: float, ci: float) -> str:
    if np.isnan(mean):
        return "--"
    main = f"{mean:.3f}"
    ci_part = f"{{\\scriptsize$\\pm{ci:.2f}$}}" if ci > 0 else ""
    return main + ci_part

def make_table(
        base: Path,
        algo: str,
        method: str,
        strat: str,
        seq_len: int,
        seeds: List[int],
        metric: str,
        levels: List[int],
        agents: int,
        y_mode: str,
        ap_window: int,
        confidence: float,
        tasks: List[int] | None,
) -> str:
    # pick task columns
    if tasks:
        cols = [t for t in tasks if 1 <= t <= seq_len]
    else:
        # 5 checkpoints evenly spaced, inclusive of seq_len
        cols = np.linspace(1, seq_len, 5, dtype=int).tolist()
        cols = sorted(list(dict.fromkeys(cols)))  # unique, stable

    # header
    col_headers = " & ".join([f"Task {t}" for t in cols])
    rows_tex = []

    for lvl in levels:
        try:
            xs, mu, ci = build_series_over_tasks(
                base=base, algo=algo, method=method, strat=strat, seq_len=seq_len,
                seeds=seeds, metric=metric, level=lvl, y_mode=y_mode,
                agents=agents, ap_window=ap_window, confidence=confidence
            )
        except RuntimeError:
            # no data → row of dashes
            row_cells = ["--"] * len(cols)
            rows_tex.append(f"Level {lvl} & " + " & ".join(row_cells) + r" \\")
            continue

        # map x -> (mean, ci)
        # xs is 1..N; we need nearest not exceeding x, i.e., direct index x-1 if exists
        row_cells = []
        max_x = int(xs[-1]) if len(xs) else 0
        for t in cols:
            if t <= max_x:
                m, c = float(mu[t-1]), float(ci[t-1])
                row_cells.append(_fmt_cell(m, c))
            else:
                row_cells.append("--")
        rows_tex.append(f"Level {lvl} & " + " & ".join(row_cells) + r" \\")

    ylabel = "Average Normalized Score" if y_mode == "performance" else "Cumulative Forgetting"
    cap = f"{method}: {ylabel} at selected task counts."
    table = (
            "\\begin{table}[t]\n"
            "\\centering\n"
            f"\\caption{{{cap}}}\n"
            "\\label{tab:levels_vs_tasks}\n"
            "\\begin{tabular}{l" + "c"*len(cols) + "}\n"
                                                   "\\toprule\n"
                                                   "Level & " + col_headers + r" \\" "\n"
                                                                              "\\midrule\n"
            + "\n".join(rows_tex) + "\n"
                                    "\\bottomrule\n"
                                    "\\end{tabular}\n"
                                    "\\end{table}\n"
    )
    return table

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="LaTeX table: rows=levels, cols=task checkpoints; values match the plot (mean ± CI across seeds)."
    )
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--algo", type=str, default="ippo")
    p.add_argument("--method", type=str, required=True)
    p.add_argument("--strategy", type=str, default="generate")
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--metric", type=str, default="soup", choices=["soup", "reward"])
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--y", type=str, default="performance", choices=["performance", "forgetting"])
    p.add_argument("--ap-window", type=int, default=10, help="final-K evals for performance")
    p.add_argument("--confidence", type=float, default=0.95, help="0.90/0.95/0.99")
    p.add_argument("--tasks", type=int, nargs="*", default=None,
                   help="specific task indices (e.g., 20 40 60 80 100). If omitted, 5 evenly spaced checkpoints are used.")
    return p.parse_args()

def main():
    args = parse_args()
    base = (Path(__file__).resolve().parent.parent / args.data_root).resolve()
    tex = make_table(
        base=base, algo=args.algo, method=args.method, strat=args.strategy, seq_len=args.seq_len,
        seeds=args.seeds, metric=args.metric, levels=args.levels, agents=args.agents,
        y_mode=args.y, ap_window=args.ap_window, confidence=args.confidence, tasks=args.tasks
    )
    print(tex)

if __name__ == "__main__":
    main()
