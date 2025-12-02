#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from experiments.results.plotting.utils import collect_env_curves, save_plot, LEVEL_COLORS

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.grid"] = False


# ------------------------- helpers (match table defs) -------------------------

def _forward_fill_1d(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs, but keep leading NaNs as NaN."""
    a = a.astype(float, copy=False)
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = np.where(mask, 0, np.arange(a.size))
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    # keep leading NaNs
    if mask[0]:
        first_valid = np.argmax(~mask)
        if first_valid > 0:
            out[:first_valid] = np.nan
    return out


def _calc_curve_based_forgetting(task_curve: np.ndarray, training_end_idx: int | None) -> float:
    """
    Weighted normalized forgetting, aligned with the table script:
      F = weighted_avg_{t>t_eot} max(0, (p_eot - p_t)/max(|p_eot|, eps))
    Early drops weigh more (exp decay).
    """
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
    """
    Return {seed: (n_train, chunk)}, where chunk = n_train // seq_len from training_soup.json length.
    Missing training file → seed omitted.
    """
    chunks = {}
    folder = base / algo / method / f"level_{level}" / f"agents_{agents}" / f"{strat}_{seq_len}"
    for seed in seeds:
        fp = folder / f"seed_{seed}" / "training_soup.json"
        if not fp.exists():
            print(f"[warn] missing data {fp}")
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


# ---------------------- series over tasks (now consistent) --------------------

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
    Return (xs=1..N, mu(x), ci(x)), where the mean/CI are across seeds.
      - performance: per-task, per-seed final (mean of last K evals) → per-seed prefix mean over tasks 1..x
      - forgetting: per-task, per-seed curve-based forgetting → per-seed cumulative sum over tasks 1..x
    CI uses normal approx: z * sd / sqrt(n) with NaN-safe stats.
    """
    # ---- load eval curves
    _, curves = collect_env_curves(base, algo, method, strat, seq_len, seeds, metric=metric, level=level)
    n_envs = min(len(curves), seq_len)
    curves = curves[:n_envs]

    # forward-fill/pad per seed
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

    # helper: mean & CI across seeds, vectorized over x
    def mean_ci_over_seeds(vals_seeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # vals_seeds: shape (S, X) with NaNs for missing seeds
        mu = np.nanmean(vals_seeds, axis=0)
        sd = np.nanstd(vals_seeds, axis=0, ddof=1)
        n = np.sum(~np.isnan(vals_seeds), axis=0)
        n = np.where(n > 1, n, 1)  # avoid div-by-zero
        z = 1.96 if abs(confidence - 0.95) < 1e-6 else {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        ci = z * sd / np.sqrt(n)
        # zero CI when only one seed
        ci = np.where(np.sum(~np.isnan(vals_seeds), axis=0) <= 1, 0.0, ci)
        return mu.astype(float), ci.astype(float)

    if y_mode == "performance":
        # per-task, per-seed finals
        K = max(1, ap_window)
        # finals[env, seed]
        finals = np.full((n_envs, len(seeds)), np.nan, dtype=float)
        for env_idx, Cf in enumerate(processed):
            # Cf: (S, T)
            for s_i in range(Cf.shape[0]):
                row = Cf[s_i]
                if np.all(np.isnan(row)):
                    continue
                # take last K finite values
                finite = row[~np.isnan(row)]
                if finite.size == 0:
                    continue
                tail = finite[-K:] if finite.size >= K else finite
                finals[env_idx, s_i] = float(np.mean(tail))

        # per-seed prefix mean across tasks 1..x
        # build (S, X) where each column x is mean(finals[:x, s]) over envs
        S = finals.shape[1]
        vals_seeds = np.full((S, n_envs), np.nan, dtype=float)
        for x in range(1, n_envs + 1):
            # mean over tasks ≤ x per seed
            vals_seeds[:, x - 1] = np.nanmean(finals[:x, :], axis=0)
        mu, ci = mean_ci_over_seeds(vals_seeds)
        return xs, mu, ci

    elif y_mode == "forgetting":
        # training chunk per seed
        seed_chunks = _load_training_chunks(base, algo, method, strat, seq_len, seeds, level, agents)

        # F_env_seed[env, seed]
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

        # per-seed cumulative sum across tasks → shape (S, X)
        vals_seeds = np.full((len(seeds), n_envs), np.nan, dtype=float)
        for s_i in range(len(seeds)):
            per_task = F[:, s_i]
            if np.all(np.isnan(per_task)):
                continue
            # nan-safe cumsum: treat NaN as 0 increase but keep NaN columns NaN if all prior NaN
            # here we simply replace NaN with 0 for cumsum; that’s fine because we’re CI-ing across seeds
            cs = np.cumsum(np.nan_to_num(per_task, nan=0.0))
            vals_seeds[s_i, :] = cs

        mu, ci = mean_ci_over_seeds(vals_seeds)
        return xs, mu, ci

    else:
        raise ValueError(f"Unknown y_mode: {y_mode}")


# --------------------------------- CLI & main --------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot performance/forgetting vs #tasks for ONE method across multiple levels (colored L1=green, L2=yellow, L3=red)."
    )
    p.add_argument("--data-root", type=str, default="data",
                   help="root folder containing algo/method/.../seed_*")
    p.add_argument("--algo", type=str, default="ippo")
    p.add_argument("--method", type=str, required=True,
                   help="Single method, e.g. Online_EWC or EWC")
    p.add_argument("--strategy", type=str, default="generate")
    p.add_argument("--seq-len", type=int, required=True,
                   help="N tasks; compute points for x=1..N, cropping to available data")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="seed indices present as seed_<k> folders (order matters)")
    p.add_argument("--metric", type=str, default="soup", choices=["soup", "reward"])
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3],
                   help="difficulty levels to plot (colors fixed: 1=green, 2=yellow, 3=red)")
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--y", type=str, default="performance",
                   choices=["performance", "forgetting"])
    p.add_argument("--ap-window", type=int, default=5,
                   help="# final eval points to average for performance")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--ylim", type=float, nargs=2, default=None)
    p.add_argument("--plot-name", type=str, default=None)
    p.add_argument("--confidence", type=float, default=0.95, help="CI level: 0.90, 0.95, or 0.99 (default: 0.95)")

    return p.parse_args()


def main():
    args = parse_args()
    base = (Path(__file__).resolve().parent.parent / args.data_root).resolve()

    plt.figure(figsize=(5, 3))
    max_x = 1
    plotted_any = False

    method = args.method
    for level in args.levels:
        try:
            xs, mu, ci = build_series_over_tasks(
                base=base,
                algo=args.algo,
                method=method,
                strat=args.strategy,
                seq_len=args.seq_len,
                seeds=args.seeds,
                metric=args.metric,
                level=level,
                y_mode=args.y,
                agents=args.agents,
                ap_window=args.ap_window,
                confidence=args.confidence,
            )
        except RuntimeError:
            continue
        if len(xs) == 0:
            continue
        max_x = max(max_x, int(xs[-1]))
        color = LEVEL_COLORS.get(level, None)
        lbl = f"Level {level}"
        plt.plot(xs, mu, label=lbl, color=color)
        plt.fill_between(xs, mu - ci, mu + ci, alpha=0.18, linewidth=0, color=color)
        plotted_any = True

    if not plotted_any:
        raise SystemExit("No matching data for the requested method/levels.")

    plt.xlabel("Number of tasks encountered")
    ylabel = "Average Normalized Score" if args.y == "performance" else "Cumulative Forgetting"
    plt.ylabel(ylabel)
    if args.title:
        plt.title(args.title)
    if args.ylim:
        plt.ylim(args.ylim)
    plt.xlim(1, max_x)
    plt.grid(False)
    plt.legend(loc="best", ncol=1)
    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "plots"
    name = args.plot_name or f"{args.y}_{method}_{args.seq_len}_tasks"
    save_plot(plt.gcf(), out_dir, name)
    plt.show()


if __name__ == "__main__":
    main()
