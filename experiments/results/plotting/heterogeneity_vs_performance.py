#!/usr/bin/env python3
"""
For a set of focal tasks, plot soup score (solid) and heterogeneity (dashed)
over the full training sequence.

Layout: rows = focal tasks, cols = CL methods.
Each subplot shares the same [0, 1] y-axis so the two curves are directly
comparable.  A vertical dashed line marks where the focal task's own training
ends — if heterogeneity drops *before* the soup score after that line it is an
early-warning signal of coordination forgetting.

Usage:
    python experiments/results/plotting/heterogeneity_vs_performance.py \\
        --data_root experiments/results/data \\
        --algo ippo \\
        --methods EWC Online_EWC MAS L2 FT \\
        --level 2 --agents 2 \\
        --seq_len 20 --strategy generate \\
        --focal_tasks 0 4 9 \\
        --seeds 1 2 3 \\
        --out runs/het_vs_perf.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from experiments.results.plotting.utils import METHOD_COLORS, METHOD_DISPLAY_NAMES

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["axes.grid"] = False


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_series(fp: Path) -> Optional[np.ndarray]:
    if not fp.exists():
        return None
    try:
        if fp.suffix == ".json":
            return np.array(json.loads(fp.read_text()), dtype=float)
        if fp.suffix == ".npz":
            return np.load(fp)["data"].astype(float)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_task_curves(
    data_root: Path,
    algo: str,
    method: str,
    level: int,
    agents: int,
    strategy: str,
    seq_len: int,
    seeds: List[int],
    focal_task: int,
) -> dict:
    """
    For a single focal task, load soup + heterogeneity series for every seed.

    Returns dict with keys 'soup' and 'het', each a list of 1-D np.arrays
    (one per available seed).
    """
    base = (
        data_root / algo / method
        / f"level_{level}" / f"agents_{agents}"
        / f"{strategy}_{seq_len}"
    )
    soup_seeds, het_seeds = [], []
    for seed in seeds:
        sd = base / f"seed_{seed}"
        s = load_series(sd / f"{focal_task}_soup.json")
        h = load_series(sd / f"{focal_task}_heterogeneity.json")
        if s is not None and h is not None:
            soup_seeds.append(s)
            het_seeds.append(h)
    return {"soup": soup_seeds, "het": het_seeds}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _align_and_mean_ci(series_list: List[np.ndarray], target_len: int):
    """
    Interpolate every series to *target_len* points, return (mean, lo, hi).
    lo/hi are the 95 % CI bounds.
    """
    if not series_list:
        nans = np.full(target_len, np.nan)
        return nans, nans, nans

    resampled = []
    x_out = np.linspace(0, 1, target_len)
    for s in series_list:
        s = np.array(s, dtype=float)
        if len(s) < 2:
            continue
        x_in = np.linspace(0, 1, len(s))
        resampled.append(np.interp(x_out, x_in, s))

    if not resampled:
        nans = np.full(target_len, np.nan)
        return nans, nans, nans

    arr = np.stack(resampled)          # (n_seeds, target_len)
    mean = np.nanmean(arr, axis=0)
    if arr.shape[0] > 1:
        ci = 1.96 * np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    else:
        ci = np.zeros_like(mean)
    return mean, mean - ci, mean + ci


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(
    data_root: Path,
    algo: str,
    methods: List[str],
    level: int,
    agents: int,
    strategy: str,
    seq_len: int,
    seeds: List[int],
    focal_tasks: List[int],
    smooth: int,
    out: Path,
    points: int = 300,
):
    n_rows = len(focal_tasks)
    n_cols = len(methods)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.6 * n_rows),
        sharex=False, sharey=True,
        squeeze=False,
    )

    # x-axis: normalised to [0, seq_len] so task boundaries are at integers
    x = np.linspace(0, seq_len, points)

    # Where each focal task's training ends (unit = 1 task)
    def end_of_task(task_idx: int) -> float:
        return float(task_idx + 1)

    for row, focal_task in enumerate(focal_tasks):
        for col, method in enumerate(methods):
            ax = axes[row][col]
            color = METHOD_COLORS.get(method, "#444444")
            display = METHOD_DISPLAY_NAMES.get(method, method)

            data = load_task_curves(
                data_root, algo, method, level, agents,
                strategy, seq_len, seeds, focal_task,
            )

            for metric, linestyle, label in [
                ("soup", "-",  "Soup score"),
                ("het",  "--", "Heterogeneity"),
            ]:
                mean, lo, hi = _align_and_mean_ci(data[metric], points)

                if smooth > 1:
                    from scipy.ndimage import gaussian_filter1d
                    sigma = smooth / 6
                    mean = gaussian_filter1d(mean, sigma)
                    lo   = gaussian_filter1d(lo,   sigma)
                    hi   = gaussian_filter1d(hi,   sigma)

                ax.plot(x, mean, linestyle=linestyle, color=color,
                        linewidth=1.4, label=label)
                ax.fill_between(x, lo, hi, color=color, alpha=0.15)

            # Vertical line: end of focal task's training
            task_end = end_of_task(focal_task)
            ax.axvline(task_end, color="black", linestyle=":", linewidth=1.0,
                       label="End of task training")

            ax.set_ylim(-0.02, 1.05)
            ax.set_xlim(0, seq_len)

            # Integer task-boundary ticks
            ax.set_xticks(range(0, seq_len + 1, max(1, seq_len // 5)))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda val, _: f"T{int(val)}"
            ))

            # Column header (method name) on top row
            if row == 0:
                ax.set_title(display, fontsize=9, fontweight="bold")

            # Row label (focal task) on leftmost column
            if col == 0:
                ax.set_ylabel(f"Task {focal_task}", fontsize=8)

            # Legend only on first subplot
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

    fig.text(0.5, -0.01, "Training task", ha="center", fontsize=9)
    fig.suptitle(
        f"Soup score vs. heterogeneity — Level {level}  (seeds {seeds})",
        fontsize=10, y=1.01,
    )

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Heterogeneity vs soup score over the task sequence")
    p.add_argument("--data_root", default="experiments/results/data")
    p.add_argument("--algo", default="ippo")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--strategy", default="generate")
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--focal_tasks", type=int, nargs="+", default=[0, 4, 9],
                   help="Which task indices to show (0-based)")
    p.add_argument("--smooth", type=int, default=10,
                   help="Gaussian smoothing window (in data points); 1 = no smoothing")
    p.add_argument("--out", type=Path, default=Path("runs/het_vs_perf.pdf"))
    args = p.parse_args()

    plot(
        data_root=Path(args.data_root),
        algo=args.algo,
        methods=args.methods,
        level=args.level,
        agents=args.agents,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
        focal_tasks=args.focal_tasks,
        smooth=args.smooth,
        out=args.out,
    )
