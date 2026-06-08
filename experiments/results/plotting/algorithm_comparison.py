#!/usr/bin/env python3
"""
Heatmap: MARL algorithm (columns) × CL method (rows), one panel per difficulty level.
Cell colour = Average Performance (soup); cell text = mean ± CI.

Usage:
    python -m experiments.results.plotting.algorithm_comparison \
        --data_root results/data \
        --algorithms ippo mappo happo vdn qmix \
        --methods FT L2 EWC MAS Online_EWC AGEM \
        --levels 1 2 3 \
        --seeds 1 2 3
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from experiments.results.numerical.algorithm_comparison import compute_metrics
from experiments.results.plotting.utils import save_plot, create_base_parser, method_display_name

ALGO_DISPLAY = {
    "ippo":  "IPPO",
    "mappo": "MAPPO",
    "happo": "HAPPO",
    "vdn":   "VDN",
    "qmix":  "QMIX",
}

LEVEL_LABELS = {1: "Easy", 2: "Medium", 3: "Hard"}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_grid(data_root, algorithms, methods, strategy, seq_len, seeds, levels, agents, window):
    grid = {}
    for level in levels:
        grid[level] = {}
        for method in methods:
            grid[level][method] = {}
            for algo in algorithms:
                grid[level][method][algo] = compute_metrics(
                    data_root=data_root, algo=algo, method=method,
                    strategy=strategy, seq_len=seq_len, seeds=seeds,
                    level=level, agents=agents, end_window_evals=window,
                )
    return grid


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(args):
    data_root = Path(__file__).resolve().parent.parent / args.data_root
    grid = _load_grid(
        data_root=data_root, algorithms=args.algorithms, methods=args.methods,
        strategy=args.strategy, seq_len=args.seq_len, seeds=args.seeds,
        levels=args.levels, agents=args.agents, window=args.window,
    )

    n_levels = len(args.levels)
    n_methods = len(args.methods)
    n_algos = len(args.algorithms)

    cell_w = max(1.0, 2.25 / n_methods)
    cell_h = max(0.55, 1.5 / n_algos)
    fig, axes = plt.subplots(
        1, n_levels,
        figsize=(n_levels * n_methods * cell_w + 0.8, n_algos * cell_h + 1.5),
        squeeze=False,
    )

    # Shared colour scale across all levels
    all_vals = [
        grid[lv][m][a]["AP"]
        for lv in args.levels
        for m in args.methods
        for a in args.algorithms
        if not np.isnan(grid[lv][m][a]["AP"])
    ]
    vmin, vmax = (min(all_vals), max(all_vals)) if all_vals else (0, 1)

    cmap = plt.get_cmap("YlGn")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for col, level in enumerate(args.levels):
        ax = axes[0, col]

        # rows = algos, cols = methods  (transposed from original)
        mat = np.array([
            [grid[level][m][a]["AP"] for m in args.methods]
            for a in args.algorithms
        ])
        ci_mat = np.array([
            [grid[level][m][a]["AP_CI"] for m in args.methods]
            for a in args.algorithms
        ])

        ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        for r in range(n_algos):
            for c in range(n_methods):
                val = mat[r, c]
                ci  = ci_mat[r, c]
                if np.isnan(val):
                    txt = "—"
                    color = "grey"
                else:
                    txt = f"{val:.2f}"
                    if not np.isnan(ci) and ci > 0:
                        txt += f"\n±{ci:.2f}"
                    brightness = norm(val)
                    color = "white" if brightness > 0.55 else "black"
                ax.text(c, r, txt, ha="center", va="center",
                        fontsize=8, color=color, linespacing=1.3)

        # x-axis: CL methods (top)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels([method_display_name(m) for m in args.methods], fontsize=10)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", which="both", bottom=False, top=False)

        # y-axis: algos (left, first panel only)
        if col == 0:
            ax.set_yticks(range(n_algos))
            ax.set_yticklabels(
                [ALGO_DISPLAY.get(a, a.upper()) for a in args.algorithms],
                fontsize=10,
            )
        else:
            ax.set_yticks([])

        ax.set_xticks(np.arange(n_methods) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_algos) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)

    fig.tight_layout(rect=[-0.02, -0.03, 1.02, 1.02])

    out_dir = Path(__file__).resolve().parent.parent / "plots"
    save_plot(fig, out_dir, args.plot_name or "algorithm_comparison")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    p = create_base_parser("Heatmap: MARL algorithm × CL method average performance.")
    p.add_argument("--data_root",  default="data")
    p.add_argument("--algorithms", nargs="+",
                   default=["ippo", "mappo", "happo", "vdn", "qmix"])
    p.add_argument("--methods",    nargs="+", required=True)
    p.add_argument("--strategy",   default="generate")
    p.add_argument("--seq_len",    type=int, default=20)
    p.add_argument("--agents",     type=int, default=2)
    p.add_argument("--seeds",      type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--levels",     type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--window",     type=int, default=10)
    p.add_argument("--plot_name",  default=None)
    return p.parse_args()


if __name__ == "__main__":
    plot(_cli())
