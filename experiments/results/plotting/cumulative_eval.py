#!/usr/bin/env python3
"""Plot cumulative average performance for the MARL continual‑learning benchmark.

The script can now work in **two modes**:

1. **Method comparison**  (default)
   Same behaviour as the original script – curves for several
   continual‑learning methods on the *same* task level directory.

2. **Level comparison**   (``--compare_by level``)
   Show how one particular method (``--method``) behaves on multiple task
   *levels* (typically ``level_2`` vs ``level_3``).

The directory structure expected on disk is compatible with the download
script we created earlier:

```
results/data/<algo>/<method>/<level>/<strategy>_<seq_len>/seed_<seed>/
```

Usage examples
--------------

*Compare EWC vs AGEM on level‑3 tasks*
```
python plot_cumulative.py \
  --algo ippo \
  --methods EWC AGEM \
  --strategy generate \
  --seq_len 10 \
```

*Compare level-1 vs level‑2 vs level‑3 for EWC*
```
python plot_cumulative.py \
  --compare_by level \
  --method EWC \
  --levels 1 2 3 \
  --algo ippo \
  --strategy generate \
  --seq_len 10 \
```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from experiments.results.plotting.utils import (
    collect_cumulative_runs,
    setup_figure,
    add_task_boundaries,
    setup_task_axes,
    smooth_and_ci,
    save_plot,
    finalize_plot,
    METHOD_COLORS,
    LEVEL_COLORS,
    create_eval_parser,
)


# ────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = create_eval_parser(
        description="Plot cumulative average performance for MARL continual‑learning benchmark",
        metric_choices=["reward", "soup"],
    )
    p.set_defaults(metric="soup")  # override utils default

    # Mode flag
    p.add_argument(
        "--compare_by",
        choices=["method", "level"],
        default="method",
        help="What the curves represent on the plot.",
    )
    # When comparing levels we want a fixed method and a list of levels.
    p.add_argument(
        "--method",
        default="EWC",
        help="Continual‑learning method to plot when --compare_by level",
    )
    p.add_argument(
        "--levels",
        nargs="+",
        default=[1, 2, 3],
        help="Which task‑level sub‑folders to include (only used when --compare_by level).",
    )

    # Fine‑tuning of the legend
    p.add_argument("--legend_anchor", type=float, default=0.79)

    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# MAIN PLOTTING LOGIC
# ────────────────────────────────────────────────────────────────────────────

def _plot_curve(
        ax: plt.Axes,
        x: np.ndarray,
        mu: np.ndarray,
        ci: np.ndarray,
        label: str,
        color: str | None,
):
    # plot returns a list; unpack the first Line2D object
    (ln,) = ax.plot(x, mu, label=label, color=color)
    ax.fill_between(x, mu - ci, mu + ci, color=ln.get_color(), alpha=0.20)


def _collect_and_plot(
        ax: plt.Axes,
        label: str,
        data_root: Path,
        algo: str,
        method: str,
        experiment: str,
        strategy: str,
        metric: str,
        seq_len: int,
        seeds: List[int],
        steps_per_task: int,
        sigma: float,
        confidence: float,
        compare_by: str = "method",
):
    data = collect_cumulative_runs(
        data_root,
        algo,
        method,
        experiment,
        strategy,
        metric,
        seq_len,
        seeds,
    )
    if len(data) == 0:
        print(f"[warn] no data for {method}")
        return

    mu, ci = smooth_and_ci(data, sigma, confidence)
    x = np.linspace(0, seq_len * steps_per_task, len(mu))
    # Use METHOD_COLORS only when comparing methods, not when comparing levels
    color = METHOD_COLORS.get(label, None) if compare_by == "method" else LEVEL_COLORS.get(label, None)
    _plot_curve(ax, x, mu, ci, label, color)


def plot():
    args = _parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root

    total_steps = args.seq_len * args.steps_per_task
    fig, ax = setup_figure(width=10, height=3)

    # Unified approach for both method and level comparison
    items_to_plot = args.methods if args.compare_by == "method" else args.levels

    for item in items_to_plot:
        if args.compare_by == "method":
            # Method comparison
            method_name = item
            label = method_name
            experiment = f"level_{args.level}"
        else:
            # Level comparison
            level_num = item
            level_str = f"level_{level_num}"
            label = level_str.replace('_', ' ').title()  # e.g., "Level 1"
            method_name = args.method
            experiment = level_str

        _collect_and_plot(
            ax,
            label=label,
            data_root=data_root,
            algo=args.algo,
            method=method_name,
            experiment=experiment,
            strategy=args.strategy,
            metric=args.metric,
            seq_len=args.seq_len,
            seeds=args.seeds,
            steps_per_task=args.steps_per_task,
            sigma=args.sigma,
            confidence=args.confidence,
            compare_by=args.compare_by,
        )

    # Add task boundaries and nice axes.
    boundaries = [i * args.steps_per_task for i in range(args.seq_len + 1)]
    if args.seq_len <= 20:
        add_task_boundaries(ax, boundaries, color="grey", linewidth=0.5)
        setup_task_axes(ax, boundaries, args.seq_len, fontsize=8)
    else:
        # ── many tasks: keep all thin boundaries, but de-clutter labels
        add_task_boundaries(ax, boundaries, color="grey", linewidth=0.4)

        # Bottom: show fewer x ticks (environment steps)
        from matplotlib.ticker import MaxNLocator, ScalarFormatter
        ax.set_xlim(0, total_steps)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))   # ~5–6 ticks
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.tick_params(axis="x", labelsize=12)

        # Left y stays as-is
        ax.tick_params(axis="y", labelsize=12)

        # Top: show every 10th task index (at task centers)
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())

        steps = args.steps_per_task
        centers = (np.arange(args.seq_len) + 0.5) * steps
        # every 10th task: 10, 20, 30, ... (1-indexed in labels)
        keep_idx = np.arange(9, args.seq_len, 10)  # 9→task 10, 19→task 20, ...
        top_ticks = centers[keep_idx]
        top_labels = [f"Task {idx+1}" for idx in keep_idx]  # 1-indexed label

        ax_top.set_xticks(top_ticks)
        ax_top.set_xticklabels(top_labels)
        ax_top.tick_params(axis="x", labelsize=11, pad=2)
        # no minor ticks up top
        ax_top.xaxis.set_minor_locator(plt.NullLocator())

    # Final decorations.
    legend_items = args.methods if args.compare_by == "method" else args.levels
    finalize_plot(
        ax,
        xlabel="Environment Steps",
        ylabel="Average Normalized Score",
        xlim=(0, total_steps),
        ylim=(0, None),
        legend_loc="lower center",
        legend_bbox_to_anchor=(0.5, args.legend_anchor),
        legend_ncol=len(legend_items),
    )

    # Save figure
    out_dir = Path(__file__).resolve().parent.parent / "plots"
    stem = args.plot_name or (
            "avg_cumulative_" + ("methods" if args.compare_by == "method" else "levels")
    )
    stem += f"_seq_{args.seq_len}"
    # Add level suffix if not already present
    if "_level" not in stem:
        stem += f"_level_{args.level}"
    save_plot(fig, out_dir, stem)

    plt.show()


if __name__ == "__main__":
    plot()
