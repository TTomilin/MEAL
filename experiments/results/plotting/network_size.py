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
        strategy: str,
        metric: str,
        seq_len: int,
        seeds: List[int],
        steps_per_task: int,
        sigma: float,
        confidence: float,
        level: int,
        agents: int,
        experiment: str,
):
    data = collect_cumulative_runs(
        data_root,
        algo,
        method,
        strategy,
        metric,
        seq_len,
        seeds,
        level,
        agents,
        experiment,
    )
    if len(data) == 0:
        print(f"[warn] no data for {method}")
        return

    mu, ci = smooth_and_ci(data, sigma, confidence)
    x = np.linspace(0, seq_len * steps_per_task, len(mu))
    _plot_curve(ax, x, mu, ci, label, None)


def plot():
    args = _parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root

    total_steps = args.seq_len * args.steps_per_task
    fig, ax = setup_figure(width=10, height=3)

    level_str = f"level_{args.level}"
    method_name = args.method

    experiments = ["big_network", "orig_network"]

    for experiment in experiments:
        label = experiment.replace('_', ' ').title()
        _collect_and_plot(
            ax,
            label=label,
            data_root=data_root,
            algo=args.algo,
            method=method_name,
            strategy=args.strategy,
            metric=args.metric,
            seq_len=args.seq_len,
            seeds=args.seeds,
            steps_per_task=args.steps_per_task,
            sigma=args.sigma,
            confidence=args.confidence,
            level=args.level,
            agents=args.agents,
            experiment=experiment,
        )

    # Add task boundaries and nice axes.
    boundaries = [i * args.steps_per_task for i in range(args.seq_len + 1)]
    add_task_boundaries(ax, boundaries, color="grey", linewidth=0.5)
    setup_task_axes(ax, boundaries, args.seq_len, fontsize=8)

    # Final decorations.
    finalize_plot(
        ax,
        xlabel="Environment Steps",
        ylabel="Average Normalized Score",
        xlim=(0, total_steps),
        ylim=(0, None),
        legend_loc="lower center",
        legend_bbox_to_anchor=(0.5, args.legend_anchor),
        legend_ncol=len(experiments),
    )

    # Save figure
    out_dir = Path(__file__).resolve().parent.parent / "plots"
    stem = args.plot_name or "networks"
    save_plot(fig, out_dir, stem)
    plt.show()


if __name__ == "__main__":
    plot()
