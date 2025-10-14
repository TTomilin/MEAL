#!/usr/bin/env python3
"""Plot a *single* aggregated bar per item (method or level) showing final performance.

This replaces the earlier histogram version – now we collapse each seed’s final
score to its mean, then take the *sample mean across seeds* for one scalar per
item, with a 95 % confidence interval as an error‑bar.

Two comparison modes (mirrors ``plot_cumulative.py``):

1. **Method comparison** ``--compare_by method`` *(default)*
   One bar per continual‑learning *method* on the same level.

2. **Level comparison** ``--compare_by level``
   One bar per *task level* for a fixed method.

Directory layout must be compatible with the repo‑download script, i.e.
``results/data/<algo>/<method>/<level>/<strategy>_<seq_len>/seed_<seed>/``.

Usage examples
--------------
*Compare EWC vs AGEM on level‑3 tasks*
```
python plot_final_bar.py \
  --algo ippo \
  --methods EWC AGEM \
  --strategy generate \
  --seq_len 10 \
  --data_root results/data
```

*Compare task levels for EWC*
```
python plot_final_bar.py \
  --compare_by level \
  --method EWC \
  --levels 1 2 3 \
  --algo ippo \
  --strategy generate \
  --seq_len 10 \
  --data_root results/data
```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.results.plotting.utils import (
    collect_cumulative_runs,
    setup_figure,
    save_plot,
    finalize_plot,
    METHOD_COLORS,
    LEVEL_COLORS,
    create_eval_parser,
    CRIT,
)

# ────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = create_eval_parser(
        description="Plot bar chart of final cumulative scores",
        metric_choices=["reward", "soup"],
    )
    p.set_defaults(metric="soup")

    p.add_argument(
        "--compare_by",
        choices=["method", "level"],
        default="method",
        help="What each bar represents.",
    )
    p.add_argument(
        "--method",
        default="EWC",
        help="Fixed method when --compare_by level.",
    )
    p.add_argument(
        "--levels",
        nargs="+",
        default=[1, 2, 3],
        help="Task levels to compare when --compare_by level.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=10,
        help="Number of last points to average for the final score.",
    )
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────
# DATA GATHERING
# ────────────────────────────────────────────────────────────────────────────

def _final_scores(
        root: Path,
        algo: str,
        method: str,
        experiment: str,
        strategy: str,
        metric: str,
        seq_len: int,
        seeds: Sequence[int],
        window: int,
) -> np.ndarray:
    """Return 1D array (one element per seed) of average of last *window* points."""
    curves = collect_cumulative_runs(
        root,
        algo,
        method,
        experiment,
        strategy,
        metric,
        seq_len,
        list(seeds),
    )  # (S, T)
    if curves.size == 0:
        return np.array([])
    window = min(window, curves.shape[1])
    finals = np.nanmean(curves[:, -window:], axis=1)
    return finals[np.isfinite(finals)]

# ────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ────────────────────────────────────────────────────────────────────────────

def _plot_bars(ax: plt.Axes, data: dict[str, np.ndarray], compare_by: str, mean_values = False):
    labels = list(data.keys())
    means = [np.mean(v) if len(v) else np.nan for v in data.values()]
    cis = [
        (CRIT[0.95] * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        for v in data.values()
    ]
    x = np.arange(len(labels))
    colors = [
        METHOD_COLORS.get(lbl) if compare_by == "method" else LEVEL_COLORS.get(lbl)
        for lbl in labels
    ]
    bars = ax.bar(x, means, yerr=cis, capsize=5, color=colors, edgecolor="black")

    # Annotate mean values on top of bars
    if mean_values:
        for rect, val in zip(bars, means):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, height, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)

# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def plot():
    args = _parse_args()

    data_root = Path(__file__).resolve().parent.parent / args.data_root

    if args.compare_by == "method":
        items = args.methods
    else:
        items = [f"Level {lvl}" for lvl in args.levels]

    data: dict[str, np.ndarray] = {}
    for item in items:
        if args.compare_by == "method":
            method = item
            experiment = f"level_{args.level}"
        else:
            method = args.method
            experiment = f"level_{item.split()[1]}"
        vals = _final_scores(
            data_root,
            algo=args.algo,
            method=method,
            experiment=experiment,
            strategy=args.strategy,
            metric=args.metric,
            seq_len=args.seq_len,
            seeds=args.seeds,
            window=args.window,
        )
        data[item] = vals

    fig, ax = setup_figure(width=1.5 * len(items), height=2.5)
    _plot_bars(ax, data, compare_by=args.compare_by)

    finalize_plot(
        ax,
        title=None,
        xlabel="",
        ylabel="Final Average Score",
    )

    out_dir = Path(__file__).resolve().parent.parent / "plots"
    stem = args.plot_name or (
            "bar_final_" + ("methods" if args.compare_by == "method" else "levels")
    )
    # Add level suffix if not already present
    if "_level" not in stem:
        stem += f"_level_{args.level}"
    save_plot(fig, out_dir, stem)
    plt.show()


if __name__ == "__main__":
    plot()
