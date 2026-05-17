#!/usr/bin/env python3
"""Bar-chart comparing partner-adaptation CL methods across architectures.

Layout: 2 rows × 4 columns.
  - Rows   : metric (Average Performance top, Forgetting bottom)
  - Columns: one Overcooked layout each

Each subplot has 6 grouped bars — 3 methods × 2 architectures:
  - Color   : method  (reuses project METHOD_COLORS)
  - Hatch   : multi-head = solid fill, single-head = hatched (//)

Usage
-----
python partner_adaptation_arch_hist.py                   # all four layouts
python partner_adaptation_arch_hist.py --layout_names cramped_room coord_ring
python partner_adaptation_arch_hist.py --no_ci           # hide error bars
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from experiments.results.numerical.partner_adaption_arch_table import (
    compute_metrics,
    LAYOUT_DISPLAY,
    DEFAULT_LAYOUT_NAMES,
    DEFAULT_METHODS,
    DEFAULT_PARTNERS,
    DEFAULT_SEEDS,
    ARCHS,
)
from experiments.results.plotting.utils import METHOD_COLORS, METHOD_DISPLAY_NAMES, save_plot

# ── visual constants ──────────────────────────────────────────────────────────

HATCH = {"multihead": "", "singlehead": "////"}
ARCH_LABEL = {"multihead": "Multi-head", "singlehead": "Single-head"}
BAR_WIDTH = 0.35
GROUP_GAP = 0.15       # extra space between method groups
EDGE_COLOR = "black"
EDGE_LW = 0.6
FALLBACK_COLOR = "#888888"


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, FALLBACK_COLOR)


def _method_label(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method)


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_subplot(
    ax: plt.Axes,
    df,
    methods: List[str],
    metric: str,
    show_ci: bool,
    y_label: bool,
) -> None:
    """Draw 6 grouped bars (3 methods × 2 archs) onto *ax*."""
    n_archs = len(ARCHS)
    group_width = n_archs * BAR_WIDTH + GROUP_GAP
    x_centers = np.arange(len(methods)) * group_width

    for method_idx, method in enumerate(methods):
        row = df[df["Method"] == method].iloc[0]
        x_group = x_centers[method_idx]
        offsets = np.linspace(
            -(n_archs - 1) * BAR_WIDTH / 2,
             (n_archs - 1) * BAR_WIDTH / 2,
            n_archs,
        )
        for arch_idx, arch in enumerate(ARCHS):
            mean = row[f"{arch}_{metric}"]
            ci   = row[f"{arch}_{metric}_ci"] if show_ci else 0.0
            color = _method_color(method)
            hatch = HATCH[arch]
            ax.bar(
                x_group + offsets[arch_idx],
                mean if np.isfinite(mean) else 0,
                width=BAR_WIDTH,
                yerr=ci if (show_ci and np.isfinite(ci) and ci > 0) else None,
                capsize=3,
                color=color,
                hatch=hatch,
                edgecolor=EDGE_COLOR,
                linewidth=EDGE_LW,
                error_kw={"elinewidth": 0.8, "ecolor": "black"},
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([_method_label(m) for m in methods], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_label:
        label = r"Avg. Norm. Soup  $\mathcal{S} ↑$" if metric == "AP" else r"Forgetting  $\mathcal{F} ↓$"
        ax.set_ylabel(label, fontsize=11)


def build_figure(
    layout_dfs: Dict[str, object],
    layout_names: List[str],
    methods: List[str],
    show_ci: bool,
) -> plt.Figure:
    n_layouts = len(layout_names)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_layouts,
        figsize=(2.8 * n_layouts, 5.0),
        constrained_layout=True,
    )
    if n_layouts == 1:
        axes = axes.reshape(2, 1)

    metric_rows = [("AP", r"Avg. Performance"), ("F", r"Forgetting")]

    for col, layout in enumerate(layout_names):
        df = layout_dfs[layout]
        display = LAYOUT_DISPLAY.get(layout, layout.replace("_", " ").title())
        for row, (metric, _) in enumerate(metric_rows):
            ax = axes[row, col]
            _plot_subplot(ax, df, methods, metric, show_ci, y_label=(col == 0))
            if row == 0:
                ax.set_title(display, fontsize=9, fontweight="bold")
            if metric == "F":
                ax.set_ylim(bottom=0)

    # ── arch legend in bottom-left subplot only ─────────────────────────────────
    arch_handles = [
        mpatches.Patch(facecolor="white", edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
                       hatch=HATCH[arch], label=ARCH_LABEL[arch])
        for arch in ARCHS
    ]
    axes[1, 0].legend(
        handles=arch_handles,
        loc="upper right",
        fontsize=10,
        frameon=True,
        framealpha=0.8,
        edgecolor="none",
    )

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Partner-adaptation architecture comparison bar chart.",
    )
    p.add_argument("--data_root",    default="results/data")
    p.add_argument("--layout_names", nargs="+", default=DEFAULT_LAYOUT_NAMES)
    p.add_argument("--methods",      nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--num_partners", type=int,  default=DEFAULT_PARTNERS)
    p.add_argument("--seeds",        type=int,  nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--end_window",   type=int,  default=10)
    p.add_argument("--no_ci",        action="store_true", help="Hide error bars")
    p.add_argument("--plot_name",    default="partner_adaptation_arch_hist")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parents[2] / args.data_root

    layout_dfs = {}
    for layout in args.layout_names:
        layout_dfs[layout] = compute_metrics(
            data_root=data_root,
            layout_name=layout,
            methods=args.methods,
            num_partners=args.num_partners,
            seeds=args.seeds,
            end_window=args.end_window,
        )

    fig = build_figure(layout_dfs, args.layout_names, args.methods, not args.no_ci)

    out_dir = Path(__file__).resolve().parent.parent / "plots"
    save_plot(fig, out_dir, args.plot_name)
    plt.show()


if __name__ == "__main__":
    main()
