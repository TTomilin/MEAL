#!/usr/bin/env python3
"""Plot cumulative average performance for partner-adaptation (BR) experiments.

Data layout (new):
  data/ppo/<method>/<layout_name>/<arch>/partners_<N>/seed_<seed>/
    eval_partner_0_soup.json  ...  eval_partner_{N-1}_soup.json

Curves represent different CL methods (--compare_by method) or
multihead vs singlehead (--compare_by arch).
When multiple --layout_names are given, results are averaged across layouts.

Usage examples
--------------
Compare methods on a single layout:
  python cumulative_eval_partners.py --layout_names cramped_room --methods FT EWC Online_EWC L2

Compare across all four layouts:
  python cumulative_eval_partners.py \
    --layout_names cramped_room coord_ring counter_circuit asymm_advantages \
    --methods FT EWC Online_EWC L2

Compare multihead vs singlehead:
  python cumulative_eval_partners.py --compare_by arch --method Online_EWC \
    --layout_names cramped_room
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from experiments.results.plotting.utils import (
    collect_br_cumulative_runs,
    setup_figure,
    add_task_boundaries,
    setup_task_axes,
    smooth_and_ci,
    save_plot,
    finalize_plot,
    METHOD_COLORS,
    method_display_name,
)

# ────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_LAYOUTS  = ["cramped_room", "coord_ring", "counter_circuit", "asymm_advantages"]
DEFAULT_ARCH     = "multihead"
DEFAULT_PARTNERS = 8
DEFAULT_METHODS  = ["FT", "EWC", "Online_EWC", "L2", "AGEM"]

ARCH_COLORS = {
    "multihead":  "#12939A",
    "singlehead": "#FF6E54",
}
ARCH_DISPLAY = {
    "multihead":  "Multi-head",
    "singlehead": "Single-head",
}

# ────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot cumulative partner-adaptation performance.",
    )
    p.add_argument("--data_root",     default="data",           help="Root folder for downloaded data")
    p.add_argument("--layout_names",  nargs="+", default=DEFAULT_LAYOUTS,
                   help="Overcooked layout name(s); results averaged across layouts if multiple")
    p.add_argument("--arch",          default=DEFAULT_ARCH,     choices=["multihead", "singlehead"],
                   help="Network architecture (used when compare_by=method)")
    p.add_argument("--num_partners",  type=int, default=DEFAULT_PARTNERS, help="Number of eval partners")
    p.add_argument("--steps_per_task", type=float, default=1e7,
                   help="Training steps per partner task (x-axis scaling)")

    p.add_argument("--compare_by",   choices=["method", "arch"], default="method",
                   help="What the curves represent")
    p.add_argument("--methods",      nargs="+", default=DEFAULT_METHODS,
                   help="CL methods to compare (used when --compare_by method)")
    p.add_argument("--method",       default="Online_EWC",
                   help="Single method when --compare_by arch")

    p.add_argument("--seeds",        type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--sigma",        type=float, default=1.5)
    p.add_argument("--confidence",   type=float, default=0.95, choices=[0.9, 0.95, 0.99])
    p.add_argument("--legend_anchor", type=float, default=0.79)
    p.add_argument("--plot_name",    default=None)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION
# ────────────────────────────────────────────────────────────────────────────

def _collect_across_layouts(
    data_root: Path,
    method: str,
    layout_names: List[str],
    arch: str,
    num_partners: int,
    seeds: List[int],
) -> np.ndarray:
    """Collect and average data across all specified layouts. Returns (S, T) or empty."""
    per_layout = []
    for layout in layout_names:
        data = collect_br_cumulative_runs(
            data_root, method, layout, arch, num_partners, seeds,
        )
        if data.size > 0:
            per_layout.append(data)

    if not per_layout:
        return np.empty((0, 0))

    # Align lengths then average across layouts
    T = max(d.shape[1] for d in per_layout)
    padded = [np.pad(d, ((0, 0), (0, T - d.shape[1])), constant_values=np.nan)
              for d in per_layout]
    return np.nanmean(np.stack(padded, axis=0), axis=0)  # (S, T)


def _collect_curve(
    data_root: Path,
    method: str,
    layout_names: List[str],
    arch: str,
    num_partners: int,
    seeds: List[int],
    steps_per_task: float,
    sigma: float,
    confidence: float,
    label: str,
    color: Optional[str],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str, Optional[str]]]:
    data = _collect_across_layouts(data_root, method, layout_names, arch, num_partners, seeds)
    if data.size == 0:
        print(f"[warn] no data for method={method} arch={arch} layouts={layout_names}")
        return None

    mu, ci = smooth_and_ci(data, sigma, confidence)
    x = np.linspace(0, num_partners * steps_per_task, len(mu))
    return x, mu, ci, label, color


# ────────────────────────────────────────────────────────────────────────────
# PLOT
# ────────────────────────────────────────────────────────────────────────────

def _plot_curve(ax, x, mu, ci, label, color):
    (ln,) = ax.plot(x, mu, label=label, color=color)
    ax.fill_between(x, mu - ci, mu + ci, color=ln.get_color(), alpha=0.20)


def _apply_decorations(ax, args, n_legend_items):
    spt = args.steps_per_task
    n = args.num_partners
    boundaries = [i * spt for i in range(n + 1)]

    add_task_boundaries(ax, boundaries, color="grey", linewidth=0.5)
    setup_task_axes(ax, boundaries, n, fontsize=8)

    finalize_plot(
        ax,
        xlabel="Environment Steps",
        ylabel="Average Normalized Soup",
        xlim=(0, n * spt),
        ylim=(0, None),
        legend_loc="lower center",
        legend_bbox_to_anchor=(0.5, args.legend_anchor),
        legend_ncol=n_legend_items,
    )


def plot():
    args = _parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root

    if args.compare_by == "method":
        items = [
            (m, args.arch, method_display_name(m), METHOD_COLORS.get(m))
            for m in args.methods
        ]
    else:
        items = [
            (args.method, arch, ARCH_DISPLAY.get(arch, arch), ARCH_COLORS.get(arch))
            for arch in ["multihead", "singlehead"]
        ]

    curves = []
    for method, arch, label, color in items:
        result = _collect_curve(
            data_root=data_root,
            method=method,
            layout_names=args.layout_names,
            arch=arch,
            num_partners=args.num_partners,
            seeds=args.seeds,
            steps_per_task=args.steps_per_task,
            sigma=args.sigma,
            confidence=args.confidence,
            label=label,
            color=color,
        )
        if result is not None:
            curves.append(result)

    if not curves:
        print("[error] no data collected — check paths and arguments")
        return

    fig, ax = setup_figure(width=10, height=3)
    for x, mu, ci, label, color in curves:
        _plot_curve(ax, x, mu, ci, label, color)
    _apply_decorations(ax, args, len(curves))

    out_dir = Path(__file__).resolve().parent.parent / "plots"
    if args.plot_name:
        stem = args.plot_name
    else:
        by = "methods" if args.compare_by == "method" else "arch"
        layout_str = "_".join(args.layout_names) if len(args.layout_names) == 1 else "all_layouts"
        stem = f"ppo_cumulative_{by}_{layout_str}_{args.arch}_partners{args.num_partners}"

    save_plot(fig, out_dir, stem)
    plt.show()


if __name__ == "__main__":
    plot()
