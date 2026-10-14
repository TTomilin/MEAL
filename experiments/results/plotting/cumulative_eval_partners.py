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

CSV mode (local run directories written by experiments/partner_adaptation/run_br.py):
  python -m experiments.results.plotting.cumulative_eval_partners --run_root cpa_results/runs \
    --layout_names cramped_room --methods Online_EWC --identity available --arch multihead \
    --output_dir /tmp/cpa_figs --no_show
  Mean team soups over all 24 partners at the measured full-bank evaluations, raw (see experiments/results/README.MD).
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
from experiments.results.plotting.utils import cpa_csv

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
    p.add_argument("--layout_names",  nargs="+", default=None,
                   help=f"Overcooked layout name(s); results averaged across layouts if multiple "
                        f"(legacy default: {' '.join(DEFAULT_LAYOUTS)}; CSV mode: exactly one, required)")
    p.add_argument("--arch",          default=DEFAULT_ARCH,     choices=["multihead", "singlehead"],
                   help="Network architecture (used when compare_by=method)")
    p.add_argument("--num_partners",  type=int, default=None,
                   help=f"Number of eval partners (legacy default {DEFAULT_PARTNERS}; "
                        f"CSV mode {cpa_csv.DEFAULT_CSV_PARTNERS})")
    p.add_argument("--steps_per_task", type=float, default=1e7,
                   help="Training steps per partner task (x-axis scaling)")

    p.add_argument("--compare_by",   choices=["method", "arch"], default="method",
                   help="What the curves represent")
    p.add_argument("--methods",      nargs="+", default=None,
                   help=f"CL methods to compare (used when --compare_by method; legacy default: "
                        f"{' '.join(DEFAULT_METHODS)}; CSV mode: {' '.join(cpa_csv.CSV_METHODS)})")
    p.add_argument("--method",       default="Online_EWC",
                   help="Single method when --compare_by arch")

    p.add_argument("--seeds",        type=int, nargs="+", default=None,
                   help="Seeds (legacy default 1 2 3 4 5; CSV mode: one seed, default 0)")
    p.add_argument("--sigma",        type=float, default=None,
                   help="Smoothing (legacy default 1.5; CSV mode 0 = none)")
    p.add_argument("--confidence",   type=float, default=0.95, choices=[0.9, 0.95, 0.99])
    p.add_argument("--legend_anchor", type=float, default=0.79)
    p.add_argument("--plot_name",    default=None)
    cpa_csv.add_csv_args(p)
    args = p.parse_args()
    if args.run_root is None:  # legacy mode keeps its original defaults
        args.layout_names = args.layout_names or DEFAULT_LAYOUTS
        args.num_partners = args.num_partners or DEFAULT_PARTNERS
        args.methods = args.methods or DEFAULT_METHODS
        args.seeds = args.seeds or [1, 2, 3, 4, 5]
        args.sigma = 1.5 if args.sigma is None else args.sigma
    return args


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


def _plot_csv(args):
    """Average over all partners of the measured full-bank evaluations of local run directories."""
    layout, seed, sigma, selected = cpa_csv.load_csv_selection(args, args.layout_names)
    methods = [m for m, _, _ in selected]
    runs = [r for _, r, _ in selected]
    out_dir, formats = cpa_csv.prepare_output(args, Path(__file__).resolve().parent.parent / "plots",
                                              cpa_csv.DEFAULT_CSV_FORMATS)
    boundaries = cpa_csv.common_boundaries(runs)

    fig, ax = setup_figure(width=11, height=5.2)
    for method, run, events in selected:
        color = METHOD_COLORS.get(method)
        ax.plot(events.env_steps, cpa_csv.smooth_line(events.mean, sigma), color=color, lw=1.6,
                label=method_display_name(method))
        ax.plot(events.env_steps, events.mean, "o", color=color, ms=4)
    cpa_csv.decorate_partner_axis(ax, runs[0])
    finalize_plot(ax, xlabel=cpa_csv.X_LABEL, ylabel=f"Mean team soups per {runs[0].episode_steps}-step episode\n"
                  f"(mean over all {runs[0].num_partners} partners)", xlim=(0, boundaries[-1]), ylim=(0, None))
    title = (f"{layout} · {', '.join(method_display_name(m) for m in methods)} · {cpa_csv.condition_text(runs[0])} · "
             f"seed {seed} · {runs[0].num_partners} partners")
    cpa_csv.finish_figure(fig, title, cpa_csv.caption(runs[0], "average", sigma))

    stem = args.plot_name or cpa_csv.csv_stem("cpa_avg", layout, methods, runs[0])
    save_plot(fig, out_dir, stem, formats)
    if not args.no_show:
        plt.show()


def plot():
    args = _parse_args()
    if args.run_root:
        return _plot_csv(args)
    out_dir, formats = cpa_csv.prepare_output(args, Path(__file__).resolve().parent.parent / "plots", ["png", "pdf"])
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

    if args.plot_name:
        stem = args.plot_name
    else:
        by = "methods" if args.compare_by == "method" else "arch"
        layout_str = "_".join(args.layout_names) if len(args.layout_names) == 1 else "all_layouts"
        stem = f"ppo_cumulative_{by}_{layout_str}_{args.arch}_partners{args.num_partners}"

    save_plot(fig, out_dir, stem, formats)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    plot()
