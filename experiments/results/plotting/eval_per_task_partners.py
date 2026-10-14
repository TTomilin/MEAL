#!/usr/bin/env python3
"""
One subplot per CL method, one coloured line per environment/task.

* Directory layout, --metric switch, baseline normalisation and every other
  CLI flag are identical to plot_avg.py.
* Colours are auto-generated; the first task is blue, the next green … (husl).

CSV mode (`--run_root`, local run directories written by experiments/partner_adaptation/run_br.py) draws one
figure per method with one curve per partner at the measured full-bank evaluations; see
experiments/results/README.MD:
  python -m experiments.results.plotting.eval_per_task_partners --run_root cpa_results/runs \
    --layout_name cramped_room --methods Online_EWC --identity available --arch multihead \
    --output_dir /tmp/cpa_figs --no_show

Additional features:
1. X-axis ticks plotted on every subplot.
2. Vertical dividing lines between tasks.
3. Top x-axis labels reading "Task 1", "Task 2", …, colored to match lines for each method.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from experiments.results.plotting.utils import (
    smooth_and_ci, add_task_boundaries,
    save_plot, create_eval_parser, METHOD_DISPLAY_NAMES
)
from experiments.results.plotting.utils.data_loading import collect_partner_curves
from experiments.results.plotting.utils import cpa_csv


def parse_args():
    """Parse command line arguments for the per-task evaluation plot script."""
    p = create_eval_parser(
        description="Plot per-task evaluation metrics for MARL continual-learning benchmark",
        metric_choices=['reward', 'soup']
    )
    p.add_argument("--layout_name",  default=None,
                   help="Overcooked layout name (legacy default cramped_room; required in CSV mode)")
    p.add_argument("--arch",         default="multihead",    choices=["multihead", "singlehead"])
    p.add_argument("--num_partners", type=int, default=None,
                   help=f"Number of eval partners (legacy default 8; CSV mode {cpa_csv.DEFAULT_CSV_PARTNERS})")
    p.set_defaults(metric='soup', seeds=None, sigma=None)
    cpa_csv.add_csv_args(p)
    args = p.parse_args()
    if args.run_root is None:  # legacy mode keeps its original defaults
        args.layout_name = args.layout_name or "cramped_room"
        args.num_partners = args.num_partners or 8
        args.seeds = args.seeds or [1, 2, 3, 4, 5]
        args.sigma = 1.5 if args.sigma is None else args.sigma
    return args


def plot_csv(args):
    """One figure per method: one curve per partner at the measured full-bank evaluations of a run directory."""
    layout, seed, sigma, selected = cpa_csv.load_csv_selection(args, [args.layout_name] if args.layout_name else None)
    out_dir, formats = cpa_csv.prepare_output(args, Path(__file__).resolve().parent.parent / 'plots',
                                              cpa_csv.DEFAULT_CSV_FORMATS)
    for method, run, events in selected:
        n = run.num_partners
        colors = sns.color_palette("hls", n)  # by partner_id: the same partner has the same colour in every figure
        labels = cpa_csv.partner_labels(run)
        kinds = [p.get("kind") for p in run.partners]
        fig, ax = plt.subplots(figsize=(13.5, 5.8))
        for i in range(n):
            ls = '-' if kinds[i] == kinds[0] else (0, (4, 1.5))
            y = events.partner_means[:, i]
            ax.plot(events.env_steps, cpa_csv.smooth_line(y, sigma), color=colors[i], lw=1.1, ls=ls, label=labels[i])
            ax.plot(events.env_steps, y, 'o', color=colors[i], ms=2.5)
        boundaries = cpa_csv.decorate_partner_axis(ax, run)
        for i in range(n):  # training interval of partner i: shaded band and a thick segment of its own curve
            ax.axvspan(boundaries[i], boundaries[i + 1], color=colors[i], alpha=0.07, lw=0)
            inside = (events.env_steps >= boundaries[i]) & (events.env_steps <= boundaries[i + 1])
            ax.plot(events.env_steps[inside], cpa_csv.smooth_line(events.partner_means[:, i], sigma)[inside],
                    color=colors[i], lw=3.2, solid_capstyle='butt')
        ax.set_xlim(0, boundaries[-1])
        ax.set_ylim(0, None)
        ax.set_xlabel(cpa_csv.X_LABEL)
        ax.set_ylabel(f"Team soups per {run.episode_steps}-step episode")
        ax.legend(title="Partner (position · name)", loc='center left', bbox_to_anchor=(1.01, 0.5),
                  fontsize=8, title_fontsize=8, ncol=1)
        title = (f"{layout} · {METHOD_DISPLAY_NAMES.get(method, method)} · {cpa_csv.condition_text(run)} · "
                 f"seed {seed} · {n} partners · per-partner evaluation")
        cpa_csv.finish_figure(fig, title, cpa_csv.caption(run, "per_partner", sigma))
        stem = (f"{args.plot_name}_{method}" if args.plot_name and len(selected) > 1
                else args.plot_name or cpa_csv.csv_stem("cpa_per_partner", layout, [method], run))
        save_plot(fig, out_dir, stem, formats)
        if not args.no_show:
            plt.show()
        plt.close(fig)


def plot():
    """
    Main plotting function for per-task evaluation metrics.

    Creates one subplot per method, with one colored line per environment/task.
    Adds task boundaries and colored task labels.
    """
    args = parse_args()
    if args.run_root:
        return plot_csv(args)
    out_dir, formats = cpa_csv.prepare_output(args, Path(__file__).resolve().parent.parent / 'plots', ['png', 'pdf'])
    data_root = Path(__file__).resolve().parent.parent / args.data_root

    # Calculate total steps and set up boundaries
    n = args.num_partners
    total_steps = n * args.steps_per_task
    task_colors = sns.color_palette("hls", n)
    boundaries = [i * args.steps_per_task for i in range(n + 1)]
    mids = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(n)]

    # Set up figure with one subplot per method
    methods = args.methods
    fig_height = 3 * len(methods) if len(methods) > 1 else 2.8
    fig, axes = plt.subplots(len(methods), 1, sharex=False, sharey=True, figsize=(12, fig_height))
    if len(methods) == 1:
        axes = [axes]

    # Create one subplot per method
    for m_idx, method in enumerate(methods):
        ax = axes[m_idx]

        # Collect data for this method
        envs, curves = collect_partner_curves(
            data_root, args.algo, method,
            args.layout_name, args.arch, args.num_partners,
            args.seeds, args.metric,
        )

        # Add task boundaries
        add_task_boundaries(ax, boundaries, color='gray', linewidth=0.5)

        # Set up x-axis ticks
        ax.set_xticks(boundaries)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

        # Plot one curve per environment/task
        for i, curve in enumerate(curves):
            mean, ci = smooth_and_ci(curve, args.sigma, args.confidence)
            x = np.linspace(0, total_steps, len(mean))
            ax.plot(x, mean, color=task_colors[i])
            ax.fill_between(x, mean - ci, mean + ci, alpha=0.2, color=task_colors[i])

        # Set axis limits and labels
        ax.set_xlim(0, total_steps)
        ax.set_ylim(0, None)
        ax.set_ylabel(f"Normalized Soup")
        # Only set title if there are multiple methods
        if len(methods) > 1:
            ax.set_title(METHOD_DISPLAY_NAMES.get(method, method), fontsize=11)

        # Set up secondary x-axis with task labels
        twin = ax.twiny()
        twin.set_xlim(ax.get_xlim())
        twin.set_xticks(mids)
        labels = [f"Partner {i + 1}" for i in range(n)]
        twin.set_xticklabels(labels, fontsize=10)
        twin.tick_params(axis='x', length=0)

        # Color task labels to match task curves
        for idx, label in enumerate(twin.get_xticklabels()):
            label.set_color(task_colors[idx])

    # Set x-axis label on the bottom subplot
    axes[-1].set_xlabel('Environment Steps')

    # Finalize and save the plot
    plt.tight_layout()
    name = args.plot_name or f"partner_per_task_score"
    # Add method name to filename if only a single method is given
    if len(methods) == 1:
        name += f"_{methods[0]}"
    save_plot(fig, out_dir, name, formats)

    # Display the plot
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    plot()
