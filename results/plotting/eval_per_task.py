#!/usr/bin/env python3
"""
One subplot per CL method, one coloured line per environment/task.

* Directory layout, --metric switch, baseline normalisation and every other
  CLI flag are identical to plot_avg.py.
* Colours are auto-generated; the first task is blue, the next green … (husl).

Additional features:
1. X-axis ticks plotted on every subplot.
2. Vertical dividing lines between tasks.
3. Top x-axis labels reading "Task 1", "Task 2", …, colored to match lines for each method.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import utilities from the utils package
try:
    # Try relative import first (when imported as a module)
    from .utils import (
        collect_env_curves, smooth_and_ci, add_task_boundaries,
        setup_task_axes, save_plot, CRIT
    )
except ImportError:
    # Fall back to absolute import (when run as a script)
    from results.plotting.utils import (
        collect_env_curves, smooth_and_ci, add_task_boundaries,
        setup_task_axes, save_plot, CRIT
    )


def parse_args():
    """Parse command line arguments for the per-task evaluation plot script."""
    p = argparse.ArgumentParser(description="Plot per-task evaluation metrics for MARL continual-learning benchmark")
    p.add_argument('--data_root', required=True, help="Root directory for data")
    p.add_argument('--algo', required=True, help="Algorithm name")
    p.add_argument('--methods', nargs='+', required=True, help="Method names to plot")
    p.add_argument('--strategy', required=True, help="Training strategy")
    p.add_argument('--seq_len', type=int, required=True, help="Sequence length")
    p.add_argument('--steps_per_task', type=float, default=1e7, help="Steps per task")
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5], help="Seeds to include")
    p.add_argument('--sigma', type=float, default=1.5, help="Smoothing parameter")
    p.add_argument('--confidence', type=float, default=0.95, choices=[0.9, 0.95, 0.99], help="Confidence level")
    p.add_argument('--metric', choices=['reward', 'soup'], default='reward', help="Metric to plot")
    p.add_argument('--plot_name', default=None, help="Custom plot name")
    return p.parse_args()


def plot():
    """
    Main plotting function for per-task evaluation metrics.

    Creates one subplot per method, with one colored line per environment/task.
    Adds task boundaries and colored task labels.
    """
    args = parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root

    # Calculate total steps and set up boundaries
    total_steps = args.seq_len * args.steps_per_task
    task_colors = sns.color_palette("hls", args.seq_len)
    boundaries = [i * args.steps_per_task for i in range(args.seq_len + 1)]
    mids = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(args.seq_len)]

    # Set up figure with one subplot per method
    methods = args.methods
    fig_height = 2.5 * len(methods) if len(methods) > 1 else 2.8
    fig, axes = plt.subplots(len(methods), 1, sharex=False, sharey=True, figsize=(12, fig_height))
    if len(methods) == 1: 
        axes = [axes]

    # Create one subplot per method
    for m_idx, method in enumerate(methods):
        ax = axes[m_idx]

        # Collect data for this method
        envs, curves = collect_env_curves(
            data_root, args.algo, method, args.strategy, 
            args.seq_len, args.seeds, args.metric
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
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"Normalized {args.metric.capitalize()}")
        ax.set_title(method, fontsize=11)

        # Set up secondary x-axis with task labels
        twin = ax.twiny()
        twin.set_xlim(ax.get_xlim())
        twin.set_xticks(mids)
        labels = [f"Task {i + 1}" for i in range(args.seq_len)]
        twin.set_xticklabels(labels, fontsize=10)
        twin.tick_params(axis='x', length=0)

        # Color task labels to match task curves
        for idx, label in enumerate(twin.get_xticklabels()):
            label.set_color(task_colors[idx])

    # Set x-axis label on the bottom subplot
    axes[-1].set_xlabel('Environment Steps')

    # Finalize and save the plot
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent.parent / 'plots'
    name = args.plot_name or f"per_task_norm_{args.metric}"
    save_plot(fig, out_dir, name)

    # Display the plot
    plt.show()


if __name__ == '__main__':
    plot()
