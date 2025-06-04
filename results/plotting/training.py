#!/usr/bin/env python3
"""
Plot average *success* or *reward* for the MARL continual-learning benchmark.

Metric semantics
----------------
success : curves are divided by per-environment baseline avg_rewards
          (0 = random agent, 1 = baseline, >1 = out-performing baseline)
reward  : raw reward curves, no normalisation

Usage (examples)
----------------
# success (default)
python plot_avg.py --metric success --data_root results ...

# reward
python plot_avg.py --metric reward --data_root results ...
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np

# Import utilities from the utils package
try:
    # Try relative import first (when imported as a module)
    from .utils import (
        collect_runs, setup_figure, add_task_boundaries, 
        setup_task_axes, smooth_and_ci, save_plot, finalize_plot,
        CRIT, METHOD_COLORS
    )
except ImportError:
    # Fall back to absolute import (when run as a script)
    from results.plotting.utils import (
        collect_runs, setup_figure, add_task_boundaries, 
        setup_task_axes, smooth_and_ci, save_plot, finalize_plot,
        CRIT, METHOD_COLORS
    )


def parse_args():
    """Parse command line arguments for the training plot script."""
    p = argparse.ArgumentParser(description="Plot training metrics for MARL continual-learning benchmark")
    p.add_argument('--data_root', required=True, help="Root directory for data")
    p.add_argument('--algo', required=True, help="Algorithm name")
    p.add_argument('--arch', required=True, help="Architecture name")
    p.add_argument('--methods', nargs='+', required=True, help="Method names to plot")
    p.add_argument('--strategy', required=True, help="Training strategy")
    p.add_argument('--seq_len', type=int, required=True, help="Sequence length")
    p.add_argument('--steps_per_task', type=float, default=1e7, help="Steps per task")
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5], help="Seeds to include")
    p.add_argument('--sigma', type=float, default=1.5, help="Smoothing parameter")
    p.add_argument('--confidence', type=float, default=0.9, choices=[0.9, 0.95, 0.99], help="Confidence level")
    p.add_argument('--metric', choices=['reward', 'soup'], default='soup', help="Metric to plot")
    p.add_argument('--plot_name', default=None, help="Custom plot name")
    p.add_argument('--legend_anchor', type=float, default=0.87, help="Legend anchor position")
    return p.parse_args()


def plot():
    """
    Main plotting function for training metrics.

    Collects data for each method, creates a plot with method curves,
    adds task boundaries, and saves the plot.
    """
    args = parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root

    # Calculate total steps and set up figure
    total_steps = args.seq_len * args.steps_per_task
    width = min(max(args.seq_len, 8), 14)
    fig, ax = setup_figure(width=width, height=4)

    # Dictionary to store data for each method
    method_data = {}

    # Collect data for each method
    for method in args.methods:
        data, env_names = collect_runs(
            data_root, args.algo, method, args.arch,
            args.strategy, args.seq_len, args.seeds, args.metric
        )
        method_data[method] = data

        # Calculate smoothed mean and confidence interval
        mu, ci = smooth_and_ci(data, args.sigma, args.confidence)

        # Plot the method curve
        x = np.linspace(0, total_steps, len(mu))
        color = METHOD_COLORS.get(method)
        ax.plot(x, mu, label=method, color=color)
        ax.fill_between(x, mu - ci, mu + ci, color=color, alpha=0.2)

    # Add task boundaries
    boundaries = [i * args.steps_per_task for i in range(args.seq_len + 1)]
    add_task_boundaries(ax, boundaries)

    # Set up task axes (primary and secondary x-axes)
    setup_task_axes(ax, boundaries, args.seq_len)

    # Finalize plot with labels, limits, and legend
    finalize_plot(
        ax,
        xlabel='Environment Steps',
        ylabel=f'{args.metric.capitalize()} Delivered Normalized',
        xlim=(0, total_steps),
        ylim=(0, None),
        legend_loc='lower center',
        legend_bbox_to_anchor=(0.5, args.legend_anchor),
        legend_ncol=len(args.methods)
    )

    # Save the plot
    out_dir = Path(__file__).resolve().parent.parent / 'plots'
    stem = args.plot_name or f"avg_norm_{args.metric}"
    save_plot(fig, out_dir, stem)

    # Display the plot
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    plot()
