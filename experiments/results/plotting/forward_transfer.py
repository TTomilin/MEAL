#!/usr/bin/env python3
"""
Plot forward-transfer curves.

For every CL method:
  • load its `training_soup.*` traces (one per seed);
  • load the matching single-task baseline traces (`cl_method == single`);
  • average across seeds, smooth with Gaussian σ, and plot both;
  • shade the region between them: blue for positive, red for negative.

Layout: one subplot per CL method (one per row), with all tasks combined into a single line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from experiments.results.plotting.utils import (
    CRIT, METHOD_COLORS, collect_training_data, save_plot, create_plasticity_parser
)


# ───────────────────────── CLI ──────────────────────────
def _cli():
    p = create_plasticity_parser("Plot forward-transfer curves.")
    p.add_argument("--baseline_method", default="single",
                   help="Folder name that stores the single-task traces")
    p.add_argument("--max_tasks", type=int, default=10,
                   help="Maximum number of tasks to plot (default: 10)")
    p.set_defaults(confidence=0.9, plot_name="forward_transfer")
    return p.parse_args()


# ────────────────────────── helpers ─────────────────────
def avg_and_smooth(curves: np.ndarray, sigma: float):
    if curves.size == 0 or curves.shape[0] == 0:
        return np.array([]), np.array([])

    # Check if all values are NaN or if we have insufficient valid data
    if np.all(np.isnan(curves)) or curves.shape[0] < 2:
        return np.array([]), np.array([])

    # Check if each column has at least one valid value
    valid_cols = ~np.all(np.isnan(curves), axis=0)
    if not np.any(valid_cols):
        return np.array([]), np.array([])

    # Compute mean and std, handling NaN values
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = np.nanmean(curves, axis=0)
        sd = np.nanstd(curves, axis=0)

        # Replace NaN values with 0 for smoothing
        mu_clean = np.where(np.isnan(mu), 0, mu)
        sd_clean = np.where(np.isnan(sd), 0, sd)

        mu = gaussian_filter1d(mu_clean, sigma)
        sd = gaussian_filter1d(sd_clean, sigma)
        ci = CRIT[args.confidence] * sd / np.sqrt(curves.shape[0])

    return mu, ci


def shade_between(ax, x, y_cl, y_base):
    pos = y_cl > y_base
    ax.fill_between(x, y_cl, y_base, where=pos, interpolate=True,
                    color="tab:green", alpha=0.30)
    ax.fill_between(x, y_cl, y_base, where=~pos, interpolate=True,
                    color="tab:red", alpha=0.20)


# ────────────────────────── main ────────────────────────
def main():
    global args
    args = _cli()
    base = Path(__file__).resolve().parent.parent
    data_root = base / args.data_root

    # Limit the number of tasks to display
    effective_seq_len = min(args.seq_len, args.max_tasks)

    # prepare Figure: one subplot per method (one row per method)
    fig, axes = plt.subplots(len(args.methods), 1,
                             figsize=(12, 2.5 * len(args.methods)),
                             squeeze=False)
    axes = axes.flatten()

    # Store legend handles and labels for the bottom legend
    legend_handles = []
    legend_labels = []

    # Store all data values for calculating ylim in single method case
    all_data_values = []

    # iterate over CL methods (row)
    for row, method in enumerate(args.methods):
        # CL traces
        cl_runs = collect_training_data(
            data_root, args.algo, method,
            args.strategy, args.seq_len, args.repeat_sequence, args.seeds, level=args.level
        )
        # Baseline traces
        base_runs = collect_training_data(
            data_root, args.algo, args.baseline_method,
            args.strategy, args.seq_len, 1, args.seeds, level=args.level
        )

        color = METHOD_COLORS.get(method, "tab:gray")
        ax = axes[row]

        # Check if we have training data
        if cl_runs.size == 0 or base_runs.size == 0:
            ax.set_visible(False)
            continue

        # Truncate training data to only include the first effective_seq_len tasks
        if effective_seq_len < args.seq_len:
            # Calculate the proportion of data points that correspond to effective_seq_len tasks
            truncate_ratio = effective_seq_len / args.seq_len

            # Truncate CL runs
            cl_truncate_points = int(cl_runs.shape[1] * truncate_ratio)
            cl_runs_truncated = cl_runs[:, :cl_truncate_points]

            # Truncate baseline runs
            base_truncate_points = int(base_runs.shape[1] * truncate_ratio)
            base_runs_truncated = base_runs[:, :base_truncate_points]

            combined_cl_array = cl_runs_truncated
            combined_base_array = base_runs_truncated
        else:
            # Training data is already in the right format: (n_seeds, n_points)
            combined_cl_array = cl_runs
            combined_base_array = base_runs

        # Smooth and average the combined curves
        mu_cl, ci_cl = avg_and_smooth(combined_cl_array, args.sigma)
        mu_base, ci_bs = avg_and_smooth(combined_base_array, args.sigma)

        # Check if smoothed curves are empty
        if len(mu_cl) == 0 or len(mu_base) == 0:
            ax.set_visible(False)
            continue

        # Collect data values for ylim calculation in single method case
        if len(args.methods) == 1:
            all_data_values.extend(mu_cl[~np.isnan(mu_cl)])
            all_data_values.extend(mu_base[~np.isnan(mu_base)])

        # Create x-axis spanning limited tasks
        total_steps = args.steps_per_task * effective_seq_len * args.repeat_sequence
        x_cl = np.linspace(0, total_steps, len(mu_cl))
        x_base = np.linspace(0, total_steps, len(mu_base))

        # plot curves
        baseline_line, = ax.plot(x_base, mu_base, color="crimson", lw=1.5, label="IPPO")
        method_line, = ax.plot(x_cl, mu_cl, color=color, lw=1.5, label=method)

        # Collect legend handles and labels (method first, baseline last)
        legend_handles.append(method_line)
        legend_labels.append(method)
        if row == 0:
            legend_handles.append(baseline_line)
            legend_labels.append("IPPO")

        # shaded transfer - interpolate to common grid for comparison
        if len(mu_cl) > 0 and len(mu_base) > 0:
            # Use the longer sequence as reference
            if len(mu_cl) >= len(mu_base):
                x_common = x_cl
                mu_base_interp = np.interp(x_common, x_base, mu_base)
                shade_between(ax, x_common, mu_cl, mu_base_interp)
            else:
                x_common = x_base
                mu_cl_interp = np.interp(x_common, x_cl, mu_cl)
                shade_between(ax, x_common, mu_cl_interp, mu_base)

        # Add vertical lines to separate tasks
        for task_idx in range(1, effective_seq_len):
            task_boundary = (args.steps_per_task * task_idx * args.repeat_sequence)
            ax.axvline(x=task_boundary, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlim(0, total_steps)

        # Set ylim based on data for single method, fixed for multiple methods
        if len(args.methods) == 1:
            # Calculate ylim from actual data values (will be set after loop)
            pass
        else:
            ax.set_ylim(0, 1.3)

        ax.set_ylabel("Normalized Score")

        # Add subtitle for each subplot with CL method name (only for multiple methods)
        if len(args.methods) > 1:
            ax.set_title(method, fontsize=12, pad=10)

        # Only show timesteps (scientific notation) on the final/bottom subplot
        if row == len(args.methods) - 1:
            # Set x-axis to use scientific notation for time steps
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        else:
            # Hide x-axis labels and ticks for non-bottom subplots
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)

        # Add task labels only on the topmost subplot
        if row == 0:
            task_centers = [(args.steps_per_task * args.repeat_sequence * (i + 0.5))
                            for i in range(effective_seq_len)]
            # Create a secondary x-axis for task labels
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(task_centers)
            ax2.set_xticklabels([f"Task {i + 1}" for i in range(effective_seq_len)])
            # Remove small vertical ticks below the task labels
            ax2.tick_params(axis='x', which='major', pad=5, length=0)

    # Handle legend and ylim for single method case
    if len(args.methods) == 1 and legend_handles and legend_labels:
        # Set ylim based on actual data values
        if all_data_values:
            y_min = min(all_data_values)
            y_max = max(all_data_values)
            # Add some padding
            y_range = y_max - y_min
            y_padding = y_range * 0.1 if y_range > 0 else 0.1
            # axes[0].set_ylim(y_min - y_padding, y_max + y_padding)
            axes[0].set_ylim(y_min, y_max)

        # Add legend directly on the plot where there is space
        # Replace underscores with spaces in legend labels
        formatted_legend_labels = [label.replace('_', ' ') for label in legend_labels]
        axes[0].legend(legend_handles, formatted_legend_labels, loc='best', frameon=True, fontsize=12)

    y_offset = 0.06 if len(args.methods) == 1 else 0.01
    fig.text(0.5, y_offset, "Environment steps", ha="center", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 1])

    # Modify filename for single method case
    if len(args.methods) == 1:
        plot_name = f"{args.plot_name}_{args.methods[0]}"
    else:
        plot_name = args.plot_name
    plot_name += f"_level_{args.level}"

    save_plot(fig, base / "plots", plot_name)
    fig.show()


if __name__ == "__main__":
    main()
