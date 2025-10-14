#!/usr/bin/env python3
"""
Plot training curves from training_soup data files.

This script loads training_soup data files and plots the training curves
for multiple methods. It supports both seq_length and repeat_sequence
parameters, multiplying them to determine the effective sequence length
for file retrieval.

Directory layout expected:
```
<data_root>/<algo>/<method>/plasticity/<strategy>_<seq_len*repeat_sequence>/seed_<seed>/training_soup.*
```
For example `--strategy generate --seq_len 10 --repeat_sequence 10` ⇒
folder `…/plasticity/generate_100/…`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from experiments.results.plotting.utils import (
    setup_figure, add_task_boundaries,
    finalize_plot, save_plot, CRIT, METHOD_COLORS, collect_training_data,
    create_plasticity_parser
)


# ───────────────────────── CLI ──────────────────────────

def _cli():
    """Parse command line arguments for the training curves plot script."""
    p = create_plasticity_parser(description="Plot training curves from training_soup data files.")
    # Set default confidence level to 0.9 (overriding the default from create_plasticity_parser)
    p.set_defaults(confidence=0.9, plot_name="training_curve")
    return p.parse_args()


# ────────────────────────── main ─────────────────────────

def main():
    args = _cli()
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / args.data_root
    total_steps = args.seq_len * args.repeat_sequence * args.steps_per_task

    # Set up figure
    fig, ax = setup_figure(width=12, height=6)

    # Dictionary to store data for each method
    method_data = {}

    # Collect and plot data for each method
    for method in args.methods:
        try:
            data = collect_training_data(
                data_dir,
                args.algo,
                method,
                args.strategy,
                args.seq_len,
                args.repeat_sequence,
                args.seeds,
            )
            method_data[method] = data

            # Calculate smoothed mean and confidence interval
            mu = gaussian_filter1d(np.nanmean(data, axis=0), sigma=args.sigma)
            sd = gaussian_filter1d(np.nanstd(data, axis=0), sigma=args.sigma)
            ci = CRIT[args.confidence] * sd / np.sqrt(data.shape[0])

            # Plot the method curve
            x = np.linspace(0, total_steps, len(mu))
            color = METHOD_COLORS.get(method)
            ax.plot(x, mu, label=method, color=color)
            ax.fill_between(x, mu - ci, mu + ci, color=color, alpha=0.2)

        except Exception as e:
            print(f"Error processing method {method}: {e}")

    # Add task boundaries
    boundaries = [i * args.steps_per_task * args.repeat_sequence for i in range(args.seq_len + 1)]
    add_task_boundaries(ax, boundaries)

    # Finalize plot with labels, limits, and legend
    finalize_plot(
        ax,
        xlabel="Environment Steps",
        ylabel="Training Performance",
        xlim=(0, total_steps),
        ylim=(0, None),
        legend_loc="best",
    )

    # Save the plot
    out_dir = base_dir / Path("plots")
    plot_name = args.plot_name
    # Add level suffix if not already present
    if "_level_" not in plot_name:
        plot_name += f"_level_{args.level}"
    save_plot(fig, out_dir, plot_name)

    plt.show()


if __name__ == "__main__":
    main()
