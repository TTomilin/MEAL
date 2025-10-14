#!/usr/bin/env python3
"""
Plot dormant ratio curves from the start of training until the very end.

This script creates a simple time-series plot of the dormant neuron ratio
throughout the entire training process, without task indices. The values
are transformed as (1 - ratio) as specified.

Directory layout expected:
```
<data_root>/<algo>/<method>/plasticity/<strategy>_<seq_len>_rep_<repeats>/seed_<seed>/dormant_ratio.json
```

Example usage:
```bash
python dormant_ratio_curves.py \
  --data_root results/data \
  --algo ippo \
  --methods FT \
  --strategy generate \
  --seq_len 10 \
  --repeat_sequence 10 \
  --seeds 6 7 8
```
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from experiments.results.plotting.utils import (
    create_plasticity_parser, load_series, smooth_and_ci, METHOD_COLORS, save_plot
)


def collect_dormant_ratio_data(
    base: Path,
    algo: str,
    method: str,
    strategy: str,
    seq_len: int,
    repeat_sequence: int,
    seeds: List[int],
    level: int = 1,
) -> List[np.ndarray]:
    """
    Collect dormant ratio data from multiple seeds.

    Args:
        base: Base data directory
        algo: Algorithm name (e.g., 'ippo')
        method: Method name (e.g., 'FT')
        strategy: Training strategy (e.g., 'generate')
        seq_len: Number of tasks in sequence
        repeat_sequence: Number of repetitions
        seeds: List of seed numbers to collect
        level: Difficulty level

    Returns:
        List of numpy arrays, one per seed containing the dormant ratio time series
    """
    all_traces = []

    # Construct folder name
    folder = f"{strategy}_{seq_len}"
    if repeat_sequence > 1:
        folder += f"_rep_{repeat_sequence}"

    for seed in seeds:
        run_dir = base / algo / method / "plasticity" / folder / f"seed_{seed}"
        dormant_file = run_dir / "dormant_ratio.json"

        if not dormant_file.exists():
            print(f"Warning: {dormant_file} not found, skipping seed {seed}")
            continue

        try:
            trace = load_series(dormant_file)
            if trace.ndim != 1:
                print(f"Warning: dormant ratio trace in {dormant_file} is not 1-D, skipping")
                continue

            # Apply the transformation: 1 - ratio
            transformed_trace = 1.0 - trace
            all_traces.append(transformed_trace)
            print(f"Loaded dormant ratio data for seed {seed}: {len(trace)} points")

        except Exception as e:
            print(f"Error loading {dormant_file}: {e}")
            continue

    return all_traces


def main():
    """Main function to create dormant ratio curve plots."""
    parser = create_plasticity_parser(
        description="Plot dormant ratio curves from start to end of training"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / args.data_root
    total_steps = args.seq_len * args.repeat_sequence * args.steps_per_task

    # Create output directory
    out_dir = base_dir / "plots"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    method_lines = []
    method_names = []
    any_data_found = False
    all_mu_values = []  # Collect all y-values to determine y-limits

    for method in args.methods:
        # Collect dormant ratio data for this method
        traces = collect_dormant_ratio_data(
            data_dir,
            args.algo,
            method,
            args.strategy,
            args.seq_len,
            args.repeat_sequence,
            args.seeds,
            args.level,
        )

        if not traces:
            print(f"No data found for method {method}")
            continue

        any_data_found = True

        # Find the minimum length to pad all traces to the same length
        min_length = min(len(trace) for trace in traces)
        max_length = max(len(trace) for trace in traces)

        print(f"Method {method}: {len(traces)} seeds, length range: {min_length}-{max_length}")

        # Pad traces to the same length (use max length, pad with NaN)
        padded_traces = []
        for trace in traces:
            if len(trace) < max_length:
                padded_trace = np.pad(trace, (0, max_length - len(trace)), constant_values=np.nan)
            else:
                padded_trace = trace[:max_length]  # Truncate if longer
            padded_traces.append(padded_trace)

        # Convert to numpy array for easier processing
        data_array = np.array(padded_traces)

        # Apply additional smoothing to individual traces for better curve appearance
        smoothed_traces = []
        for trace in data_array:
            # Apply Gaussian smoothing to each individual trace
            smoothed_trace = gaussian_filter1d(trace, sigma=args.sigma)
            smoothed_traces.append(smoothed_trace)
        smoothed_data_array = np.array(smoothed_traces)

        # Calculate smoothed mean and confidence intervals
        mu, ci = smooth_and_ci(smoothed_data_array, args.sigma, args.confidence)

        # Create x-axis (training steps) - scale to total_steps
        x = np.linspace(0, total_steps, len(mu))

        # Collect y-values for determining y-limits
        all_mu_values.extend(mu[~np.isnan(mu)])  # Only non-NaN values

        # Get color for this method
        color = METHOD_COLORS.get(method, '#1f77b4')  # Default blue if method not in colors

        # Plot the curve
        line, = ax.plot(x, mu, color=color, label=method, linewidth=2)
        ax.fill_between(x, mu - ci, mu + ci, color=color, alpha=0.2)

        method_lines.append(line)
        method_names.append(method)

        print(f"Plotted {method}: mean range {np.nanmin(mu):.3f}-{np.nanmax(mu):.3f}")

        # Format x-axis to use scientific notation if numbers are large
        if len(mu) > 10000:
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    # Check if any data was found
    if not any_data_found:
        print("No data found for any method. Please check your data paths and arguments.")
        return

    # Customize the plot
    ax.set_xlabel("Training Steps", fontsize=14)
    ax.set_ylabel("Dormant Ratio", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Set x-limits to (0, total_steps)
    ax.set_xlim(0, total_steps)

    # Set y-limits based on actual data range
    if all_mu_values:
        y_min = min(all_mu_values)
        y_max = max(all_mu_values)
        # Add some padding around the data
        y_range = y_max - y_min
        y_padding = y_range * 0.05 if y_range > 0 else 0.05
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    else:
        # Fallback to default if no data
        ax.set_ylim(0, 1)

    # Add legend if multiple methods
    if len(method_names) > 1:
        ax.legend(method_lines, method_names, loc='best', frameon=True)

    plt.tight_layout()

    # Save the plot
    plot_name = args.plot_name or "dormant_ratio_curves"
    if len(args.methods) == 1:
        plot_name += f"_{args.methods[0]}"
    plot_name += f"_level_{args.level}"

    save_plot(fig, out_dir, plot_name)

    print(f"Plot saved to {out_dir}/{plot_name}.png")
    plt.show()


if __name__ == "__main__":
    main()
