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

import argparse
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Import utilities from the utils package
try:
    # Try relative import first (when imported as a module)
    from .utils import (
        load_series, smooth_and_ci, setup_figure, add_task_boundaries,
        finalize_plot, save_plot, CRIT, METHOD_COLORS
    )
except ImportError:
    # Fall back to absolute import (when run as a script)
    from results.plotting.utils import (
        load_series, smooth_and_ci, setup_figure, add_task_boundaries,
        finalize_plot, save_plot, CRIT, METHOD_COLORS
    )


# ───────────────────────── CLI ──────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot training curves from training_soup data files.",
    )
    p.add_argument("--data_root", required=True, help="Root folder with algo/method runs.")
    p.add_argument("--algo", required=True)
    p.add_argument("--strategy", required=True, help="Prefix of the data folder, e.g. 'generate'.")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--seq_len", type=int, required=True, help="Tasks per sequence.")
    p.add_argument("--repeat_sequence", type=int, default=1, help="Sequence repetitions inside the file.")
    p.add_argument("--steps_per_task", type=float, default=1e7, help="x‑axis scaling.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--sigma", type=float, default=1.5, help="Gaussian smoothing σ.")
    p.add_argument("--confidence", type=float, default=0.9, choices=[0.9, 0.95, 0.99])
    p.add_argument("--plot_name", default="training_curve")
    return p.parse_args()


# ─────────────────── helpers ───────────────────

def collect_training_data(
        base: Path,
        algo: str,
        method: str,
        strat: str,
        seq_len: int,
        repeats: int,
        seeds: List[int],
) -> np.ndarray:
    """
    Collect training data from training_soup files.
    
    Args:
        base: Base directory for data
        algo: Algorithm name
        method: Method name
        strat: Strategy name
        seq_len: Sequence length
        repeats: Number of sequence repetitions
        seeds: List of seeds to collect
        
    Returns:
        Array of shape (n_seeds, n_points) containing the training curves
    """
    folder = f"{strat}_{seq_len * repeats}"
    runs = []
    
    for seed in seeds:
        run_dir = base / algo / method / "plasticity" / folder / f"seed_{seed}"
        if not run_dir.exists():
            print(f"Warning: no directory {run_dir}")
            continue
            
        # Look for training_soup.json or training_soup.npz
        for ext in [".json", ".npz"]:
            fp = run_dir / f"training_soup{ext}"
            if fp.exists():
                try:
                    data = load_series(fp)
                    runs.append(data)
                    break
                except Exception as e:
                    print(f"Error loading {fp}: {e}")
        else:
            print(f"Warning: no training_soup file found in {run_dir}")
    
    if not runs:
        raise RuntimeError(f"No training data found for method {method}")
    
    # Pad shorter runs with NaNs so we can average
    T = max(map(len, runs))
    padded = [np.pad(r, (0, T - len(r)), constant_values=np.nan) for r in runs]
    return np.vstack(padded)


# ────────────────────────── main ─────────────────────────

def main():
    args = _cli()
    data_dir = Path(__file__).resolve().parent.parent / args.data_root
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
        title="Training Curve",
        xlabel="Environment Steps",
        ylabel="Training Performance",
        xlim=(0, total_steps),
        ylim=(0, None),
        legend_loc="best",
    )
    
    # Save the plot
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{args.plot_name}.{ext}")
    
    plt.show()


if __name__ == "__main__":
    main()