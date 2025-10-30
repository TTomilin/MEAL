"""
Utility functions for plasticity plotting scripts.

This module contains functions for loading and processing data for plasticity plots,
including collecting runs and processing time series data.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .common import load_series, smooth_and_ci


def collect_plasticity_runs(
        base: Path,
        algo: str,
        method: str,
        strat: str,
        seq_len: int,
        repeats: int,
        seeds: List[int],
        level: int = 1,
) -> list[np.ndarray]:
    """Return a list where item *i* contains all curves for task *i* across seeds.

    Each trace ("*_soup.*") is assumed to contain *repeats* consecutive
    sequences of *seq_len* tasks.  For every seed we build **one** curve per
    task by concatenating all its segments across repetitions, then taking
    the cumulative average (normalised to 1.0 at the end).

    Args:
        base: Base directory for data
        algo: Algorithm name
        method: Method name
        strat: Strategy name
        seq_len: Sequence length
        repeats: Number of sequence repetitions
        seeds: List of seeds to collect
        level: Difficulty level (default: 1)

    Returns:
        List of arrays, where each array contains curves for a specific task
    """

    task_runs: list[list[np.ndarray]] = [[] for _ in range(seq_len)]

    # Try to find data with higher repetitions first, then fall back to exact match
    possible_repeats = sorted([r for r in range(repeats, 21) if r >= repeats], reverse=True)

    for seed in seeds:
        # Try to find data with higher repetitions first, then fall back to exact match
        found_data = False

        for try_repeats in possible_repeats:
            folder = f"{strat}_{seq_len}"
            if try_repeats > 1:
                folder += f"_rep_{try_repeats}"

            # Use different directory structure for single baseline vs CL methods
            run_dir = base / algo / method / "plasticity" / folder / f"seed_{seed}"
            if not run_dir.exists():
                continue

            for fp in sorted(run_dir.glob("*_soup.*")):
                trace = load_series(fp)
                if trace.ndim != 1:
                    raise ValueError(f"Trace in {fp} is not 1‑D (shape {trace.shape})")

                total_chunks = seq_len * try_repeats
                L_est = len(trace) // total_chunks
                if L_est == 0:
                    print(f"Warning: trace in {fp} shorter than expected; skipped.")
                    continue

                # If we're reusing data from higher repetitions, inform the user
                if try_repeats > repeats:
                    print(f"Info: Reusing data from {folder}/seed_{seed} for rep_{repeats} (trimming {try_repeats - repeats} repetitions)")

                # build one long segment per task by concatenating its occurrences
                # but only use the first 'repeats' repetitions
                for task_idx in range(seq_len):
                    slices = []
                    for rep in range(repeats):  # Only use the requested number of repetitions
                        start = (rep * seq_len + task_idx) * L_est
                        end = start + L_est
                        if end > len(trace):  # safety for ragged endings
                            break
                        slices.append(trace[start:end])
                    if not slices:
                        continue

                    task_trace = np.concatenate(slices)
                    task_runs[task_idx].append(task_trace)

                found_data = True
                break  # Found data for this seed, move to next seed

            if found_data:
                break  # Found data for this seed, move to next seed

        if not found_data:
            print(f"Warning: no data found for seed {seed} with any repetition >= {repeats}")

    # pad to equal length so we can average ----------------------------------
    processed: list[np.ndarray] = []
    for idx, runs in enumerate(task_runs):
        if not runs:
            processed.append(np.array([]))
            continue
        T = max(len(r) for r in runs)
        padded = [np.pad(r, (0, T - len(r)), constant_values=np.nan) for r in runs]
        processed.append(np.vstack(padded))
    return processed


def setup_plasticity_grid(seq_len: int, figsize_scale: Tuple[int, int] = (4, 3), n_rows: int = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Set up a grid of subplots for plasticity visualization.

    Args:
        seq_len: Number of tasks in the sequence
        figsize_scale: Tuple of (width, height) scale factors per subplot
        n_rows: Optional number of rows. If provided, columns will be seq_len.
                If not provided, will auto-calculate based on seq_len.

    Returns:
        Tuple of (figure, flattened axes array)
    """
    if n_rows is not None:
        # For forward transfer plots: n_rows = number of methods, n_cols = seq_len
        n_cols = seq_len
    elif seq_len == 10:
        n_rows, n_cols = 2, 5
    else:
        n_rows = int(np.ceil(np.sqrt(seq_len)))
        n_cols = int(np.ceil(seq_len / n_rows))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_scale[0] * n_cols, figsize_scale[1] * n_rows),
        squeeze=False
    )
    return fig, axes.flatten()


def collect_training_data(
        base: Path,
        algo: str,
        method: str,
        strat: str,
        seq_len: int,
        repeats: int,
        seeds: List[int],
        level: int = 1
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
        run_dir = base / algo / method / f"level_{level}" / folder / f"seed_{seed}"
        if not run_dir.exists():
            print(f"Warning: no directory {run_dir}")
            continue

        # Look for training_soup files - try single file first, then numbered files
        training_data = None

        # First try single training_soup file (for CL methods)
        for ext in [".json", ".npz"]:
            fp = run_dir / f"training_soup{ext}"
            if fp.exists():
                try:
                    training_data = load_series(fp)
                    break
                except Exception as e:
                    print(f"Error loading {fp}: {e}")

        # If no single file found, try numbered training_soup files (for baseline methods)
        if training_data is None:
            training_files = []
            for ext in [".json", ".npz"]:
                pattern = f"*_training_soup{ext}"
                files = sorted(run_dir.glob(pattern))
                if files:
                    training_files = files
                    break

            if training_files:
                try:
                    # Load and concatenate all training files in order
                    segments = []
                    for fp in training_files:
                        data = load_series(fp)
                        segments.append(data)

                    # Concatenate all segments to form one continuous training curve
                    training_data = np.concatenate(segments)
                except Exception as e:
                    print(f"Error loading training files in {run_dir}: {e}")

        if training_data is not None:
            runs.append(training_data)
        else:
            print(f"Warning: no training_soup files found in {run_dir}")

    if not runs:
        raise RuntimeError(f"No training data found for method {method}")

    # Pad shorter runs with NaNs so we can average
    T = max(map(len, runs))
    padded = [np.pad(r, (0, T - len(r)), constant_values=np.nan) for r in runs]
    return np.vstack(padded)
