"""
Common utilities for plotting scripts.

This module contains shared constants, imports, and basic utility functions
used across different plotting scripts.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Set default plotting style
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams['axes.grid'] = False

# Critical values for confidence intervals
CRIT = {0.9: 1.833, 0.95: 1.96, 0.99: 2.576}

# Standard colors for different methods
METHOD_COLORS = {
    'EWC': '#12939A', 'MAS': '#2CA02C', 'AGEM': '#BC5090', 'Online_EWC': '#FF6E54', 'FT': '#FFA600',
    'L2': '#003F5C', 'PackNet': '#2F4B7C', 'ReDo': '#D62728', 'CBP': '#58508D'
}

LEVEL_COLORS = {
    'Level 1': '#2ECC71',  # green
    'Level 2': '#F1C40F',  # yellow
    'Level 3': '#E74C3C',  # red
}


def load_series(fp: Path) -> np.ndarray:
    """
    Load a time series from a file.

    Args:
        fp: Path to the file (.json or .npz)

    Returns:
        numpy array containing the time series data

    Raises:
        ValueError: If the file has an unsupported extension
    """
    if fp.suffix == '.json':
        return np.array(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == '.npz':
        return np.load(fp)['data'].astype(float)
    raise ValueError(f'Unsupported file suffix: {fp.suffix}')


def smooth_and_ci(data: np.ndarray, sigma: float, conf: float):
    """
    Calculate smoothed mean and confidence intervals.

    Args:
        data: Input data array of shape (n_samples, n_points)
        sigma: Smoothing parameter for Gaussian filter
        conf: Confidence level (0.9, 0.95, or 0.99)

    Returns:
        Tuple of (smoothed_mean, confidence_interval)
    """
    # Handle case where data is all zeros or NaNs
    if np.all(np.isnan(data)) or np.all(data == 0):
        n_points = data.shape[1]
        return np.zeros(n_points), np.zeros(n_points)

    # Calculate mean with NaN handling
    mean = gaussian_filter1d(np.nanmean(data, axis=0), sigma=sigma)

    # Calculate standard deviation with NaN handling
    # Use np.nanstd which ignores NaN values
    sd = gaussian_filter1d(np.nanstd(data, axis=0), sigma=sigma)

    # Avoid division by zero when calculating confidence intervals
    # If standard deviation is zero or NaN, set confidence interval to zero
    n_samples = np.sum(~np.isnan(data), axis=0)
    n_samples = np.where(n_samples > 0, n_samples, 1)  # Avoid division by zero

    # Calculate confidence intervals with safety checks
    ci = np.where(sd > 0, CRIT[conf] * sd / np.sqrt(n_samples), 0)

    return mean, ci


def get_output_path(filename: str = None, default_name: str = "plot") -> Path:
    """
    Get the output path for saving plots.

    Args:
        filename: Optional custom filename
        default_name: Default name to use if filename is None

    Returns:
        Path object for the output directory
    """
    out_dir = Path(__file__).resolve().parent.parent.parent / 'plots'
    out_dir.mkdir(exist_ok=True)
    return out_dir, filename or default_name


def forward_fill(a: np.ndarray) -> np.ndarray:
    """
    Vectorised 1-d forward-fill that leaves NaNs before the first valid.

    Args:
        a: Input array with potential NaN values

    Returns:
        Array with NaN values filled forward
    """
    mask = np.isnan(a)
    idx = np.where(mask, 0, np.arange(len(a)))
    np.maximum.accumulate(idx, out=idx)
    filled = a[idx]
    filled[mask & (idx == 0)] = np.nan
    return filled
