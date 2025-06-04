"""
Common utilities for plotting scripts.

This module contains shared constants, imports, and basic utility functions
used across different plotting scripts.
"""

import json
from pathlib import Path
from typing import Dict, List, Union, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Set default plotting style
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams['axes.grid'] = False

# Critical values for confidence intervals
CRIT = {0.9: 1.833, 0.95: 1.96, 0.99: 2.576}

# Standard colors for different methods
METHOD_COLORS = {
    'EWC': '#12939A', 'MAS': '#FF6E54', 'AGEM': '#FFA600',
    'L2': '#003F5C', 'PackNet': '#BC5090', 'ReDo': '#58508D', 'CBP': '#2F4B7C'
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
    mean = gaussian_filter1d(np.nanmean(data, axis=0), sigma=sigma)
    sd = gaussian_filter1d(np.nanstd(data, axis=0), sigma=sigma)
    ci = CRIT[conf] * sd / np.sqrt(data.shape[0])
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