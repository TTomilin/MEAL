"""
Plotting utilities for visualization scripts.

This module contains functions for creating and configuring plots,
including setting up axes, adding task boundaries, and other common
plotting operations.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

from .common import CRIT, METHOD_COLORS, smooth_and_ci

def setup_figure(width: int = 10, height: int = 4) -> Tuple[plt.Figure, plt.Axes]:
    """
    Set up a figure with the given dimensions.
    
    Args:
        width: Figure width in inches
        height: Figure height in inches
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax

def add_task_boundaries(ax: plt.Axes, boundaries: List[float], 
                        color: str = 'gray', linestyle: str = '--', linewidth: float = 0.5):
    """
    Add vertical lines at task boundaries.
    
    Args:
        ax: Matplotlib axes to add lines to
        boundaries: List of x-coordinates for boundaries
        color: Line color
        linestyle: Line style
        linewidth: Line width
    """
    for b in boundaries[1:-1]:  # Skip first and last boundaries
        ax.axvline(b, color=color, ls=linestyle, lw=linewidth)

def setup_task_axes(ax: plt.Axes, boundaries: List[float], task_count: int, 
                    fontsize: int = 10, task_colors: Optional[List[str]] = None):
    """
    Set up primary and secondary axes for task visualization.
    
    Args:
        ax: Primary matplotlib axes
        boundaries: List of x-coordinates for task boundaries
        task_count: Number of tasks
        fontsize: Font size for task labels
        task_colors: Optional list of colors for task labels
    
    Returns:
        Secondary axes object
    """
    # Primary x-axis (environment steps)
    ax.set_xticks(boundaries)
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    
    # Secondary x-axis (task labels)
    secax = ax.secondary_xaxis('top')
    mids = [(boundaries[i] + boundaries[i + 1]) / 2.0 for i in range(task_count)]
    secax.set_xticks(mids)
    secax.set_xticklabels([f"Task {i + 1}" for i in range(task_count)], fontsize=fontsize)
    secax.tick_params(axis='x', length=0)
    
    # Color task labels if colors are provided
    if task_colors:
        for idx, label in enumerate(secax.get_xticklabels()):
            label.set_color(task_colors[idx])
    
    return secax

def plot_method_curves(ax: plt.Axes, methods: List[str], data_dict: Dict[str, np.ndarray], 
                      x_values: np.ndarray, sigma: float, confidence: float):
    """
    Plot curves for multiple methods with confidence intervals.
    
    Args:
        ax: Matplotlib axes to plot on
        methods: List of method names
        data_dict: Dictionary mapping method names to data arrays
        x_values: X-axis values
        sigma: Smoothing parameter
        confidence: Confidence level for intervals
    """
    for method in methods:
        data = data_dict[method]
        mu, ci = smooth_and_ci(data, sigma, confidence)
        
        color = METHOD_COLORS.get(method)
        ax.plot(x_values, mu, label=method, color=color)
        ax.fill_between(x_values, mu - ci, mu + ci, color=color, alpha=0.2)

def save_plot(fig: plt.Figure, output_dir: Path, filename: str, formats: List[str] = ['png', 'pdf']):
    """
    Save a plot in multiple formats.
    
    Args:
        fig: Matplotlib figure to save
        output_dir: Directory to save to
        filename: Base filename (without extension)
        formats: List of file formats to save as
    """
    output_dir.mkdir(exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{filename}.{fmt}")

def finalize_plot(ax: plt.Axes, title: Optional[str] = None, 
                 xlabel: str = "Environment Steps", 
                 ylabel: str = "Score", 
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 legend_loc: str = 'best',
                 legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
                 legend_ncol: int = 1):
    """
    Finalize a plot with labels, limits, and legend.
    
    Args:
        ax: Matplotlib axes to finalize
        title: Optional plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        xlim: Optional x-axis limits as (min, max)
        ylim: Optional y-axis limits as (min, max)
        legend_loc: Legend location
        legend_bbox_to_anchor: Optional legend anchor point
        legend_ncol: Number of columns in the legend
    """
    if title:
        ax.set_title(title)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if xlim:
        ax.set_xlim(xlim)
    
    if ylim:
        ax.set_ylim(ylim)
    
    if legend_bbox_to_anchor:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol)
    else:
        ax.legend(loc=legend_loc, ncol=legend_ncol)
    
    plt.tight_layout()