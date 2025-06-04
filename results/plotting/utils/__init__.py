"""
Plotting utilities package for JAXOvercooked.

This package contains common utilities for plotting results from JAXOvercooked experiments.
"""

# Import key functions and constants from modules
from .common import (
    CRIT, METHOD_COLORS, 
    load_series, smooth_and_ci, get_output_path, forward_fill
)

from .data_loading import (
    collect_runs, collect_env_curves, collect_cumulative_runs
)

from .plotting import (
    setup_figure, add_task_boundaries, setup_task_axes,
    plot_method_curves, save_plot, finalize_plot
)

__all__ = [
    # From common
    'CRIT', 'METHOD_COLORS', 'load_series', 'smooth_and_ci', 
    'get_output_path', 'forward_fill',
    
    # From data_loading
    'collect_runs', 'collect_env_curves', 'collect_cumulative_runs',
    
    # From plotting
    'setup_figure', 'add_task_boundaries', 'setup_task_axes',
    'plot_method_curves', 'save_plot', 'finalize_plot'
]