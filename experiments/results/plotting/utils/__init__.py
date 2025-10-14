"""
Plotting utilities package for JAXOvercooked.

This package contains common utilities for plotting results from JAXOvercooked experiments.
"""

# Import key functions and constants from modules
from .common import (
    CRIT, METHOD_COLORS, LEVEL_COLORS,
    load_series, smooth_and_ci, get_output_path, forward_fill
)

from .data_loading import (
    collect_runs, collect_env_curves, collect_cumulative_runs
)

from .plotting import (
    setup_figure, add_task_boundaries, setup_task_axes,
    plot_method_curves, save_plot, finalize_plot
)

from .plasticity_utils import (
    collect_plasticity_runs, setup_plasticity_grid, collect_training_data
)

from .cli import (
    create_base_parser, add_common_args, add_metric_arg, add_repeat_sequence_arg,
    create_parser_with_common_args, create_plasticity_parser, create_eval_parser
)

__all__ = [
    # From common
    'CRIT', 'METHOD_COLORS', 'LEVEL_COLORS', 'load_series', 'smooth_and_ci',
    'get_output_path', 'forward_fill',

    # From data_loading
    'collect_runs', 'collect_env_curves', 'collect_cumulative_runs',

    # From plotting
    'setup_figure', 'add_task_boundaries', 'setup_task_axes',
    'plot_method_curves', 'save_plot', 'finalize_plot',

    # From plasticity_utils
    'collect_plasticity_runs', 'setup_plasticity_grid', 'collect_training_data',

    # From cli
    'create_base_parser', 'add_common_args', 'add_metric_arg', 'add_repeat_sequence_arg',
    'create_parser_with_common_args', 'create_plasticity_parser', 'create_eval_parser'
]
