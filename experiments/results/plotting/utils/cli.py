"""
Command-line interface utilities for plotting scripts.

This module provides functions for creating argument parsers with common
arguments used across different plotting scripts.
"""

import argparse
from typing import List, Optional, Dict, Any, Union


def create_base_parser(description: str = "Plot data from MARL continual-learning benchmark") -> argparse.ArgumentParser:
    """
    Create a base argument parser with common formatting.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        ArgumentParser with common formatting
    """
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common arguments used across most plotting scripts.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument("--data_root", default="data", help="Root folder with algo/method runs")
    parser.add_argument("--algo", default="ippo", help="Algorithm name")
    parser.add_argument("--methods", nargs="+", help="Method names to plot")
    parser.add_argument("--strategy", default='generate', help="Training strategy (e.g., 'generate', 'ordered')")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length (number of tasks)")
    parser.add_argument("--steps_per_task", type=float, default=1e8, help="Steps per task (x-axis scaling)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Seeds to include")
    parser.add_argument("--sigma", type=float, default=1.5, help="Gaussian smoothing parameter")
    parser.add_argument("--level", type=int, default=1, help="Difficulty level of the environment")
    parser.add_argument("--agents", type=int, default=2, help="Number of agents in the environment")
    parser.add_argument("--confidence", type=float, default=0.95, choices=[0.9, 0.95, 0.99], help="Confidence level")
    parser.add_argument("--plot_name", default=None, help="Custom plot name (default: auto-generated)")


def add_numerical_data_args(
    parser: argparse.ArgumentParser,
    seq_len_default: int = 20,
    seeds_default: List[int] = None,
    required: bool = False,
) -> None:
    """
    Add the data-selection arguments shared by every experiments/results/numerical/*.py
    script: --data_root, --algo, --strategy, --seq_len, --seeds. Deliberately omits
    --methods/--method and any --level(s)/--agents/--num_partners variant, since those
    differ too much in naming and defaults across scripts to share; add those in the
    calling script after this.

    Args:
        parser: ArgumentParser to add arguments to
        seq_len_default: Default --seq_len value
        seeds_default: Default --seeds value (default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        required: If True, --data_root/--algo are required with no default (matches the
            handful of scripts that historically used required=True instead of a default)
    """
    if seeds_default is None:
        seeds_default = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    parser.add_argument("--data_root", required=required, default=None if required else "data",
                        help="Root directory containing the data")
    parser.add_argument("--algo", required=required, default=None if required else "ippo",
                        help="Algorithm name")
    parser.add_argument("--strategy", default="generate", help="Strategy name")
    parser.add_argument("--seq_len", type=int, default=seq_len_default, help="Sequence length")
    parser.add_argument("--seeds", type=int, nargs="+", default=seeds_default, help="Seeds to include")


def add_forgetting_args(parser: argparse.ArgumentParser, default: str = "peak_final",
                        end_window_default: int = 10) -> None:
    """
    Add --end_window_evals / --forgetting_formula, shared by every script that reports
    Forgetting (F).

    Args:
        parser: ArgumentParser to add arguments to
        default: Which formula this script's own results were actually generated with
            ('weighted' or 'peak_final') -- confirm via git history before changing it,
            see experiments/results/plotting/utils/metrics.py for the two formulas.
        end_window_default: Default --end_window_evals value
    """
    parser.add_argument(
        "--end_window_evals", type=int, default=end_window_default,
        help="How many final eval points to average for Forgetting (only used by the 'peak_final' formula)",
    )
    parser.add_argument(
        "--forgetting_formula", choices=["weighted", "peak_final"], default=default,
        help=f"Forgetting formula ('{default}' is what actually generated this script's "
             "results, confirmed via git history). 'weighted' is normalized and decay-"
             "weighted; 'peak_final' is the unnormalized peak-minus-final drop. See "
             "experiments/results/plotting/utils/metrics.py.",
    )


def add_metric_arg(parser: argparse.ArgumentParser, choices: List[str] = None, default: str = None) -> None:
    """
    Add a metric argument with customizable choices.
    
    Args:
        parser: ArgumentParser to add the argument to
        choices: List of valid metric choices
        default: Default metric value
    """
    if choices is None:
        choices = ["reward", "soup"]
    if default is None:
        default = choices[0]
    
    parser.add_argument("--metric", choices=choices, default=default, help="Metric to plot")


def add_repeat_sequence_arg(parser: argparse.ArgumentParser, default: int = 1) -> None:
    """
    Add a repeat_sequence argument for plasticity plots.
    
    Args:
        parser: ArgumentParser to add the argument to
        default: Default value for repeat_sequence
    """
    parser.add_argument("--repeat_sequence", type=int, default=default, 
                        help="Sequence repetitions inside the file")


def create_parser_with_common_args(description: str = "Plot data from MARL continual-learning benchmark") -> argparse.ArgumentParser:
    """
    Create a parser with common arguments already added.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        ArgumentParser with common arguments
    """
    parser = create_base_parser(description)
    add_common_args(parser)
    return parser


def create_plasticity_parser(description: str = "Plot plasticity data") -> argparse.ArgumentParser:
    """
    Create a parser specifically for plasticity plotting scripts.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        ArgumentParser with plasticity-specific arguments
    """
    parser = create_parser_with_common_args(description)
    add_repeat_sequence_arg(parser)
    return parser


def create_eval_parser(description: str = "Plot evaluation metrics", 
                      metric_choices: List[str] = None) -> argparse.ArgumentParser:
    """
    Create a parser specifically for evaluation plotting scripts.
    
    Args:
        description: Description for the argument parser
        metric_choices: List of valid metric choices
        
    Returns:
        ArgumentParser with evaluation-specific arguments
    """
    parser = create_parser_with_common_args(description)
    add_metric_arg(parser, choices=metric_choices)
    return parser