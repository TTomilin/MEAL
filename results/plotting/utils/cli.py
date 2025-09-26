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
    parser.add_argument("--steps_per_task", type=float, default=1e7, help="Steps per task (x-axis scaling)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Seeds to include")
    parser.add_argument("--sigma", type=float, default=1.5, help="Gaussian smoothing parameter")
    parser.add_argument("--level", type=int, default=1, help="Difficulty level of the environment")
    parser.add_argument("--confidence", type=float, default=0.95, choices=[0.9, 0.95, 0.99], help="Confidence level")
    parser.add_argument("--plot_name", default=None, help="Custom plot name (default: auto-generated)")


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