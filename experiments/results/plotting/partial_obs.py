"""
Plotting script to compare performance of:
1. Full observable IPPO
2. Partially observable IPPO  
3. Partially observable MAPPO

Creates a bar chart with average normalized scores for each level.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Dict

import matplotlib.pyplot as plt
import numpy as np

from experiments.results.plotting.utils import (
    collect_cumulative_runs,
    setup_figure,
    save_plot,
    finalize_plot,
    create_base_parser,
    CRIT,
)


def _parse_args():
    """Parse command line arguments"""
    parser = create_base_parser(
        description="Compare performance of Full Observable IPPO, Partially Observable IPPO, and Partially Observable MAPPO"
    )

    parser.add_argument(
        "--data_root",
        required=True,
        help="Root folder with algo/method runs"
    )

    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Difficulty levels to compare (default: [1, 2, 3])"
    )

    parser.add_argument(
        "--strategy",
        required=True,
        help="Training strategy (e.g., 'generate', 'ordered')"
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        required=True,
        help="Sequence length (number of tasks)"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Seeds to include (default: [1, 2, 3, 4, 5])"
    )

    parser.add_argument(
        "--metric",
        choices=["reward", "soup"],
        default="soup",
        help="Metric to plot (default: reward)"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Window size for averaging final scores (default: 10)"
    )

    parser.add_argument(
        "--plot_name",
        default="partial_observability",
        help="Custom plot name"
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize scores to [0, 1] range"
    )

    parser.add_argument(
        "--show_values",
        action="store_true",
        help="Show mean values on top of bars"
    )

    return parser.parse_args()


def _final_scores(
        root: Path,
        algo: str,
        method: str,
        experiment: str,
        strategy: str,
        metric: str,
        seq_len: int,
        seeds: Sequence[int],
        window: int,
) -> np.ndarray:
    """Return 1D array (one element per seed) of average of last *window* points."""
    try:
        curves = collect_cumulative_runs(
            root,
            algo,
            method,
            experiment,
            strategy,
            metric,
            seq_len,
            list(seeds),
        )  # (S, T)
        if curves.size == 0:
            return np.array([])
        window = min(window, curves.shape[1])
        finals = np.nanmean(curves[:, -window:], axis=1)
        return finals[np.isfinite(finals)]
    except Exception as e:
        print(f"Warning: Could not load data for {method} {experiment}: {e}")
        return np.array([])


def _collect_method_data(
        data_root: Path,
        levels: List[int],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        metric: str,
        window: int,
        normalize: bool = False
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Collect data for all three methods across all levels.

    Returns:
        Dictionary with structure: {method_name: {level_name: scores_array}}
    """

    # Define the three methods to compare with their specific configurations
    methods_config = {
        "IPPO (FO)": {
            "algo": "ippo",
            "method": "EWC",
            "strategy": "generate",
            "seq_len": 20,
        },
        "IPPO (PO)": {
            "algo": "ippo",
            "method": "EWC_partial",
            "strategy": "generate",
            "seq_len": 20,
        },
        "MAPPO (PO)": {
            "algo": "mappo",
            "method": "EWC_partial",
            "strategy": "generate",
            "seq_len": 20,
        }
    }

    data = {}
    all_scores = []  # For normalization

    # First pass: collect all data
    for method_name, config in methods_config.items():
        data[method_name] = {}
        for level in levels:
            experiment = f"level_{level}"
            scores = _final_scores(
                data_root,
                algo=config["algo"],
                method=config["method"],
                experiment=experiment,
                strategy=config["strategy"],
                metric=metric,
                seq_len=config["seq_len"],
                seeds=seeds,
                window=window,
            )
            data[method_name][f"Level {level}"] = scores
            if len(scores) > 0:
                all_scores.extend(scores)

    # Normalize if requested
    if normalize and len(all_scores) > 0:
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        score_range = max_score - min_score

        if score_range > 0:
            for method_name in data:
                for level_name in data[method_name]:
                    if len(data[method_name][level_name]) > 0:
                        data[method_name][level_name] = (
                                                                data[method_name][level_name] - min_score
                                                        ) / score_range

    return data


def _plot_comparison_bars(
        ax: plt.Axes,
        data: Dict[str, Dict[str, np.ndarray]],
        show_values: bool = False
):
    """
    Plot comparison bars for the three methods across levels.
    """
    methods = list(data.keys())
    levels = list(data[methods[0]].keys()) if methods else []

    if not methods or not levels:
        print("Warning: No data to plot")
        return

    # Set up bar positions
    n_levels = len(levels)
    n_methods = len(methods)
    bar_width = 0.25
    x = np.arange(n_levels)

    # Define colors for methods
    method_colors = {
        "IPPO (FO)": "#2E86AB",  # Blue
        "IPPO (PO)": "#A23B72",  # Purple
        "MAPPO (PO)": "#F18F01"  # Orange
    }

    # Plot bars for each method
    for i, method in enumerate(methods):
        means = []
        cis = []

        for level in levels:
            scores = data[method][level]
            if len(scores) > 0:
                mean_score = np.mean(scores)
                ci = (CRIT[0.95] * np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
            else:
                mean_score = 0.0
                ci = 0.0

            means.append(mean_score)
            cis.append(ci)

        # Plot bars
        bars = ax.bar(
            x + i * bar_width,
            means,
            bar_width,
            yerr=cis,
            capsize=5,
            label=method,
            color=method_colors.get(method, f"C{i}"),
            edgecolor="black",
            linewidth=0.5
        )

        # Add value labels on bars if requested
        if show_values:
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + bar.get_height() * 0.01,
                    f"{mean_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

    # Customize the plot
    ax.set_ylabel("Average Score")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(levels)
    ax.legend()

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')


def plot():
    """Main plotting function"""
    args = _parse_args()

    data_root = Path(__file__).resolve().parent.parent / args.data_root

    # Collect data for all methods and levels
    data = _collect_method_data(
        data_root=data_root,
        levels=args.levels,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
        metric=args.metric,
        window=args.window,
        normalize=args.normalize
    )

    # Create the plot
    fig, ax = setup_figure(width=5, height=3.5)
    _plot_comparison_bars(ax, data, show_values=args.show_values)

    # Set title and labels

    finalize_plot(
        ax,
        xlabel="",
        ylabel="Normalized Average Score",
    )

    # Save the plot
    out_dir = Path(__file__).resolve().parent.parent / "plots"
    stem = args.plot_name or "ippo_mappo_comparison"
    if args.normalize:
        stem += "_normalized"

    save_plot(fig, out_dir, stem)
    plt.show()


if __name__ == "__main__":
    plot()
