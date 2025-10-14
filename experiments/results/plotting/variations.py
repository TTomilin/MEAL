#!/usr/bin/env python3
"""
Visualization script for comparing continual learning performance across different experimental variations.
This script generates plots to analyze how different variations (reward settings, complementary restrictions, 
ablations, etc.) affect learning performance across methods and levels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_series(fp: Path) -> List[float]:
    """Load a time series from JSON file."""
    if fp.suffix == ".json":
        with fp.open() as f:
            return json.load(f)
    elif fp.suffix == ".npz":
        return np.load(fp)["data"].tolist()
    else:
        raise ValueError(f"Unsupported file format: {fp.suffix}")


def get_experiment_folder(variation: str, level: Optional[int] = None) -> str:
    """Get the experiment folder name based on variation and level."""
    if variation == 'default':
        if level is not None:
            return f"level_{level}"
        else:
            return "main"
    elif variation == 'sparse':
        return "sparse_rewards"
    elif variation == 'individual':
        return "individual_rewards"
    elif variation == 'complementary_restrictions':
        return "complementary_restrictions"
    else:
        # For any other variation, use it directly as folder name
        return variation


def format_variation_name(variation: str) -> str:
    """Format variation name for display in plots."""
    if variation == 'default':
        return 'Shared Rewards'
    elif variation == 'sparse':
        return 'Sparse Rewards'
    elif variation == 'individual':
        return 'Individual Rewards'
    elif variation == 'complementary_restrictions':
        return 'Complementary Restrictions'
    else:
        # For any other variation, use title case with underscores replaced
        return variation.replace('_', ' ').title()


def load_variation_data(
        data_root: Path,
        algo: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        variations: List[str] = ['default', 'sparse', 'individual'],
        level: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Load data for all variations and methods.

    Args:
        data_root: Root directory containing experimental data
        algo: Algorithm name (e.g., 'ippo')
        methods: List of continual learning methods
        strategy: Strategy name (e.g., 'generate')
        seq_len: Sequence length
        seeds: List of random seeds
        variations: List of experimental variations to compare
        level: Difficulty level for default setting

    Returns:
        Nested dict: {variation: {method: {seed: training_curve}}}
    """
    data = {}

    for variation in variations:
        data[variation] = {}
        experiment_folder = get_experiment_folder(variation, level)

        for method in methods:
            data[variation][method] = {}

            base_folder = (
                data_root / algo / method / experiment_folder / f"{strategy}_{seq_len}"
            )

            for seed in seeds:
                seed_dir = base_folder / f"seed_{seed}"
                training_file = seed_dir / "training_soup.json"

                if training_file.exists():
                    try:
                        training_curve = load_series(training_file)
                        data[variation][method][seed] = np.array(training_curve)
                    except Exception as e:
                        print(f"[warn] Failed to load {training_file}: {e}")
                        continue
                else:
                    print(f"[warn] Missing training file: {training_file}")

    return data


def plot_training_curves_comparison(
        data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        methods: List[str],
        variations: List[str],
        seq_len: int,
        output_path: Path,
        title_suffix: str = "",
) -> None:
    """Plot training curves comparing different variations across methods."""

    # Create a single plot with multiple lines
    plt.figure(figsize=(12, 8))

    # Generate colors for variations and line styles for methods
    variation_colors = plt.cm.Set1(np.linspace(0, 1, len(variations)))
    method_styles = ['-', '--', '-.', ':']

    # Create color mapping for variations
    color_map = {var: variation_colors[i] for i, var in enumerate(variations)}

    for method_idx, method in enumerate(methods):
        line_style = method_styles[method_idx % len(method_styles)]

        for variation in variations:
            if variation not in data or method not in data[variation]:
                continue

            # Collect all curves for this method and variation
            curves = []
            for seed, curve in data[variation][method].items():
                if len(curve) > 0:
                    curves.append(curve)

            if not curves:
                continue

            # Pad curves to same length
            max_len = max(len(curve) for curve in curves)
            padded_curves = []
            for curve in curves:
                if len(curve) < max_len:
                    # Pad with last value
                    padded = np.pad(curve, (0, max_len - len(curve)), 
                                  mode='constant', constant_values=curve[-1])
                    padded_curves.append(padded)
                else:
                    padded_curves.append(curve)

            curves_array = np.array(padded_curves)
            mean_curve = np.mean(curves_array, axis=0)
            std_curve = np.std(curves_array, axis=0)

            x = np.arange(len(mean_curve))

            # Plot mean curve
            color = color_map[variation]
            label = f'{method} - {format_variation_name(variation)}'
            plt.plot(x, mean_curve, color=color, linestyle=line_style, 
                    label=label, linewidth=2, alpha=0.8)

            # Plot confidence interval
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
                          color=color, alpha=0.15)

    # Add vertical lines to separate tasks (use the first available curve for reference)
    if data and any(data.values()):
        first_method_data = next(iter(next(iter(data.values())).values()))
        if first_method_data:
            first_curve = next(iter(first_method_data.values()))
            if len(first_curve) > 0:
                chunk_size = len(first_curve) // seq_len
                for i in range(1, seq_len):
                    plt.axvline(x=i * chunk_size, color='gray', linestyle='--', alpha=0.5)

    plt.title(f'Training Curves Comparison{title_suffix}', fontsize=16)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'training_curves_comparison.pdf', bbox_inches='tight')
    plt.close()


def plot_final_performance_comparison(
        data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        methods: List[str],
        variations: List[str],
        output_path: Path,
        title_suffix: str = "",
) -> None:
    """Plot final performance comparison across variations."""

    # Collect final performance data
    performance_data = []

    for variation in variations:
        if variation not in data:
            continue

        for method in methods:
            if method not in data[variation]:
                continue

            final_performances = []
            for seed, curve in data[variation][method].items():
                if len(curve) > 0:
                    final_performances.append(curve[-1])

            for perf in final_performances:
                performance_data.append({
                    'Method': method,
                    'Variation': format_variation_name(variation),
                    'Final Performance': perf
                })

    if not performance_data:
        print("[warn] No performance data found for plotting")
        return

    df = pd.DataFrame(performance_data)

    # Create smaller bar chart
    plt.figure(figsize=(6, 4))

    # Box plot
    sns.boxplot(data=df, x='Method', y='Final Performance', hue='Variation')

    plt.title(f'Final Performance Comparison{title_suffix}', fontsize=14)
    plt.ylabel('Final Performance', fontsize=12)
    plt.xlabel('Continual Learning Method', fontsize=12)
    plt.legend(title='Variation', fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / 'final_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'final_performance_comparison.pdf', bbox_inches='tight')
    plt.close()


def plot_learning_efficiency(
        data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        methods: List[str],
        variations: List[str],
        seq_len: int,
        output_path: Path,
        title_suffix: str = "",
) -> None:
    """Plot learning efficiency (area under curve) comparison."""

    efficiency_data = []

    for variation in variations:
        if variation not in data:
            continue

        for method in methods:
            if method not in data[variation]:
                continue

            aucs = []
            for seed, curve in data[variation][method].items():
                if len(curve) > 0:
                    # Calculate AUC (area under curve)
                    auc = np.trapz(curve) / len(curve)
                    aucs.append(auc)

            for auc in aucs:
                efficiency_data.append({
                    'Method': method,
                    'Variation': format_variation_name(variation),
                    'Learning Efficiency (AUC)': auc
                })

    if not efficiency_data:
        print("[warn] No efficiency data found for plotting")
        return

    df = pd.DataFrame(efficiency_data)

    # Create smaller bar chart
    plt.figure(figsize=(6, 4))

    # Bar plot with error bars
    sns.barplot(data=df, x='Method', y='Learning Efficiency (AUC)', hue='Variation')

    plt.title(f'Learning Efficiency Comparison{title_suffix}', fontsize=14)
    plt.ylabel('Learning Efficiency (AUC)', fontsize=12)
    plt.xlabel('Continual Learning Method', fontsize=12)
    plt.legend(title='Variation', fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / 'learning_efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'learning_efficiency_comparison.pdf', bbox_inches='tight')
    plt.close()


def plot_task_specific_performance(
        data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        methods: List[str],
        variations: List[str],
        seq_len: int,
        output_path: Path,
        title_suffix: str = "",
) -> None:
    """Plot task-specific performance across variations."""

    # Create a single plot with multiple lines
    plt.figure(figsize=(10, 6))

    # Generate colors for variations and line styles for methods
    variation_colors = plt.cm.Set1(np.linspace(0, 1, len(variations)))
    method_styles = ['-', '--', '-.', ':']

    # Create color mapping for variations
    color_map = {var: variation_colors[i] for i, var in enumerate(variations)}

    for method_idx, method in enumerate(methods):
        line_style = method_styles[method_idx % len(method_styles)]

        for variation in variations:
            if variation not in data or method not in data[variation]:
                continue

            # Calculate task-specific performance
            task_performances = []

            for seed, curve in data[variation][method].items():
                if len(curve) == 0:
                    continue

                chunk_size = len(curve) // seq_len
                task_perfs = []

                for task in range(seq_len):
                    start_idx = task * chunk_size
                    end_idx = (task + 1) * chunk_size
                    task_curve = curve[start_idx:end_idx]

                    if len(task_curve) > 0:
                        # Use final performance of the task
                        task_perfs.append(task_curve[-1])
                    else:
                        task_perfs.append(0.0)

                task_performances.append(task_perfs)

            if task_performances:
                task_performances = np.array(task_performances)
                mean_perfs = np.mean(task_performances, axis=0)
                std_perfs = np.std(task_performances, axis=0)

                x = np.arange(seq_len)
                color = color_map[variation]
                label = f'{method} - {format_variation_name(variation)}'

                plt.plot(x, mean_perfs, 'o-', color=color, linestyle=line_style, 
                        label=label, linewidth=2, markersize=5, alpha=0.8)
                plt.fill_between(x, mean_perfs - std_perfs, mean_perfs + std_perfs, 
                              color=color, alpha=0.15)

    plt.title(f'Task-Specific Performance{title_suffix}', fontsize=16)
    plt.xlabel('Task Index', fontsize=12)
    plt.ylabel('Task Performance', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(seq_len))
    plt.tight_layout()
    plt.savefig(output_path / 'task_specific_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'task_specific_performance.pdf', bbox_inches='tight')
    plt.close()


def generate_summary_report(
        data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        methods: List[str],
        variations: List[str],
        output_path: Path,
) -> None:
    """Generate a summary report of the experimental variations comparison."""

    report_lines = []
    report_lines.append("# Experimental Variations Comparison Report\n")
    report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"Methods analyzed: {', '.join(methods)}\n")
    report_lines.append(f"Variations compared: {', '.join([v.replace('_', ' ').title() for v in variations])}\n\n")

    # Calculate summary statistics
    for variation in variations:
        if variation not in data:
            continue

        report_lines.append(f"## {format_variation_name(variation)}\n")

        for method in methods:
            if method not in data[variation]:
                continue

            curves = list(data[variation][method].values())
            if not curves:
                continue

            final_perfs = [curve[-1] for curve in curves if len(curve) > 0]
            aucs = [np.trapz(curve) / len(curve) for curve in curves if len(curve) > 0]

            if final_perfs and aucs:
                report_lines.append(f"### {method}\n")
                report_lines.append(f"- Final Performance: {np.mean(final_perfs):.3f} ± {np.std(final_perfs):.3f}\n")
                report_lines.append(f"- Learning Efficiency (AUC): {np.mean(aucs):.3f} ± {np.std(aucs):.3f}\n")
                report_lines.append(f"- Number of runs: {len(final_perfs)}\n\n")

    # Write report
    with open(output_path / 'summary_report.md', 'w') as f:
        f.writelines(report_lines)

    print(f"Summary report saved to: {output_path / 'summary_report.md'}")


def main():
    parser = argparse.ArgumentParser(description="Generate experimental variations comparison plots")
    parser.add_argument("--data_root", required=True, help="Root directory containing experimental data")
    parser.add_argument("--algo", required=True, help="Algorithm name (e.g., ippo)")
    parser.add_argument("--methods", nargs="+", required=True, help="List of CL methods")
    parser.add_argument("--strategy", required=True, help="Strategy name (e.g., generate)")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Random seeds")
    parser.add_argument("--variations", nargs="+", 
                       default=["default", "sparse", "individual", "complementary_restrictions"],
                       help="Experimental variations to compare (e.g., default, sparse, individual, complementary_restrictions)")
    parser.add_argument("--level", type=int, default=None, 
                       help="Difficulty level (1, 2, 3) for default setting")
    parser.add_argument("--output", default="plots/variation_comparison",
                       help="Output directory for plots")
    parser.add_argument("--title_suffix", default="", 
                       help="Suffix to add to plot titles")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(__file__).resolve().parent.parent / Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for variations: {args.variations}")
    print(f"Methods: {args.methods}")
    print(f"Output directory: {output_path}")

    # Load data
    data = load_variation_data(
        data_root=Path(args.data_root),
        algo=args.algo,
        methods=args.methods,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
        variations=args.variations,
        level=args.level,
    )

    # Check if data was loaded
    total_runs = sum(
        len(method_data) 
        for setting_data in data.values() 
        for method_data in setting_data.values()
    )

    if total_runs == 0:
        print("[error] No data found. Please check your data paths and parameters.")
        return

    print(f"Loaded data for {total_runs} runs")

    # Generate plots
    print("Generating training curves comparison...")
    plot_training_curves_comparison(
        data, args.methods, args.variations, args.seq_len, 
        output_path, args.title_suffix
    )

    print("Generating final performance comparison...")
    plot_final_performance_comparison(
        data, args.methods, args.variations, 
        output_path, args.title_suffix
    )

    print("Generating learning efficiency comparison...")
    plot_learning_efficiency(
        data, args.methods, args.variations, args.seq_len,
        output_path, args.title_suffix
    )

    print("Generating task-specific performance plots...")
    plot_task_specific_performance(
        data, args.methods, args.variations, args.seq_len,
        output_path, args.title_suffix
    )

    print("Generating summary report...")
    generate_summary_report(
        data, args.methods, args.variations, output_path
    )

    print(f"\nAll plots and reports saved to: {output_path}")
    print("Generated files:")
    for file in output_path.glob("*"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
