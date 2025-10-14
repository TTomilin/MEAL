#!/usr/bin/env python3
"""
Plot forward transfer vs forgetting scatter plot for the MARL continual-learning benchmark.

This script creates a scatter plot where:
- X-axis: Forward Transfer
- Y-axis: Forgetting
- Each method is represented as a dot on the graph

The metrics are calculated using the same logic as in results/numerical/results_table.py
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.results.plotting.utils import METHOD_COLORS, get_output_path


def load_series(fp: Path) -> np.ndarray:
    """Load a time series from a JSON file."""
    if fp.suffix == '.json':
        return np.array(json.loads(fp.read_text()), dtype=float)
    raise ValueError(f'Unsupported file suffix: {fp.suffix}')


def _mean_ci(series: List[float]) -> tuple:
    """Calculate mean and confidence interval."""
    if not series:
        return np.nan, np.nan
    mean = float(np.mean(series))
    if len(series) == 1:
        return mean, 0.0
    ci = 1.96 * np.std(series, ddof=1) / np.sqrt(len(series))
    return mean, float(ci)


def compute_metrics_simplified(
        data_root: Path,
        algo: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        end_window_evals: int = 10,
        level: int = 1,
) -> pd.DataFrame:
    """
    Compute metrics exactly like results_table.py with proper forward transfer calculation.
    """
    rows: list[dict[str, float]] = []

    # Load baseline data once for forward transfer calculation
    baseline_data = {}
    baseline_folder = (
        data_root
        / algo
        / "single"
        / f"level_{level}"
        / f"{strategy}_{seq_len}"
    )

    for seed in seeds:
        baseline_seed_dir = baseline_folder / f"seed_{seed}"
        if baseline_seed_dir.exists():
            # Load baseline training data for each task
            baseline_training_files = []
            for i in range(seq_len):
                baseline_file = baseline_seed_dir / f"{i}_training_soup.json"
                if baseline_file.exists():
                    baseline_training_files.append(load_series(baseline_file))
                else:
                    baseline_training_files.append(None)
            baseline_data[seed] = baseline_training_files

    for method in methods:
        AP_seeds, F_seeds, FT_seeds = [], [], []

        base_folder = (
                data_root
                / algo
                / method
                / f"level_{level}"
                / f"{strategy}_{seq_len}"
        )

        for seed in seeds:
            sd = base_folder / f"seed_{seed}"
            if not sd.exists():
                continue

            # 1) Plasticity training curve
            training_fp = sd / "training_soup.json"
            if not training_fp.exists():
                print(f"[warn] missing training_soup.json for {method} seed {seed}")
                continue
            training = load_series(training_fp)
            n_train = len(training)
            chunk = n_train // seq_len

            # 2) Per‑environment evaluation curves
            env_files = sorted([
                f for f in sd.glob("*_soup.*") if "training" not in f.name
            ])
            if len(env_files) != seq_len:
                print(
                    f"[warn] expected {seq_len} env files, found {len(env_files)} "
                    f"for {method} seed {seed}"
                )
                continue
            env_series = [load_series(f) for f in env_files]
            L = max(len(s) for s in env_series)
            env_mat = np.vstack([
                np.pad(s, (0, L - len(s)), constant_values=s[-1]) for s in env_series
            ])

            # Average Performance (AP) – last eval of mean curve
            AP_seeds.append(env_mat.mean(axis=0)[-1])

            # Forward Transfer (FT) – normalized area between CL and baseline curves
            if seed not in baseline_data:
                print(f"[warn] missing baseline data for seed {seed}")
                FT_seeds.append(np.nan)
                continue

            ft_vals = []
            for i in range(seq_len):
                # Calculate AUC for CL method (task i)
                start_idx = i * chunk
                end_idx = (i + 1) * chunk
                cl_task_curve = training[start_idx:end_idx]

                # AUCi = (1/τ) * ∫ pi(t) dt, where τ is the task duration
                # Using trapezoidal rule for numerical integration
                if len(cl_task_curve) > 1:
                    auc_cl = np.trapz(cl_task_curve) / len(cl_task_curve)
                else:
                    auc_cl = cl_task_curve[0] if len(cl_task_curve) == 1 else 0.0

                # Check if CL AUC is NaN or inf/-inf
                if np.isnan(auc_cl) or np.isinf(auc_cl):
                    print(f"[warn] CL AUC is NaN/inf/-inf for task {i}, seed {seed}, method {method}")
                    continue  # Skip this task

                # Calculate AUC for baseline method (task i)
                baseline_task_curve = baseline_data[seed][i]
                if baseline_task_curve is not None:
                    # Check if baseline data contains all NaN or inf/-inf values
                    if np.all(np.isnan(baseline_task_curve)):
                        print(f"[warn] baseline data contains all NaN for task {i}, seed {seed}")
                        continue  # Skip this task
                    elif np.all(np.isinf(baseline_task_curve)):
                        print(f"[warn] baseline data contains all inf/-inf for task {i}, seed {seed}")
                        continue  # Skip this task
                    elif np.all(np.isnan(baseline_task_curve) | np.isinf(baseline_task_curve)):
                        print(f"[warn] baseline data contains all NaN/inf/-inf for task {i}, seed {seed}")
                        continue  # Skip this task

                    if len(baseline_task_curve) > 1:
                        auc_baseline = np.trapz(baseline_task_curve) / len(baseline_task_curve)
                    else:
                        auc_baseline = baseline_task_curve[0] if len(baseline_task_curve) == 1 else 0.0

                    # Check if calculated AUC is NaN or inf/-inf
                    if np.isnan(auc_baseline):
                        print(f"[warn] baseline AUC is NaN for task {i}, seed {seed}")
                        continue  # Skip this task
                    elif np.isinf(auc_baseline):
                        print(f"[warn] baseline AUC is inf/-inf for task {i}, seed {seed}")
                        continue  # Skip this task

                    # Check if baseline performance is effectively 0
                    if abs(auc_baseline) < 1e-8:
                        print(f"[info] baseline AUC is effectively 0 ({auc_baseline}) for task {i}, seed {seed}, method {method} - skipping forward transfer calculation")
                        continue  # Skip this task

                    # Use direct ratio approach for forward transfer calculation
                    # FT_i = (AUC_CL - AUC_baseline) / max(|AUC_baseline|, ε)
                    epsilon = 1e-8
                    denominator = max(abs(auc_baseline), epsilon)
                    ft_i = (auc_cl - auc_baseline) / denominator

                    # Check if the final ft_i is inf/-inf or NaN
                    if np.isnan(ft_i) or np.isinf(ft_i):
                        print(f"[warn] Forward Transfer result is NaN/inf/-inf for task {i}, seed {seed}, method {method}")
                        # Skip this task - don't append to ft_vals
                    else:
                        ft_vals.append(ft_i)

                else:
                    print(f"[warn] missing baseline data for task {i}, seed {seed}")
                    # Don't append anything to ft_vals - skip this task

            if ft_vals:
                FT_seeds.append(float(np.nanmean(ft_vals)))
            else:
                FT_seeds.append(np.nan)

            # Forgetting (F) – drop from best‑ever to final performance
            f_vals = []
            final_idx = env_mat.shape[1] - 1
            fw_start = max(0, final_idx - end_window_evals + 1)
            for i in range(seq_len):
                final_avg = np.nanmean(env_mat[i, fw_start : final_idx + 1])
                best_perf = np.nanmax(env_mat[i, : final_idx + 1])
                f_vals.append(max(best_perf - final_avg, 0.0))
            F_seeds.append(float(np.nanmean(f_vals)))

        # Aggregate across seeds
        A_mean, A_ci = _mean_ci(AP_seeds)
        F_mean, F_ci = _mean_ci(F_seeds)
        FT_mean, FT_ci = _mean_ci(FT_seeds)

        rows.append(
            {
                "Method": method,
                "AveragePerformance": A_mean,
                "AveragePerformance_CI": A_ci,
                "Forgetting": F_mean,
                "Forgetting_CI": F_ci,
                "ForwardTransfer": FT_mean,
                "ForwardTransfer_CI": FT_ci,
            }
        )

    return pd.DataFrame(rows)


def parse_args():
    """Parse command line arguments for the forward transfer vs forgetting scatter plot."""
    parser = argparse.ArgumentParser(
        description="Plot forward transfer vs forgetting scatter plot for MARL continual-learning benchmark"
    )

    parser.add_argument("--data_root", required=True, help="Root directory containing the data")
    parser.add_argument("--algo", required=True, help="Algorithm name (e.g., 'ippo')")
    parser.add_argument("--arch", required=True, help="Architecture name (e.g., 'mlp')")
    parser.add_argument("--methods", nargs="+", required=True, help="List of methods to compare")
    parser.add_argument("--strategy", required=True, help="Strategy name (e.g., 'generate')")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="List of seeds")
    parser.add_argument("--levels", type=int, nargs="+", default=[1], help="Difficulty levels of the benchmark (can specify multiple levels)")
    parser.add_argument(
        "--end_window_evals",
        type=int,
        default=10,
        help="How many final eval points to average for forgetting calculation",
    )
    parser.add_argument("--plot_name", help="Custom name for the output plot")
    parser.add_argument("--title", help="Custom title for the plot")

    return parser.parse_args()


def main():
    """Main function to create the scatter plot."""
    args = parse_args()

    # Define marker shapes for different levels
    level_markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']  # circle, triangle_up, square, diamond, etc.

    # Create the scatter plot
    width = 4 if len(args.levels) == 1 else 5.25  # Increased width for multiple levels to accommodate legends
    fig, ax = plt.subplots(figsize=(width, 3.25))

    # Collect all data for summary statistics
    all_dfs = []

    # Keep track of unique methods and levels for legend creation
    unique_methods = set()
    plotted_method_level_combinations = set()

    # Process each level
    for level_idx, level in enumerate(args.levels):
        # Compute metrics using simplified logic for available data structure
        df = compute_metrics_simplified(
            data_root=Path(args.data_root),
            algo=args.algo,
            methods=args.methods,
            strategy=args.strategy,
            seq_len=args.seq_len,
            seeds=args.seeds,
            end_window_evals=args.end_window_evals,
            level=level,
        )

        # Pretty-print method names (same as in results_table.py)
        df["Method"] = df["Method"].replace({"Online_EWC": "Online EWC"})
        df["Level"] = level  # Add level information to the dataframe
        all_dfs.append(df)

        # Get marker shape for this level
        marker = level_markers[level_idx % len(level_markers)]

        # Plot each method as a dot
        for _, row in df.iterrows():
            method = row["Method"]
            ft = row["ForwardTransfer"]
            forgetting = row["Forgetting"]

            # Skip if either metric is NaN
            if np.isnan(ft) or np.isnan(forgetting):
                print(f"Warning: Skipping {method} level {level} due to NaN values (FT: {ft}, F: {forgetting})")
                continue

            # Get color for the method
            # Handle special case for Online EWC to match METHOD_COLORS key
            if method == "Online EWC":
                color_key = "Online_EWC"
            else:
                color_key = method.upper().replace(" ", "_")
            color = METHOD_COLORS.get(color_key, '#333333')

            # Track unique methods for legend
            unique_methods.add(method)
            plotted_method_level_combinations.add((method, level))

            # For single level, use method name as label; for multiple levels, don't use label (we'll create custom legend)
            if len(args.levels) == 1:
                label = method
            else:
                label = None  # No label - we'll create custom legend

            # Plot the point with level-specific marker
            ax.scatter(ft, forgetting, color=color, s=150, alpha=0.8, label=label, 
                      edgecolors='black', linewidth=1, marker=marker)

            # Add method name as text annotation only for single level
            if len(args.levels) == 1:
                ax.annotate(method, (ft, forgetting), xytext=(0, 8), textcoords='offset points', 
                           fontsize=10, alpha=0.8, ha='center')

    # Create custom legend for multiple levels
    if len(args.levels) > 1:
        # Create legend handles for methods (colors)
        method_handles = []
        for method in sorted(unique_methods):
            # Get color for the method
            if method == "Online EWC":
                color_key = "Online_EWC"
            else:
                color_key = method.upper().replace(" ", "_")
            color = METHOD_COLORS.get(color_key, '#333333')

            # Create a dummy scatter point for the legend
            handle = ax.scatter([], [], color=color, s=150, alpha=0.8, 
                              edgecolors='black', linewidth=1, marker='o', label=method)
            method_handles.append(handle)

        # Create legend handles for levels (shapes)
        level_handles = []
        for level_idx, level in enumerate(sorted(args.levels)):
            marker = level_markers[level_idx % len(level_markers)]
            # Use a neutral gray color for shape legend
            handle = ax.scatter([], [], color='gray', s=150, alpha=0.8,
                              edgecolors='black', linewidth=1, marker=marker, label=f'Level {level}')
            level_handles.append(handle)

        # Create the legend with two columns
        method_legend = ax.legend(handles=method_handles, title='Methods', 
                                loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
        level_legend = ax.legend(handles=level_handles, title='Levels',
                               loc='upper left', bbox_to_anchor=(1.02, 0.6), frameon=True)

        # Add both legends to the plot
        ax.add_artist(method_legend)  # Add the first legend back since the second one replaces it

    # Customize the plot
    ax.set_xlabel('Forward Transfer ↑', fontsize=12)
    ax.set_ylabel('Forgetting ↓', fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Add reference lines at zero
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Adjust layout to prevent legend cutoff
    if len(args.levels) > 1:
        # For multiple levels with external legends, manually adjust subplot to make room
        # plt.subplots_adjust(right=0.6)  # Leave space on the right for legends
        plt.tight_layout(rect=[0, 0, 0.95, 1])
    else:
        # For single level, use tight_layout as normal
        plt.tight_layout()

    # Save the plot
    out_dir, plot_name = get_output_path(args.plot_name, "forward_transfer_vs_forgetting")

    # Create filename suffix based on levels
    if len(args.levels) == 1:
        level_suffix = f"_level{args.levels[0]}"
    else:
        level_suffix = f"_levels{'_'.join(map(str, args.levels))}"

    # Save with different bbox_inches settings depending on layout
    if len(args.levels) > 1:
        # For multiple levels, don't use bbox_inches='tight' to preserve manual layout adjustments
        plt.savefig(out_dir / f"{plot_name}{level_suffix}.png", dpi=300)
        plt.savefig(out_dir / f"{plot_name}{level_suffix}.pdf")
    else:
        # For single level, use bbox_inches='tight' for optimal cropping
        plt.savefig(out_dir / f"{plot_name}{level_suffix}.png", dpi=300, bbox_inches='tight')
        plt.savefig(out_dir / f"{plot_name}{level_suffix}.pdf", bbox_inches='tight')

    print(f"Plot saved to {out_dir / plot_name}{level_suffix}.png and {out_dir / plot_name}{level_suffix}.pdf")

    # Display summary statistics
    print("\nSummary Statistics:")
    print("=" * 70)
    if len(args.levels) > 1:
        print(f"{'Method':<15} {'Level':<6} {'Forward Transfer':<18} {'Forgetting':<12}")
        print("-" * 70)
        for df in all_dfs:
            level = df['Level'].iloc[0]  # All rows in df have the same level
            for _, row in df.iterrows():
                method = row["Method"]
                ft = row["ForwardTransfer"]
                ft_ci = row["ForwardTransfer_CI"]
                forgetting = row["Forgetting"]
                forgetting_ci = row["Forgetting_CI"]

                ft_str = f"{ft:.3f} ± {ft_ci:.3f}" if not np.isnan(ft) and not np.isnan(ft_ci) else "N/A"
                f_str = f"{forgetting:.3f} ± {forgetting_ci:.3f}" if not np.isnan(forgetting) and not np.isnan(forgetting_ci) else "N/A"

                print(f"{method:<15} {level:<6} {ft_str:<18} {f_str:<12}")
    else:
        print(f"{'Method':<15} {'Forward Transfer':<18} {'Forgetting':<12}")
        print("-" * 70)
        df = all_dfs[0]  # Only one level
        for _, row in df.iterrows():
            method = row["Method"]
            ft = row["ForwardTransfer"]
            ft_ci = row["ForwardTransfer_CI"]
            forgetting = row["Forgetting"]
            forgetting_ci = row["Forgetting_CI"]

            ft_str = f"{ft:.3f} ± {ft_ci:.3f}" if not np.isnan(ft) and not np.isnan(ft_ci) else "N/A"
            f_str = f"{forgetting:.3f} ± {forgetting_ci:.3f}" if not np.isnan(forgetting) and not np.isnan(forgetting_ci) else "N/A"

            print(f"{method:<15} {ft_str:<18} {f_str:<12}")

    plt.show()


if __name__ == "__main__":
    main()
