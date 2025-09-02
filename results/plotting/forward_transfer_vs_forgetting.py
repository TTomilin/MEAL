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

from results.plotting.utils import METHOD_COLORS, get_output_path


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

                # Calculate AUC for baseline method (task i)
                baseline_task_curve = baseline_data[seed][i]
                if baseline_task_curve is not None:
                    if len(baseline_task_curve) > 1:
                        auc_baseline = np.trapz(baseline_task_curve) / len(baseline_task_curve)
                    else:
                        auc_baseline = baseline_task_curve[0] if len(baseline_task_curve) == 1 else 0.0

                    # Calculate Forward Transfer: FTi = (AUCi - AUCb_i) / (1 - AUCb_i)
                    denominator = 1.0 - auc_baseline
                    if abs(denominator) > 1e-8:  # Avoid division by zero
                        ft_i = (auc_cl - auc_baseline) / denominator
                        ft_vals.append(ft_i)
                    else:
                        # If baseline AUC is 1.0, forward transfer is undefined
                        print(f"[warn] baseline AUC = 1.0 for task {i}, seed {seed}, method {method}")
                else:
                    print(f"[warn] missing baseline data for task {i}, seed {seed}")

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
    parser.add_argument("--level", type=int, default=1, help="Difficulty level of the benchmark")
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

    # Compute metrics using simplified logic for available data structure
    df = compute_metrics_simplified(
        data_root=Path(args.data_root),
        algo=args.algo,
        methods=args.methods,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
        end_window_evals=args.end_window_evals,
        level=args.level,
    )

    # Pretty-print method names (same as in results_table.py)
    df["Method"] = df["Method"].replace({"Online_EWC": "Online EWC"})

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(4.5, 3.25))

    # Plot each method as a dot
    for _, row in df.iterrows():
        method = row["Method"]
        ft = row["ForwardTransfer"]
        forgetting = row["Forgetting"]

        # Skip if either metric is NaN
        if np.isnan(ft) or np.isnan(forgetting):
            print(f"Warning: Skipping {method} due to NaN values (FT: {ft}, F: {forgetting})")
            continue

        # Get color for the method
        # Handle special case for Online EWC to match METHOD_COLORS key
        if method == "Online EWC":
            color_key = "Online_EWC"
        else:
            color_key = method.upper().replace(" ", "_")
        color = METHOD_COLORS.get(color_key, '#333333')

        # Plot the point
        ax.scatter(ft, forgetting, color=color, s=150, alpha=0.8, label=method, edgecolors='black', linewidth=1)

        # Add method name as text annotation
        ax.annotate(method, (ft, forgetting), xytext=(5, 5), textcoords='offset points', fontsize=10, alpha=0.8)

    # Customize the plot
    ax.set_xlabel('Forward Transfer ↑', fontsize=12)
    ax.set_ylabel('Forgetting ↓', fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Add reference lines at zero
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save the plot
    out_dir, plot_name = get_output_path(args.plot_name, "forward_transfer_vs_forgetting")

    plt.savefig(out_dir / f"{plot_name}_level{args.level}.png", dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / f"{plot_name}_level{args.level}.pdf", bbox_inches='tight')

    print(f"Plot saved to {out_dir / plot_name}.png and {out_dir / plot_name}.pdf")

    # Display summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    print(f"{'Method':<15} {'Forward Transfer':<18} {'Forgetting':<12}")
    print("-" * 50)
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
