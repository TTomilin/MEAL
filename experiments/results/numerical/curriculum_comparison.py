from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd


def load_series(fp: Path) -> List[float]:
    """Load a time series from JSON file."""
    with fp.open() as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _mean_ci(series: List[float]) -> Tuple[float, float]:
    """Compute mean and 95% confidence interval."""
    if not series:
        return np.nan, np.nan
    arr = np.array(series)
    mean = np.mean(arr)
    ci = 1.96 * np.std(arr) / np.sqrt(len(arr))
    return mean, ci


def compute_metrics_for_task_range(
        data_root: Path,
        algo: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        task_range: Tuple[int, int],
        level: Optional[int] = None,
        end_window_evals: int = 10,
) -> pd.DataFrame:
    """
    Compute continual learning metrics for experiments with a specific strategy on a specific task range.

    Args:
        data_root: Root directory containing experimental data
        algo: Algorithm name (e.g., 'ippo')
        methods: List of continual learning methods (e.g., ['EWC', 'MAS', 'L2'])
        strategy: Strategy name (e.g., 'curriculum', 'generate')
        seq_len: Sequence length
        seeds: List of random seeds
        task_range: Tuple of (start_idx, end_idx) for task range (inclusive)
        level: Difficulty level (1, 2, 3) for level-based experiments
        end_window_evals: Number of final evaluations to average for forgetting metric

    Returns:
        DataFrame with computed metrics for the specified task range
    """
    rows: list[dict[str, float]] = []

    # Determine experiment folder based on level
    experiment_folder = f"level_{level}" if level is not None else "main"

    start_idx, end_idx = task_range
    task_indices = list(range(start_idx, end_idx + 1))

    print(f"Computing metrics for {strategy} strategy on tasks {start_idx+1}-{end_idx+1} (indices {start_idx}-{end_idx})")

    # Load baseline data once for forward transfer calculation
    baseline_data = {}

    # For generate strategy, use appropriate difficulty levels based on task range
    if strategy == "generate":
        for seed in seeds:
            baseline_training_files = []
            for i in task_indices:
                # Determine the appropriate difficulty level for this task
                if 0 <= i <= 4:  # Tasks 1-5 (easy)
                    task_experiment_folder = "level_1"
                elif 5 <= i <= 9:  # Tasks 6-10 (medium)
                    task_experiment_folder = "level_2"
                elif 10 <= i <= 14:  # Tasks 11-15 (hard)
                    task_experiment_folder = "level_3"
                else:
                    # Fallback to main for other task indices
                    task_experiment_folder = "main"

                baseline_folder = (
                    data_root
                    / algo
                    / "single"
                    / task_experiment_folder
                    / f"{strategy}_{seq_len}"
                )
                baseline_seed_dir = baseline_folder / f"seed_{seed}"

                if baseline_seed_dir.exists():
                    baseline_file = baseline_seed_dir / f"{i}_training_soup.json"
                    if baseline_file.exists():
                        baseline_training_files.append(load_series(baseline_file))
                    else:
                        baseline_training_files.append(None)
                else:
                    baseline_training_files.append(None)
            baseline_data[seed] = baseline_training_files
    else:
        # For other strategies (like curriculum), use the original logic
        baseline_folder = (
            data_root
            / algo
            / "single"
            / experiment_folder
            / f"{strategy}_{seq_len}"
        )

        for seed in seeds:
            baseline_seed_dir = baseline_folder / f"seed_{seed}"
            if baseline_seed_dir.exists():
                # Load baseline training data for each task in the range
                baseline_training_files = []
                for i in task_indices:
                    baseline_file = baseline_seed_dir / f"{i}_training_soup.json"
                    if baseline_file.exists():
                        baseline_training_files.append(load_series(baseline_file))
                    else:
                        baseline_training_files.append(None)
                baseline_data[seed] = baseline_training_files

    for method in methods:
        AP_seeds, F_seeds, FT_seeds, AUC_seeds = [], [], [], []

        # For generate strategy, we need to load evaluation data from different difficulty levels
        if strategy == "generate":
            # For generate strategy, collect evaluation data from appropriate difficulty levels
            for seed in seeds:
                # 1) Load training curve from main experiment folder
                base_folder = (
                    data_root
                    / algo
                    / method
                    / experiment_folder
                    / f"{strategy}_{seq_len}"
                )
                sd = base_folder / f"seed_{seed}"

                if not sd.exists():
                    print(f"[debug] seed directory does not exist: {sd}")
                    continue

                training_fp = sd / "training_soup.json"
                if not training_fp.exists():
                    print(f"[warn] missing training_soup.json for {method} seed {seed}")
                    continue
                print(f"[debug] found training file for {method} seed {seed}: {training_fp}")
                training = load_series(training_fp)
                n_train = len(training)
                chunk = n_train // seq_len

                # 2) Load evaluation data from appropriate difficulty level folders
                env_files = []
                for i in task_indices:
                    # Determine the appropriate difficulty level for this task
                    if 0 <= i <= 4:  # Tasks 1-5 (easy)
                        task_experiment_folder = "level_1"
                    elif 5 <= i <= 9:  # Tasks 6-10 (medium)
                        task_experiment_folder = "level_2"
                    elif 10 <= i <= 14:  # Tasks 11-15 (hard)
                        task_experiment_folder = "level_3"
                    else:
                        # Fallback to main for other task indices
                        task_experiment_folder = "main"

                    # Load evaluation file from the appropriate difficulty level folder
                    task_base_folder = (
                        data_root
                        / algo
                        / method
                        / task_experiment_folder
                        / f"{strategy}_{seq_len}"
                    )
                    task_sd = task_base_folder / f"seed_{seed}"

                    if task_sd.exists():
                        matching_files = list(task_sd.glob(f"{i}_*_soup.*"))
                        if matching_files:
                            env_files.append(matching_files[0])  # Take the first match
                        else:
                            print(f"[warn] missing evaluation file for task {i} in {method} seed {seed} from {task_experiment_folder}")
                            env_files.append(None)
                    else:
                        print(f"[warn] task seed directory does not exist: {task_sd}")
                        env_files.append(None)

                # Filter out None values and check if we have enough files
                valid_env_files = [f for f in env_files if f is not None]
                if len(valid_env_files) != len(task_indices):
                    print(f"[warn] expected {len(task_indices)} env files for task range, got {len(valid_env_files)} for {method} seed {seed}")
                    continue

                # Load evaluation data for the specified task range
                eval_data = []
                for env_file in valid_env_files:
                    eval_data.append(load_series(env_file))

                # 3) Compute metrics on the task range
                # Average Performance (AP): mean of final performance on tasks in range
                final_performances = [eval_data[i][-1] for i in range(len(eval_data))]
                AP = np.mean(final_performances)

                # Forgetting (F): average drop from peak to final performance on tasks in range
                forgetting_values = []
                for i in range(len(eval_data)):
                    peak_perf = max(eval_data[i])
                    final_perf = np.mean(eval_data[i][-end_window_evals:])
                    forgetting_values.append(peak_perf - final_perf)
                F = np.mean(forgetting_values)

                # Forward Transfer (FT): normalized area between CL and baseline curves for tasks in range
                if seed not in baseline_data:
                    print(f"[warn] missing baseline data for seed {seed}")
                    FT = np.nan
                else:
                    ft_vals = []
                    for idx, task_idx in enumerate(task_indices):
                        # Calculate AUC for CL method (task task_idx)
                        start_train_idx = task_idx * chunk
                        end_train_idx = (task_idx + 1) * chunk
                        cl_task_curve = training[start_train_idx:end_train_idx]

                        # AUCi = (1/τ) * ∫ pi(t) dt, where τ is the task duration
                        # Using trapezoidal rule for numerical integration
                        if len(cl_task_curve) > 1:
                            auc_cl = np.trapz(cl_task_curve) / len(cl_task_curve)
                        else:
                            auc_cl = cl_task_curve[0] if len(cl_task_curve) == 1 else 0.0

                        # Calculate AUC for baseline method (task task_idx)
                        baseline_task_curve = baseline_data[seed][idx]  # Use idx for baseline_data indexing
                        if baseline_task_curve is not None:
                            if len(baseline_task_curve) > 1:
                                auc_baseline = np.trapz(baseline_task_curve) / len(baseline_task_curve)
                            else:
                                auc_baseline = baseline_task_curve[0] if len(baseline_task_curve) == 1 else 0.0
                        else:
                            auc_baseline = 0.0

                        # Forward transfer for task task_idx
                        if auc_baseline > 0:
                            ft_i = (auc_cl - auc_baseline) / auc_baseline
                        else:
                            ft_i = 0.0
                        ft_vals.append(ft_i)

                    FT = np.mean(ft_vals)

                # Average AUC: area under the curve for all evaluation curves in range
                auc_values = []
                for i in range(len(eval_data)):
                    auc_values.append(np.mean(eval_data[i]))
                AUC = np.mean(auc_values)

                AP_seeds.append(AP)
                F_seeds.append(F)
                FT_seeds.append(FT)
                AUC_seeds.append(AUC)
        else:
            # For other strategies (like curriculum), use the original logic
            base_folder = (
                    data_root
                    / algo
                    / method
                    / experiment_folder
                    / f"{strategy}_{seq_len}"
            )

            for seed in seeds:
                sd = base_folder / f"seed_{seed}"
                if not sd.exists():
                    print(f"[debug] seed directory does not exist: {sd}")
                    continue

                # 1) Plasticity training curve
                training_fp = sd / "training_soup.json"
                if not training_fp.exists():
                    print(f"[warn] missing training_soup.json for {method} seed {seed}")
                    continue
                print(f"[debug] found training file for {method} seed {seed}: {training_fp}")
                training = load_series(training_fp)
                n_train = len(training)
                chunk = n_train // seq_len

                # 2) Per‑environment evaluation curves - only load files for the specified task range
                env_files = []
                for i in task_indices:
                    matching_files = list(sd.glob(f"{i}_*_soup.*"))
                    if matching_files:
                        env_files.append(matching_files[0])  # Take the first match
                    else:
                        print(f"[warn] missing evaluation file for task {i} in {method} seed {seed}")

                if len(env_files) != len(task_indices):
                    print(f"[warn] expected {len(task_indices)} env files for task range, got {len(env_files)} for {method} seed {seed}")
                    continue

                # Load evaluation data for the specified task range
                eval_data = []
                for env_file in env_files:
                    eval_data.append(load_series(env_file))

                # 3) Compute metrics on the task range
                # Average Performance (AP): mean of final performance on tasks in range
                final_performances = [eval_data[i][-1] for i in range(len(eval_data))]
                AP = np.mean(final_performances)

                # Forgetting (F): average drop from peak to final performance on tasks in range
                forgetting_values = []
                for i in range(len(eval_data)):
                    peak_perf = max(eval_data[i])
                    final_perf = np.mean(eval_data[i][-end_window_evals:])
                    forgetting_values.append(peak_perf - final_perf)
                F = np.mean(forgetting_values)

                # Forward Transfer (FT): normalized area between CL and baseline curves for tasks in range
                if seed not in baseline_data:
                    print(f"[warn] missing baseline data for seed {seed}")
                    FT = np.nan
                else:
                    ft_vals = []
                    for idx, task_idx in enumerate(task_indices):
                        # Calculate AUC for CL method (task task_idx)
                        start_train_idx = task_idx * chunk
                        end_train_idx = (task_idx + 1) * chunk
                        cl_task_curve = training[start_train_idx:end_train_idx]

                        # AUCi = (1/τ) * ∫ pi(t) dt, where τ is the task duration
                        # Using trapezoidal rule for numerical integration
                        if len(cl_task_curve) > 1:
                            auc_cl = np.trapz(cl_task_curve) / len(cl_task_curve)
                        else:
                            auc_cl = cl_task_curve[0] if len(cl_task_curve) == 1 else 0.0

                        # Calculate AUC for baseline method (task task_idx)
                        baseline_task_curve = baseline_data[seed][idx]  # Use idx for baseline_data indexing
                        if baseline_task_curve is not None:
                            if len(baseline_task_curve) > 1:
                                auc_baseline = np.trapz(baseline_task_curve) / len(baseline_task_curve)
                            else:
                                auc_baseline = baseline_task_curve[0] if len(baseline_task_curve) == 1 else 0.0
                        else:
                            auc_baseline = 0.0

                        # Forward transfer for task task_idx
                        if auc_baseline > 0:
                            ft_i = (auc_cl - auc_baseline) / auc_baseline
                        else:
                            ft_i = 0.0
                        ft_vals.append(ft_i)

                    FT = np.mean(ft_vals)

                # Average AUC: area under the curve for all evaluation curves in range
                auc_values = []
                for i in range(len(eval_data)):
                    auc_values.append(np.mean(eval_data[i]))
                AUC = np.mean(auc_values)

                AP_seeds.append(AP)
                F_seeds.append(F)
                FT_seeds.append(FT)
                AUC_seeds.append(AUC)

        # Compute mean and CI across seeds
        AP_mean, AP_ci = _mean_ci(AP_seeds)
        F_mean, F_ci = _mean_ci(F_seeds)
        FT_mean, FT_ci = _mean_ci(FT_seeds)
        AUC_mean, AUC_ci = _mean_ci(AUC_seeds)

        rows.append({
            "Method": method,
            "AveragePerformance": AP_mean,
            "AveragePerformance_CI": AP_ci,
            "Forgetting": F_mean,
            "Forgetting_CI": F_ci,
            "ForwardTransfer": FT_mean,
            "ForwardTransfer_CI": FT_ci,
            "AverageAUC": AUC_mean,
            "AverageAUC_CI": AUC_ci,
        })

    return pd.DataFrame(rows)


def compare_strategies_on_task_range(
        data_root: Path,
        algo: str,
        methods: List[str],
        strategies: List[str],
        seq_len: int,
        seeds: List[int],
        task_range: Tuple[int, int],
        level: Optional[int] = None,
        end_window_evals: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Compare continual learning metrics across different strategies on a specific task range.

    Returns:
        Dictionary mapping strategy names to DataFrames with metrics
    """
    results = {}
    for strategy in strategies:
        print(f"Computing metrics for strategy: {strategy}")
        df = compute_metrics_for_task_range(
            data_root=data_root,
            algo=algo,
            methods=methods,
            strategy=strategy,
            seq_len=seq_len,
            seeds=seeds,
            task_range=task_range,
            level=level,
            end_window_evals=end_window_evals,
        )
        results[strategy] = df
    return results


def _fmt(mean: float, ci: float, best: bool, better: str = "max") -> str:
    """Format mean ± CI, bolding if best."""
    if np.isnan(mean) or np.isnan(ci):
        return "—"
    formatted = f"{mean:.3f} ± {ci:.3f}"
    return f"\\textbf{{{formatted}}}" if best else formatted


def generate_task_range_comparison_table(
        data_root: Path,
        algo: str,
        methods: List[str],
        strategies: List[str],
        seq_len: int,
        seeds: List[int],
        task_range: Tuple[int, int],
        level: Optional[int] = None,
        end_window_evals: int = 10,
) -> str:
    """
    Generate a LaTeX table comparing strategies on a specific task range.

    Returns:
        LaTeX table string
    """
    # Compute results for all strategies
    results = compare_strategies_on_task_range(
        data_root=data_root,
        algo=algo,
        methods=methods,
        strategies=strategies,
        seq_len=seq_len,
        seeds=seeds,
        task_range=task_range,
        level=level,
        end_window_evals=end_window_evals,
    )

    # Assume we're working with a single method for now (can be extended)
    method = methods[0] if methods else "EWC"

    # Create comparison table with strategies as rows
    rows = []
    strategy_names = {
        "curriculum": "Curriculum", 
        "generate": "Generated",
        "generated_15": "Generated"  # Handle both naming conventions
    }

    # Collect all metric values for comparison to find best
    all_values = {}
    metrics = ["AveragePerformance", "Forgetting", "ForwardTransfer", "AverageAUC"]
    better_direction = {"AveragePerformance": "max", "Forgetting": "min", 
                      "ForwardTransfer": "max", "AverageAUC": "max"}

    for metric in metrics:
        all_values[metric] = {}
        for strategy in strategies:
            df = results[strategy]
            method_row = df[df["Method"] == method]
            if not method_row.empty:
                mean_val = method_row[metric].iloc[0]
                ci_val = method_row[f"{metric}_CI"].iloc[0]
                all_values[metric][strategy] = (mean_val, ci_val)

    # Find best values for each metric
    best_strategies = {}
    for metric in metrics:
        if all_values[metric]:
            if better_direction[metric] == "max":
                best_strategies[metric] = max(all_values[metric].keys(), 
                                         key=lambda k: all_values[metric][k][0] if not np.isnan(all_values[metric][k][0]) else -np.inf)
            else:
                best_strategies[metric] = min(all_values[metric].keys(), 
                                         key=lambda k: all_values[metric][k][0] if not np.isnan(all_values[metric][k][0]) else np.inf)

    # Create rows for each strategy
    for strategy in strategies:
        row = {"Strategy": strategy_names.get(strategy, strategy.capitalize())}

        for metric in metrics:
            if strategy in all_values[metric]:
                mean_val, ci_val = all_values[metric][strategy]
                is_best = (strategy == best_strategies.get(metric))
                row[metric] = _fmt(mean_val, ci_val, is_best, better_direction[metric])
            else:
                row[metric] = "—"

        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Rename columns to LaTeX format
    metric_symbols = {
        "AveragePerformance": r"$\mathcal{A}\!\uparrow$",
        "Forgetting": r"$\mathcal{F}\!\downarrow$", 
        "ForwardTransfer": r"$\mathcal{FT}\!\uparrow$",
        "AverageAUC": r"$\mathcal{AUC}\!\uparrow$"
    }

    df_out.columns = ["Strategy"] + [metric_symbols[metric] for metric in metrics]

    # Generate LaTeX table
    column_format = "lcccc"  # Strategy + 4 metrics

    start_idx, end_idx = task_range
    task_range_str = f"tasks {start_idx+1}-{end_idx+1}"

    latex_table = df_out.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        label=f"tab:cmarl_metrics_task_range_{start_idx+1}_{end_idx+1}_comparison",
        caption=f"Comparison of continual learning metrics between curriculum and generated strategies on {task_range_str}. "
                "Bold values indicate the best performance for each metric. "
                r"$\uparrow$ indicates higher is better, $\downarrow$ indicates lower is better."
    )

    return latex_table


def generate_ap_only_comparison_table(
        data_root: Path,
        algo: str,
        methods: List[str],
        strategies: List[str],
        seq_len: int,
        seeds: List[int],
        medium_range: Tuple[int, int],
        hard_range: Tuple[int, int],
        level: Optional[int] = None,
        end_window_evals: int = 10,
) -> str:
    """
    Generate a LaTeX table comparing strategies on both medium and hard ranges,
    showing only Average Performance (AP) with confidence intervals.

    Returns:
        LaTeX table string
    """
    # Compute results for both ranges
    medium_results = compare_strategies_on_task_range(
        data_root=data_root,
        algo=algo,
        methods=methods,
        strategies=strategies,
        seq_len=seq_len,
        seeds=seeds,
        task_range=medium_range,
        level=level,
        end_window_evals=end_window_evals,
    )

    hard_results = compare_strategies_on_task_range(
        data_root=data_root,
        algo=algo,
        methods=methods,
        strategies=strategies,
        seq_len=seq_len,
        seeds=seeds,
        task_range=hard_range,
        level=level,
        end_window_evals=end_window_evals,
    )

    # Assume we're working with a single method for now (can be extended)
    method = methods[0] if methods else "EWC"

    # Create comparison table with strategies as rows and difficulty ranges as columns
    rows = []
    strategy_names = {
        "curriculum": "Curriculum", 
        "generate": "Generated",
        "generated_15": "Generated"  # Handle both naming conventions
    }

    # Collect all AP values for comparison to find best
    all_ap_values = {}
    for difficulty, results in [("Medium", medium_results), ("Hard", hard_results)]:
        all_ap_values[difficulty] = {}
        for strategy in strategies:
            df = results[strategy]
            method_row = df[df["Method"] == method]
            if not method_row.empty:
                mean_val = method_row["AveragePerformance"].iloc[0]
                ci_val = method_row["AveragePerformance_CI"].iloc[0]
                all_ap_values[difficulty][strategy] = (mean_val, ci_val)

    # Find best values for each difficulty
    best_strategies = {}
    for difficulty in ["Medium", "Hard"]:
        if all_ap_values[difficulty]:
            best_strategies[difficulty] = max(all_ap_values[difficulty].keys(), 
                                           key=lambda k: all_ap_values[difficulty][k][0] if not np.isnan(all_ap_values[difficulty][k][0]) else -np.inf)

    # Create rows for each strategy
    for strategy in strategies:
        row = {"Strategy": strategy_names.get(strategy, strategy.capitalize())}

        for difficulty in ["Medium", "Hard"]:
            if strategy in all_ap_values[difficulty]:
                mean_val, ci_val = all_ap_values[difficulty][strategy]
                is_best = (strategy == best_strategies.get(difficulty))
                row[f"AP_{difficulty}"] = _fmt(mean_val, ci_val, is_best, "max")
            else:
                row[f"AP_{difficulty}"] = "—"

        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Rename columns to LaTeX format
    medium_start, medium_end = medium_range
    hard_start, hard_end = hard_range

    df_out.columns = [
        "Strategy", 
        f"AP Tasks {medium_start+1}-{medium_end+1}",
        f"AP Tasks {hard_start+1}-{hard_end+1}"
    ]

    # Generate LaTeX table
    column_format = "lcc"  # Strategy + 2 AP columns

    latex_table = df_out.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        label="tab:cmarl_ap_medium_hard_comparison",
        caption=f"Comparison of Average Performance (AP) with 95\\% confidence intervals between curriculum and generated strategies "
                f"on medium tasks ({medium_start+1}-{medium_end+1}) and hard tasks ({hard_start+1}-{hard_end+1}). "
                "Bold values indicate the best performance for each task range."
    )

    return latex_table


def main():
    parser = argparse.ArgumentParser(description="Compare curriculum vs generated strategies on specific task ranges")
    parser.add_argument("--data_root", type=Path, default="results/data", 
                       help="Root directory containing experimental data")
    parser.add_argument("--algo", type=str, default="ippo", 
                       help="Algorithm name")
    parser.add_argument("--methods", nargs="+", default=["EWC"], 
                       help="Continual learning methods to analyze")
    parser.add_argument("--strategies", nargs="+", default=["curriculum", "generate"], 
                       help="Strategies to compare")
    parser.add_argument("--seq_len", type=int, default=15, 
                       help="Sequence length")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3],
                       help="Random seeds")
    parser.add_argument("--level", type=int, default=None, 
                       help="Difficulty level (1, 2, 3)")
    parser.add_argument("--end_window_evals", type=int, default=10, 
                       help="Number of final evaluations for forgetting metric")
    parser.add_argument("--output_prefix", type=str, default=None, 
                       help="Output file prefix for LaTeX tables")

    # Task range arguments
    parser.add_argument("--medium_range", nargs=2, type=int, default=[5, 9],
                       help="Task range for medium difficulty comparison (0-indexed, inclusive)")
    parser.add_argument("--hard_range", nargs=2, type=int, default=[10, 14],
                       help="Task range for hard difficulty comparison (0-indexed, inclusive)")
    parser.add_argument("--compare_both", action="store_true",
                       help="Compare both medium and hard ranges")
    parser.add_argument("--range_type", choices=["medium", "hard"], default=None,
                       help="Specify which range to compare (medium or hard). If not specified, defaults to medium unless --compare_both is used.")
    parser.add_argument("--ap_only", action="store_true",
                       help="Generate a single table comparing only Average Performance (AP) with CI for both medium and hard ranges")

    args = parser.parse_args()

    print(f"Comparing strategies: {args.strategies}")
    print(f"Algorithm: {args.algo}")
    print(f"Methods: {args.methods}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Seeds: {args.seeds}")
    print(f"Level: {args.level}")

    if args.ap_only:
        # Generate AP-only comparison table for both medium and hard ranges
        print(f"\n{'='*80}")
        print("COMPARING AP ONLY: MEDIUM AND HARD TASKS")
        print(f"{'='*80}")

        latex_table = generate_ap_only_comparison_table(
            data_root=args.data_root,
            algo=args.algo,
            methods=args.methods,
            strategies=args.strategies,
            seq_len=args.seq_len,
            seeds=args.seeds,
            medium_range=tuple(args.medium_range),
            hard_range=tuple(args.hard_range),
            level=args.level,
            end_window_evals=args.end_window_evals,
        )

        print("\nLATEX TABLE (AP ONLY)")
        print("-"*40)
        print(latex_table)

        if args.output_prefix:
            output_file = f"{args.output_prefix}_ap_only_medium_hard.tex"
            with open(output_file, 'w') as f:
                f.write(latex_table)
            print(f"\nTable saved to: {output_file}")

        return

    if args.compare_both:
        # Compare both medium and hard ranges
        ranges = [
            (tuple(args.medium_range), "medium"),
            (tuple(args.hard_range), "hard")
        ]
    elif args.range_type == "hard":
        # Compare only hard range
        ranges = [(tuple(args.hard_range), "hard")]
    elif args.range_type == "medium":
        # Compare only medium range
        ranges = [(tuple(args.medium_range), "medium")]
    else:
        # Default to medium range if not specified
        ranges = [(tuple(args.medium_range), "medium")]

    for task_range, difficulty_name in ranges:
        print(f"\n{'='*80}")
        print(f"COMPARING {difficulty_name.upper()} TASKS ({task_range[0]+1}-{task_range[1]+1})")
        print(f"{'='*80}")

        # Generate comparison table
        latex_table = generate_task_range_comparison_table(
            data_root=args.data_root,
            algo=args.algo,
            methods=args.methods,
            strategies=args.strategies,
            seq_len=args.seq_len,
            seeds=args.seeds,
            task_range=task_range,
            level=args.level,
            end_window_evals=args.end_window_evals,
        )

        print("\nLATEX TABLE")
        print("-"*40)
        print(latex_table)

        if args.output_prefix:
            output_file = f"{args.output_prefix}_{difficulty_name}_tasks_{task_range[0]+1}_{task_range[1]+1}.tex"
            with open(output_file, 'w') as f:
                f.write(latex_table)
            print(f"\nTable saved to: {output_file}")

        # Also compute and display raw metrics for debugging
        results = compare_strategies_on_task_range(
            data_root=args.data_root,
            algo=args.algo,
            methods=args.methods,
            strategies=args.strategies,
            seq_len=args.seq_len,
            seeds=args.seeds,
            task_range=task_range,
            level=args.level,
            end_window_evals=args.end_window_evals,
        )

        print(f"\nRAW METRICS FOR {difficulty_name.upper()} TASKS")
        print("-"*40)
        for strategy, df in results.items():
            print(f"\n{strategy.upper()} STRATEGY:")
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
