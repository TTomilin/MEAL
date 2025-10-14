from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

ConfInt = Tuple[float, float]


def load_series(fp: Path) -> List[float]:
    """Load a JSON file containing a list of floats."""
    with open(fp, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    else:
        raise ValueError(f"Expected list, got {type(data)}")


def _mean_ci(series: List[float]) -> ConfInt:
    if not series:
        return np.nan, np.nan
    mean = float(np.mean(series))
    if len(series) == 1:
        return mean, 0.0
    ci = 1.96 * np.std(series, ddof=1) / np.sqrt(len(series))
    return mean, float(ci)


def get_experiment_folder(restriction_setting: str, level: Optional[int] = None) -> str:
    """
    Get the experiment folder name based on restriction setting and level.

    Args:
        restriction_setting: One of 'default', 'complementary_restrictions'
        level: Difficulty level (1, 2, 3) or None

    Returns:
        Folder name for the experiment
    """
    if restriction_setting == 'default':
        if level is not None:
            return f"level_{level}"
        else:
            return "main"
    elif restriction_setting == 'complementary_restrictions':
        return "complementary_restrictions"
    else:
        raise ValueError(f"Unknown restriction setting: {restriction_setting}")


def compute_metrics_with_restriction_settings(
        data_root: Path,
        algo: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        restriction_setting: str = 'default',
        level: Optional[int] = None,
        end_window_evals: int = 10,
) -> pd.DataFrame:
    """
    Compute continual learning metrics for experiments with restriction settings.

    Args:
        data_root: Root directory containing experimental data
        algo: Algorithm name (e.g., 'ippo')
        methods: List of continual learning methods (e.g., ['EWC', 'MAS', 'L2'])
        strategy: Strategy name (e.g., 'generate')
        seq_len: Sequence length
        seeds: List of random seeds
        restriction_setting: Restriction setting ('default', 'complementary_restrictions')
        level: Difficulty level (1, 2, 3) for default setting, ignored for others
        end_window_evals: Number of final evaluations to average for forgetting metric

    Returns:
        DataFrame with computed metrics
    """
    rows: list[dict[str, float]] = []
    experiment_folder = get_experiment_folder(restriction_setting, level)

    # Load baseline data once for forward transfer calculation
    baseline_data = {}
    baseline_folder = (
        data_root
        / algo
        / "single"
        / get_experiment_folder('default', level)  # Baselines are always default
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
        AP_seeds, F_seeds, FT_seeds, AUC_seeds = [], [], [], []

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

            # 2) Per‑environment evaluation curves
            env_files = sorted([
                f for f in sd.glob("*_soup.*") if "training" not in f.name
            ])

            if len(env_files) != seq_len:
                print(f"[warn] expected {seq_len} env files, got {len(env_files)} for {method} seed {seed}")
                continue

            # Load evaluation data
            eval_data = []
            for env_file in env_files:
                eval_data.append(load_series(env_file))

            # 3) Compute metrics
            # Average Performance (AP): mean of final performance on all tasks
            final_performances = [eval_data[i][-1] for i in range(seq_len)]
            AP = np.mean(final_performances)

            # Forgetting (F): average drop from peak to final performance
            forgetting_values = []
            for i in range(seq_len):
                peak_perf = max(eval_data[i])
                final_perf = np.mean(eval_data[i][-end_window_evals:])
                forgetting_values.append(peak_perf - final_perf)
            F = np.mean(forgetting_values)

            # Forward Transfer (FT): normalized area between CL and baseline curves
            if seed not in baseline_data:
                print(f"[warn] missing baseline data for seed {seed}")
                FT = np.nan
            else:
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
                    else:
                        auc_baseline = 0.0

                    # Forward transfer for task i
                    if auc_baseline > 0:
                        ft_i = (auc_cl - auc_baseline) / auc_baseline
                    else:
                        ft_i = 0.0
                    ft_vals.append(ft_i)

                FT = np.mean(ft_vals)

            # Average AUC: area under the curve for all evaluation curves
            auc_values = []
            for i in range(seq_len):
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


def compare_restriction_settings(
        data_root: Path,
        algo: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        restriction_settings: List[str] = ['default', 'complementary_restrictions'],
        level: Optional[int] = None,
        end_window_evals: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Compare continual learning metrics across different restriction settings.

    Returns:
        Dictionary mapping restriction setting names to DataFrames with metrics
    """
    results = {}
    for setting in restriction_settings:
        print(f"Computing metrics for restriction setting: {setting}")
        df = compute_metrics_with_restriction_settings(
            data_root=data_root,
            algo=algo,
            methods=methods,
            strategy=strategy,
            seq_len=seq_len,
            seeds=seeds,
            restriction_setting=setting,
            level=level,
            end_window_evals=end_window_evals,
        )
        results[setting] = df
    return results


def _fmt(mean: float, ci: float, best: bool, better: str = "max") -> str:
    """Format a metric with confidence interval, bolding if best."""
    if np.isnan(mean):
        return "—"
    formatted = f"{mean:.2f} ± {ci:.2f}"
    return f"\\textbf{{{formatted}}}" if best else formatted


def generate_restriction_comparison_table(
        data_root: Path,
        algo: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        restriction_settings: List[str] = ['default', 'complementary_restrictions'],
        level: Optional[int] = None,
        end_window_evals: int = 10,
) -> str:
    """
    Generate a LaTeX table comparing metrics across restriction settings.
    Default results on first row, complementary restrictions on second row.

    Returns:
        LaTeX table string
    """
    # Get results for all restriction settings
    results = compare_restriction_settings(
        data_root=data_root,
        algo=algo,
        methods=methods,
        strategy=strategy,
        seq_len=seq_len,
        seeds=seeds,
        restriction_settings=restriction_settings,
        level=level,
        end_window_evals=end_window_evals,
    )

    # Assume we're working with a single method for now (can be extended)
    method = methods[0] if methods else "EWC"

    # Create comparison table with restriction settings as rows
    rows = []
    setting_names = {"default": "Default", "complementary_restrictions": "Restricted"}

    # Collect all metric values for comparison to find best
    all_values = {}
    metrics = ["AveragePerformance", "Forgetting", "ForwardTransfer", "AverageAUC"]
    better_direction = {"AveragePerformance": "max", "Forgetting": "min", 
                      "ForwardTransfer": "max", "AverageAUC": "max"}

    for metric in metrics:
        all_values[metric] = {}
        for setting in restriction_settings:
            df = results[setting]
            method_row = df[df["Method"] == method]
            if not method_row.empty:
                mean_val = method_row[metric].iloc[0]
                ci_val = method_row[f"{metric}_CI"].iloc[0]
                all_values[metric][setting] = (mean_val, ci_val)

    # Find best values for each metric
    best_settings = {}
    for metric in metrics:
        if all_values[metric]:
            if better_direction[metric] == "max":
                best_settings[metric] = max(all_values[metric].keys(), 
                                         key=lambda k: all_values[metric][k][0] if not np.isnan(all_values[metric][k][0]) else -np.inf)
            else:
                best_settings[metric] = min(all_values[metric].keys(), 
                                         key=lambda k: all_values[metric][k][0] if not np.isnan(all_values[metric][k][0]) else np.inf)

    # Create rows for each restriction setting
    for setting in restriction_settings:
        row = {"Setting": setting_names.get(setting, setting)}

        for metric in metrics:
            if setting in all_values[metric]:
                mean_val, ci_val = all_values[metric][setting]
                is_best = (setting == best_settings.get(metric))
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

    df_out.columns = ["Setting"] + [metric_symbols[metric] for metric in metrics]

    # Generate LaTeX table
    column_format = "lcccc"  # Setting + 4 metrics

    latex_table = df_out.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        label="tab:cmarl_metrics_restriction_comparison",
        caption="Continual learning metrics comparison: Default vs Complementary Restrictions",
    )

    return latex_table


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare continual learning metrics across restriction settings")
    p.add_argument("--data_root", type=Path, default="data",
                   help="Root directory containing experimental data")
    p.add_argument("--algo", default="ippo", help="Algorithm name")
    p.add_argument("--methods", nargs="+", default=["EWC"],
                   help="Continual learning methods to compare")
    p.add_argument("--strategy", default="generate", help="Strategy name")
    p.add_argument("--seq_len", type=int, default=10, help="Sequence length")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="Random seeds to average over")
    p.add_argument("--level", type=int, help="Difficulty level (1, 2, 3)")
    p.add_argument("--end_window_evals", type=int, default=10,
                   help="Number of final evaluations to average for forgetting metric")
    p.add_argument("--single_setting", action="store_true",
                   help="Generate table for single restriction setting instead of comparison")
    p.add_argument("--restriction_setting", choices=["default", "complementary_restrictions"],
                   default="default", help="Single restriction setting to analyze")

    args = p.parse_args()

    if args.single_setting:
        # Generate table for single restriction setting
        df = compute_metrics_with_restriction_settings(
            data_root=Path(args.data_root),
            algo=args.algo,
            methods=args.methods,
            strategy=args.strategy,
            seq_len=args.seq_len,
            seeds=args.seeds,
            restriction_setting=args.restriction_setting,
            level=args.level,
            end_window_evals=args.end_window_evals,
        )

        # Pretty‑print method names
        df["Method"] = df["Method"].replace({"Online_EWC": "Online EWC"})

        # Identify best means (ignoring CI)
        best_A = df["AveragePerformance"].max()
        best_F = df["Forgetting"].min()
        best_FT = df["ForwardTransfer"].max()
        best_AUC = df["AverageAUC"].max()

        # Build human‑readable strings with CI
        df_out = pd.DataFrame()
        df_out["Method"] = df["Method"]
        df_out["AveragePerformance"] = df.apply(
            lambda r: _fmt(r.AveragePerformance, r.AveragePerformance_CI, r.AveragePerformance == best_A, "max"),
            axis=1,
        )
        df_out["Forgetting"] = df.apply(
            lambda r: _fmt(r.Forgetting, r.Forgetting_CI, r.Forgetting == best_F, "min"),
            axis=1,
        )
        df_out["ForwardTransfer"] = df.apply(
            lambda r: _fmt(r.ForwardTransfer, r.ForwardTransfer_CI, r.ForwardTransfer == best_FT, "max"),
            axis=1,
        )
        df_out["AverageAUC"] = df.apply(
            lambda r: _fmt(r.AverageAUC, r.AverageAUC_CI, r.AverageAUC == best_AUC, "max"),
            axis=1,
        )

        # Rename columns to mathy headers
        df_out.columns = [
            "Method",
            r"$\mathcal{A}\!\uparrow$",
            r"$\mathcal{F}\!\downarrow$",
            r"$\mathcal{FT}\!\uparrow$",
            r"$\mathcal{AUC}\!\uparrow$",
        ]

        latex_table = df_out.to_latex(
            index=False,
            escape=False,
            column_format="lcccc",
            label=f"tab:cmarl_metrics_{args.restriction_setting}",
            caption=f"Continual learning metrics for {args.restriction_setting} restriction setting",
        )

        print(latex_table)

    else:
        # Generate comparison table
        latex_table = generate_restriction_comparison_table(
            data_root=Path(args.data_root),
            algo=args.algo,
            methods=args.methods,
            strategy=args.strategy,
            seq_len=args.seq_len,
            seeds=args.seeds,
            restriction_settings=["default", "complementary_restrictions"],
            level=args.level,
            end_window_evals=args.end_window_evals,
        )

        print(latex_table)
