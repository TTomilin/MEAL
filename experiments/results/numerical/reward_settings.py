#!/usr/bin/env python3
"""
Enhanced results table script that supports reward settings experiments.
This extends the original results_table.py to handle sparse_rewards and individual_rewards experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

ConfInt = Tuple[float, float]


def load_series(fp: Path) -> List[float]:
    """Load a time series from JSON file."""
    if fp.suffix == ".json":
        with fp.open() as f:
            return json.load(f)
    elif fp.suffix == ".npz":
        return np.load(fp)["data"].tolist()
    else:
        raise ValueError(f"Unsupported file format: {fp.suffix}")


def _mean_ci(series: List[float]) -> ConfInt:
    if not series:
        return np.nan, np.nan
    mean = float(np.mean(series))
    if len(series) == 1:
        return mean, 0.0
    ci = 1.96 * np.std(series, ddof=1) / np.sqrt(len(series))
    return mean, float(ci)


def get_experiment_folder(reward_setting: str, level: Optional[int] = None) -> str:
    """
    Get the experiment folder name based on reward setting and level.

    Args:
        reward_setting: One of 'shared', 'sparse', 'individual'
        level: Difficulty level (1, 2, 3) or None

    Returns:
        Folder name for the experiment
    """
    if reward_setting == 'shared':
        if level is not None:
            return f"level_{level}"
        else:
            return "main"
    elif reward_setting == 'sparse':
        return "sparse_rewards"
    elif reward_setting == 'individual':
        return "individual_rewards"
    else:
        raise ValueError(f"Unknown reward setting: {reward_setting}")


def compute_metrics_with_reward_settings(
        data_root: Path,
        algo: str,
        arch: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        reward_setting: str = 'shared',
        level: Optional[int] = None,
        end_window_evals: int = 10,
) -> pd.DataFrame:
    """
    Compute continual learning metrics for experiments with reward settings.

    Args:
        data_root: Root directory containing experimental data
        algo: Algorithm name (e.g., 'ippo')
        arch: Architecture name (not used in current implementation)
        methods: List of continual learning methods (e.g., ['EWC', 'MAS', 'L2'])
        strategy: Strategy name (e.g., 'generate')
        seq_len: Sequence length
        seeds: List of random seeds
        reward_setting: Reward setting ('shared', 'sparse', 'individual')
        level: Difficulty level (1, 2, 3) for shared setting, ignored for others
        end_window_evals: Number of final evaluations to average for forgetting metric

    Returns:
        DataFrame with computed metrics
    """
    rows: list[dict[str, float]] = []
    experiment_folder = get_experiment_folder(reward_setting, level)

    # Load baseline data once for forward transfer calculation
    baseline_data = {}
    baseline_folder = (
        data_root
        / algo
        / "single"
        / get_experiment_folder('shared', level)  # Baselines are always shared
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
                print(
                    f"[warn] expected {seq_len} env files, found {len(env_files)} "
                    f"for {method} seed {seed}"
                )
                print(f"[debug] found files: {[f.name for f in env_files]}")
                all_soup_files = list(sd.glob("*_soup.*"))
                print(f"[debug] all *_soup.* files: {[f.name for f in all_soup_files]}")
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
                else:
                    auc_baseline = 0.0

                # Forward transfer for task i
                if auc_baseline > 0:
                    ft_i = (auc_cl - auc_baseline) / auc_baseline
                else:
                    ft_i = 0.0
                ft_vals.append(ft_i)

            FT_seeds.append(np.mean(ft_vals))

            # Forgetting (F) – drop from peak to final performance
            f_vals = []
            for i in range(seq_len):
                curve = env_mat[i, :]
                peak = np.max(curve)
                final = np.mean(curve[-end_window_evals:])
                f_vals.append(peak - final)
            F_seeds.append(np.mean(f_vals))

            # Average AUC – average area under curve of evaluation curves across all tasks
            auc_vals = []
            for i in range(seq_len):
                curve = env_mat[i, :]
                # Calculate AUC using trapezoidal rule, normalized by curve length
                if len(curve) > 1:
                    auc_task = np.trapz(curve) / len(curve)
                else:
                    auc_task = curve[0] if len(curve) == 1 else 0.0
                auc_vals.append(auc_task)
            AUC_seeds.append(np.mean(auc_vals))

        # Compute mean ± CI for this method
        AP_mean, AP_ci = _mean_ci(AP_seeds)
        F_mean, F_ci = _mean_ci(F_seeds)
        FT_mean, FT_ci = _mean_ci(FT_seeds)
        AUC_mean, AUC_ci = _mean_ci(AUC_seeds)

        rows.append({
            "Method": method,
            "RewardSetting": reward_setting,
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


def compare_reward_settings(
        data_root: Path,
        algo: str,
        arch: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        reward_settings: List[str] = ['shared', 'sparse', 'individual'],
        level: Optional[int] = None,
        end_window_evals: int = 10,
) -> pd.DataFrame:
    """
    Compare different reward settings across methods.

    Returns:
        DataFrame with results for all reward settings
    """
    all_results = []

    for reward_setting in reward_settings:
        df = compute_metrics_with_reward_settings(
            data_root=data_root,
            algo=algo,
            arch=arch,
            methods=methods,
            strategy=strategy,
            seq_len=seq_len,
            seeds=seeds,
            reward_setting=reward_setting,
            level=level,
            end_window_evals=end_window_evals,
        )
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def _fmt(mean: float, ci: float, best: bool, better: str = "max") -> str:
    """Format mean ± CI, highlighting the best value."""
    if np.isnan(mean):
        return "—"

    formatted = f"{mean:.2f} ± {ci:.2f}"
    if best:
        formatted = f"\\textbf{{{formatted}}}"
    return formatted


def generate_reward_comparison_table(
        data_root: Path,
        algo: str,
        arch: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        reward_settings: List[str] = ['shared', 'sparse', 'individual'],
        level: Optional[int] = None,
        end_window_evals: int = 10,
) -> str:
    """
    Generate a LaTeX table comparing different reward settings.

    Returns:
        LaTeX table string
    """
    df = compare_reward_settings(
        data_root=data_root,
        algo=algo,
        arch=arch,
        methods=methods,
        strategy=strategy,
        seq_len=seq_len,
        seeds=seeds,
        reward_settings=reward_settings,
        level=level,
        end_window_evals=end_window_evals,
    )

    # Pretty‑print method names
    df["Method"] = df["Method"].replace({"Online_EWC": "Online EWC"})

    # Create a pivot table for better comparison
    results = []

    for reward_setting in reward_settings:
        setting_df = df[df["RewardSetting"] == reward_setting].copy()

        if setting_df.empty:
            continue

        # Identify best means (ignoring CI) within this reward setting
        best_A = setting_df["AveragePerformance"].max()
        best_F = setting_df["Forgetting"].min()
        best_FT = setting_df["ForwardTransfer"].max()
        best_AUC = setting_df["AverageAUC"].max()

        # Build human‑readable strings with CI
        df_out = pd.DataFrame()
        df_out["Method"] = setting_df["Method"]
        # Special handling for display names
        display_name = "Shared" if reward_setting == "shared" else reward_setting.title()
        df_out["RewardSetting"] = display_name
        df_out["AveragePerformance"] = setting_df.apply(
            lambda r: _fmt(r.AveragePerformance, r.AveragePerformance_CI, r.AveragePerformance == best_A, "max"),
            axis=1,
        )
        df_out["Forgetting"] = setting_df.apply(
            lambda r: _fmt(r.Forgetting, r.Forgetting_CI, r.Forgetting == best_F, "min"),
            axis=1,
        )
        df_out["ForwardTransfer"] = setting_df.apply(
            lambda r: _fmt(r.ForwardTransfer, r.ForwardTransfer_CI, r.ForwardTransfer == best_FT, "max"),
            axis=1,
        )
        df_out["AverageAUC"] = setting_df.apply(
            lambda r: _fmt(r.AverageAUC, r.AverageAUC_CI, r.AverageAUC == best_AUC, "max"),
            axis=1,
        )

        results.append(df_out)

    # Combine all results
    final_df = pd.concat(results, ignore_index=True)

    # Rename columns to mathy headers
    final_df.columns = [
        "Method",
        "Reward Setting",
        r"$\mathcal{A}\!\uparrow$",
        r"$\mathcal{F}\!\downarrow$",
        r"$\mathcal{FT}\!\uparrow$",
        r"$\mathcal{AUC}\!\uparrow$",
    ]

    latex_table = final_df.to_latex(
        index=False,
        escape=False,
        column_format="llcccc",
        label="tab:reward_settings_comparison",
        caption="Comparison of continual learning methods across different reward settings",
    )

    return latex_table


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate results table with reward settings support")
    p.add_argument("--data_root", required=True, help="Root directory containing experimental data")
    p.add_argument("--algo", required=True, help="Algorithm name (e.g., ippo)")
    p.add_argument("--arch", required=True, help="Architecture name")
    p.add_argument("--methods", nargs="+", required=True, help="List of CL methods")
    p.add_argument("--strategy", required=True, help="Strategy name (e.g., generate)")
    p.add_argument("--seq_len", type=int, default=10, help="Sequence length")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Random seeds")
    p.add_argument("--reward_settings", nargs="+", 
                   choices=["shared", "sparse", "individual"],
                   default=["shared", "sparse", "individual"],
                   help="Reward settings to compare")
    p.add_argument("--level", type=int, default=None, 
                   help="Difficulty level (1, 2, 3) for shared setting")
    p.add_argument("--end_window_evals", type=int, default=10,
                   help="Number of final eval points to average for forgetting")
    p.add_argument("--single_setting", action="store_true",
                   help="Generate table for single reward setting instead of comparison")
    p.add_argument("--reward_setting", choices=["shared", "sparse", "individual"],
                   default="shared", help="Single reward setting to analyze")

    args = p.parse_args()

    if args.single_setting:
        # Generate table for single reward setting
        df = compute_metrics_with_reward_settings(
            data_root=Path(args.data_root),
            algo=args.algo,
            arch=args.arch,
            methods=args.methods,
            strategy=args.strategy,
            seq_len=args.seq_len,
            seeds=args.seeds,
            reward_setting=args.reward_setting,
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
            label=f"tab:cmarl_metrics_{args.reward_setting}",
            caption=f"Continual learning metrics for {args.reward_setting} reward setting",
        )

        print(latex_table)

    else:
        # Generate comparison table
        latex_table = generate_reward_comparison_table(
            data_root=Path(args.data_root),
            algo=args.algo,
            arch=args.arch,
            methods=args.methods,
            strategy=args.strategy,
            seq_len=args.seq_len,
            seeds=args.seeds,
            reward_settings=args.reward_settings,
            level=args.level,
            end_window_evals=args.end_window_evals,
        )

        print(latex_table)
