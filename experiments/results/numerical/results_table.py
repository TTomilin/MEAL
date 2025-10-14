#!/usr/bin/env python3
"""Build a LaTeX table with mean ±95% CI (smaller font) for CL metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_series(fp: Path) -> np.ndarray:
    """Load a 1‑D float array from *.json or *.npz."""
    if fp.suffix == ".json":
        return np.array(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == ".npz":
        return np.load(fp)["data"].astype(float)
    raise ValueError(f"Unsupported file suffix: {fp.suffix}")


# -----------------------------------------------------------------------------
# Metric aggregation
# -----------------------------------------------------------------------------

ConfInt = tuple[float, float]  # (mean, 95% CI)

def _mean_ci(series: List[float]) -> ConfInt:
    if not series:
        return np.nan, np.nan
    mean = float(np.mean(series))
    if len(series) == 1:
        return mean, 0.0
    ci = 1.96 * np.std(series, ddof=1) / np.sqrt(len(series))
    return mean, float(ci)


def _calculate_curve_based_forgetting(task_curve: np.ndarray, training_end_idx: int = None) -> float:
    """
    Calculate normalized forgetting where 0 = no forgetting and 1 = complete forgetting.

    Forgetting is normalized such that:
    - 0 means performance never drops below the end-of-training performance
    - 1 means performance drops to 0 right after training finishes and stays there

    Args:
        task_curve: Performance curve for a single task over time
        training_end_idx: Index where training for this task ends. If None, uses the last index.

    Returns:
        Normalized forgetting score between 0 and 1
    """
    if len(task_curve) <= 1:
        return 0.0

    # Determine the end-of-training index
    if training_end_idx is None or training_end_idx >= len(task_curve):
        training_end_idx = len(task_curve) - 1

    # Use the performance at the end of training as the baseline reference
    end_of_training_performance = task_curve[training_end_idx - 1]

    # Only consider performance after the end of training for forgetting calculation
    if training_end_idx >= len(task_curve) - 1:
        # Training ends at the last point, so no forgetting can be measured
        return 0.0

    post_training_curve = task_curve[training_end_idx + 1:]

    # Calculate forgetting at each time step after training ends
    # Forgetting = max(0, end_of_training_performance - current_performance)
    forgetting_at_each_step = np.maximum(end_of_training_performance - post_training_curve, 0.0)

    # Normalize forgetting by end_of_training_performance to get values between 0 and 1
    # This makes 1.0 represent complete forgetting (performance drops to 0)
    normalized_forgetting_at_each_step = forgetting_at_each_step / end_of_training_performance

    # Weight forgetting by how early it occurs (earlier forgetting gets higher weight)
    # Use exponential decay: weight = exp(-λ * (t / T)) where t is time step, T is total time
    lambda_decay = 2.0  # Higher values penalize early forgetting more
    time_steps = np.arange(len(post_training_curve))
    total_time = len(post_training_curve) - 1

    if total_time > 0:
        # Normalize time to [0, 1] and apply exponential decay
        normalized_time = time_steps / total_time
        weights = np.exp(-lambda_decay * normalized_time)
    else:
        weights = np.ones(len(post_training_curve))

    # Calculate weighted normalized forgetting
    weighted_forgetting = normalized_forgetting_at_each_step * weights

    # Calculate the weighted average forgetting score
    # Use the sum of weights to properly normalize the weighted average
    if len(weighted_forgetting) > 0 and np.sum(weights) > 0:
        curve_based_forgetting = np.sum(weighted_forgetting) / np.sum(weights)
    else:
        curve_based_forgetting = 0.0

    # Ensure the result is between 0 and 1
    return float(np.clip(curve_based_forgetting, 0.0, 1.0))


def compute_metrics(
        data_root: Path,
        algo: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        end_window_evals: int = 10,
        level: int = 1,
) -> pd.DataFrame:
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
                    baseline_series = load_series(baseline_file)
                    # Validate the loaded data
                    if len(baseline_series) == 0:
                        print(f"[warn] empty baseline data for task {i}, seed {seed}")
                        baseline_training_files.append(None)
                    elif np.all(np.isnan(baseline_series)):
                        print(f"[warn] baseline data contains all NaN for task {i}, seed {seed}")
                        baseline_training_files.append(None)
                    elif np.all(np.isinf(baseline_series)):
                        print(f"[warn] baseline data contains all inf/-inf for task {i}, seed {seed}")
                        baseline_training_files.append(None)
                    elif np.all(np.isnan(baseline_series) | np.isinf(baseline_series)):
                        print(f"[warn] baseline data contains all NaN/inf/-inf for task {i}, seed {seed}")
                        baseline_training_files.append(None)
                    else:
                        baseline_training_files.append(baseline_series)
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
            # Handle missing files by creating expected file paths and loading them
            # This ensures we always have seq_len series, even if some files are missing
            env_series = []
            missing_files = []
            for i in range(seq_len):
                expected_file = sd / f"{i}_gen_soup.json"
                if expected_file.exists():
                    env_series.append(load_series(expected_file))
                else:
                    # Try alternative naming patterns
                    alt_file = sd / f"{i}_soup.json"
                    if alt_file.exists():
                        env_series.append(load_series(alt_file))
                    else:
                        print(f"[warn] missing env file for task {i}, seed {seed}, method {method}, using zeros")
                        missing_files.append(i)
                        # Create a default array of zeros with reasonable length
                        env_series.append(np.zeros(100))

            # Replace NaN and inf/-inf values with zeros in env_series
            processed_env_series = []
            for i, series in enumerate(env_series):
                # Check for NaN and inf/-inf values
                has_nan = np.any(np.isnan(series))
                has_inf = np.any(np.isinf(series))

                if np.all(np.isnan(series)) and not has_inf:
                    print(f"[warn] env series {i} contains all NaN values for {method} seed {seed}, replacing with zeros")
                    processed_series = np.zeros_like(series)
                elif np.all(np.isinf(series)) and not has_nan:
                    print(f"[warn] env series {i} contains all inf/-inf values for {method} seed {seed}, replacing with zeros")
                    processed_series = np.zeros_like(series)
                elif np.all(np.isnan(series) | np.isinf(series)):
                    print(f"[warn] env series {i} contains all NaN/inf/-inf values for {method} seed {seed}, replacing with zeros")
                    processed_series = np.zeros_like(series)
                elif has_nan and has_inf:
                    print(f"[warn] env series {i} contains some NaN and inf/-inf values for {method} seed {seed}, replacing with zeros")
                    processed_series = np.where(np.isnan(series) | np.isinf(series), 0.0, series)
                elif has_nan:
                    print(f"[warn] env series {i} contains some NaN values for {method} seed {seed}, replacing NaN with zeros")
                    processed_series = np.where(np.isnan(series), 0.0, series)
                elif has_inf:
                    print(f"[warn] env series {i} contains some inf/-inf values for {method} seed {seed}, replacing with zeros")
                    processed_series = np.where(np.isinf(series), 0.0, series)
                else:
                    processed_series = series
                processed_env_series.append(processed_series)

            L = max(len(s) for s in processed_env_series)
            env_mat = np.vstack([
                np.pad(s, (0, L - len(s)), constant_values=s[-1]) for s in processed_env_series
            ])

            # Average Performance (AP) – last eval of mean curve
            AP_seeds.append(np.nanmean(env_mat, axis=0)[-1])

            # Forgetting (F) – curve-based forgetting that considers when forgetting occurs
            f_vals = []
            final_idx = env_mat.shape[1] - 1

            # Process all series (NaN values have been replaced with zeros)
            for i in range(seq_len):
                task_curve = env_mat[i, : final_idx + 1]

                # Calculate when training for task i ends in the evaluation timeline
                # Training for task i ends at (i + 1) * chunk in the training timeline
                # Map this proportionally to the evaluation timeline
                training_end_step = (i + 1) * chunk
                if n_train > 0:
                    # Map training timeline to evaluation timeline proportionally
                    training_end_idx = int((training_end_step / n_train) * len(task_curve))
                    # Ensure the index is within bounds
                    training_end_idx = min(training_end_idx, len(task_curve) - 1)
                else:
                    # Fallback: use the end of the curve
                    training_end_idx = len(task_curve) - 1

                # Calculate curve-based forgetting using end-of-training performance
                if any(task_curve > 0.0):  # If end-of-training performance is 0 or negative, no meaningful forgetting can be calculated
                    curve_forgetting = _calculate_curve_based_forgetting(task_curve, training_end_idx)
                    f_vals.append(curve_forgetting)

            F_seeds.append(float(np.nanmean(f_vals)))

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


# -----------------------------------------------------------------------------
# LaTeX formatting helpers
# -----------------------------------------------------------------------------

def _fmt(mean: float, ci: float, best: bool, better: str = "max", show_confidence_intervals: bool = True) -> str:
    """Return *mean ±CI* formatted for LaTeX, with CI in \scriptsize."""
    if np.isnan(mean):
        return "--"
    main = f"{mean:.3f}"
    if best:
        main = rf"\textbf{{{main}}}"
    ci_part = rf"{{\scriptsize$\pm{ci:.2f}$}}" if show_confidence_intervals and not np.isnan(ci) and ci > 0 else ""
    return main + ci_part


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--algo", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--strategy", default='generate')
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--level", type=int, default=None, help="Difficulty level of the environment (if not provided, generates table for all levels 1, 2, 3)")
    p.add_argument(
        "--end_window_evals",
        type=int,
        default=10,
        help="How many final eval points to average for F (Forgetting)",
    )
    p.add_argument(
        "--confidence-intervals",
        action="store_true",
        default=True,
        help="Show confidence intervals in table (default: True).",
    )
    p.add_argument(
        "--no-confidence-intervals",
        dest="confidence_intervals",
        action="store_false",
        help="Hide confidence intervals in table.",
    )
    args = p.parse_args()

    # Handle single level or all levels
    if args.level is not None:
        # Single level case (original behavior)
        df = compute_metrics(
            data_root=Path(args.data_root),
            algo=args.algo,
            methods=args.methods,
            strategy=args.strategy,
            seq_len=args.seq_len,
            seeds=args.seeds,
            end_window_evals=args.end_window_evals,
            level=args.level,
        )
        # Pretty‑print method names
        df["Method"] = df["Method"].replace({"Online_EWC": "Online EWC"})
    else:
        # All levels case (new behavior) - pivot so each method is one row with columns for each level
        level_data = {}
        for level in [1, 2, 3]:
            level_df = compute_metrics(
                data_root=Path(args.data_root),
                algo=args.algo,
                methods=args.methods,
                strategy=args.strategy,
                seq_len=args.seq_len,
                seeds=args.seeds,
                end_window_evals=args.end_window_evals,
                level=level,
            )
            # Pretty‑print method names
            level_df["Method"] = level_df["Method"].replace({"Online_EWC": "Online EWC"})
            level_data[level] = level_df

        # Create pivoted structure: one row per method, columns for each level
        methods = level_data[1]["Method"].tolist()
        rows = []

        for method in methods:
            row = {"Method": method}

            # Add columns for each level and metric
            for level in [1, 2, 3]:
                level_df = level_data[level]
                method_row = level_df[level_df["Method"] == method].iloc[0]

                # Add columns with level suffix
                row[f"AveragePerformance_L{level}"] = method_row["AveragePerformance"]
                row[f"AveragePerformance_CI_L{level}"] = method_row["AveragePerformance_CI"]
                row[f"Forgetting_L{level}"] = method_row["Forgetting"]
                row[f"Forgetting_CI_L{level}"] = method_row["Forgetting_CI"]
                row[f"ForwardTransfer_L{level}"] = method_row["ForwardTransfer"]
                row[f"ForwardTransfer_CI_L{level}"] = method_row["ForwardTransfer_CI"]

            rows.append(row)

        df = pd.DataFrame(rows)


    if args.level is not None:
        # Single level case - original formatting
        # Identify best means (ignoring CI)
        best_A = df["AveragePerformance"].max()
        best_F = df["Forgetting"].min()
        best_FT = df["ForwardTransfer"].max()

        # Build human‑readable strings with CI
        df_out = pd.DataFrame()
        df_out["Method"] = df["Method"]
        df_out["AveragePerformance"] = df.apply(
            lambda r: _fmt(r.AveragePerformance, r.AveragePerformance_CI, r.AveragePerformance == best_A, "max", args.confidence_intervals),
            axis=1,
        )
        df_out["Forgetting"] = df.apply(
            lambda r: _fmt(r.Forgetting, r.Forgetting_CI, r.Forgetting == best_F, "min", args.confidence_intervals),
            axis=1,
        )
        df_out["ForwardTransfer"] = df.apply(
            lambda r: _fmt(r.ForwardTransfer, r.ForwardTransfer_CI, r.ForwardTransfer == best_FT, "max", args.confidence_intervals),
            axis=1,
        )

        # Rename columns to mathy headers
        df_out.columns = [
            "Method",
            r"$\mathcal{A}\!\uparrow$",
            r"$\mathcal{F}\!\downarrow$",
            r"$\mathcal{FT}\!\uparrow$",
        ]

        column_format = "lccc"
    else:
        # All levels case - new formatting with columns for each level
        # Identify best means for each level (ignoring CI)
        best_values = {}
        for level in [1, 2, 3]:
            best_values[f"A_L{level}"] = df[f"AveragePerformance_L{level}"].max()
            best_values[f"F_L{level}"] = df[f"Forgetting_L{level}"].min()
            best_values[f"FT_L{level}"] = df[f"ForwardTransfer_L{level}"].max()

        # Build human‑readable strings with CI
        df_out = pd.DataFrame()
        df_out["Method"] = df["Method"]

        # Add formatted columns grouped by level: for each level, add A, F, FT columns
        for level in [1, 2, 3]:
            # Average Performance column for this level
            df_out[f"AveragePerformance_L{level}"] = df.apply(
                lambda r: _fmt(
                    r[f"AveragePerformance_L{level}"], 
                    r[f"AveragePerformance_CI_L{level}"], 
                    r[f"AveragePerformance_L{level}"] == best_values[f"A_L{level}"], 
                    "max",
                    args.confidence_intervals
                ),
                axis=1,
            )

            # Forgetting column for this level
            df_out[f"Forgetting_L{level}"] = df.apply(
                lambda r: _fmt(
                    r[f"Forgetting_L{level}"], 
                    r[f"Forgetting_CI_L{level}"], 
                    r[f"Forgetting_L{level}"] == best_values[f"F_L{level}"], 
                    "min",
                    args.confidence_intervals
                ),
                axis=1,
            )

            # Forward Transfer column for this level
            df_out[f"ForwardTransfer_L{level}"] = df.apply(
                lambda r: _fmt(
                    r[f"ForwardTransfer_L{level}"], 
                    r[f"ForwardTransfer_CI_L{level}"], 
                    r[f"ForwardTransfer_L{level}"] == best_values[f"FT_L{level}"], 
                    "max",
                    args.confidence_intervals
                ),
                axis=1,
            )

        # Rename columns to mathy headers grouped by level
        new_columns = ["Method"]
        for level in [1, 2, 3]:
            new_columns.extend([
                rf"$\mathcal{{A}}\!\uparrow$",
                rf"$\mathcal{{F}}\!\downarrow$", 
                rf"$\mathcal{{FT}}\!\uparrow$"
            ])
        df_out.columns = new_columns

        # Column format: Method + 3 levels × 3 metrics = 10 columns
        column_format = "l" + "c" * 9

    if args.level is not None:
        # Single level case - use standard LaTeX table
        latex_table = df_out.to_latex(
            index=False,
            escape=False,
            column_format=column_format,
            label="tab:cmarl_metrics",
        )
    else:
        # All levels case - custom LaTeX table with multicolumn headers
        latex_table = df_out.to_latex(
            index=False,
            escape=False,
            column_format=column_format,
            label="tab:cmarl_metrics",
            caption="Continual learning metrics across three difficulty levels.",
        )

        # Replace the header to add multicolumn structure
        lines = latex_table.split('\n')

        # Find the header line (contains the column names)
        header_line_idx = None
        for i, line in enumerate(lines):
            if '$\\mathcal{A}' in line:
                header_line_idx = i
                break

        if header_line_idx is not None:
            # Create the multicolumn header
            multicolumn_header = (
                "\\multirow{2}{*}{Method} &\n"
                "\\multicolumn{3}{c}{Level 1} &\n"
                "\\multicolumn{3}{c}{Level 2} &\n"
                "\\multicolumn{3}{c}{Level 3} \\\\\n"
                "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n"
                " & $\\mathcal{A}\\!\\uparrow$ & $\\mathcal{F}\\!\\downarrow$ & $\\mathcal{FT}\\!\\uparrow$ "
                " & $\\mathcal{A}\\!\\uparrow$ & $\\mathcal{F}\\!\\downarrow$ & $\\mathcal{FT}\\!\\uparrow$"
                " & $\\mathcal{A}\\!\\uparrow$ & $\\mathcal{F}\\!\\downarrow$ & $\\mathcal{FT}\\!\\uparrow$ \\\\"
            )

            # Replace the original header line
            lines[header_line_idx] = multicolumn_header

            # Reconstruct the table
            latex_table = '\n'.join(lines)

    print(latex_table)
