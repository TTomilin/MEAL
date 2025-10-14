from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Type alias for confidence intervals
ConfInt = tuple[float, float]


def load_series(fp: Path) -> List[float]:
    """Load a JSON series from file."""
    with open(fp, 'r') as f:
        data = json.load(f)
    return [float(x) for x in data]


def _mean_ci(series: List[float]) -> ConfInt:
    """Compute mean and 95% confidence interval."""
    if not series:
        return np.nan, np.nan
    mean = float(np.mean(series))
    if len(series) == 1:
        return mean, 0.0
    ci = 1.96 * np.std(series, ddof=1) / np.sqrt(len(series))
    return mean, float(ci)


def compute_metrics(
        data_root: Path,
        algo: str,
        method: str,
        strategy: str,
        seq_len: int,
        seeds: List[int],
        num_agents: int,
        end_window_evals: int = 10,
        level: int = 1,
) -> dict:
    """Compute metrics for a single algorithm/method/num_agents combination."""
    AP_seeds, F_seeds, FT_seeds = [], [], []

    # Load baseline data once for forward transfer calculation
    repo_root = Path(__file__).resolve().parent.parent
    baseline_data = {}
    baseline_folder = (
        repo_root
        / data_root
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

    base_folder = (
            repo_root
            / data_root
            / algo
            / method
            / f"level_{level}"
            / f"agents_{num_agents}"
            / f"{strategy}_{seq_len}"
    )

    for seed in seeds:
        sd = base_folder / f"seed_{seed}"
        if not sd.exists():
            print(f"[debug] seed directory does not exist: {sd}")
            continue



        # Per‑environment evaluation curves
        env_series: List[List[float]] = []
        present_task_ids: List[int] = []
        for i in range(seq_len):
            # find eval file for task i; ignore training files
            cand = sorted([p for p in sd.glob(f"{i}_*_soup.json") if "training" not in p.name])
            if not cand:
                # try a looser pattern (older dumps)
                cand = sorted([p for p in sd.glob(f"{i}_*soup.*") if "training" not in p.name])
            if not cand:
                # no eval file for this task; skip it
                print(f"[info] missing eval for task {i}, seed {seed} — skipping this task")
                continue
            env_series.append(load_series(cand[0]))
            present_task_ids.append(i)

        if len(env_series) == 0:
            print(f"[warn] no eval curves found for seed {seed}; skipping seed")
            continue






        L = max(len(s) for s in env_series)
        env_mat = np.vstack([
            np.pad(s, (0, L - len(s)), constant_values=s[-1]) for s in env_series
        ])

        # Average Performance (AP) – last eval of mean curve
        AP_seeds.append(env_mat.mean(axis=0)[-1])

        # Load training data for forward transfer calculation
        training_fp = sd / "training_soup.json"
        if not training_fp.exists():
            print(f"[warn] missing training_soup.json for {method} {num_agents}agents seed {seed}")
            FT_seeds.append(np.nan)
        else:
            training = load_series(training_fp)
            n_train = len(training)
            chunk = max(1, n_train // seq_len)

            # Forward Transfer (FT) – normalized area between CL and baseline curves
            if seed not in baseline_data:
                print(f"[warn] missing baseline data for seed {seed}")
                FT_seeds.append(np.nan)
            else:
                ft_vals = []
                for i in present_task_ids:
                    # Calculate AUC for CL method (task i)
                    start_idx = i * chunk
                    end_idx = (i + 1) * chunk
                    end_idx = min(end_idx, n_train)  # clamp
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

                    # guard index in baseline array
                    if i >= len(baseline_data[seed]):
                        print(f"[warn] missing baseline index for task {i}, seed {seed}")
                        continue
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
        for i in range(env_mat.shape[0]):
            final_avg = np.nanmean(env_mat[i, fw_start : final_idx + 1])
            best_perf = np.nanmax(env_mat[i, : final_idx + 1])
            f_vals.append(max(best_perf - final_avg, 0.0))
        F_seeds.append(float(np.nanmean(f_vals)))

    # Aggregate across seeds
    A_mean, A_ci = _mean_ci(AP_seeds)
    F_mean, F_ci = _mean_ci(F_seeds)
    FT_mean, FT_ci = _mean_ci(FT_seeds)

    return {
        "AveragePerformance": A_mean,
        "AveragePerformance_CI": A_ci,
        "Forgetting": F_mean,
        "Forgetting_CI": F_ci,
        "ForwardTransfer": FT_mean,
        "ForwardTransfer_CI": FT_ci,
    }


def compare_agents(
        data_root: Path,
        algorithm: str,
        method: str,
        strategy: str,
        seq_len: int,
        seeds: List[int],
        num_agents_list: List[int],
        levels: List[int],
        end_window_evals: int = 10,
) -> pd.DataFrame:
    """Compare results between different numbers of agents."""
    rows = []

    for num_agents in num_agents_list:
        row_data = {"NumAgents": num_agents}

        for level in levels:
            # Compute metrics for this level
            metrics = compute_metrics(
                data_root=data_root,
                algo=algorithm,
                method=method,
                strategy=strategy,
                seq_len=seq_len,
                seeds=seeds,
                num_agents=num_agents,
                end_window_evals=end_window_evals,
                level=level,
            )

            # Add metrics to row with level prefix
            row_data[f"LEVEL{level}_AveragePerformance"] = metrics["AveragePerformance"]
            row_data[f"LEVEL{level}_AveragePerformance_CI"] = metrics["AveragePerformance_CI"]
            row_data[f"LEVEL{level}_Forgetting"] = metrics["Forgetting"]
            row_data[f"LEVEL{level}_Forgetting_CI"] = metrics["Forgetting_CI"]
            row_data[f"LEVEL{level}_ForwardTransfer"] = metrics["ForwardTransfer"]
            row_data[f"LEVEL{level}_ForwardTransfer_CI"] = metrics["ForwardTransfer_CI"]

        rows.append(row_data)

    return pd.DataFrame(rows)


def _fmt(mean: float, ci: float, best: bool, better: str = "max", show_ci: bool = True) -> str:
    """Return *mean ±CI* formatted for LaTeX, with CI in \scriptsize."""
    if np.isnan(mean) or np.isinf(mean):
        return "--"
    main = f"{mean:.3f}"
    if best:
        main = rf"\textbf{{{main}}}"
    ci_part = rf"{{\scriptsize$\pm{ci:.2f}$}}" if show_ci and not np.isnan(ci) and ci > 0 else ""
    return main + ci_part


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare results between different numbers of agents")
    p.add_argument("--data_root", default="data", help="Root directory containing the data")
    p.add_argument("--algorithm", default="ippo", help="Algorithm to analyze")
    p.add_argument("--method", default="EWC", help="Continual learning method to compare")
    p.add_argument("--strategy", default="generate", help="Strategy name")
    p.add_argument("--seq_len", type=int, default=20, help="Sequence length")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Seeds to include")
    p.add_argument("--num_agents", type=int, nargs="+", default=[1, 2, 3], help="Number of agents to compare")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3], help="Difficulty levels to compare")
    p.add_argument(
        "--end_window_evals",
        type=int,
        default=10,
        help="How many final eval points to average for F (Forgetting)",
    )
    p.add_argument(
        "--confidence_intervals",
        action="store_true",
        help="Include confidence intervals in the output table",
    )
    args = p.parse_args()

    print(f"Comparing num_agents: {args.num_agents}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Method: {args.method}")
    print(f"Strategy: {args.strategy}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Seeds: {args.seeds}")
    print(f"Levels: {args.levels}")

    # Compute comparison metrics
    df = compare_agents(
        data_root=Path(args.data_root),
        algorithm=args.algorithm,
        method=args.method,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
        num_agents_list=args.num_agents,
        levels=args.levels,
        end_window_evals=args.end_window_evals,
    )

    # Find best values per column (across all agent configurations)
    best_values = {}
    for level in args.levels:
        # Get all values for this level across all agent configurations
        ap_values = [row[f"LEVEL{level}_AveragePerformance"] for _, row in df.iterrows() 
                     if not (np.isnan(row[f"LEVEL{level}_AveragePerformance"]) or np.isinf(row[f"LEVEL{level}_AveragePerformance"]))]
        f_values = [row[f"LEVEL{level}_Forgetting"] for _, row in df.iterrows() 
                    if not (np.isnan(row[f"LEVEL{level}_Forgetting"]) or np.isinf(row[f"LEVEL{level}_Forgetting"]))]
        ft_values = [row[f"LEVEL{level}_ForwardTransfer"] for _, row in df.iterrows() 
                     if not (np.isnan(row[f"LEVEL{level}_ForwardTransfer"]) or np.isinf(row[f"LEVEL{level}_ForwardTransfer"]))]

        best_values[f"A_L{level}"] = max(ap_values) if ap_values else np.nan
        best_values[f"F_L{level}"] = min(f_values) if f_values else np.nan
        best_values[f"FT_L{level}"] = max(ft_values) if ft_values else np.nan

    # For each num_agents, format the table
    df_out_rows = []

    for _, row in df.iterrows():
        num_agents = row["NumAgents"]

        # Extract values for each level
        levels_values = {}
        for level in args.levels:
            levels_values[level] = {
                'ap': row[f"LEVEL{level}_AveragePerformance"],
                'ap_ci': row[f"LEVEL{level}_AveragePerformance_CI"],
                'f': row[f"LEVEL{level}_Forgetting"],
                'f_ci': row[f"LEVEL{level}_Forgetting_CI"],
                'ft': row[f"LEVEL{level}_ForwardTransfer"],
                'ft_ci': row[f"LEVEL{level}_ForwardTransfer_CI"],
            }

        # Create formatted row
        agent_text = f"{int(num_agents)} Agent" + ("s" if num_agents > 1 else "")
        formatted_row = {"Agents": agent_text}

        # Add formatted columns grouped by level: for each level, add A, F, FT columns
        for level in args.levels:
            values = levels_values[level]

            # Average Performance column for this level
            formatted_row[f"AveragePerformance_L{level}"] = _fmt(
                values['ap'], 
                values['ap_ci'], 
                values['ap'] == best_values[f"A_L{level}"], 
                "max",
                args.confidence_intervals
            )

            # Forgetting column for this level
            formatted_row[f"Forgetting_L{level}"] = _fmt(
                values['f'], 
                values['f_ci'], 
                values['f'] == best_values[f"F_L{level}"], 
                "min",
                args.confidence_intervals
            )

            # Forward Transfer column for this level
            formatted_row[f"ForwardTransfer_L{level}"] = _fmt(
                values['ft'], 
                values['ft_ci'], 
                values['ft'] == best_values[f"FT_L{level}"], 
                "max",
                args.confidence_intervals
            )

        df_out_rows.append(formatted_row)

    df_out = pd.DataFrame(df_out_rows)

    # Rename columns to mathy headers grouped by levels
    new_columns = ["Agents"]
    for level in args.levels:
        new_columns.extend([
            rf"$\mathcal{{A}}\!\uparrow$",
            rf"$\mathcal{{F}}\!\downarrow$", 
            rf"$\mathcal{{FT}}\!\uparrow$"
        ])
    df_out.columns = new_columns

    # Column format: Agents + levels × 3 metrics
    column_format = "l" + "c" * (len(args.levels) * 3)

    # Generate LaTeX table with proper structure following results_table.py format
    caption_text = (f"Comparison of {args.method} with {args.algorithm.upper()} across "
                   f"{' and '.join([f'Level {level}' for level in args.levels])} "
                   f"for {' and '.join([f'{n} agent' + ('s' if n > 1 else '') for n in args.num_agents])}. "
                   f"Bold values indicate the best performance for each metric. "
                   f"$\\mathcal{{A}}$ represents Average Performance (higher is better), "
                   f"$\\mathcal{{F}}$ represents Forgetting (lower is better), "
                   f"$\\mathcal{{FT}}$ represents Forward Transfer (higher is better).")

    # Build the LaTeX table manually with correct structure
    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{caption_text}}}")
    latex_lines.append("\\label{tab:agents_comparison}")
    latex_lines.append(f"\\begin{{tabular}}{{{column_format}}}")
    latex_lines.append("\\toprule")

    # Create the multicolumn header following the format from results_table.py
    multicolumn_header = "\\multirow{2}{*}[-0.7ex]{Agents} &"

    # Add multicolumn headers for each level
    for i, level in enumerate(args.levels):
        level_text = f"Level {level}"
        if i < len(args.levels) - 1:
            multicolumn_header += f"\n\\multicolumn{{3}}{{c}}{{{level_text}}} &"
        else:
            multicolumn_header += f"\n\\multicolumn{{3}}{{c}}{{{level_text}}} \\\\"

    latex_lines.append(multicolumn_header)

    # Add cmidrule parts
    cmidrule_parts = []
    for i, _ in enumerate(args.levels):
        start_col = 2 + i * 3
        end_col = start_col + 2
        cmidrule_parts.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")

    latex_lines.append(" ".join(cmidrule_parts))

    # Add the metric symbols row
    metric_row = " & "
    for i, _ in enumerate(args.levels):
        if i < len(args.levels) - 1:
            metric_row += "$\\mathcal{A}\\!\\uparrow$ & $\\mathcal{F}\\!\\downarrow$ & $\\mathcal{FT}\\!\\uparrow$ & "
        else:
            metric_row += "$\\mathcal{A}\\!\\uparrow$ & $\\mathcal{F}\\!\\downarrow$ & $\\mathcal{FT}\\!\\uparrow$ \\\\"

    latex_lines.append(metric_row)
    latex_lines.append("\\midrule")

    # Add data rows
    for _, row in df_out.iterrows():
        row_data = []
        for i in range(len(df_out.columns)):
            row_data.append(str(row.iloc[i]))
        latex_lines.append(" & ".join(row_data) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_table = "\n".join(latex_lines)

    print("\nComparison Results:")
    print("=" * 80)
    print(df_out.to_string(index=False))

    print(f"\nLATEX TABLE:")
    print("-" * 40)
    print(latex_table)
