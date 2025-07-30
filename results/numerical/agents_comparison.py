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

    base_folder = (
            data_root
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
        env_files = sorted([
            f for f in sd.glob("*_soup.*") if "training" not in f.name
        ])
        if len(env_files) != seq_len:
            print(
                f"[warn] expected {seq_len} env files, found {len(env_files)} "
                f"for {algo} {method} {num_agents}agents seed {seed}"
            )
            continue

        env_series = [load_series(f) for f in env_files]
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
            chunk = n_train // seq_len

            # Forward Transfer (FT) – normalized area between CL and baseline curves
            if seed not in baseline_data:
                print(f"[warn] missing baseline data for seed {seed}")
                FT_seeds.append(np.nan)
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

                        # Calculate Forward Transfer: FTi = (AUCi - AUCb_i) / (1 - AUCb_i)
                        denominator = 1.0 - auc_baseline

                        # More robust checks for problematic denominators
                        if np.isnan(denominator) or np.isinf(denominator) or abs(denominator) < 1e-8:
                            if abs(auc_baseline - 1.0) < 1e-8:
                                print(f"[warn] baseline AUC ≈ 1.0 for task {i}, seed {seed}, method {method}")
                            elif np.isnan(auc_baseline):
                                print(f"[warn] baseline AUC is NaN for task {i}, seed {seed}, method {method}")
                            elif np.isinf(denominator):
                                print(f"[warn] denominator is inf/-inf ({denominator}) for task {i}, seed {seed}, method {method}")
                            else:
                                print(f"[warn] denominator too small ({denominator}) for task {i}, seed {seed}, method {method}")
                            # Skip this task - don't append to ft_vals
                        else:
                            ft_i = (auc_cl - auc_baseline) / denominator
                            # Check if the final ft_i is inf/-inf
                            if np.isinf(ft_i):
                                print(f"[warn] Forward Transfer result is inf/-inf for task {i}, seed {seed}, method {method}")
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

    for level in levels:
        row_data = {"Level": level}

        for num_agents in num_agents_list:
            # Compute metrics for this num_agents
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

            # Add metrics to row with num_agents prefix
            row_data[f"{num_agents}AGENTS_AveragePerformance"] = metrics["AveragePerformance"]
            row_data[f"{num_agents}AGENTS_AveragePerformance_CI"] = metrics["AveragePerformance_CI"]
            row_data[f"{num_agents}AGENTS_Forgetting"] = metrics["Forgetting"]
            row_data[f"{num_agents}AGENTS_Forgetting_CI"] = metrics["Forgetting_CI"]
            row_data[f"{num_agents}AGENTS_ForwardTransfer"] = metrics["ForwardTransfer"]
            row_data[f"{num_agents}AGENTS_ForwardTransfer_CI"] = metrics["ForwardTransfer_CI"]

        rows.append(row_data)

    return pd.DataFrame(rows)


def _fmt(mean: float, ci: float, best: bool, better: str = "max") -> str:
    """Return *mean ±CI* formatted for LaTeX, with CI in \scriptsize."""
    if np.isnan(mean) or np.isinf(mean):
        return "--"
    main = f"{mean:.3f}"
    if best:
        main = rf"\textbf{{{main}}}"
    ci_part = rf"{{\scriptsize$\pm{ci:.2f}$}}" if not np.isnan(ci) and ci > 0 else ""
    return main + ci_part


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare results between different numbers of agents")
    p.add_argument("--data_root", default="results/data", help="Root directory containing the data")
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

    # For each level, identify best performance and format the table
    df_out_rows = []

    for _, row in df.iterrows():
        level = row["Level"]

        # Extract values for each num_agents
        agents_values = {}
        for num_agents in args.num_agents:
            agents_values[num_agents] = {
                'ap': row[f"{num_agents}AGENTS_AveragePerformance"],
                'ap_ci': row[f"{num_agents}AGENTS_AveragePerformance_CI"],
                'f': row[f"{num_agents}AGENTS_Forgetting"],
                'f_ci': row[f"{num_agents}AGENTS_Forgetting_CI"],
                'ft': row[f"{num_agents}AGENTS_ForwardTransfer"],
                'ft_ci': row[f"{num_agents}AGENTS_ForwardTransfer_CI"],
            }

        # Find best values across num_agents for this level
        valid_a_values = [v['ap'] for v in agents_values.values() if not (np.isnan(v['ap']) or np.isinf(v['ap']))]
        valid_f_values = [v['f'] for v in agents_values.values() if not (np.isnan(v['f']) or np.isinf(v['f']))]
        valid_ft_values = [v['ft'] for v in agents_values.values() if not (np.isnan(v['ft']) or np.isinf(v['ft']))]

        best_a = max(valid_a_values) if valid_a_values else np.nan
        best_f = min(valid_f_values) if valid_f_values else np.nan
        best_ft = max(valid_ft_values) if valid_ft_values else np.nan

        # Create formatted row
        formatted_row = {"Level": f"Level {int(level)}"}

        # First add all average performance columns
        for num_agents in args.num_agents:
            values = agents_values[num_agents]

            formatted_row[f"{num_agents}AGENTS_AveragePerformance"] = _fmt(
                values['ap'], 
                values['ap_ci'], 
                values['ap'] == best_a, 
                "max"
            )

        # Then add all forgetting columns
        for num_agents in args.num_agents:
            values = agents_values[num_agents]

            formatted_row[f"{num_agents}AGENTS_Forgetting"] = _fmt(
                values['f'], 
                values['f_ci'], 
                values['f'] == best_f, 
                "min"
            )

        # Finally add all forward transfer columns
        for num_agents in args.num_agents:
            values = agents_values[num_agents]

            formatted_row[f"{num_agents}AGENTS_ForwardTransfer"] = _fmt(
                values['ft'], 
                values['ft_ci'], 
                values['ft'] == best_ft, 
                "max"
            )

        df_out_rows.append(formatted_row)

    df_out = pd.DataFrame(df_out_rows)

    # Create column headers - first all average performance, then all forgetting, then all forward transfer
    columns = ["Level"]
    # Add all average performance columns first
    for num_agents in args.num_agents:
        columns.append(rf"$\mathcal{{A}}\!\uparrow$ {num_agents} Agents")
    # Then add all forgetting columns
    for num_agents in args.num_agents:
        columns.append(rf"$\mathcal{{F}}\!\downarrow$ {num_agents} Agents")
    # Finally add all forward transfer columns
    for num_agents in args.num_agents:
        columns.append(rf"$\mathcal{{FT}}\!\uparrow$ {num_agents} Agents")

    df_out.columns = columns

    # Generate LaTeX table
    column_format = "l" + "ccc" * len(args.num_agents)
    latex_table = df_out.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        label="tab:agents_comparison",
        caption=f"Comparison of {args.method} with {args.algorithm.upper()} between "
                f"{' and '.join([f'{n} agent' + ('s' if n > 1 else '') for n in args.num_agents])}. "
                f"Bold values indicate the best performance for each metric. "
                f"$\\mathcal{{A}}$ represents Average Performance (higher is better), "
                f"$\\mathcal{{F}}$ represents Forgetting (lower is better), "
                f"$\\mathcal{{FT}}$ represents Forward Transfer (higher is better).",
    )

    print("\nComparison Results:")
    print("=" * 80)
    print(df_out.to_string(index=False))

    print(f"\nLATEX TABLE:")
    print("-" * 40)
    print(latex_table)
