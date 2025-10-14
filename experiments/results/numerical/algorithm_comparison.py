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
        end_window_evals: int = 10,
        level: int = 1,
) -> dict:
    """Compute metrics for a single algorithm/method combination."""
    AP_seeds, F_seeds = [], []

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

        # Per‑environment evaluation curves
        env_files = sorted([
            f for f in sd.glob("*_soup.*") if "training" not in f.name
        ])
        if len(env_files) != seq_len:
            print(
                f"[warn] expected {seq_len} env files, found {len(env_files)} "
                f"for {algo} {method} seed {seed}"
            )
            continue

        env_series = [load_series(f) for f in env_files]
        L = max(len(s) for s in env_series)
        env_mat = np.vstack([
            np.pad(s, (0, L - len(s)), constant_values=s[-1]) for s in env_series
        ])

        # Average Performance (AP) – last eval of mean curve
        AP_seeds.append(env_mat.mean(axis=0)[-1])

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

    return {
        "AveragePerformance": A_mean,
        "AveragePerformance_CI": A_ci,
        "Forgetting": F_mean,
        "Forgetting_CI": F_ci,
    }


def compare_algorithms(
        data_root: Path,
        algorithms: List[str],
        method: str,
        strategy: str,
        seq_len: int,
        seeds: List[int],
        levels: List[int],
        end_window_evals: int = 10,
) -> pd.DataFrame:
    """Compare EWC_partial results between different algorithms."""
    rows = []

    for level in levels:
        row_data = {"Level": level}

        for algo in algorithms:
            # Compute metrics for this algorithm
            metrics = compute_metrics(
                data_root=data_root,
                algo=algo,
                method=method,
                strategy=strategy,
                seq_len=seq_len,
                seeds=seeds,
                end_window_evals=end_window_evals,
                level=level,
            )

            # Add metrics to row with algorithm prefix
            row_data[f"{algo.upper()}_AveragePerformance"] = metrics["AveragePerformance"]
            row_data[f"{algo.upper()}_AveragePerformance_CI"] = metrics["AveragePerformance_CI"]
            row_data[f"{algo.upper()}_Forgetting"] = metrics["Forgetting"]
            row_data[f"{algo.upper()}_Forgetting_CI"] = metrics["Forgetting_CI"]

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
    p = argparse.ArgumentParser(description="Compare EWC_partial results between IPPO and MAPPO algorithms")
    p.add_argument("--data_root", default="results/data", help="Root directory containing the data")
    p.add_argument("--algorithms", nargs="+", default=["ippo", "mappo"], help="Algorithms to compare")
    p.add_argument("--method", default="EWC_partial", help="Continual learning method to compare")
    p.add_argument("--strategy", default="generate", help="Strategy name")
    p.add_argument("--seq_len", type=int, default=20, help="Sequence length")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Seeds to include")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3], help="Difficulty levels to compare")
    p.add_argument(
        "--end_window_evals",
        type=int,
        default=10,
        help="How many final eval points to average for F (Forgetting)",
    )
    args = p.parse_args()

    print(f"Comparing algorithms: {args.algorithms}")
    print(f"Method: {args.method}")
    print(f"Strategy: {args.strategy}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Seeds: {args.seeds}")
    print(f"Levels: {args.levels}")

    # Compute comparison metrics
    df = compare_algorithms(
        data_root=Path(args.data_root),
        algorithms=args.algorithms,
        method=args.method,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
        levels=args.levels,
        end_window_evals=args.end_window_evals,
    )

    # For each level, identify best performance and format the table
    df_out_rows = []

    for _, row in df.iterrows():
        level = row["Level"]

        # Extract values for each algorithm
        algo_values = {}
        for algo in args.algorithms:
            algo_upper = algo.upper()
            algo_values[algo] = {
                'ap': row[f"{algo_upper}_AveragePerformance"],
                'ap_ci': row[f"{algo_upper}_AveragePerformance_CI"],
                'f': row[f"{algo_upper}_Forgetting"],
                'f_ci': row[f"{algo_upper}_Forgetting_CI"],
            }

        # Find best values across algorithms for this level
        valid_a_values = [v['ap'] for v in algo_values.values() if not (np.isnan(v['ap']) or np.isinf(v['ap']))]
        valid_f_values = [v['f'] for v in algo_values.values() if not (np.isnan(v['f']) or np.isinf(v['f']))]

        best_a = max(valid_a_values) if valid_a_values else np.nan
        best_f = min(valid_f_values) if valid_f_values else np.nan

        # Create formatted row
        formatted_row = {"Level": f"Level {int(level)}"}

        # First add all average performance columns
        for algo in args.algorithms:
            algo_upper = algo.upper()
            values = algo_values[algo]

            formatted_row[f"{algo_upper}_AveragePerformance"] = _fmt(
                values['ap'], 
                values['ap_ci'], 
                values['ap'] == best_a, 
                "max"
            )

        # Then add all forgetting columns
        for algo in args.algorithms:
            algo_upper = algo.upper()
            values = algo_values[algo]

            formatted_row[f"{algo_upper}_Forgetting"] = _fmt(
                values['f'], 
                values['f_ci'], 
                values['f'] == best_f, 
                "min"
            )

        df_out_rows.append(formatted_row)

    df_out = pd.DataFrame(df_out_rows)

    # Create column headers - first all average performance, then all forgetting
    columns = ["Level"]
    # Add all average performance columns first
    for algo in args.algorithms:
        algo_upper = algo.upper()
        columns.append(rf"$\mathcal{{A}}\!\uparrow$ {algo_upper}")
    # Then add all forgetting columns
    for algo in args.algorithms:
        algo_upper = algo.upper()
        columns.append(rf"$\mathcal{{F}}\!\downarrow$ {algo_upper}")

    df_out.columns = columns

    # Generate LaTeX table
    column_format = "l" + "cc" * len(args.algorithms)
    latex_table = df_out.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        label="tab:algorithm_comparison",
        caption=f"Comparison of {args.method} between {' and '.join([algo.upper() for algo in args.algorithms])} algorithms. "
                f"Bold values indicate the best performance for each metric. "
                f"$\\mathcal{{A}}$ represents Average Performance (higher is better), "
                f"$\\mathcal{{F}}$ represents Forgetting (lower is better).",
    )

    print("\nComparison Results:")
    print("=" * 80)
    print(df_out.to_string(index=False))

    print(f"\nLATEX TABLE:")
    print("-" * 40)
    print(latex_table)
