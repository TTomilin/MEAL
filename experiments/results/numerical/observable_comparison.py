#!/usr/bin/env python3
"""Compare results between partially observable and fully observable settings."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiments.results.plotting.utils import (
    load_series, mean_ci, build_task_matrix, add_numerical_data_args, add_forgetting_args,
)
from experiments.results.plotting.utils.metrics import (
    compute_forgetting, training_end_idx_for_task,
)

ConfInt = tuple[float, float]  # (mean, 95% CI)


def compute_metrics(
        data_root: Path,
        algo: str,
        method: str,
        strategy: str,
        seq_len: int,
        seeds: List[int],
        end_window_evals: int = 10,
        level: int = 1,
        forgetting_formula: str = "peak_final",
) -> dict:
    """Compute metrics for a single method/setting combination."""
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
                f"for {method} seed {seed}"
            )
            continue

        env_series = [load_series(f) for f in env_files]
        env_mat = build_task_matrix(env_series, sanitize=False)

        # Average Performance (AP) – last eval of mean curve
        AP_seeds.append(env_mat.mean(axis=0)[-1])

        # Forgetting
        training_fp = sd / "training_soup.json"
        n_train = len(load_series(training_fp)) if training_fp.exists() else 0

        final_idx = env_mat.shape[1] - 1
        f_vals = []
        for i in range(seq_len):
            task_curve = env_mat[i, : final_idx + 1]
            n_train_ref = n_train or len(task_curve)
            training_end_idx = training_end_idx_for_task(i, len(task_curve), n_train_ref, seq_len)
            f_vals.append(compute_forgetting(
                task_curve, forgetting_formula, training_end_idx=training_end_idx,
                end_window_evals=end_window_evals,
            ))
        F_seeds.append(float(np.nanmean(f_vals)))

    # Aggregate across seeds
    A_mean, A_ci = mean_ci(AP_seeds)
    F_mean, F_ci = mean_ci(F_seeds)

    return {
        "AveragePerformance": A_mean,
        "AveragePerformance_CI": A_ci,
        "Forgetting": F_mean,
        "Forgetting_CI": F_ci,
    }


def compare_observability_settings(
        data_root: Path,
        algo: str,
        strategy: str,
        seq_len: int,
        seeds: List[int],
        levels: List[int],
        end_window_evals: int = 10,
        forgetting_formula: str = "peak_final",
) -> pd.DataFrame:
    """Compare EWC results between fully and partially observable settings."""
    rows = []

    for level in levels:
        # Compute metrics for fully observable (EWC)
        full_metrics = compute_metrics(
            data_root=data_root,
            algo=algo,
            method="EWC",
            strategy=strategy,
            seq_len=seq_len,
            seeds=seeds,
            end_window_evals=end_window_evals,
            level=level,
            forgetting_formula=forgetting_formula,
        )

        # Compute metrics for partially observable (EWC_partial)
        partial_metrics = compute_metrics(
            data_root=data_root,
            algo=algo,
            method="EWC_partial",
            strategy=strategy,
            seq_len=seq_len,
            seeds=seeds,
            end_window_evals=end_window_evals,
            level=level,
            forgetting_formula=forgetting_formula,
        )

        # Create one row per level with FO and PO as columns
        rows.append({
            "Level": level,
            "FO_AveragePerformance": full_metrics["AveragePerformance"],
            "FO_AveragePerformance_CI": full_metrics["AveragePerformance_CI"],
            "FO_Forgetting": full_metrics["Forgetting"],
            "FO_Forgetting_CI": full_metrics["Forgetting_CI"],
            "PO_AveragePerformance": partial_metrics["AveragePerformance"],
            "PO_AveragePerformance_CI": partial_metrics["AveragePerformance_CI"],
            "PO_Forgetting": partial_metrics["Forgetting"],
            "PO_Forgetting_CI": partial_metrics["Forgetting_CI"],
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# LaTeX formatting helpers
# -----------------------------------------------------------------------------

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
    p = argparse.ArgumentParser(description="Compare EWC results between fully and partially observable settings")
    add_numerical_data_args(p, required=False)
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3], help="Difficulty levels to compare")
    add_forgetting_args(p, default="peak_final")
    args = p.parse_args()

    # Compute comparison metrics
    df = compare_observability_settings(
        data_root=Path(args.data_root),
        algo=args.algo,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
        levels=args.levels,
        end_window_evals=args.end_window_evals,
        forgetting_formula=args.forgetting_formula,
    )

    # For each level, identify best performance and format the table
    df_out_rows = []

    for _, row in df.iterrows():
        level = row["Level"]

        # Find best values across FO and PO for this level
        fo_a = row["FO_AveragePerformance"]
        po_a = row["PO_AveragePerformance"]
        fo_f = row["FO_Forgetting"]
        po_f = row["PO_Forgetting"]

        # Determine best values (excluding NaN and inf)
        valid_a_values = [v for v in [fo_a, po_a] if not (np.isnan(v) or np.isinf(v))]
        valid_f_values = [v for v in [fo_f, po_f] if not (np.isnan(v) or np.isinf(v))]

        best_a = max(valid_a_values) if valid_a_values else np.nan
        best_f = min(valid_f_values) if valid_f_values else np.nan

        df_out_rows.append({
            "Level": f"Level {int(level)}",
            "FO_AveragePerformance": _fmt(
                fo_a, 
                row["FO_AveragePerformance_CI"], 
                fo_a == best_a, 
                "max"
            ),
            "PO_AveragePerformance": _fmt(
                po_a, 
                row["PO_AveragePerformance_CI"], 
                po_a == best_a, 
                "max"
            ),
            "FO_Forgetting": _fmt(
                fo_f, 
                row["FO_Forgetting_CI"], 
                fo_f == best_f, 
                "min"
            ),
            "PO_Forgetting": _fmt(
                po_f, 
                row["PO_Forgetting_CI"], 
                po_f == best_f, 
                "min"
            ),
        })

    df_out = pd.DataFrame(df_out_rows)

    # Rename columns to mathy headers
    df_out.columns = [
        "Level",
        r"$\mathcal{A}\!\uparrow$ FO",
        r"$\mathcal{A}\!\uparrow$ PO",
        r"$\mathcal{F}\!\downarrow$ FO",
        r"$\mathcal{F}\!\downarrow$ PO",
    ]

    latex_table = df_out.to_latex(
        index=False,
        escape=False,
        column_format="lcccc",
        label="tab:observability_comparison",
    )

    print("EWC Observability Comparison Results")
    print("=" * 50)
    print(latex_table)
