#!/usr/bin/env python3
"""Compare results between partially observable and fully observable settings."""
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


def compare_observability_settings(
        data_root: Path,
        algo: str,
        strategy: str,
        seq_len: int,
        seeds: List[int],
        levels: List[int],
        end_window_evals: int = 10,
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
    p.add_argument("--data_root", required=True, help="Root directory containing the data")
    p.add_argument("--algo", default="ippo", help="Algorithm name")
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

    # Compute comparison metrics
    df = compare_observability_settings(
        data_root=Path(args.data_root),
        algo=args.algo,
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
