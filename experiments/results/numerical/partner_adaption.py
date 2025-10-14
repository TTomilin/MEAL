#!/usr/bin/env python3
"""Build a LaTeX table with mean ±95% CI (smaller font) for partner adaptation CL metrics."""
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
        methods: List[str],
        num_partners: int,
        seeds: List[int],
        end_window_evals: int = 10,
) -> pd.DataFrame:
    """Compute partner adaptation metrics (Average Performance and Forgetting only)."""
    rows: list[dict[str, float]] = []

    for method in methods:
        AP_seeds, F_seeds = [], []

        # Partner adaptation uses "partners_8" directory structure
        base_folder = (
            data_root
            / algo
            / method
            / f"partners_{num_partners}"
        )

        for seed in seeds:
            sd = base_folder / f"seed_{seed}"
            if not sd.exists():
                print(f"[debug] seed directory does not exist: {sd}")
                continue

            # Load training curve (not used for metrics but kept for consistency)
            training_fp = sd / "training_soup.json"
            if not training_fp.exists():
                print(f"[warn] missing training_soup.json for {method} seed {seed}")
                continue
            print(f"[debug] found training file for {method} seed {seed}: {training_fp}")

            # Load per-partner evaluation curves
            env_series = []
            for i in range(num_partners):
                # Partner adaptation files are named eval_partner_{i}_soup.json
                expected_file = sd / f"eval_partner_{i}_soup.json"

                if expected_file.exists():
                    env_series.append(load_series(expected_file))
                else:
                    print(f"[warn] missing partner file {i} for seed {seed}, method {method}, using zeros")
                    # Create a default array of zeros with reasonable length
                    env_series.append(np.zeros(100))

            # Replace NaN and inf/-inf values with zeros in env_series
            processed_env_series = []
            for i, series in enumerate(env_series):
                # Check for NaN and inf/-inf values
                has_nan = np.any(np.isnan(series))
                has_inf = np.any(np.isinf(series))

                if np.all(np.isnan(series)) and not has_inf:
                    print(f"[warn] partner {i} series contains all NaN values for {method} seed {seed}, replacing with zeros")
                    processed_series = np.zeros_like(series)
                elif np.all(np.isinf(series)) and not has_nan:
                    print(f"[warn] partner {i} series contains all inf/-inf values for {method} seed {seed}, replacing with zeros")
                    processed_series = np.zeros_like(series)
                elif np.all(np.isnan(series) | np.isinf(series)):
                    print(f"[warn] partner {i} series contains all NaN/inf/-inf values for {method} seed {seed}, replacing with zeros")
                    processed_series = np.zeros_like(series)
                elif has_nan and has_inf:
                    print(f"[warn] partner {i} series contains some NaN and inf/-inf values for {method} seed {seed}, replacing with zeros")
                    processed_series = np.where(np.isnan(series) | np.isinf(series), 0.0, series)
                elif has_nan:
                    print(f"[warn] partner {i} series contains some NaN values for {method} seed {seed}, replacing NaN with zeros")
                    processed_series = np.where(np.isnan(series), 0.0, series)
                elif has_inf:
                    print(f"[warn] partner {i} series contains some inf/-inf values for {method} seed {seed}, replacing with zeros")
                    processed_series = np.where(np.isinf(series), 0.0, series)
                else:
                    processed_series = series
                processed_env_series.append(processed_series)

            # Pad all series to the same length
            L = max(len(s) for s in processed_env_series)
            env_mat = np.vstack([
                np.pad(s, (0, L - len(s)), constant_values=s[-1]) for s in processed_env_series
            ])

            # Average Performance (AP) – last eval of mean curve across all partners
            AP_seeds.append(np.nanmean(env_mat, axis=0)[-1])

            # Forgetting (F) – drop from best‑ever to final performance across all partners
            f_vals = []
            final_idx = env_mat.shape[1] - 1
            fw_start = max(0, final_idx - end_window_evals + 1)
            for i in range(num_partners):
                final_avg = np.nanmean(env_mat[i, fw_start : final_idx + 1])
                best_perf = np.nanmax(env_mat[i, : final_idx + 1])
                f_vals.append(max(best_perf - final_avg, 0.0))
            F_seeds.append(float(np.nanmean(f_vals)))

        # Aggregate across seeds
        A_mean, A_ci = _mean_ci(AP_seeds)
        F_mean, F_ci = _mean_ci(F_seeds)

        rows.append(
            {
                "Method": method,
                "AveragePerformance": A_mean,
                "AveragePerformance_CI": A_ci,
                "Forgetting": F_mean,
                "Forgetting_CI": F_ci,
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
    p = argparse.ArgumentParser(description="Generate LaTeX table for partner adaptation continual learning metrics")
    p.add_argument("--data_root", required=True, help="Root directory containing the data")
    p.add_argument("--algo", required=True, help="Algorithm name (e.g., ippo, ppo)")
    p.add_argument("--methods", nargs="+", required=True, help="CL methods to compare (e.g., ewc mas l2 ft)")
    p.add_argument("--num_partners", type=int, default=8, help="Number of partners (default: 8)")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Random seeds to aggregate over")
    p.add_argument(
        "--end_window_evals",
        type=int,
        default=10,
        help="How many final eval points to average for Forgetting calculation",
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

    # Compute partner adaptation metrics
    df = compute_metrics(
        data_root=Path(args.data_root),
        algo=args.algo,
        methods=args.methods,
        num_partners=args.num_partners,
        seeds=args.seeds,
        end_window_evals=args.end_window_evals,
    )

    # Pretty‑print method names
    df["Method"] = df["Method"].replace({"Online_EWC": "Online EWC"})

    # Identify best means (ignoring CI)
    best_A = df["AveragePerformance"].max()
    best_F = df["Forgetting"].min()

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

    # Rename columns to mathy headers (only Average Performance and Forgetting for partner adaptation)
    df_out.columns = [
        "Method",
        r"$\mathcal{A}\!\uparrow$",
        r"$\mathcal{F}\!\downarrow$",
    ]

    # Generate LaTeX table
    latex_table = df_out.to_latex(
        index=False,
        escape=False,
        column_format="lcc",
        label="tab:partner_adaptation_metrics",
        caption="Partner adaptation continual learning metrics with 95\\% confidence intervals.",
    )

    print(latex_table)
