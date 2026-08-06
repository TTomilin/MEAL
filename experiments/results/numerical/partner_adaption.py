#!/usr/bin/env python3
"""Build a LaTeX table with mean ±95% CI (smaller font) for partner adaptation CL metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiments.results.plotting.utils import (
    METHOD_DISPLAY_NAMES, load_series, mean_ci, build_task_matrix, add_forgetting_args,
)
from experiments.results.plotting.utils.metrics import (
    compute_forgetting, training_end_idx_for_task,
)

ConfInt = tuple[float, float]  # (mean, 95% CI)


def compute_metrics(
        data_root: Path,
        algo: str,
        methods: List[str],
        num_partners: int,
        seeds: List[int],
        end_window_evals: int = 10,
        forgetting_formula: str = "peak_final",
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

            # Load training curve. Gives the forgetting metric (below) the task/partner
            # boundary location within each partner's own eval curve.
            training_fp = sd / "training_soup.json"
            if not training_fp.exists():
                print(f"[warn] missing training_soup.json for {method} seed {seed}")
                continue
            print(f"[debug] found training file for {method} seed {seed}: {training_fp}")
            n_train = len(load_series(training_fp))

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

            env_mat = build_task_matrix(
                env_series,
                sanitize_label_fn=lambda i: f"partner {i} series for {method} seed {seed}",
            )

            # Average Performance (AP) – last eval of mean curve across all partners
            AP_seeds.append(np.nanmean(env_mat, axis=0)[-1])

            # Forgetting (F), across all partners.
            final_idx = env_mat.shape[1] - 1
            f_vals = []
            for i in range(num_partners):
                task_curve = env_mat[i, : final_idx + 1]
                training_end_idx = training_end_idx_for_task(i, len(task_curve), n_train, num_partners)
                f_vals.append(compute_forgetting(
                    task_curve, forgetting_formula, training_end_idx=training_end_idx,
                    end_window_evals=end_window_evals,
                ))
            F_seeds.append(float(np.nanmean(f_vals)))

        # Aggregate across seeds
        A_mean, A_ci = mean_ci(AP_seeds)
        F_mean, F_ci = mean_ci(F_seeds)

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
    add_forgetting_args(p, default="peak_final")
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
        forgetting_formula=args.forgetting_formula,
    )

    # Pretty‑print method names
    df["Method"] = df["Method"].replace(METHOD_DISPLAY_NAMES)

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
