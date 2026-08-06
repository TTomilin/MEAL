from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiments.results.plotting.utils import (
    METHOD_DISPLAY_NAMES, load_series, mean_ci, build_task_matrix, load_baseline_task_curves,
    average_forward_transfer, add_numerical_data_args, add_forgetting_args,
)
from experiments.results.plotting.utils.metrics import compute_forgetting

ConfInt = tuple[float, float]  # (mean, 95% CI)


def compute_metrics(
        data_root: Path,
        algo: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        end_window_evals: int = 10,
        level: int = 1,
        agents: int = 2,
        lambda_decay: float = 2.0,
        truncate_tasks: int = None,
        forgetting_formula: str = "weighted",
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    effective_tasks = truncate_tasks if truncate_tasks is not None else seq_len

    # Load baseline data once for forward transfer calculation
    baseline_folder = (
        data_root
        / algo
        / "single"
        / f"level_{level}"
        / f"agents_{agents}"
        / f"{strategy}_{seq_len}"
    )
    baseline_data = load_baseline_task_curves(baseline_folder, seeds, effective_tasks)

    for method in methods:
        AP_seeds, F_seeds, FT_seeds = [], [], []

        base_folder = (
                data_root
                / algo
                / method
                / f"level_{level}"
                / f"agents_{agents}"
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
            # Truncate training curve to only cover effective_tasks
            training = training[: effective_tasks * chunk]
            n_train = len(training)

            # 2) Per‑environment evaluation curves
            # Handle missing files by creating expected file paths and loading them
            # This ensures we always have effective_tasks series, even if some files are missing
            env_series = []
            for i in range(effective_tasks):
                fp = sd / f"{i}_soup.json"
                if fp.exists():
                    env_series.append(load_series(fp))
                else:
                    print(f"[warn] missing env file for task {i}, seed {seed}, method {method}, using zeros")
                    env_series.append(np.zeros(100))

            env_mat = build_task_matrix(
                env_series,
                sanitize_label_fn=lambda i: f"env series {i} for {method} seed {seed}",
            )

            # Average Performance (AP) – last eval of mean curve
            AP_seeds.append(np.nanmean(env_mat, axis=0)[-1])

            # Forgetting (F) – curve-based forgetting that considers when forgetting occurs
            f_vals = []
            final_idx = env_mat.shape[1] - 1

            # Formula: sum over i=1..N-1 (exclude last task — no post-training window)
            for i in range(effective_tasks - 1):
                task_curve = env_mat[i, : final_idx + 1]

                training_end_step = (i + 1) * chunk
                if n_train > 0:
                    training_end_idx = int((training_end_step / n_train) * len(task_curve))
                    training_end_idx = min(training_end_idx, len(task_curve) - 1)
                else:
                    training_end_idx = len(task_curve) - 1

                if any(task_curve > 0.0):
                    curve_forgetting = compute_forgetting(
                        task_curve, forgetting_formula, training_end_idx=training_end_idx,
                        end_window_evals=end_window_evals, lambda_decay=lambda_decay,
                    )
                    f_vals.append(curve_forgetting)

            F_seeds.append(float(np.nanmean(f_vals)))

            # Forward Transfer (FT) – normalized area between CL and baseline curves
            if seed not in baseline_data:
                print(f"[warn] missing baseline data for seed {seed}")
                FT_seeds.append(np.nan)
                continue

            FT_seeds.append(average_forward_transfer(training, chunk, effective_tasks, baseline_data[seed]))

        # Aggregate across seeds
        A_mean, A_ci = mean_ci(AP_seeds)
        F_mean, F_ci = mean_ci(F_seeds)
        FT_mean, FT_ci = mean_ci(FT_seeds)

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
    main = f"{mean:.2f}"
    if best:
        main = rf"\textbf{{{main}}}"
    ci_part = rf"{{\scriptsize$\pm{ci:.2f}$}}" if show_confidence_intervals and not np.isnan(ci) and ci > 0 else ""
    return main + ci_part


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_numerical_data_args(p, seq_len_default=10, seeds_default=[1, 2, 3, 4, 5], required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--level", type=int, default=None, help="Difficulty level of the environment (if not provided, generates table for all levels 1, 2, 3)")
    p.add_argument("--agents", type=int, default=2, help="Number of agents in the environment")
    add_forgetting_args(p, default="weighted")
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
    p.add_argument(
        "--truncate_tasks",
        type=int,
        default=None,
        help="Only use the first N tasks from the sequence (e.g. use 10 from a 20-task run).",
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
            agents=args.agents,
            truncate_tasks=args.truncate_tasks,
            forgetting_formula=args.forgetting_formula,
        )
        # Pretty‑print method names
        df["Method"] = df["Method"].replace(METHOD_DISPLAY_NAMES)
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
                truncate_tasks=args.truncate_tasks,
                forgetting_formula=args.forgetting_formula,
            )
            # Pretty‑print method names
            level_df["Method"] = level_df["Method"].replace(METHOD_DISPLAY_NAMES)
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
