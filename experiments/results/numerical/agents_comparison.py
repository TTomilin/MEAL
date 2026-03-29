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
        env: str,
        strategy: str,
        seq_len: int,
        seeds: List[int],
        num_agents: int,
        end_window_evals: int = 10,
        level: int = 1,
) -> dict:
    """Compute metrics for a single algorithm/method/num_agents combination."""
    AP_seeds, F_seeds, FT_seeds = [], [], []

    _eval_suffixes = {"jaxnav": "success", "mpe": "coverage_fraction", "smax": "return"}
    eval_suffix = _eval_suffixes.get(env, "soup")
    _training_metrics = {"jaxnav": "return"}
    training_metric = _training_metrics.get(env, None if env in ("mpe", "smax") else "soup")
    _level_strings = {"jaxnav": "jaxnav", "mpe": "mpe", "smax": "smax"}
    level_string = _level_strings.get(env, f"level_{level}")

    # Load baseline data once for forward transfer calculation (only when training metric exists)
    repo_root = Path(__file__).resolve().parent.parent
    baseline_data = {}
    if training_metric is not None:
        baseline_folder = (
            repo_root
            / data_root
            / algo
            / "single"
            / level_string
            / f"{strategy}_{seq_len}"
        )

        for seed in seeds:
            baseline_seed_dir = baseline_folder / f"seed_{seed}"
            if baseline_seed_dir.exists():
                # Load baseline training data for each task
                baseline_training_files = []
                for i in range(seq_len):
                    baseline_file = baseline_seed_dir / f"{i}_training_{training_metric}.json"
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
            / level_string
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
            direct = sd / f"{i}_{eval_suffix}.json"
            if direct.exists():
                cand = [direct]
            else:
                # fallback: glob for older naming conventions
                cand = sorted([p for p in sd.glob(f"{i}_*_{eval_suffix}.*") if "training" not in p.name])
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

        # Forward Transfer (FT) – skipped for envs without a training metric
        if training_metric is None:
            FT_seeds.append(np.nan)
        else:
            training_fp = sd / f"training_{training_metric}.json"
            if not training_fp.exists():
                print(f"[warn] missing training_{training_metric}.json for {method} {num_agents}agents seed {seed}")
                FT_seeds.append(np.nan)
            else:
                training = load_series(training_fp)
                n_train = len(training)
                chunk = max(1, n_train // seq_len)

                if seed not in baseline_data:
                    print(f"[warn] missing baseline data for seed {seed}")
                    FT_seeds.append(np.nan)
                else:
                    ft_vals = []
                    for i in present_task_ids:
                        start_idx = i * chunk
                        end_idx = min((i + 1) * chunk, n_train)
                        cl_task_curve = training[start_idx:end_idx]

                        if len(cl_task_curve) > 1:
                            auc_cl = np.trapz(cl_task_curve) / len(cl_task_curve)
                        else:
                            auc_cl = cl_task_curve[0] if len(cl_task_curve) == 1 else 0.0

                        if np.isnan(auc_cl) or np.isinf(auc_cl):
                            print(f"[warn] CL AUC is NaN/inf/-inf for task {i}, seed {seed}, method {method}")
                            continue

                        if i >= len(baseline_data[seed]):
                            print(f"[warn] missing baseline index for task {i}, seed {seed}")
                            continue
                        baseline_task_curve = baseline_data[seed][i]
                        if baseline_task_curve is None:
                            print(f"[warn] missing baseline data for task {i}, seed {seed}")
                            continue

                        if np.all(np.isnan(baseline_task_curve) | np.isinf(baseline_task_curve)):
                            print(f"[warn] baseline data contains all NaN/inf for task {i}, seed {seed}")
                            continue

                        if len(baseline_task_curve) > 1:
                            auc_baseline = np.trapz(baseline_task_curve) / len(baseline_task_curve)
                        else:
                            auc_baseline = baseline_task_curve[0] if len(baseline_task_curve) == 1 else 0.0

                        if np.isnan(auc_baseline) or np.isinf(auc_baseline):
                            print(f"[warn] baseline AUC is NaN/inf for task {i}, seed {seed}")
                            continue

                        if abs(auc_baseline) < 1e-8:
                            print(f"[info] baseline AUC ~0 for task {i}, seed {seed}, method {method} – skipping FT")
                            continue

                        epsilon = 1e-8
                        ft_i = (auc_cl - auc_baseline) / max(abs(auc_baseline), epsilon)
                        if np.isnan(ft_i) or np.isinf(ft_i):
                            print(f"[warn] FT result is NaN/inf for task {i}, seed {seed}, method {method}")
                        else:
                            ft_vals.append(ft_i)

                    FT_seeds.append(float(np.nanmean(ft_vals)) if ft_vals else np.nan)

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
        env: str,
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
                env=env,
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
    p.add_argument("--env", default="overcooked", help="Environment name")
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

    # MPE and SMAX have no difficulty levels and no forward transfer metric
    NO_LEVELS_ENVS = {"mpe", "smax"}
    has_levels = args.env not in NO_LEVELS_ENVS
    has_ft = args.env not in NO_LEVELS_ENVS
    levels = args.levels if has_levels else [1]

    print(f"Comparing num_agents: {args.num_agents}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Method: {args.method}")
    print(f"Strategy: {args.strategy}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Seeds: {args.seeds}")
    if has_levels:
        print(f"Levels: {levels}")

    # Compute comparison metrics
    df = compare_agents(
        data_root=Path(args.data_root),
        algorithm=args.algorithm,
        method=args.method,
        env=args.env,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
        num_agents_list=args.num_agents,
        levels=levels,
        end_window_evals=args.end_window_evals,
    )

    # Find best values per column (across all agent configurations)
    best_values = {}
    for level in levels:
        ap_values = [row[f"LEVEL{level}_AveragePerformance"] for _, row in df.iterrows()
                     if not (np.isnan(row[f"LEVEL{level}_AveragePerformance"]) or np.isinf(row[f"LEVEL{level}_AveragePerformance"]))]
        f_values = [row[f"LEVEL{level}_Forgetting"] for _, row in df.iterrows()
                    if not (np.isnan(row[f"LEVEL{level}_Forgetting"]) or np.isinf(row[f"LEVEL{level}_Forgetting"]))]
        best_values[f"A_L{level}"] = max(ap_values) if ap_values else np.nan
        best_values[f"F_L{level}"] = min(f_values) if f_values else np.nan
        if has_ft:
            ft_values = [row[f"LEVEL{level}_ForwardTransfer"] for _, row in df.iterrows()
                         if not (np.isnan(row[f"LEVEL{level}_ForwardTransfer"]) or np.isinf(row[f"LEVEL{level}_ForwardTransfer"]))]
            best_values[f"FT_L{level}"] = max(ft_values) if ft_values else np.nan

    # Build formatted rows
    df_out_rows = []
    for _, row in df.iterrows():
        num_agents = row["NumAgents"]
        agent_text = f"{int(num_agents)} Agent" + ("s" if num_agents > 1 else "")
        formatted_row = {"Agents": agent_text}

        for level in levels:
            ap  = row[f"LEVEL{level}_AveragePerformance"]
            ap_ci = row[f"LEVEL{level}_AveragePerformance_CI"]
            f   = row[f"LEVEL{level}_Forgetting"]
            f_ci  = row[f"LEVEL{level}_Forgetting_CI"]

            formatted_row[f"AveragePerformance_L{level}"] = _fmt(
                ap, ap_ci, ap == best_values[f"A_L{level}"], "max", args.confidence_intervals)
            formatted_row[f"Forgetting_L{level}"] = _fmt(
                f, f_ci, f == best_values[f"F_L{level}"], "min", args.confidence_intervals)

            if has_ft:
                ft    = row[f"LEVEL{level}_ForwardTransfer"]
                ft_ci = row[f"LEVEL{level}_ForwardTransfer_CI"]
                formatted_row[f"ForwardTransfer_L{level}"] = _fmt(
                    ft, ft_ci, ft == best_values[f"FT_L{level}"], "max", args.confidence_intervals)

        df_out_rows.append(formatted_row)

    df_out = pd.DataFrame(df_out_rows)

    # Rename columns
    metrics_per_level = [rf"$\mathcal{{A}}\!\uparrow$", rf"$\mathcal{{F}}\!\downarrow$"]
    if has_ft:
        metrics_per_level.append(rf"$\mathcal{{FT}}\!\uparrow$")
    new_columns = ["Agents"]
    for _ in levels:
        new_columns.extend(metrics_per_level)
    df_out.columns = new_columns

    n_metrics = len(metrics_per_level)
    column_format = "l" + "c" * (len(levels) * n_metrics)

    # Caption
    if has_levels:
        level_str = ' and '.join([f'Level {l}' for l in levels])
        caption_text = (f"Comparison of {args.method} with {args.algorithm.upper()} across "
                        f"{level_str} for {args.env}. ")
    else:
        caption_text = (f"Comparison of {args.method} with {args.algorithm.upper()} on {args.env.upper()}. ")
    caption_text += (f"Bold values indicate the best performance for each metric. "
                     f"$\\mathcal{{A}}$ represents Average Performance (higher is better), "
                     f"$\\mathcal{{F}}$ represents Forgetting (lower is better)")
    if has_ft:
        caption_text += f", $\\mathcal{{FT}}$ represents Forward Transfer (higher is better)"
    caption_text += "."

    # Build LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{caption_text}}}")
    latex_lines.append("\\label{tab:agents_comparison}")
    latex_lines.append(f"\\begin{{tabular}}{{{column_format}}}")
    latex_lines.append("\\toprule")

    if has_levels:
        # Two-row header: level multicolumns + metric symbols
        multicolumn_header = "\\multirow{2}{*}[-0.7ex]{Agents}"
        for i, level in enumerate(levels):
            sep = " &" if i < len(levels) - 1 else " \\\\"
            multicolumn_header += f" & \\multicolumn{{{n_metrics}}}{{c}}{{Level {level}}}{sep}"
        latex_lines.append(multicolumn_header)

        cmidrule_parts = []
        for i in range(len(levels)):
            start_col = 2 + i * n_metrics
            end_col = start_col + n_metrics - 1
            cmidrule_parts.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        latex_lines.append(" ".join(cmidrule_parts))

        metric_sym = " & $\\mathcal{A}\\!\\uparrow$ & $\\mathcal{F}\\!\\downarrow$"
        if has_ft:
            metric_sym += " & $\\mathcal{FT}\\!\\uparrow$"
        latex_lines.append((metric_sym * len(levels)).lstrip(" & ").rstrip() + " \\\\")
    else:
        # Single-row header: Agents + metric symbols
        header = "Agents & " + " & ".join(metrics_per_level * len(levels)) + " \\\\"
        latex_lines.append(header)

    latex_lines.append("\\midrule")

    for _, row in df_out.iterrows():
        latex_lines.append(" & ".join(str(row.iloc[i]) for i in range(len(df_out.columns))) + " \\\\")

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
