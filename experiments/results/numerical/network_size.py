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
        method: str,
        strategy: str,
        seq_len: int,
        seeds: List[int],
        level: int = 1,
        agents: int = 2,
        variant: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    AP_seeds = []
    base_folder = data_root / algo / method / f"level_{level}" / f"agents_{agents}" / f"{strategy}_{seq_len}" / variant

    for seed in seeds:
        sd = base_folder / f"seed_{seed}"
        if not sd.exists():
            print(f"[debug] seed directory does not exist: {sd}")
            continue

        # 1) Per‑environment evaluation curves
        # Handle missing files by creating expected file paths and loading them
        # This ensures we always have seq_len series, even if some files are missing
        env_series = []
        missing_files = []
        for i in range(seq_len):
            expected_file = sd / f"{i}_gen_soup.json"
            if expected_file.exists():
                env_series.append(load_series(expected_file))
            else:
                # Try an alternative naming pattern
                alt_file = sd / f"{i}_soup.json"
                if alt_file.exists():
                    env_series.append(load_series(alt_file))
                else:
                    # Try yet another alternative naming pattern
                    difficulty = "easy" if level == 1 else "medium" if level == 2 else "hard"
                    alt_file = sd / f"{i}_{difficulty}_gen_soup.json"
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

    # Aggregate across seeds
    A_mean, A_ci = _mean_ci(AP_seeds)

    rows.append(
        {
            "Method": method,
            "AveragePerformance": A_mean,
            "AveragePerformance_CI": A_ci,
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
    p.add_argument("--method", type=str, default='EWC')
    p.add_argument("--strategy", type=str, default='generate')
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--level", type=int, default=None, help="Difficulty level of the environment (if not provided, generates table for all levels 1, 2, 3)")
    p.add_argument("--agents", type=int, default=2, help="Number of agents in the environment")
    p.add_argument(
        "--variants",
        nargs="+",
        default=["orig_network", "big_network"],
        help="Network variants (subdirectories) to compare, e.g. orig_network big_network",
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

    root_dir = Path(__file__).resolve().parent.parent
    rows = []
    for variant in args.variants:
        # collect metrics for this variant across all levels
        level_means = {}
        level_cis = {}
        for level in [1, 2, 3]:
            level_df = compute_metrics(
                data_root=root_dir / Path(args.data_root),
                algo=args.algo,
                method=args.method,
                strategy=args.strategy,
                seq_len=args.seq_len,
                seeds=args.seeds,
                level=level,
                variant=variant,
            )
            # Pretty‑print method names
            level_df["Method"] = level_df["Method"].replace({"Online_EWC": "Online EWC"})
            if level_df.empty:
                print(f"[warn] no data for {args.method}, variant {variant}, level {level}")
                level_means[level] = np.nan
                level_cis[level] = np.nan
                continue

            method_row = level_df.iloc[0]
            level_means[level] = method_row["AveragePerformance"]
            level_cis[level] = method_row["AveragePerformance_CI"]

        # nicer label for row name
        pretty_variant = {
            "orig_network": "Original network",
            "big_network": "Big network",
        }.get(variant, variant)

        rows.append(
            {
                "Variant": pretty_variant,
                "A_L1": level_means[1],
                "A_CI_L1": level_cis[1],
                "A_L2": level_means[2],
                "A_CI_L2": level_cis[2],
                "A_L3": level_means[3],
                "A_CI_L3": level_cis[3],
            }
        )

    df = pd.DataFrame(rows)

    # All levels case - format table: rows = network variants, columns = AveragePerformance per level.
    # Identify best AveragePerformance for each level (ignoring CI)
    best_L1 = df["A_L1"].max()
    best_L2 = df["A_L2"].max()
    best_L3 = df["A_L3"].max()

    df_out = pd.DataFrame()
    df_out["Network"] = df["Variant"]

    df_out["Level 1"] = df.apply(
        lambda r: _fmt(
            r.A_L1,
            r.A_CI_L1,
            r.A_L1 == best_L1,
            "max",
            args.confidence_intervals,
            ),
        axis=1,
    )
    df_out["Level 2"] = df.apply(
        lambda r: _fmt(
            r.A_L2,
            r.A_CI_L2,
            r.A_L2 == best_L2,
            "max",
            args.confidence_intervals,
            ),
        axis=1,
    )
    df_out["Level 3"] = df.apply(
        lambda r: _fmt(
            r.A_L3,
            r.A_CI_L3,
            r.A_L3 == best_L3,
            "max",
            args.confidence_intervals,
            ),
        axis=1,
    )

    # column format: Network + 3 levels
    column_format = "lccc"

    latex_table = df_out.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        label="tab:network_sizes_levels",
        caption="Average normalized performance across difficulty levels for different network sizes.",
    )

    print(latex_table)
