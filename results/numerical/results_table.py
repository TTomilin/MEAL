#!/usr/bin/env python3
"""Build a LaTeX table with mean ±95% CI (smaller font) for CL metrics."""
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
        arch: str,
        methods: List[str],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        end_window_evals: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for method in methods:
        AP_seeds, F_seeds, PL_seeds = [], [], []

        base_folder = (
                Path(__file__).resolve().parent.parent
                / data_root
                / algo
                / method
                / "main"
                / f"{strategy}_{seq_len}"
        )

        for seed in seeds:
            sd = base_folder / f"seed_{seed}"
            if not sd.exists():
                continue

            # 1) Plasticity training curve
            training_fp = sd / "training_soup.json"
            if not training_fp.exists():
                print(f"[warn] missing training_soup.json for {method} seed {seed}")
                continue
            training = load_series(training_fp)
            n_train = len(training)
            chunk = n_train // seq_len

            # 2) Per‑environment evaluation curves
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

            # Plasticity (PL) – mean of last *end_window_evals* of each chunk
            pl_vals = []
            for i in range(seq_len):
                end_idx = (i + 1) * chunk - 1
                start_idx = max(0, end_idx - end_window_evals + 1)
                pl_vals.append(np.nanmean(training[start_idx : end_idx + 1]))
            PL_seeds.append(float(np.nanmean(pl_vals)))

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
        P_mean, P_ci = _mean_ci(PL_seeds)

        rows.append(
            {
                "Method": method,
                "AveragePerformance": A_mean,
                "AveragePerformance_CI": A_ci,
                "Forgetting": F_mean,
                "Forgetting_CI": F_ci,
                "Plasticity": P_mean,
                "Plasticity_CI": P_ci,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# LaTeX formatting helpers
# -----------------------------------------------------------------------------

def _fmt(mean: float, ci: float, best: bool, better: str = "max") -> str:
    """Return *mean ±CI* formatted for LaTeX, with CI in \scriptsize."""
    if np.isnan(mean):
        return "--"
    main = f"{mean:.3f}"
    if best:
        main = rf"\textbf{{{main}}}"
    ci_part = rf"{{\scriptsize$\pm{ci:.2f}$}}" if not np.isnan(ci) and ci > 0 else ""
    return main + ci_part


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--algo", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument(
        "--end_window_evals",
        type=int,
        default=10,
        help="How many final eval points to average for PL and F",
    )
    args = p.parse_args()

    df = compute_metrics(
        data_root=Path(args.data_root),
        algo=args.algo,
        arch=args.arch,
        methods=args.methods,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
        end_window_evals=args.end_window_evals,
    )

    # Pretty‑print method names
    df["Method"] = df["Method"].replace({"Online_EWC": "Online EWC"})

    # Identify best means (ignoring CI)
    best_A = df["AveragePerformance"].max()
    best_F = df["Forgetting"].min()
    best_P = df["Plasticity"].max()

    # Build human‑readable strings with CI
    df_out = pd.DataFrame()
    df_out["Method"] = df["Method"]
    df_out["AveragePerformance"] = df.apply(
        lambda r: _fmt(r.AveragePerformance, r.AveragePerformance_CI, r.AveragePerformance == best_A, "max"),
        axis=1,
    )
    df_out["Forgetting"] = df.apply(
        lambda r: _fmt(r.Forgetting, r.Forgetting_CI, r.Forgetting == best_F, "min"),
        axis=1,
    )
    df_out["Plasticity"] = df.apply(
        lambda r: _fmt(r.Plasticity, r.Plasticity_CI, r.Plasticity == best_P, "max"),
        axis=1,
    )

    # Rename columns to mathy headers
    df_out.columns = [
        "Method",
        r"$\mathcal{A}\!\uparrow$",
        r"$\mathcal{F}\!\downarrow$",
        r"$\mathcal{P}\!\uparrow$",
    ]

    latex_table = df_out.to_latex(
        index=False,
        escape=False,
        column_format="lccc",
        label="tab:cmarl_metrics",
    )

    print(latex_table)
