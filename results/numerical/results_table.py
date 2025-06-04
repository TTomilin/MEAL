#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_series(fp: Path) -> np.ndarray:
    if fp.suffix == ".json":
        return np.array(json.loads(fp.read_text()), dtype=float)
    elif fp.suffix == ".npz":
        return np.load(fp)["data"].astype(float)
    else:
        raise ValueError(f"Unsupported file suffix: {fp.suffix}")


def compute_metrics(
        data_root: Path,
        algo: str,
        arch: str,
        methods: list[str],
        strategy: str,
        seq_len: int,
        seeds: list[int],
        end_window_evals: int = 10,
) -> pd.DataFrame:
    rows = []

    for method in methods:
        AP_seeds, F_seeds, PL_seeds = [], [], []

        base_folder = Path(
            __file__).resolve().parent.parent / data_root / algo / method / arch / f"{strategy}_{seq_len}"
        for seed in seeds:
            sd = base_folder / f"seed_{seed}"
            if not sd.exists():
                continue

            # 1) Load training curve for Plasticity
            training_fp = sd / "training_reward.json"
            if not training_fp.exists():
                print(f"[warn] missing training_reward.json for {method} seed {seed}")
                continue
            training = load_series(training_fp)
            n_train = len(training)
            chunk = n_train // seq_len

            # 2) Load per‐env evaluation series
            env_files = sorted([f for f in sd.glob("*_reward.*") if "training" not in f.name])
            if len(env_files) != seq_len:
                print(f"[warn] expected {seq_len} env files, found {len(env_files)} for {method} seed {seed}")
                continue
            env_series = [load_series(f) for f in env_files]
            # ensure equal length
            L = max(len(s) for s in env_series)
            env_mat = np.vstack([np.pad(s, (0, L - len(s)), constant_values=np.nan) for s in env_series])

            # --- Average Performance (AP) ---
            AP_seeds.append(np.nanmean(env_mat))

            # --- Plasticity (PL): mean training return over last end_window_evals evals of each task chunk ---
            pl_vals = []
            for i in range(seq_len):
                end_idx = (i + 1) * chunk - 1
                start_idx = max(0, end_idx - end_window_evals + 1)
                pl_vals.append(np.nanmean(training[start_idx: end_idx + 1]))
            PL_seeds.append(np.nanmean(pl_vals))

            # --- Forgetting (F): drop from end-of-task to final eval ---
            f_vals = []
            final_idx = env_mat.shape[1] - 1
            for i in range(seq_len):
                end_idx = (i + 1) * chunk - 1
                # average over last window of final performance for that env
                fw_start = max(0, final_idx - end_window_evals + 1)
                final_avg = np.nanmean(env_mat[i, fw_start: final_idx + 1])
                f_vals.append(env_mat[i, end_idx] - final_avg)
            F_seeds.append(np.nanmean(f_vals))

        # aggregate across seeds
        rows.append({
            "Method": method,
            "AveragePerformance": np.mean(AP_seeds) if AP_seeds else np.nan,
            "Forgetting": np.mean(F_seeds) if F_seeds else np.nan,
            "Plasticity": np.mean(PL_seeds) if PL_seeds else np.nan,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="e.g. results")
    p.add_argument("--algo", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--seq_len", type=int, nargs="?", default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--end_window_evals",
                   type=int, default=10,
                   help="How many final evaluation points to average for PL and F")
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

    # ------------------------------------------------------------------
    #  Post-processing for LaTeX output
    # ------------------------------------------------------------------

    # 1) nicer method label
    df["Method"] = df["Method"].replace({"Online_EWC": "Online EWC"})

    # 2) round for compactness (optional)
    df = df.round(3)

    # 3) identify best values (max ↑, min ↓)
    best_A = df["AveragePerformance"].max()
    best_F = df["Forgetting"].min()
    best_P = df["Plasticity"].max()

    def bold(v, best, better="max"):
        if (better == "max" and v == best) or (better == "min" and v == best):
            return rf"\textbf{{{v}}}"
        return f"{v}"

    df["AveragePerformance"] = df["AveragePerformance"].apply(
        lambda v: bold(v, best_A, "max"))
    df["Forgetting"] = df["Forgetting"].apply(
        lambda v: bold(v, best_F, "min"))
    df["Plasticity"] = df["Plasticity"].apply(
        lambda v: bold(v, best_P, "max"))

    # 4) rename columns to mathy headers with arrows
    df.columns = [
        "Method",
        r"$\mathcal{A}\!\uparrow$",
        r"$\mathcal{F}\!\downarrow$",
        r"$\mathcal{P}\!\uparrow$",
    ]

    latex_table = df.to_latex(
        index=False,
        escape=False,               # keep math + \textbf
        column_format="lccc",
        caption=r"CMARL metrics: "
                r"$\mathcal{A}$ (avg.\ performance), "
                r"$\mathcal{F}$ (forgetting), "
                r"$\mathcal{P}$ (plasticity).",
        label="tab:cmarl_metrics",
    )

    print(latex_table)

