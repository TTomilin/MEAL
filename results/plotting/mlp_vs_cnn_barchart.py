#!/usr/bin/env python3
"""
Compare MLP vs. CNN for several CL methods with a bar-chart (mean + 95 % CI).

Example
-------
python plot_bar.py --data_root results --algo ippo \
                   --methods EWC MAS L2 --strategy ordered \
                   --seq_len 10 --metric reward
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t

sns.set_theme(style="whitegrid", context="notebook")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="root folder: results/<algo>/<method>/<arch>/strategy_len/seed_*")
    p.add_argument("--algo", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--metric", choices=["reward", "success"], default="reward")
    p.add_argument("--seeds", nargs="+", type=int, default=[3])
    return p.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_series(fp: Path) -> np.ndarray:
    if fp.suffix == ".json":
        return np.asarray(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == ".npz":
        return np.load(fp)["data"].astype(float)
    raise ValueError(fp)


def final_scores(folder: Path, metric: str, seeds: list[int]) -> list[float]:
    scores = []
    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists(): continue
        files = sorted(sd.glob(f"*_{metric}.*"))
        if not files: continue

        env_vals = [load_series(f)[-1] for f in files]
        if env_vals:
            scores.append(np.nanmean(env_vals))
    return scores


def ci95(vals: np.ndarray) -> float:
    if len(vals) < 2: return np.nan
    return vals.std(ddof=1) / np.sqrt(len(vals)) * t.ppf(0.975, len(vals) - 1)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    base = root / args.data_root / args.algo
    rows = []

    for method in args.methods:
        for arch in ("MLP", "CNN"):
            run_dir = base / method / arch / f"{args.strategy}_{args.seq_len}"
            vals = final_scores(run_dir, args.metric, args.seeds)
            for v in vals:
                rows.append(dict(method=method, arch=arch, score=v))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No matching data found; check paths/arguments.")

    # aggregate: mean + 95 % CI
    agg = (df.groupby(["method", "arch"])["score"]
           .agg(["mean", "count", "std"])
           .reset_index())
    agg["ci95"] = agg.apply(
        lambda r: r["std"] / np.sqrt(r["count"]) * t.ppf(0.975, r["count"] - 1)
        if r["count"] > 1 else np.nan, axis=1)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    width = max(6, len(args.methods) * 1.5)
    fig, ax = plt.subplots(figsize=(width, 4))

    palette = {"MLP": "#4C72B0", "CNN": "#DD8452"}
    bar_w = 0.35
    x = np.arange(len(args.methods))

    for i, arch in enumerate(("MLP", "CNN")):
        sub = agg[agg.arch == arch]
        offsets = x - bar_w / 2 + i * bar_w
        ax.bar(offsets, sub["mean"], bar_w,
               yerr=sub["ci95"], capsize=5,
               color=palette[arch], label=arch, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(args.methods)
    ax.set_ylabel(f"Normalized Score")
    ax.set_xlabel("CL Method")
    ax.legend(title="Architecture")
    plt.tight_layout()
    out = root / 'plots'
    out.mkdir(exist_ok=True)
    stem = "mlp_vs_cnn"
    plt.savefig(out / f"{stem}.png")
    plt.savefig(out / f"{stem}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
