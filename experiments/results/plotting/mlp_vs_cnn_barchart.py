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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t

# Import utilities from the utils package
try:
    # Try relative import first (when imported as a module)
    from .utils import (
        create_base_parser, add_common_args, add_metric_arg, load_series
    )
except ImportError:
    # Fall back to absolute import (when run as a script)
    import sys
    import os
    # Add the parent directory to sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from experiments.results.plotting.utils import (
        create_base_parser, add_common_args, add_metric_arg, load_series
    )

sns.set_theme(style="whitegrid", context="notebook")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    """Parse command line arguments for the MLP vs CNN bar chart script."""
    p = create_base_parser(description="Compare MLP vs. CNN for several CL methods with a bar-chart")

    # Add only the arguments needed for this script
    p.add_argument("--data_root", required=True,
                   help="root folder: results/<algo>/<method>/<arch>/strategy_len/seed_*")
    p.add_argument("--algo", required=True, help="Algorithm name")
    p.add_argument("--methods", nargs="+", required=True, help="Method names to plot")
    p.add_argument("--strategy", required=True, help="Training strategy")
    p.add_argument("--seq_len", type=int, required=True, help="Sequence length")
    p.add_argument("--seeds", nargs="+", type=int, default=[3], help="Seeds to include")
    p.add_argument("--level", type=int, default=1, help="Difficulty level of the environment")

    # Add metric argument
    add_metric_arg(p, choices=["reward", "soup"], default="soup")

    return p.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


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
        for arch in (f"level_{args.level}", "cnn"):
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

    palette = {f"level_{args.level}": "#4C72B0", "cnn": "#DD8452"}
    bar_w = 0.35
    x = np.arange(len(args.methods))

    for i, arch in enumerate((f"level_{args.level}", "cnn")):
        sub = agg[agg.arch == arch]
        offsets = x - bar_w / 2 + i * bar_w
        label = "MLP" if arch == f"level_{args.level}" else "CNN"
        ax.bar(offsets, sub["mean"], bar_w,
               yerr=sub["ci95"], capsize=5,
               color=palette[arch], label=label, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(args.methods)
    ax.set_ylabel(f"Normalized Score")
    ax.set_xlabel("CL Method")
    ax.legend(title="Architecture")
    plt.tight_layout()
    out = root / 'plots'
    out.mkdir(exist_ok=True)
    stem = "mlp_vs_cnn"
    # Add level suffix if not already present
    if "_level_" not in stem:
        stem += f"_level_{args.level}"
    plt.savefig(out / f"{stem}.png")
    plt.savefig(out / f"{stem}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
