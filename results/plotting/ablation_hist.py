#!/usr/bin/env python3
"""Plot histogram comparing original vs ablated performance."""
import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

# Import utilities to load series
try:
    from .utils import load_series
except ImportError:  # pragma: no cover - direct script execution
    from results.plotting.utils import load_series


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare ablated runs against main")
    p.add_argument("--data_root", required=True,
                   help="root folder with algorithm subfolders")
    p.add_argument("--algo", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--experiment", required=True,
                   help="name of the ablation folder, e.g. no_task_id")
    p.add_argument("--strategy", required=True)
    p.add_argument("--seq_len", type=int, required=True,
                   help="sequence length of the ablated runs")
    p.add_argument("--main_seq_len", type=int, default=20,
                   help="sequence length of the main runs")
    p.add_argument("--metric", choices=["reward", "soup"], default="soup")
    p.add_argument("--seeds", nargs="+", type=int, default=[1,2,3,4,5])
    p.add_argument("--plot_name", default=None)
    return p.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def final_scores(run_dir: Path, metric: str, seeds: List[int], n_tasks: int | None = None) -> List[float]:
    scores = []
    for seed in seeds:
        sd = run_dir / f"seed_{seed}"
        if not sd.exists():
            continue
        files = sorted(sd.glob(f"*_{metric}.*"))
        if n_tasks is not None:
            files = [f for f in files if not f.name.startswith("training")]  # exclude training
            files = files[:n_tasks]
        else:
            files = [f for f in files if not f.name.startswith("training")]
        if not files:
            continue
        vals = [load_series(f)[-1] for f in files]
        if vals:
            scores.append(float(np.nanmean(vals)))
    return scores


def ci95(vals: np.ndarray) -> float:
    if len(vals) < 2:
        return float('nan')
    return vals.std(ddof=1) / np.sqrt(len(vals)) * t.ppf(0.975, len(vals) - 1)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    base = root / args.data_root / args.algo

    rows = []
    for method in args.methods:
        # main results
        main_dir = base / method / "main" / f"{args.strategy}_{args.main_seq_len}"
        main_vals = final_scores(main_dir, args.metric, args.seeds, n_tasks=10)
        for v in main_vals:
            rows.append(dict(method=method, version="main", score=v))

        # ablation results
        abl_dir = base / method / args.experiment / f"{args.strategy}_{args.seq_len}"
        abl_vals = final_scores(abl_dir, args.metric, args.seeds)
        for v in abl_vals:
            rows.append(dict(method=method, version=args.experiment, score=v))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No matching data found; check paths/arguments.")

    agg = (df.groupby(["method", "version"])['score']
              .agg(['mean','count','std'])
              .reset_index())
    agg['ci95'] = agg.apply(lambda r: ci95(df[(df.method==r['method']) & (df.version==r['version'])]['score'].values), axis=1)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    width = max(6, len(args.methods) * 1.5)
    fig, ax = plt.subplots(figsize=(width, 4))

    versions = ["main", args.experiment]
    palette = {"main": "#4C72B0", args.experiment: "#DD8452"}
    bar_w = 0.35
    x = np.arange(len(args.methods))

    for i, ver in enumerate(versions):
        sub = agg[agg.version == ver]
        offsets = x - bar_w/2 + i*bar_w
        ax.bar(offsets, sub['mean'], bar_w,
               yerr=sub['ci95'], capsize=5,
               color=palette[ver], label=ver, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(args.methods)
    ax.set_ylabel('Normalized Score')
    ax.set_xlabel('CL Method')
    ax.legend(title='Version')

    plt.tight_layout()
    out = root / 'plots'
    out.mkdir(exist_ok=True)
    stem = args.plot_name or f"ablation_{args.experiment}"
    plt.savefig(out / f"{stem}.png")
    plt.savefig(out / f"{stem}.pdf")
    plt.show()


if __name__ == '__main__':
    main()
