#!/usr/bin/env python3
"""Plot histogram comparing original vs ablated performance.

Features
--------
* Supports **multiple** ablations on one figure (`--experiments`).
* Bars are grouped in the order given by `--methods`.
* Y‑axis starts at zero so every bar sits on the bottom axis.
* Legend is a single row centred beneath the plot, one column per variant.
* Legend labels are prettified (``main`` → *Original*, underscores → spaces, title‑case).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from results.plotting.utils import load_series


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare ablated runs against main")
    p.add_argument("--data_root", required=True,
                   help="root folder with algorithm subfolders")
    p.add_argument("--algo", required=True)
    p.add_argument("--methods", nargs="+", required=True,
                   help="ordered list of CL methods to show on x‑axis")
    p.add_argument("--experiments", nargs="+", required=True,
                   help="one or more ablation folder names (no 'main')")
    p.add_argument("--strategy", required=True)
    p.add_argument("--seq_len", type=int, required=True,
                   help="sequence length of the ablated runs")
    p.add_argument("--main_seq_len", type=int, default=20,
                   help="sequence length of the main runs (sliced to 10 tasks)")
    p.add_argument("--metric", choices=["reward", "soup"], default="soup")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--plot_name", default=None,
                   help="stem for the output file names (png/pdf)")
    return p.parse_args()


# -------------------------------------------------------------
# helpers
# -------------------------------------------------------------

def final_scores(run_dir: Path, metric: str, seeds: List[int], *, n_tasks: int | None = None) -> List[float]:
    """Return one score per seed (mean across envs)."""
    scores = []
    for seed in seeds:
        sd = run_dir / f"seed_{seed}"
        if not sd.exists():
            continue
        files = sorted(sd.glob(f"*_{metric}.*"))
        files = [f for f in files if not f.name.startswith("training")]
        if n_tasks is not None:
            files = files[:n_tasks]
        if not files:
            continue
        vals = [load_series(f)[-1] for f in files]  # last entry = final score per env
        if vals:
            scores.append(float(np.nanmean(vals)))  # env‑average
    return scores


def ci95(arr: np.ndarray) -> float:
    """Half‑width of the 95 % t‑CI."""
    if len(arr) < 2:
        return float("nan")
    return arr.std(ddof=1) / np.sqrt(len(arr)) * t.ppf(0.975, len(arr) - 1)


def nice_label(ver: str) -> str:
    """Human‑friendly label for legend."""
    if ver == "main":
        return "Original"
    return ver.replace("_", " ").title()


# -------------------------------------------------------------
# main
# -------------------------------------------------------------

def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent.parent
    base = root / args.data_root / args.algo

    rows: list[dict[str, str | float]] = []

    for method in args.methods:
        # ---------- original (main) ----------
        main_dir = base / method / "main" / f"{args.strategy}_{args.main_seq_len}"
        main_vals = final_scores(main_dir, args.metric, args.seeds, n_tasks=10)
        rows.extend(dict(method=method, version="main", score=v) for v in main_vals)

        # ---------- every ablation ----------
        for exp in args.experiments:
            abl_dir = base / method / exp / f"{args.strategy}_{args.seq_len}"
            abl_vals = final_scores(abl_dir, args.metric, args.seeds)
            rows.extend(dict(method=method, version=exp, score=v) for v in abl_vals)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No matching data found. Check paths / arguments.")

    # enforce x‑axis order
    df["method"] = pd.Categorical(df["method"], categories=args.methods, ordered=True)

    # aggregate mean + CI95
    agg = (
        df.groupby(["method", "version"], observed=True)["score"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    agg["ci95"] = agg.apply(lambda r: ci95(df[(df.method == r["method"]) & (df.version == r["version"])]["score"].values), axis=1)

    # pivot to align bars
    piv = agg.pivot(index="method", columns="version", values="mean")
    ci_piv = agg.pivot(index="method", columns="version", values="ci95")

    versions = ["main", *args.experiments]
    n_ver = len(versions)
    bar_w = 0.8 / n_ver
    x = np.arange(len(args.methods))

    # dynamic palette (simple but distinct)
    base_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    while len(base_colors) < n_ver:
        base_colors.extend(base_colors)  # recycle if too many
    palette = {ver: col for ver, col in zip(versions, base_colors)}

    fig, ax = plt.subplots(figsize=(max(6, len(args.methods) * 1.5), 4))

    for i, ver in enumerate(versions):
        offsets = x - (n_ver - 1) * bar_w / 2 + i * bar_w
        means = piv[ver].values
        errs = ci_piv[ver].values
        ax.bar(offsets, means, bar_w, yerr=errs, capsize=5,
               color=palette[ver], label=nice_label(ver), alpha=0.9)

    # Axis & legend tweaks --------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(args.methods, rotation=15)
    ax.set_ylabel("Normalized Score")
    ax.set_xlabel("CL Method")
    ax.set_ylim(bottom=0)  # bars start from bottom

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=n_ver // 2, frameon=False, title="Variant")

    plt.tight_layout()
    out_dir = root / "plots"
    out_dir.mkdir(exist_ok=True)
    stem = args.plot_name or "ablation_" + "_".join(args.experiments)
    plt.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.savefig(out_dir / f"{stem}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
