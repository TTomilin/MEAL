#!/usr/bin/env python3
"""
Plot a “plasticity” curve without caring where tasks start or end.

For each timestep t:
    avg(t) = cumsum(reward)[t] / (⌊t / L_est⌋ + 1)

where L_est = len(trace) / seq_len.

The curve is then normalised so that it ends at 1.0, giving the tidy 0→1
trajectory you wanted—no boundary detection, no divisibility headaches.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# z-scores for confidence bands
CRIT = {0.9: 1, 0.95: 1.96, 0.99: 2.576}

# add colours as you please
COL = {
    "CBP": "#2F4B7C",
    "EWC": "#12939A",
    "MAS": "#FF6E54",
    "AGEM": "#FFA600",
    "L2": "#003F5C",
    "PackNet": "#BC5090",
    "ReDo": "#58508D",
}


# ───────────────────────── CLI ──────────────────────────
def _cli():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root", required=True, help="folder that contains algo/method/... runs")
    p.add_argument("--algo", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--steps_per_task", type=float, default=1e7, help="x-axis scaling")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--sigma", type=float, default=1.5, help="Gaussian smoothing σ")
    p.add_argument("--confidence", type=float, default=0.9, choices=[0.9, 0.95, 0.99])
    p.add_argument("--plot_name", default="plasticity_curve")
    return p.parse_args()


# ─────────────────── helpers ───────────────────
def _load(fp: Path) -> np.ndarray:
    if fp.suffix == ".json":
        return np.asarray(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == ".npz":
        return np.load(fp)["data"].astype(float)
    raise ValueError(f"Unsupported file type: {fp.suffix}")


def _collect_runs(base: Path, algo: str, method: str, arch: str, strat: str,
                  seq_len: int, seeds: List[int]) -> np.ndarray:
    """Return array (n_seeds, T) of normalised cumulative-avg curves."""
    runs = []
    for seed in seeds:
        run_dir = base / algo / method / arch / f"{strat}_{seq_len}" / f"seed_{seed}"
        if not run_dir.exists():
            continue

        files = sorted(run_dir.glob("*_reward.*"))
        if not files:
            continue

        # one file ⇒ use it; many files ⇒ concatenate them in lexicographic order
        trace = _load(files[0]) if len(files) == 1 else np.concatenate([_load(f) for f in files])

        L_est = len(trace) / seq_len                       # estimated task length
        denom = np.floor(np.arange(len(trace)) / L_est) + 1
        curve = np.cumsum(trace) / denom

        # force end-point to 1.0
        curve /= curve[-1] if curve[-1] > 0 else 1.0
        runs.append(curve)

    if not runs:
        raise RuntimeError(f"No runs found for {method}")

    # pad shorter runs with NaNs so we can average
    T = max(map(len, runs))
    padded = [np.pad(r, (0, T - len(r)), constant_values=np.nan) for r in runs]
    return np.vstack(padded)


# ────────────────────────── main ─────────────────────────
def main():
    args = _cli()
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = Path(__file__).resolve().parent.parent / args.data_root
    total_steps = args.seq_len * args.steps_per_task

    fig, ax = plt.subplots(figsize=(12, 4))

    for method in args.methods:
        data = _collect_runs(data_dir, args.algo, method, args.arch,
                             args.strategy, args.seq_len, args.seeds)

        mu = gaussian_filter1d(np.nanmean(data, axis=0), sigma=args.sigma)
        sd = gaussian_filter1d(np.nanstd(data, axis=0), sigma=args.sigma)
        ci = CRIT[args.confidence] * sd / np.sqrt(data.shape[0])

        x = np.linspace(0, total_steps, len(mu))
        color = COL.get(method)
        ax.plot(x, mu, label=method, color=color)
        ax.fill_between(x, mu - ci, mu + ci, color=color, alpha=0.2)

    ax.set_xlim(0, total_steps)
    ax.set_ylim(0, None)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Normalised cumulative average")
    ax.set_title("Plasticity curve (task-agnostic)")
    ax.legend(frameon=False, ncol=len(args.methods))
    fig.tight_layout()

    out_dir = Path("plots"); out_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"{args.plot_name}.{ext}")

    plt.show()


if __name__ == "__main__":
    main()
