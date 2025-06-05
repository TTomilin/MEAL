#!/usr/bin/env python3
"""
Plot “plasticity” curves aggregated **by task position** across multiple
repetitions of the same task sequence that are stored *inside a single
file* (e.g. `training_soup.npz`).

Given:
  • `seq_len`  – number of distinct tasks in one sequence;
  • `repeat_sequence` – how many times that sequence was run back‑to‑back;

…this script chops the long trace into `seq_len × repeat_sequence` equal
segments, concatenates all occurrences of *Task i* in time order, and
computes a cumulative‑average curve for that concatenated trace.  The
result is exactly **`seq_len`** sub‑plots (2×5 when `seq_len==10`).

Directory layout expected:
```
<data_root>/<algo>/<method>/plasticity/<strategy>_<seq_len*repeat_sequence>/seed_<seed>/training_soup.*
```
For example `--strategy generate --seq_len 10 --repeat_sequence 10` ⇒
folder `…/plasticity/generate_100/…`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# z‑scores for confidence bands
CRIT = {0.9: 1, 0.95: 1.96, 0.99: 2.576}

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
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot plasticity curves aggregated by task position.",
    )
    p.add_argument("--data_root", required=True, help="Root folder with algo/method runs.")
    p.add_argument("--algo", required=True)
    p.add_argument("--strategy", required=True, help="Prefix of the data folder, e.g. 'generate'.")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--seq_len", type=int, required=True, help="Tasks per sequence.")
    p.add_argument("--repeat_sequence", type=int, default=1, help="Sequence repetitions inside the file.")
    p.add_argument("--steps_per_task", type=float, default=1e7, help="x‑axis scaling.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--sigma", type=float, default=1.5, help="Gaussian smoothing σ.")
    p.add_argument("--confidence", type=float, default=0.9, choices=[0.9, 0.95, 0.99])
    p.add_argument("--plot_name", default="plasticity_curve")
    return p.parse_args()


# ─────────────────── helpers ───────────────────

def _load(fp: Path) -> np.ndarray:
    """Load a 1‑D trace from *fp* (.json or .npz with key 'data')."""
    if fp.suffix == ".json":
        return np.asarray(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == ".npz":
        return np.load(fp)["data"].astype(float)
    raise ValueError(f"Unsupported file type: {fp.suffix}")


def _collect_runs(
        base: Path,
        algo: str,
        method: str,
        strat: str,
        seq_len: int,
        repeats: int,
        seeds: List[int],
) -> list[np.ndarray]:
    """Return a list where item *i* contains all curves for task *i* across seeds.

    Each trace ("*_soup.*") is assumed to contain *repeats* consecutive
    sequences of *seq_len* tasks.  For every seed we build **one** curve per
    task by concatenating all its segments across repetitions, then taking
    the cumulative average (normalised to 1.0 at the end).
    """

    task_runs: list[list[np.ndarray]] = [[] for _ in range(seq_len)]
    folder = f"{strat}_{seq_len * repeats}"

    for seed in seeds:
        run_dir = base / algo / method / "plasticity" / folder / f"seed_{seed}"
        if not run_dir.exists():
            print(f"Warning: no directory {run_dir}")
            continue

        for fp in sorted(run_dir.glob("*_soup.*")):
            trace = _load(fp)
            if trace.ndim != 1:
                raise ValueError(f"Trace in {fp} is not 1‑D (shape {trace.shape})")

            total_chunks = seq_len * repeats
            L_est = len(trace) // total_chunks
            if L_est == 0:
                print(f"Warning: trace in {fp} shorter than expected; skipped.")
                continue

            # build one long segment per task by concatenating its occurrences
            for task_idx in range(seq_len):
                slices = []
                for rep in range(repeats):
                    start = (rep * seq_len + task_idx) * L_est
                    end = start + L_est
                    if end > len(trace):  # safety for ragged endings
                        break
                    slices.append(trace[start:end])
                if not slices:
                    continue

                task_trace = np.concatenate(slices)
                task_runs[task_idx].append(task_trace)

    # pad to equal length so we can average ----------------------------------
    processed: list[np.ndarray] = []
    for idx, runs in enumerate(task_runs):
        if not runs:
            processed.append(np.array([]))
            continue
        T = max(len(r) for r in runs)
        padded = [np.pad(r, (0, T - len(r)), constant_values=np.nan) for r in runs]
        processed.append(np.vstack(padded))
    return processed


# ────────────────────────── main ─────────────────────────

def main():
    args = _cli()
    data_dir = Path(__file__).resolve().parent.parent / args.data_root

    # figure grid ------------------------------------------------------------
    if args.seq_len == 10:
        n_rows, n_cols = 2, 5
    else:
        n_rows = int(np.ceil(np.sqrt(args.seq_len)))
        n_cols = int(np.ceil(args.seq_len / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.flatten()

    method_lines, method_names = [], []

    for method in args.methods:
        task_data = _collect_runs(
            data_dir,
            args.algo,
            method,
            args.strategy,
            args.seq_len,
            args.repeat_sequence,
            args.seeds,
        )
        color = COL.get(method)

        for idx, curves in enumerate(task_data):
            if idx >= args.seq_len or curves.size == 0:
                continue
            ax = axes[idx]

            mu = gaussian_filter1d(np.nanmean(curves, axis=0), args.sigma)
            sd = gaussian_filter1d(np.nanstd(curves, axis=0), args.sigma)
            ci = CRIT[args.confidence] * sd / np.sqrt(curves.shape[0])

            x = np.linspace(0, args.steps_per_task * args.repeat_sequence, len(mu))
            (line,) = ax.plot(x, mu, color=color, label=method)
            ax.fill_between(x, mu - ci, mu + ci, color=color, alpha=0.2)

            ax.set_title(f"Task {idx + 1}")
            ax.set_xlim(0, args.steps_per_task * args.repeat_sequence)
            ax.set_ylim(0, 1.05)

            if idx == 0:
                method_lines.append(line)
                method_names.append(method)

    # labels & legend --------------------------------------------------------
    fig.text(0.5, 0.04, "Environment steps", ha="center", va="center", fontsize=14)
    fig.text(0.01, 0.5, "Normalised Score", ha="center", va="center", rotation="vertical", fontsize=14)

    for i in range(args.seq_len, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout(rect=[0.01, 0.03, 1, 0.98])

    Path("plots").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(Path("plots") / f"{args.plot_name}.{ext}")

    fig.show()


if __name__ == "__main__":
    main()
