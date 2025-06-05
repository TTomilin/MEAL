#!/usr/bin/env python3
"""plasticity_metrics.py
Compute and visualise *loss‑of‑plasticity* metrics from the traces used by
`plasticity.py`.

Metrics implemented
───────────────────
1. **Time‑to‑τ (% final perf.)**  – steps needed until the cumulative
   average reaches τ·final‑performance.  We report it per task and normalise
   by the *first* task’s value so a value >1 means slower learning.
2. **Area‑Under‑Plasticity‑Curve (AUPC)** – integral of the normalised
   cumulative‑average curve (∈[0,1]).  The higher, the more overall
   learning progress across the task.  We report (1 − AUPC) as *plasticity
   loss* and again normalise to the first task.

Both metrics are commonplace in continual‑learning RL papers
(e.g. Abbas *et al.*, 2023; Nature 2024).  They quantify how the *speed of
adaptation* degrades instead of just final performance.

Usage example
─────────────
```bash
python plasticity_metrics.py \
  --data_root results \
  --algo atari_dqn \
  --strategy generate \
  --methods CBP MAS \
  --seq_len 10 \
  --repeat_sequence 10 \
  --steps_per_task 1e6 \
  --threshold 0.9
```
which emits a CSV to *metrics.csv* and a bar‑plot *metrics.png* under
*plots/*.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from results.plotting.plasticity import _load


# ───────────────────────── helpers ──────────────────────────

def _collect_concat(
        base: Path,
        algo: str,
        method: str,
        strat: str,
        seq_len: int,
        repeats: int,
        seeds: List[int],
) -> list[np.ndarray]:
    """Return list[task] → list[seed curves] (concatenated raw traces)."""
    buckets: list[list[np.ndarray]] = [[] for _ in range(seq_len)]
    folder = f"{strat}_{seq_len * repeats}"
    for seed in seeds:
        run_dir = base / algo / method / "plasticity" / folder / f"seed_{seed}"
        for fp in run_dir.glob("*_soup.*"):
            trace = _load(fp)
            seg_len = len(trace) // (seq_len * repeats)
            for t in range(seq_len):
                segs = [
                    trace[(r * seq_len + t) * seg_len:(r * seq_len + t + 1) * seg_len]
                    for r in range(repeats)
                ]
                buckets[t].append(np.concatenate(segs))
    return buckets


def _cumavg(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x) / (np.arange(1, len(x) + 1))

# ────────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute plasticity‑loss metrics.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--repeat_sequence", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Seeds to include in the analysis.")
    ap.add_argument("--threshold", type=float, default=0.9, help="τ for time‑to‑τ metric (0<τ<1)")
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian σ before metric computation")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent / args.data_root
    out_dir = Path("plots"); out_dir.mkdir(exist_ok=True)

    csv_rows = [("method", "task", "time_to_τ", "time_norm", "aupc_loss", "aupc_norm")]

    for method in args.methods:
        task_traces = _collect_concat(base, args.algo, method, args.strategy, args.seq_len, args.repeat_sequence, args.seeds)

        time_to_tau, aupc_loss = [], []
        for trace in task_traces:
            if not trace:
                time_to_tau.append(np.nan)
                aupc_loss.append(np.nan)
                continue
            # smooth → cumavg → normalise
            smoothed = gaussian_filter1d(np.mean(trace, axis=0), args.sigma)
            ca = _cumavg(smoothed)
            ca /= ca[-1] if ca[-1] > 0 else 1.0
            # τ metric
            try:
                step_idx = np.where(ca >= args.threshold)[0][0]
            except IndexError:
                step_idx = len(ca)  # never reached
            time_to_tau.append(step_idx)
            # AUPC (trapezoidal)
            auc = np.trapz(ca) / len(ca)
            aupc_loss.append(1 - auc)

        # normalise by task‑0 (first task)
        norm_time = np.array(time_to_tau) / time_to_tau[0]
        norm_aupc = np.array(aupc_loss) / aupc_loss[0] if aupc_loss[0] else np.nan

        # save rows
        for t in range(args.seq_len):
            csv_rows.append((method, t + 1, time_to_tau[t], norm_time[t], aupc_loss[t], norm_aupc[t]))

        # plot ---------------------------------------------------------------
        x = np.arange(1, args.seq_len + 1)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x, norm_time, marker="o", label=f"Time‑to‑{args.threshold}")
        ax.plot(x, norm_aupc, marker="s", label="AUPC loss", linestyle="--")
        ax.set_xlabel("Task index")
        ax.set_ylabel("Normalised metric (≧1 => plasticity↓)")
        ax.set_title(f"Plasticity degradation – {method}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{method}_plasticity_metrics.png", dpi=300)

    # write CSV --------------------------------------------------------------
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"Metrics saved to {out_dir}/metrics.csv and PNG plots per method.")


if __name__ == "__main__":
    main()
