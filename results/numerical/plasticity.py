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

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d  # Still needed for plotting

# Import utilities from the plotting package
from results.plotting.utils import (
    CRIT, METHOD_COLORS, load_series, smooth_and_ci,
    collect_plasticity_runs, create_plasticity_parser
)


# ───────────────────────── helpers ──────────────────────────

def _cumavg(x: np.ndarray) -> np.ndarray:
    """Calculate cumulative average of an array."""
    return np.cumsum(x) / (np.arange(1, len(x) + 1))

# ────────────────────────── main ─────────────────────────

def main():
    # Use the plasticity parser from the plotting utilities
    ap = create_plasticity_parser(description="Compute plasticity‑loss metrics.")
    ap.add_argument("--threshold", type=float, default=0.9, help="τ for time‑to‑τ metric (0<τ<1)")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_dir = base / args.data_root
    out_dir = base / Path("plots")
    data_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    csv_rows = [("method", "task", "time_to_τ", "time_norm", "aupc_loss", "aupc_norm")]

    for method in args.methods:
        task_traces = collect_plasticity_runs(data_dir, args.algo, method, args.strategy, args.seq_len, args.repeat_sequence, args.seeds)

        time_to_tau, aupc_loss = [], []
        for trace in task_traces:
            if trace.size == 0:
                time_to_tau.append(np.nan)
                aupc_loss.append(np.nan)
                continue
            # smooth → cumavg → normalise
            smoothed, _ = smooth_and_ci(trace, args.sigma, 0.95)  # We only need the mean, not the CI
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

        norm_time = np.array(time_to_tau, dtype=float)
        norm_aupc = np.array(aupc_loss,  dtype=float) / aupc_loss[0]

        # save rows
        for t in range(args.seq_len):
            csv_rows.append((method, t + 1, time_to_tau[t], norm_time[t], aupc_loss[t], norm_aupc[t]))

        # plot ---------------------------------------------------------------
        x = np.arange(1, args.seq_len + 1)
        # fig, ax = plt.subplots(figsize=(8, 3))

        # Check if we have valid data to plot
        has_time_data = not np.all(np.isnan(norm_time))
        has_aupc_data = not np.all(np.isnan(norm_aupc))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharex=True)

        ax1.plot(x, norm_time, 'o-')
        ax1.set_title(f'Time-to-{args.threshold}')
        ax1.set_ylabel('× slower than task 1')

        ax2.plot(x, norm_aupc, 's--')
        ax2.set_title('AUPC loss')
        ax2.set_ylabel('× higher than task 1')

        for ax in (ax1, ax2):
            ax.set_xlabel('Task index')
            ax.grid(True)
        fig.tight_layout()


        # Add a note to the title if data is missing
        title = f"Plasticity degradation – {method}"
        if not has_time_data and not has_aupc_data:
            title += " (No data available)"
        elif not has_time_data:
            title += f" (No Time‑to‑{args.threshold} data)"
        elif not has_aupc_data:
            title += " (No AUPC loss data)"

        ax.set_title(title)

        # Only add a legend if there's at least one valid metric to plot
        if has_time_data or has_aupc_data:
            ax.legend()

        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{method}_plasticity_metrics.png", dpi=300)
        fig.show()

    # write CSV --------------------------------------------------------------
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"Metrics saved to {out_dir}/metrics.csv and PNG plots per method.")


if __name__ == "__main__":
    main()
