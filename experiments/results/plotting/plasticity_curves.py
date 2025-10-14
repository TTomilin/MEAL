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

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Try relative import first (when imported as a module)
from experiments.results.plotting.utils import (
    CRIT, METHOD_COLORS, collect_plasticity_runs, setup_plasticity_grid, save_plot,
    create_plasticity_parser
)


# ───────────────────────── CLI ──────────────────────────

def _cli():
    """Parse command line arguments for the plasticity plot script."""
    p = create_plasticity_parser(description="Plot plasticity curves aggregated by task position.")
    # Set default confidence level to 0.9 (overriding the default from create_plasticity_parser)
    p.set_defaults(confidence=0.9, plot_name="plasticity_curve")
    return p.parse_args()


# ────────────────────────── main ─────────────────────────

def main():
    args = _cli()
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / args.data_root

    # Set up figure grid
    fig, axes = setup_plasticity_grid(args.seq_len)

    method_lines, method_names = [], []

    for method in args.methods:
        task_data = collect_plasticity_runs(
            data_dir,
            args.algo,
            method,
            args.strategy,
            args.seq_len,
            args.repeat_sequence,
            args.seeds,
            args.level,
        )
        color = METHOD_COLORS.get(method)

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

    # Save the plot
    out_dir = base_dir / Path("plots")
    plot_name = args.plot_name
    # Add level suffix if not already present
    if "_level_" not in plot_name:
        plot_name += f"_level_{args.level}"
    save_plot(fig, out_dir, plot_name)

    fig.show()


if __name__ == "__main__":
    main()
