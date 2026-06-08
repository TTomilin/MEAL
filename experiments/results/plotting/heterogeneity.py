#!/usr/bin/env python3
"""
Plot role-heterogeneity index (mean JSD between agent action distributions)
as a grouped bar chart.

Groups = difficulty levels (with a small gap between them).
Bars within each group = CL methods.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.results.numerical.heterogeneity_table import build_table
from experiments.results.plotting.utils import (
    METHOD_COLORS, save_plot, create_parser_with_common_args, method_display_name,
)


def _cli():
    p = create_parser_with_common_args("Plot heterogeneity index as grouped bar chart.")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.set_defaults(plot_name="heterogeneity")
    return p.parse_args()


def main():
    args = _cli()
    base = Path(__file__).resolve().parent.parent
    data_root = base / args.data_root

    df = build_table(
        data_root=data_root,
        algo=args.algo,
        methods=args.methods,
        levels=args.levels,
        agents=args.agents,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
    )

    n_methods = len(args.methods)
    n_levels = len(args.levels)
    bar_w = 0.8 / n_methods
    level_gap = 0.1            # extra space between level groups

    # Build x positions: each level group is width 1, then a gap
    group_width = 1.0 + level_gap
    group_centers = np.array([i * group_width for i in range(n_levels)])

    fig, ax = plt.subplots(figsize=(max(6, n_levels * n_methods * 0.6 + 1), 4))

    for m_idx, method in enumerate(args.methods):
        means, cis = [], []
        row = df.iloc[m_idx]
        for level in args.levels:
            means.append(float(row[f"level_{level}_mean"]))
            cis.append(float(row[f"level_{level}_ci"]))

        offsets = (m_idx - (n_methods - 1) / 2) * bar_w
        x = group_centers + offsets
        color = METHOD_COLORS.get(method, f"C{m_idx}")

        ax.bar(
            x, means, bar_w,
            yerr=cis, capsize=4,
            label=method_display_name(method),
            color=color, edgecolor="black", linewidth=0.5, alpha=0.9,
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"Level {l}" for l in args.levels], fontsize=12)
    ax.set_ylabel("Heterogeneity Index", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    save_plot(fig, base / "plots", args.plot_name)
    fig.show()


if __name__ == "__main__":
    main()
