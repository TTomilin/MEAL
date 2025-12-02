"""
Plot horizontal bar charts for comparison the number of agents:
one plot for Average Performance, one for Forgetting.

Use `compute_metrics` from the numerical/agents_comparison script.
Bars are horizontal, with 95% CI as error bars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from experiments.results.numerical.agents_comparison import compute_metrics
from experiments.results.plotting.utils import save_plot


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Horizontal bar plots of Average Performance and Forgetting vs #agents (e.g., JaxNav)."
    )
    p.add_argument("--data_root", default="data", help="Root directory containing the data (relative to repo root).")
    p.add_argument("--algorithm", default="ippo", help="Algorithm name (e.g., ippo).")
    p.add_argument("--method", default="Online_EWC", help="Continual learning method to visualize.")
    p.add_argument("--env", default="jaxnav", help="Environment name (use 'jaxnav' for JaxNav paths).")
    p.add_argument("--strategy", default="generate", help="Strategy name (e.g., generate).")
    p.add_argument("--seq_len", type=int, default=10, help="Sequence length.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="Seeds to include in the metrics.")
    p.add_argument("--num_agents", type=int, nargs="+", default=[2, 3, 4],
                   help="Agent counts to compare.")
    p.add_argument(
        "--end_window_evals",
        type=int,
        default=10,
        help="How many final eval points to average for Forgetting.",
    )
    p.add_argument(
        "--no_ci",
        action="store_true",
        help="If set, do not draw confidence interval error bars.",
    )
    p.add_argument(
        "--show_values",
        action="store_true",
        help="If set, show numerical values next to bars.",
    )
    p.add_argument(
        "--plot_name",
        default="jaxnav_agents_horizontal",
        help="Base name for saved plots (without extension).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve repo root similar to your metric script
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / args.data_root

    agent_counts: list[int] = sorted(args.num_agents)
    A_means, A_cis = [], []
    F_means, F_cis = [], []

    level = 1
    for n_agents in agent_counts:
        metrics = compute_metrics(
            data_root=data_root,
            algo=args.algorithm,
            method=args.method,
            env=args.env,
            strategy=args.strategy,
            seq_len=args.seq_len,
            seeds=args.seeds,
            num_agents=n_agents,
            end_window_evals=args.end_window_evals,
            level=level,
        )

        A_means.append(metrics["AveragePerformance"])
        A_cis.append(metrics["AveragePerformance_CI"])
        F_means.append(metrics["Forgetting"])
        F_cis.append(metrics["Forgetting_CI"])

    A_means = np.asarray(A_means, dtype=float)
    A_cis = np.asarray(A_cis, dtype=float)
    F_means = np.asarray(F_means, dtype=float)
    F_cis = np.asarray(F_cis, dtype=float)

    # Filter out NaN rows (in case some agent configs are missing)
    mask = np.isfinite(A_means) & np.isfinite(F_means)
    agent_counts = [a for a, m in zip(agent_counts, mask) if m]
    A_means = A_means[mask]
    A_cis = A_cis[mask]
    F_means = F_means[mask]
    F_cis = F_cis[mask]

    if len(agent_counts) == 0:
        raise RuntimeError("No valid metrics found for any agent count.")

    y = np.arange(len(agent_counts))
    labels = [f"{n} agents" for n in agent_counts]

    # Color palette: one color per agent count, shared across both subplots
    cmap = plt.get_cmap("Dark2")
    colors = [cmap(i) for i in range(len(agent_counts))]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=(7.0, 2.8),
        constrained_layout=False,
    )
    axA, axF = axes

    # Average Performance (higher is better)
    xerr_A = A_cis if not args.no_ci else None
    axA.barh(
        y,
        A_means,
        xerr=xerr_A,
        capsize=4,
        color=colors,
        edgecolor="black",
    )
    axA.set_yticks(y)
    axA.set_yticklabels(labels)
    axA.invert_yaxis()  # so 2 agents appear at top
    axA.set_title("Average Score", fontsize=12)

    # Annotate values *after* the bar + CI, so they don't overlap error bars
    A_span = np.max(A_means + (A_cis if not args.no_ci else 0))
    A_offset_base = 0.01 * A_span if A_span > 0 else 0.01
    if args.show_values:
        for yi, val, ci in zip(
                y,
                A_means,
                A_cis if not args.no_ci else np.zeros_like(A_means),
        ):
            offset = ci + A_offset_base
            axA.text(
                val + offset,
                yi,
                f"{val:.3f}",
                va="center",
                ha="left",
                fontsize=12,
            )

    # Forgetting (lower is better)
    xerr_F = F_cis if not args.no_ci else None
    axF.barh(
        y,
        F_means,
        xerr=xerr_F,
        capsize=4,
        color=colors,
        edgecolor="black",
    )
    axF.set_yticks(y)  # shared
    axF.set_yticklabels([])  # hide duplicate labels on right
    axF.set_title("Forgetting", fontsize=12)

    F_span = np.max(F_means + (F_cis if not args.no_ci else 0))
    F_offset_base = 0.01 * F_span if F_span > 0 else 0.01
    # Optionally annotate values

    if args.show_values:
        for yi, val, ci in zip(
                y,
                F_means,
                F_cis if not args.no_ci else np.zeros_like(F_means),
        ):
            offset = ci + F_offset_base
            axF.text(
                val + offset,
                yi,
                f"{val:.3f}",
                va="center",
                ha="left",
                fontsize=12,
            )

    # Shared legend: one entry per agent count
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="black", label=labels[i])
        for i in range(len(agent_counts))
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(agent_counts),
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.2,
        top=0.89,
        wspace=0.15
    )

    # Save
    out_dir = repo_root / "plots"
    save_plot(fig, out_dir, args.plot_name)
    plt.close(fig)


if __name__ == "__main__":
    main()
