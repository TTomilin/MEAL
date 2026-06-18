"""
Plot vertical bar charts comparing agent counts:
one plot for Average Performance (and optionally Forgetting).

Supports JaxNav (single level) and Overcooked (multiple levels, averaged).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.results.numerical.agents_comparison import compute_metrics
from experiments.results.plotting.utils import save_plot


_ENV_DEFAULT_SEQ_LEN: dict[str, int] = {
    "jaxnav": 10,
    "overcooked": 20,
    "mpe": 10,
    "smax": 10,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vertical bar plots of Average Performance and Forgetting vs #agents."
    )
    p.add_argument("--data_root", default="data", help="Root directory containing the data (relative to repo root).")
    p.add_argument("--algorithm", default="ippo", help="Algorithm name (e.g., ippo).")
    p.add_argument("--method", default="Online_EWC", help="Continual learning method to visualize.")
    p.add_argument("--env", default="jaxnav", help="Environment name (e.g., jaxnav, overcooked).")
    p.add_argument("--strategy", default="generate", help="Strategy name (e.g., generate).")
    p.add_argument("--seq_len", type=int, default=None,
                   help="Sequence length. Defaults per env: jaxnav=10, overcooked=20, mpe=10, smax=10.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="Seeds to include in the metrics.")
    p.add_argument("--num_agents", type=int, nargs="+", default=[2, 3, 4],
                   help="Agent counts to compare.")
    p.add_argument("--levels", type=int, nargs="+", default=[1],
                   help="Levels to include. Multiple levels are averaged (e.g., overcooked layouts).")
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
        help="If set, show numerical values above bars.",
    )
    p.add_argument(
        "--plot_name",
        default="agents_vertical",
        help="Base name for saved plots (without extension).",
    )
    p.add_argument(
        "--no_avg_soup_only",
        dest="avg_soup_only",
        action="store_false",
        help="Show Forgetting subplot in addition to Average Performance.",
    )
    p.set_defaults(avg_soup_only=True)
    return p.parse_args()


def _aggregate_levels(metrics_list: list[dict]) -> tuple[float, float, float, float]:
    """Average AP and F across levels (approximate CI by averaging)."""
    ap_means = [m["AveragePerformance"] for m in metrics_list if np.isfinite(m["AveragePerformance"])]
    ap_cis   = [m["AveragePerformance_CI"] for m in metrics_list if np.isfinite(m["AveragePerformance_CI"])]
    f_means  = [m["Forgetting"] for m in metrics_list if np.isfinite(m["Forgetting"])]
    f_cis    = [m["Forgetting_CI"] for m in metrics_list if np.isfinite(m["Forgetting_CI"])]

    ap_mean = float(np.mean(ap_means)) if ap_means else np.nan
    ap_ci   = float(np.mean(ap_cis))   if ap_cis   else np.nan
    f_mean  = float(np.mean(f_means))  if f_means  else np.nan
    f_ci    = float(np.mean(f_cis))    if f_cis    else np.nan
    return ap_mean, ap_ci, f_mean, f_ci


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / args.data_root

    seq_len = args.seq_len if args.seq_len is not None else _ENV_DEFAULT_SEQ_LEN.get(args.env, 10)

    agent_counts: list[int] = sorted(args.num_agents)
    A_means, A_cis = [], []
    F_means, F_cis = [], []

    for n_agents in agent_counts:
        metrics_list = []
        for level in args.levels:
            metrics = compute_metrics(
                data_root=data_root,
                algo=args.algorithm,
                method=args.method,
                env=args.env,
                strategy=args.strategy,
                seq_len=seq_len,
                seeds=args.seeds,
                num_agents=n_agents,
                end_window_evals=args.end_window_evals,
                level=level,
            )
            metrics_list.append(metrics)

        ap_mean, ap_ci, f_mean, f_ci = _aggregate_levels(metrics_list)
        A_means.append(ap_mean)
        A_cis.append(ap_ci)
        F_means.append(f_mean)
        F_cis.append(f_ci)

    A_means = np.asarray(A_means, dtype=float)
    A_cis   = np.asarray(A_cis,   dtype=float)
    F_means = np.asarray(F_means, dtype=float)
    F_cis   = np.asarray(F_cis,   dtype=float)

    mask = np.isfinite(A_means) if args.avg_soup_only else np.isfinite(A_means) & np.isfinite(F_means)
    agent_counts = [a for a, m in zip(agent_counts, mask) if m]
    A_means = A_means[mask]
    A_cis   = A_cis[mask]
    F_means = F_means[mask]
    F_cis   = F_cis[mask]

    if len(agent_counts) == 0:
        raise RuntimeError("No valid metrics found for any agent count.")

    x = np.arange(len(agent_counts))
    labels = [f"{n} agents" for n in agent_counts]

    cmap   = plt.get_cmap("Dark2")
    colors = [cmap(i) for i in range(len(agent_counts))]

    n_plots = 1 if args.avg_soup_only else 2
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_plots,
        figsize=(5.0 * n_plots, 3.5),
        constrained_layout=False,
    )
    if n_plots == 1:
        axes = [axes]
    axA = axes[0]

    yerr_A = A_cis if not args.no_ci else None
    axA.bar(x, A_means, yerr=yerr_A, capsize=4, color=colors, edgecolor="black")
    axA.set_xticks(x)
    axA.set_xticklabels(labels)
    axA.set_title("Average Score", fontsize=12)

    if args.show_values:
        A_span = float(np.max(A_means + (A_cis if not args.no_ci else 0)))
        A_offset = 0.01 * A_span if A_span > 0 else 0.01
        for xi, val, ci in zip(x, A_means, A_cis if not args.no_ci else np.zeros_like(A_means)):
            axA.text(xi, val + ci + A_offset, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    if not args.avg_soup_only:
        axF = axes[1]
        yerr_F = F_cis if not args.no_ci else None
        axF.bar(x, F_means, yerr=yerr_F, capsize=4, color=colors, edgecolor="black")
        axF.set_xticks(x)
        axF.set_xticklabels(labels)
        axF.set_title("Forgetting", fontsize=12)

        if args.show_values:
            F_span = float(np.max(F_means + (F_cis if not args.no_ci else 0)))
            F_offset = 0.01 * F_span if F_span > 0 else 0.01
            for xi, val, ci in zip(x, F_means, F_cis if not args.no_ci else np.zeros_like(F_means)):
                axF.text(xi, val + ci + F_offset, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.18,
        top=0.89,
        wspace=0.3,
    )

    out_dir = repo_root / "plots"
    save_plot(fig, out_dir, args.plot_name)
    plt.close(fig)


if __name__ == "__main__":
    main()
