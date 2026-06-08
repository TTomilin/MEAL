"""
Plotting script to compare full-observability vs partial-observability performance.

For each algo in --algos, plots four bars per level:
  - FO MLP:  <algo> + <cl_method>
  - PO MLP:  <algo> + <cl_method>_partial
  - FO CNN:  <algo> + <cl_method>          (experiment=cnn)
  - PO CNN:  <algo> + <cl_method>_partial  (experiment=cnn)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from experiments.results.plotting.utils import (
    collect_cumulative_runs,
    setup_figure,
    save_plot,
    finalize_plot,
    create_base_parser,
    CRIT,
)


def _parse_args():
    parser = create_base_parser(
        description="Compare FO vs PO, MLP vs CNN performance across algos and difficulty levels"
    )
    parser.add_argument("--data_root", required=True, help="Root folder with algo/method runs")
    parser.add_argument("--algos", nargs="+", required=True, help="Algorithms to compare (e.g. ippo mappo)")
    parser.add_argument("--cl_method", default="EWC", help="CL method (PO auto-suffixed with _partial)")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3], help="Difficulty levels")
    parser.add_argument("--strategy", default="generate", help="Training strategy")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Seeds to include")
    parser.add_argument("--agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--metric", choices=["reward", "soup"], default="soup", help="Metric to plot")
    parser.add_argument("--window", type=int, default=10, help="Final window size for score averaging")
    parser.add_argument("--show_values", action="store_true", help="Show mean values on bars")
    parser.add_argument("--normalize", action="store_true", help="Normalize scores to [0, 1]")
    parser.add_argument("--plot_name", default="partial_observability", help="Output plot name")
    return parser.parse_args()


def _final_scores(
        root: Path,
        algo: str,
        method: str,
        strategy: str,
        metric: str,
        seq_len: int,
        seeds: Sequence[int],
        level: int,
        agents: int,
        window: int,
        experiment: str = "",
) -> np.ndarray:
    """Return 1-D array (one value per seed) of mean of last *window* eval points."""
    try:
        curves = collect_cumulative_runs(
            root, algo, method, strategy, metric, seq_len, list(seeds), level, agents,
            experiment=experiment,
        )  # (S, T)
        if curves.size == 0:
            return np.array([])
        window = min(window, curves.shape[1])
        finals = np.nanmean(curves[:, -window:], axis=1)
        return finals[np.isfinite(finals)]
    except Exception as e:
        print(f"Warning: could not load {algo}/{method}/level_{level} (exp={experiment!r}): {e}")
        return np.array([])


# Each entry: (obs_label, method_suffix, experiment, hatch)
_VARIANTS: List[Tuple[str, str, str, str]] = [
    ("FO",     "",         "",    ""),
    ("PO",     "_partial", "",    "//"),
    # ("FO CNN", "",         "cnn", ".."),
    ("PO CNN", "_partial", "cnn", "//"),
]

_SERIES_PER_ALGO = len(_VARIANTS)


def _collect_data(
        data_root: Path,
        algos: List[str],
        cl_method: str,
        levels: List[int],
        strategy: str,
        seq_len: int,
        seeds: List[int],
        metric: str,
        window: int,
        agents: int,
        normalize: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns {series_label: {level_label: scores_array}}.
    Four series per algo: FO MLP, PO MLP, FO CNN, PO CNN.
    """
    data: Dict[str, Dict[str, np.ndarray]] = {}
    all_scores: List[float] = []

    for algo in algos:
        for obs_label, method_suffix, experiment, _ in _VARIANTS:
            method = f"{cl_method}{method_suffix}"
            label = f"{algo.upper()} ({obs_label})"
            data[label] = {}
            for level in levels:
                scores = _final_scores(
                    data_root, algo, method, strategy, metric,
                    seq_len, seeds, level, agents, window, experiment,
                )
                data[label][f"Level {level}"] = scores
                all_scores.extend(scores)

    if normalize and all_scores:
        lo, hi = np.min(all_scores), np.max(all_scores)
        rng = hi - lo
        if rng > 0:
            for label in data:
                for lv in data[label]:
                    if len(data[label][lv]) > 0:
                        data[label][lv] = (data[label][lv] - lo) / rng

    return data


def _plot_bars(ax: plt.Axes, data: Dict[str, Dict[str, np.ndarray]], show_values: bool = False):
    series = list(data.keys())
    levels = list(data[series[0]].keys()) if series else []

    if not series or not levels:
        print("Warning: no data to plot")
        return

    n_levels = len(levels)
    n_series = len(series)
    bar_width = 0.8 / n_series
    x = np.arange(n_levels)

    base_colors = ["#2E86AB", "#F18F01", "#A23B72", "#44BBA4", "#E94F37"]
    hatches = [v[3] for v in _VARIANTS]  # "", "//", "..", "xx"

    for i, label in enumerate(series):
        algo_idx = i // _SERIES_PER_ALGO
        variant_idx = i % _SERIES_PER_ALGO
        color = base_colors[algo_idx % len(base_colors)]
        hatch = hatches[variant_idx]
        # CNN variants slightly lighter
        alpha = 0.65 if "CNN" in label else 0.88

        means, cis = [], []
        for lv in levels:
            scores = data[label][lv]
            if len(scores) > 0:
                means.append(np.mean(scores))
                cis.append((CRIT[0.95] * np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0)
            else:
                means.append(0.0)
                cis.append(0.0)

        bars = ax.bar(
            x + i * bar_width, means, bar_width,
            yerr=cis, capsize=2,
            label=label, color=color, hatch=hatch,
            edgecolor="black", linewidth=0.5, alpha=alpha,
        )

        if show_values:
            for bar, val in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6,
                )

    ax.set_xticks(x + bar_width * (n_series - 1) / 2)
    ax.set_xticklabels(levels, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(fontsize=13, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3, axis="y")


def plot():
    args = _parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root

    data = _collect_data(
        data_root=data_root,
        algos=args.algos,
        cl_method=args.cl_method,
        levels=args.levels,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
        metric=args.metric,
        window=args.window,
        agents=args.agents,
        normalize=args.normalize,
    )

    n_series = len(args.algos) * _SERIES_PER_ALGO
    fig, ax = setup_figure(width=max(6, n_series * 0.8), height=3.0)
    _plot_bars(ax, data, show_values=args.show_values)
    ax.set_xlabel("")
    ax.set_ylabel("Average Soup Delivery", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 1])

    out_dir = Path(__file__).resolve().parent.parent / "plots"
    stem = args.plot_name
    if args.normalize:
        stem += "_normalized"
    save_plot(fig, out_dir, stem)
    plt.show()


if __name__ == "__main__":
    plot()
