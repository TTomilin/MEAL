#!/usr/bin/env python3
"""Plot cumulative average performance for the MARL continual‑learning benchmark.

The script can now work in **two modes**:

1. **Method comparison**  (default)
   Same behaviour as the original script – curves for several
   continual‑learning methods on the *same* task level directory.

2. **Level comparison**   (``--compare_by level``)
   Show how one particular method (``--method``) behaves on multiple task
   *levels* (typically ``level_2`` vs ``level_3``).

The directory structure expected on disk is compatible with the download
script we created earlier:

```
results/data/<algo>/<method>/<level>/<strategy>_<seq_len>/seed_<seed>/
```

Usage examples
--------------

*Compare EWC vs AGEM on level‑3 tasks*
```
python plot_cumulative.py \
  --algo ippo \
  --methods EWC AGEM \
  --strategy generate \
  --seq_len 10 \
```

*Compare level-1 vs level‑2 vs level‑3 for EWC*
```
python plot_cumulative.py \
  --compare_by level \
  --method EWC \
  --levels 1 2 3 \
  --algo ippo \
  --strategy generate \
  --seq_len 10 \
```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from experiments.results.plotting.utils import (
    collect_cumulative_runs,
    setup_figure,
    add_task_boundaries,
    setup_task_axes,
    smooth_and_ci,
    save_plot,
    finalize_plot,
    METHOD_COLORS,
    LEVEL_COLORS,
    create_eval_parser,
)


# ────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = create_eval_parser(
        description="Plot cumulative average performance for MARL continual‑learning benchmark",
        metric_choices=["reward", "soup"],
    )
    p.set_defaults(metric="soup")  # override utils default

    # Mode flag
    p.add_argument(
        "--compare_by",
        choices=["method", "level"],
        default="method",
        help="What the curves represent on the plot.",
    )
    # When comparing levels we want a fixed method and a list of levels.
    p.add_argument(
        "--method",
        default="EWC",
        help="Continual‑learning method to plot when --compare_by level",
    )
    p.add_argument(
        "--levels",
        nargs="+",
        default=[1, 2, 3],
        help="Which task‑level sub‑folders to include (only used when --compare_by level).",
    )

    # Fine‑tuning of the legend
    p.add_argument("--legend_anchor", type=float, default=0.79)

    # GIF animation — for presentation slides
    p.add_argument(
        "--vid_segments", nargs="+", default=None,
        metavar="START:END:FRAMES",
        help=(
            "Generate video(s). Each segment is START:END:FRAMES (e.g. '0:5:30 5:100:60' produces two videos: "
            "the first slowly reveals tasks 0-5 in 30 frames, the second continues from 5 to 100 in 60 frames)."
        ),
    )
    p.add_argument("--vid_fps", type=int, default=10, help="Frames per second for the videos.")
    p.add_argument("--vid_format", choices=["mp4", "gif"], default="mp4",
                    help="Output format for the video segments.")
    p.add_argument("--vid_hold", type=int, default=15,
                    help="Number of duplicate frames appended at the end of GIFs to create a visual pause.")
    p.add_argument("--vid_ease", type=float, default=2.0,
                    help="Easing exponent for the reveal speed (1 = linear, "
                         ">1 = slow start that accelerates, <1 = fast start that decelerates).")

    # Data slicing
    p.add_argument(
        "--data_seq_len", type=int, default=None,
        help=(
            "Load data from this sequence-length folder, then truncate to "
            "--seq_len tasks.  E.g. --seq_len 10 --data_seq_len 100 plots "
            "the first 10 tasks using data stored under the 100-task run."
        ),
    )

    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# CURVE HELPERS
# ────────────────────────────────────────────────────────────────────────────

def _plot_curve(
        ax: plt.Axes,
        x: np.ndarray,
        mu: np.ndarray,
        ci: np.ndarray,
        label: str,
        color: str | None,
):
    # plot returns a list; unpack the first Line2D object
    (ln,) = ax.plot(x, mu, label=label, color=color)
    ax.fill_between(x, mu - ci, mu + ci, color=ln.get_color(), alpha=0.20)


def _collect_curve(
        level: int,
        data_root: Path,
        algo: str,
        method: str,
        strategy: str,
        metric: str,
        seq_len: int,
        seeds: List[int],
        steps_per_task: int,
        sigma: float,
        confidence: float,
        agents: int,
        compare_by: str = "method",
        data_seq_len: int | None = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str, Optional[str]]]:
    """Collect smoothed curve data without plotting.

    When *data_seq_len* is set, data is loaded from the folder for that
    sequence length and then truncated to *seq_len* tasks.

    Returns (x, mu, ci, label, color) or *None* when no data is found.
    """
    load_seq = data_seq_len if data_seq_len is not None else seq_len
    data = collect_cumulative_runs(
        data_root, algo, method, strategy, metric, load_seq, seeds, level, agents,
    )
    if len(data) == 0:
        print(f"[warn] no data for {method}")
        return None

    # Truncate to the first seq_len tasks if we loaded from a longer run.
    # The loader averages over all data_seq_len envs, so rescale to a
    # denominator of seq_len (i.e. only the environments actually shown).
    if data_seq_len is not None and data_seq_len > seq_len:
        keep = int(data.shape[1] * seq_len / data_seq_len)
        data = data[:, :keep] * (data_seq_len / seq_len)

    mu, ci = smooth_and_ci(data, sigma, confidence)
    x = np.linspace(0, seq_len * steps_per_task, len(mu))
    label = method if compare_by == "method" else f"Level {level}"
    color = (METHOD_COLORS.get(method) if compare_by == "method"
             else LEVEL_COLORS.get(level))
    return x, mu, ci, label, color


def _apply_decorations(ax, args, total_steps, ylim_max=None, visible_tasks=None):
    """Add task boundaries, tick formatting, legend, and axis labels.

    Parameters
    ----------
    visible_tasks : int, optional
        Number of tasks whose x-range should be visible.  When *None* the
        full ``args.seq_len`` is used (static plot).  The GIF passes a
        growing count so the x-axis expands frame by frame.
    """
    n_tasks = visible_tasks if visible_tasks is not None else args.seq_len
    visible_steps = n_tasks * args.steps_per_task
    boundaries = [i * args.steps_per_task for i in range(n_tasks + 1)]

    if args.seq_len <= 20:
        add_task_boundaries(ax, boundaries, color="grey", linewidth=0.5)
        setup_task_axes(ax, boundaries, n_tasks, fontsize=8)
    else:
        # ── many tasks: keep all thin boundaries, but de-clutter labels
        add_task_boundaries(ax, boundaries, color="grey", linewidth=0.4)

        from matplotlib.ticker import MaxNLocator, ScalarFormatter
        ax.set_xlim(0, visible_steps)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        steps = args.steps_per_task
        centers = (np.arange(n_tasks) + 0.5) * steps
        keep_idx = np.arange(9, n_tasks, 10)
        if len(keep_idx) > 0:
            top_ticks = centers[keep_idx]
            top_labels = [f"Task {idx + 1}" for idx in keep_idx]
            ax_top.set_xticks(top_ticks)
            ax_top.set_xticklabels(top_labels)
        else:
            ax_top.set_xticks([])
        ax_top.tick_params(axis="x", labelsize=11, pad=2)
        ax_top.xaxis.set_minor_locator(plt.NullLocator())

    legend_items = args.methods if args.compare_by == "method" else args.levels
    finalize_plot(
        ax,
        xlabel="Environment Steps",
        ylabel="Average Normalized Score",
        xlim=(0, visible_steps),
        ylim=(0, ylim_max),
        legend_loc="lower center",
        legend_bbox_to_anchor=(0.5, args.legend_anchor),
        legend_ncol=len(legend_items),
    )


# ────────────────────────────────────────────────────────────────────────────
# VIDEO ANIMATION (presentation segments)
# ────────────────────────────────────────────────────────────────────────────

def _parse_segment(s: str) -> Tuple[int, int, int]:
    """Parse a ``START:END:FRAMES`` segment string."""
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"Segment must be START:END:FRAMES, got '{s}'")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _save_video_segment(args, curves, start_task, end_task, n_frames, out_path):
    """Save one animated video segment.

    The x-axis grows from *start_task* to *end_task* over *n_frames* frames.
    The y-axis scales dynamically (monotonically increasing so it never
    shrinks).  Consecutive segments share their boundary frame for a seamless
    look when placed on back-to-back presentation slides.
    """
    import matplotlib.animation as animation

    seq_len = args.seq_len
    spt = args.steps_per_task

    # Build the reveal schedule (fractional task counts per frame).
    # A power curve (ease > 1) makes the x-axis crawl at first and
    # progressively speed up.
    ease = args.vid_ease
    if start_task == 0:
        # t goes from 1/n to 1  (skip 0 to avoid an empty first frame).
        t = np.linspace(0, 1, n_frames + 1)[1:]
        reveals = end_task * (t ** ease)
    else:
        # t goes from 0 to 1; first frame reproduces the previous segment's end.
        t = np.linspace(0, 1, n_frames)
        reveals = start_task + (end_task - start_task) * (t ** ease)

    # Precompute monotonically-increasing y-limits so the axis never shrinks.
    y_maxes: List[float] = []
    running_y = 0.0
    for reveal in reveals:
        frame_y = 0.0
        for _, mu, ci, _, _ in curves:
            n = len(mu)
            cut = max(min(int(reveal * n / seq_len), n), 1)
            frame_y = max(frame_y, float(np.max(mu[:cut] + ci[:cut])))
        running_y = max(running_y, frame_y * 1.05)
        y_maxes.append(running_y)

    # Choose axis style once for the whole segment to avoid mid-animation
    # jumps (few-task style with individual labels vs. many-task style with
    # every-10th label).
    use_many_tasks_style = end_task > 20

    fig, ax = setup_figure(width=10, height=3)
    fig.set_dpi(200)

    def update(frame):
        ax.clear()
        for extra in fig.axes[1:]:
            extra.remove()

        reveal = reveals[frame]

        # ── curves ────────────────────────────────────────────────────
        xlim_right = 0.0
        for x, mu, ci, label, color in curves:
            n = len(mu)
            cut = max(min(int(reveal * n / seq_len), n), 1)
            _plot_curve(ax, x[:cut], mu[:cut], ci[:cut], label, color)
            xlim_right = max(xlim_right, float(x[cut - 1]))

        # ── task boundaries (complete tasks only) ─────────────────────
        n_complete = int(reveal)
        boundaries = [i * spt for i in range(n_complete + 1)]
        add_task_boundaries(ax, boundaries, color="grey", linewidth=0.5)

        # ── axis style & task labels ──────────────────────────────────
        if not use_many_tasks_style:
            ax.set_xticks(boundaries)
            ax.ticklabel_format(
                style="scientific", axis="x", scilimits=(0, 0),
            )
            if n_complete > 0:
                secax = ax.secondary_xaxis("top")
                mids = [
                    (boundaries[i] + boundaries[i + 1]) / 2
                    for i in range(n_complete)
                ]
                secax.set_xticks(mids)
                secax.set_xticklabels(
                    [f"Task {i + 1}" for i in range(n_complete)],
                    fontsize=8,
                )
                secax.tick_params(axis="x", length=0)
        else:
            from matplotlib.ticker import MaxNLocator, ScalarFormatter
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=12)
            if n_complete > 0:
                ax_top = ax.twiny()
                ax_top.set_xlim(0, xlim_right)
                centers = (np.arange(n_complete) + 0.5) * spt
                keep = np.arange(9, n_complete, 10)
                if len(keep) > 0:
                    ax_top.set_xticks(centers[keep])
                    ax_top.set_xticklabels(
                        [f"Task {i + 1}" for i in keep], fontsize=11,
                    )
                else:
                    ax_top.set_xticks([])
                ax_top.tick_params(axis="x", labelsize=11, pad=2)
                ax_top.xaxis.set_minor_locator(plt.NullLocator())

        # ── limits, labels, legend ────────────────────────────────────
        ax.set_xlim(0, xlim_right)
        ax.set_ylim(0, y_maxes[frame])
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Average Normalized Score")

        legend_items = (args.methods if args.compare_by == "method"
                        else args.levels)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, args.legend_anchor),
                ncol=len(legend_items),
            )
        plt.tight_layout()

    # Append hold frames (repeat the last frame) so GIFs visually stop.
    total_frames = n_frames
    if args.vid_format == "gif":
        total_frames += args.vid_hold

    def update_with_hold(frame):
        # Clamp to the last real frame for hold frames.
        return update(min(frame, n_frames - 1))

    anim = animation.FuncAnimation(
        fig, update_with_hold, frames=total_frames, blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.vid_format == "gif":
        anim.save(str(out_path), writer="pillow", fps=args.vid_fps, dpi=200)
    else:
        anim.save(str(out_path), writer="ffmpeg", fps=args.vid_fps, dpi=200)
    plt.close(fig)
    print(f"Saved {args.vid_format} to {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# MAIN PLOTTING LOGIC
# ────────────────────────────────────────────────────────────────────────────

def plot():
    args = _parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root
    total_steps = args.seq_len * args.steps_per_task

    # ── Collect curve data ────────────────────────────────────────────────
    curves = []
    items_to_plot = args.methods if args.compare_by == "method" else args.levels
    for item in items_to_plot:
        if args.compare_by == "method":
            method, level = item, args.level
        else:
            method, level = args.method, item

        result = _collect_curve(
            level=level, data_root=data_root, algo=args.algo, method=method,
            strategy=args.strategy, metric=args.metric, seq_len=args.seq_len,
            seeds=args.seeds, steps_per_task=args.steps_per_task, sigma=args.sigma,
            confidence=args.confidence, agents=args.agents,
            compare_by=args.compare_by,
            data_seq_len=args.data_seq_len,
        )
        if result is not None:
            curves.append(result)

    # ── Static plot ───────────────────────────────────────────────────────
    fig, ax = setup_figure(width=10, height=3)
    for x, mu, ci, label, color in curves:
        _plot_curve(ax, x, mu, ci, label, color)
    _apply_decorations(ax, args, total_steps)

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent.parent / "plots"
    stem = args.plot_name or (
            "avg_cumulative_" + ("methods" if args.compare_by == "method" else "levels")
    )
    stem += f"_seq_{args.seq_len}"
    # Add level suffix if not already present
    if "_level" not in stem:
        stem += f"_level_{args.level}"
    # save_plot(fig, out_dir, stem)

    if args.vid_segments and curves:
        for idx, seg_str in enumerate(args.vid_segments, 1):
            start, end, n_frames = _parse_segment(seg_str)
            vid_path = out_dir / f"{stem}_part{idx}.{args.vid_format}"
            _save_video_segment(args, curves, start, end, n_frames, vid_path)

    plt.show()


if __name__ == "__main__":
    plot()
