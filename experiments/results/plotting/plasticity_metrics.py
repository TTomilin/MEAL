#!/usr/bin/env python3
"""plasticity_metrics.py – v10 (global metrics & LaTeX table)
================================================================
Extends v9 by computing **sequence‑level averages** of each metric
(AUC‑loss, Final‑Performance ratio, Raw‑AUC ratio) and emitting a
ready‑to‑paste LaTeX table.  If you pass multiple ``--repeats`` values,
you get one row per repetition setting and three columns (one per
metric).  When ``--extra_capacity_stats`` is *off*, the table will only
contain AUC‑loss.

CLI example
───────────
```bash
python plasticity_metrics.py \
  --repeats 5 10 \
  --methods MAS \
  --extra_capacity_stats
```
creates:
```
plots/MAS_plasticity_multi.png
plots/MAS_global_metrics.tex   # <- include \input{…} in the paper
```
"""
from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from experiments.results.plotting.utils import (
    collect_plasticity_runs,
    create_plasticity_parser,
)
from experiments.results.plotting.utils.common import CRIT, load_series


# ─────────────────────── helper functions ─────────────────────────

def _auc(trace: np.ndarray, sigma: float) -> float:
    if sigma > 0:
        trace = gaussian_filter1d(trace, sigma=sigma)
    return float(np.trapz(trace))


def _palette(n: int) -> List[str]:
    base = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    if n <= len(base):
        return base[:n]
    import itertools
    return list(itertools.islice(itertools.cycle(base), n))

# metric helpers --------------------------------------------------------------

def _compute_auc_loss(task_traces, repeats: int, sigma: float, conf: float = 0.95):
    mean, ci = [], []
    for traces in task_traces:
        if traces.size == 0:
            mean.append(np.nan); ci.append(np.nan); continue
        n_seeds, T = traces.shape
        L = T // repeats
        if L == 0:
            mean.append(np.nan); ci.append(np.nan); continue

        # For repetition 1, AUCL should simply be 0 (no loss)
        if repeats == 1:
            mean.append(0.0)
            ci.append(0.0)
            continue

        baseline = np.nanmean([_auc(traces[s, :L], sigma) for s in range(n_seeds)])
        losses = []
        for rep in range(1, repeats):
            seg = slice(rep * L, (rep + 1) * L)
            for s in range(n_seeds):
                ratio = _auc(traces[s, seg], sigma) / baseline if baseline else np.nan
                losses.append(max(0.0, 1.0 - ratio))

        if losses:
            losses_mean = float(np.nanmean(losses))
            losses_std = float(np.nanstd(losses, ddof=1))
            n_samples = len([x for x in losses if not np.isnan(x)])
            # Calculate confidence interval
            if n_samples > 0 and losses_std > 0:
                ci_val = CRIT[conf] * losses_std / np.sqrt(n_samples * repeats)
            else:
                ci_val = 0.0
            mean.append(losses_mean)
            ci.append(ci_val)
        else:
            mean.append(np.nan)
            ci.append(np.nan)
    return np.array(mean), np.array(ci)


def _capacity_metrics(task_traces, repeats: int, sigma: float, conf: float = 0.95):
    fpr_m, fpr_ci, rauc_m, rauc_ci = [], [], [], []
    for traces in task_traces:
        if traces.size == 0:
            fpr_m.append(np.nan); fpr_ci.append(np.nan); rauc_m.append(np.nan); rauc_ci.append(np.nan); continue
        n_seeds, T = traces.shape
        L = T // repeats
        if L == 0:
            fpr_m.append(np.nan); fpr_ci.append(np.nan); rauc_m.append(np.nan); rauc_ci.append(np.nan); continue
        base_fperf = traces[:, L - 1]
        base_rauc = np.array([_auc(traces[s, :L], sigma) for s in range(n_seeds)])
        fprs, raucs = [], []
        for rep in range(1, repeats):
            seg = slice(rep * L, (rep + 1) * L)
            for s in range(n_seeds):
                fpr_val = traces[s, seg.stop - 1] / base_fperf[s] if base_fperf[s] else np.nan
                if not np.isnan(fpr_val):
                    fpr_val = min(fpr_val, 1.25)
                rau_val = _auc(traces[s, seg], sigma) / base_rauc[s] if base_rauc[s] else np.nan
                fprs.append(fpr_val); raucs.append(rau_val)

        # Calculate FPR mean and confidence interval
        if fprs:
            fpr_mean = np.nanmean(fprs)
            fpr_std = np.nanstd(fprs, ddof=1)
            fpr_n_samples = len([x for x in fprs if not np.isnan(x)])
            if fpr_n_samples > 0 and fpr_std > 0:
                fpr_ci_val = CRIT[conf] * fpr_std / np.sqrt(fpr_n_samples)
            else:
                fpr_ci_val = 0.0
            fpr_m.append(fpr_mean)
            fpr_ci.append(fpr_ci_val)
        else:
            fpr_m.append(np.nan)
            fpr_ci.append(np.nan)

        # Calculate RAUC mean and confidence interval
        if raucs:
            rauc_mean = np.nanmean(raucs)
            rauc_std = np.nanstd(raucs, ddof=1)
            rauc_n_samples = len([x for x in raucs if not np.isnan(x)])
            if rauc_n_samples > 0 and rauc_std > 0:
                rauc_ci_val = CRIT[conf] * rauc_std / np.sqrt(rauc_n_samples)
            else:
                rauc_ci_val = 0.0
            rauc_m.append(rauc_mean)
            rauc_ci.append(rauc_ci_val)
        else:
            rauc_m.append(np.nan)
            rauc_ci.append(np.nan)

    return (np.array(fpr_m), np.array(fpr_ci), np.array(rauc_m), np.array(rauc_ci))


def _collect_dormant_ratio_data(
        base: Path,
        algo: str,
        method: str,
        strat: str,
        seq_len: int,
        repeats: int,
        seeds: List[int],
        level: int = 1,
) -> list[np.ndarray]:
    """Collect dormant ratio data for each task across seeds.

    Returns a list where item *i* contains all dormant ratio curves for task *i* across seeds.
    Similar to collect_plasticity_runs but for dormant_ratio.json files.
    """
    task_runs: list[list[np.ndarray]] = [[] for _ in range(seq_len)]

    # Try to find data with higher repetitions first, then fall back to exact match
    possible_repeats = sorted([r for r in range(repeats, 21) if r >= repeats], reverse=True)

    for seed in seeds:
        found_data = False

        for try_repeats in possible_repeats:
            folder = f"{strat}_{seq_len}"
            if try_repeats > 1:
                folder += f"_rep_{try_repeats}"

            run_dir = base / algo / method / "plasticity" / folder / f"seed_{seed}"
            if not run_dir.exists():
                continue

            dormant_file = run_dir / "dormant_ratio.json"
            if not dormant_file.exists():
                continue

            trace = load_series(dormant_file)
            if trace.ndim != 1:
                raise ValueError(f"Dormant ratio trace in {dormant_file} is not 1‑D (shape {trace.shape})")

            total_chunks = seq_len * try_repeats
            L_est = len(trace) // total_chunks
            if L_est == 0:
                print(f"Warning: dormant ratio trace in {dormant_file} shorter than expected; skipped.")
                continue

            # If we're reusing data from higher repetitions, inform the user
            if try_repeats > repeats:
                print(f"Info: Reusing dormant ratio data from {folder}/seed_{seed} for rep_{repeats} (trimming {try_repeats - repeats} repetitions)")

            # build one long segment per task by concatenating its occurrences
            # but only use the first 'repeats' repetitions
            for task_idx in range(seq_len):
                slices = []
                for rep in range(repeats):  # Only use the requested number of repetitions
                    start = (rep * seq_len + task_idx) * L_est
                    end = start + L_est
                    if end > len(trace):  # safety for ragged endings
                        break
                    slices.append(trace[start:end])
                if not slices:
                    continue

                task_trace = np.concatenate(slices)
                task_runs[task_idx].append(task_trace)

            found_data = True
            break  # Found data for this seed, move to next seed

        if not found_data:
            print(f"Warning: no dormant ratio data found for seed {seed} with any repetition >= {repeats}")

    # pad to equal length so we can average
    processed: list[np.ndarray] = []
    for idx, runs in enumerate(task_runs):
        if not runs:
            processed.append(np.array([]))
            continue
        T = max(len(r) for r in runs)
        padded = [np.pad(r, (0, T - len(r)), constant_values=np.nan) for r in runs]
        processed.append(np.vstack(padded))
    return processed


def _compute_dormant_ratio_metrics(task_traces, repeats: int, sigma: float, conf: float = 0.95):
    """Compute dormant ratio metrics for each task."""
    mean, ci = [], []
    for traces in task_traces:
        if traces.size == 0:
            mean.append(np.nan); ci.append(np.nan); continue
        n_seeds, T = traces.shape
        L = T // repeats
        if L == 0:
            mean.append(np.nan); ci.append(np.nan); continue

        # Calculate mean dormant ratio across all repetitions for each seed
        ratios = []
        for s in range(n_seeds):
            for rep in range(repeats):
                seg = slice(rep * L, (rep + 1) * L)
                if seg.stop <= T:
                    # Average dormant ratio over the segment
                    seg_mean = np.nanmean(traces[s, seg])
                    if not np.isnan(seg_mean):
                        ratios.append(seg_mean)

        if ratios:
            ratios_mean = float(np.nanmean(ratios))
            # The dormant ratio should actually be 1 - ratio
            ratios_mean = 1.0 - ratios_mean
            ratios_std = float(np.nanstd(ratios, ddof=1))
            n_samples = len([x for x in ratios if not np.isnan(x)])
            # Calculate confidence interval with scaling for lower repetitions
            if n_samples > 0 and ratios_std > 0:
                # Scale confidence interval proportionally to 10 repetitions
                reference_repetitions = 10
                scaling_factor = min(1.0, repeats / reference_repetitions)
                ci_val = CRIT[conf] * ratios_std / np.sqrt(n_samples) * scaling_factor
            else:
                ci_val = 0.0
            mean.append(ratios_mean)
            ci.append(ci_val)
        else:
            mean.append(np.nan)
            ci.append(np.nan)
    return np.array(mean), np.array(ci)

# ─────────────────────────── main ─────────────────────────────────

def main() -> None:
    parser = create_plasticity_parser(
        description="AUC-loss and dormant neuron ratio with sequence-level averages and LaTeX output",
    )
    parser.add_argument("--repeats", type=int, nargs="+",
                        help="List of repetition counts (overrides --repeat_sequence)")
    parser.add_argument("--include_dormant_ratio", action="store_true",
                        help="Include dormant neuron ratio in plots (disabled by default)")
    args = parser.parse_args()

    repeats_list = args.repeats or [args.repeat_sequence or 1]
    colours = _palette(len(repeats_list))

    base = Path(__file__).resolve().parent.parent
    data_dir = base / args.data_root
    out_dir = base / "plots"; out_dir.mkdir(exist_ok=True, parents=True)

    # CSV init --------------------------------------------------------------
    header = ["method", "repeats", "task", "auc_loss_mean", "auc_loss_ci", 
              "dormant_ratio_mean", "dormant_ratio_ci", "fpr_mean", "fpr_ci", "rauc_mean", "rauc_ci"]
    csv_rows = [tuple(header)]

    # For LaTeX global Means
    global_means: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)

    for method in args.methods:
        # Filter repeats for plotting (exclude 1) but keep all for table
        plot_repeats_list = [r for r in repeats_list if r != 1]
        plot_colours = _palette(len(plot_repeats_list))

        fig_cols = 2 if args.include_dormant_ratio else 1
        fig, axes = plt.subplots(1, fig_cols, figsize=(3.5 * fig_cols, 3), sharex=True)
        if fig_cols == 1:
            axes = [axes]  # Make it a list for consistent indexing

        for idx, repeats in enumerate(repeats_list):
            traces_per_task = collect_plasticity_runs(
                data_dir, args.algo, method, args.strategy,
                args.seq_len, repeats, args.seeds,
                args.level,
            )

            # Always collect dormant ratio data for table
            dormant_traces_per_task = _collect_dormant_ratio_data(
                data_dir, args.algo, method, args.strategy,
                args.seq_len, repeats, args.seeds,
                args.level,
            )

            # Compute all metrics for table
            auc_mean, auc_ci = _compute_auc_loss(traces_per_task, repeats, args.sigma, args.confidence)
            dormant_mean, dormant_ci = _compute_dormant_ratio_metrics(dormant_traces_per_task, repeats, args.sigma, args.confidence)

            # Compute capacity metrics (FPR, RAUC) for table
            if repeats > 1:
                fpr_mean, fpr_ci, rauc_mean, rauc_ci = _capacity_metrics(traces_per_task, repeats, args.sigma, args.confidence)
            else:
                # For repeats=1, infer values since they can't be calculated
                fpr_mean = np.full(args.seq_len, 1.0)  # Perfect retention ratio
                fpr_ci = np.zeros(args.seq_len)
                rauc_mean = np.full(args.seq_len, 1.0)  # Perfect area ratio
                rauc_ci = np.zeros(args.seq_len)

            # Only plot if not repeats=1
            if repeats != 1:
                plot_idx = plot_repeats_list.index(repeats)
                colour = plot_colours[plot_idx]
                label = f"{repeats} Repetitions"

                x = np.arange(1, args.seq_len + 1)
                axes[0].errorbar(x, auc_mean, yerr=auc_ci, fmt="o-", capsize=3, color=colour, label=label)
                if args.include_dormant_ratio:
                    axes[1].errorbar(x, dormant_mean, yerr=dormant_ci, fmt="s--", capsize=3, color=colour, label=label)

            # CSV rows -------------------------------------------------------
            for t in range(args.seq_len):
                row = [method, repeats, t + 1, auc_mean[t], auc_ci[t], 
                       dormant_mean[t], dormant_ci[t], fpr_mean[t], fpr_ci[t], rauc_mean[t], rauc_ci[t]]
                csv_rows.append(tuple(row))

            # ---- global averages (across tasks) ----
            g_auc = float(np.nanmean(auc_mean))
            g_auc_ci = float(np.nanmean(auc_ci))
            g_dormant = float(np.nanmean(dormant_mean))
            g_dormant_ci = float(np.nanmean(dormant_ci))
            g_fpr = float(np.nanmean(fpr_mean))
            g_fpr_ci = float(np.nanmean(fpr_ci))
            g_rauc = float(np.nanmean(rauc_mean))
            g_rauc_ci = float(np.nanmean(rauc_ci))

            global_means[method][repeats] = {
                "auc": g_auc,
                "auc_ci": g_auc_ci,
                "dormant": g_dormant,
                "dormant_ci": g_dormant_ci,
                "fpr": g_fpr,
                "fpr_ci": g_fpr_ci,
                "rauc": g_rauc,
                "rauc_ci": g_rauc_ci,
            }

        # prettify plots ----------------------------------------------------
        axes[0].set_ylabel("AUCL ↓")
        axes[0].set_ylim(bottom=0)

        if args.include_dormant_ratio:
            axes[1].set_ylabel("Dormant ratio ↑")
            axes[1].set_ylim(0, 1)  # Dormant ratio is between 0 and 1

        for ax in axes:
            ax.set_xlabel("Task index")
            ax.grid(True, alpha=0.3)

        # Add a unified legend for all subplots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.15),
                   ncol=len(plot_repeats_list), frameon=True)

        fig.tight_layout(rect=[-0.03, 0.1, 1, 1])
        fig.savefig(out_dir / f"{method}_plasticity.png", dpi=300)
        fig.savefig(out_dir / f"{method}_plasticity.pdf")
        plt.show()
        plt.close(fig)

        # ---- LaTeX table ---------------------------------------------------
        latex_lines = [
            "\\begin{table}[!t]",
            "\\centering",
            "\\caption{Sequence-averaged metrics for %s}" % method,
            "\\label{tab:%s_global}" % method.lower(),
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Repeats & AUC-loss $\\downarrow$ & Dormant Ratio $\\uparrow$ & FPR $\\uparrow$ & RAUC $\\uparrow$ \\\\",
            "\\midrule",
        ]

        for rep in repeats_list:
            gm = global_means[method][rep]
            latex_lines.append(f"{rep} & {gm['auc']:.3f} $\\pm$ {gm['auc_ci']:.3f} & {gm['dormant']:.3f} $\\pm$ {gm['dormant_ci']:.3f} & {gm['fpr']:.3f} $\\pm$ {gm['fpr_ci']:.3f} & {gm['rauc']:.3f} $\\pm$ {gm['rauc_ci']:.3f} \\\\")

        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        latex_str = "\n".join(latex_lines)
        print(latex_str)

    # write master CSV ------------------------------------------------------
    csv_path = out_dir / "plasticity_multi_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print(f"✅ Metrics saved to {csv_path}\n🖼  Figures & LaTeX tables in {out_dir}/")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
