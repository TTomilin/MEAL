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
plots/MAS_global_metrics.tex   # <- include \input{…} in your paper
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

from results.plotting.utils import (
    collect_plasticity_runs,
    create_plasticity_parser,
)
from results.plotting.utils.common import CRIT


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
                ci_val = CRIT[conf] * losses_std / np.sqrt(n_samples)
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

# ─────────────────────────── main ─────────────────────────────────

def main() -> None:
    parser = create_plasticity_parser(
        description="AUC-loss with sequence-level averages and LaTeX output",
    )
    parser.add_argument("--repeats", type=int, nargs="+",
                        help="List of repetition counts (overrides --repeat_sequence)")
    parser.add_argument("--extra_capacity_stats", action="store_true",
                        help="Also compute Final-Perf and Raw-AUC ratios")
    args = parser.parse_args()

    repeats_list = args.repeats or [args.repeat_sequence or 1]
    colours = _palette(len(repeats_list))

    base = Path(__file__).resolve().parent.parent
    data_dir = base / args.data_root
    out_dir = base / "plots"; out_dir.mkdir(exist_ok=True, parents=True)

    # CSV init --------------------------------------------------------------
    header = ["method", "repeats", "task", "auc_loss_mean", "auc_loss_ci"]
    if args.extra_capacity_stats:
        header += ["fpr_mean", "fpr_ci", "rauc_mean", "rauc_ci"]
    csv_rows = [tuple(header)]

    # For LaTeX global Means
    global_means: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)

    for method in args.methods:
        fig_cols = 3 if args.extra_capacity_stats else 1
        fig, axes = plt.subplots(1, fig_cols, figsize=(3.5 * fig_cols, 3), sharex=True)
        if fig_cols == 1:
            axes = [axes]

        for idx, repeats in enumerate(repeats_list):
            traces_per_task = collect_plasticity_runs(
                data_dir, args.algo, method, args.strategy,
                args.seq_len, repeats, args.seeds,
                args.level,
            )
            colour = colours[idx]
            label = f"{repeats} Repetitions"

            auc_mean, auc_ci = _compute_auc_loss(traces_per_task, repeats, args.sigma, args.confidence)
            x = np.arange(1, args.seq_len + 1)
            axes[0].errorbar(x, auc_mean, yerr=auc_ci, fmt="o-", capsize=3, color=colour, label=label)

            # capacity stats ------------------------------------------------
            if args.extra_capacity_stats:
                fpr_m, fpr_ci, rauc_m, rauc_ci = _capacity_metrics(traces_per_task, repeats, args.sigma, args.confidence)
                axes[1].errorbar(x, fpr_m, yerr=fpr_ci, fmt="s--", capsize=3, color=colour, label=label)
                axes[2].errorbar(x, rauc_m, yerr=rauc_ci, fmt="^:", capsize=3, color=colour, label=label)

            # CSV rows -------------------------------------------------------
            for t in range(args.seq_len):
                row = [method, repeats, t + 1, auc_mean[t], auc_ci[t]]
                if args.extra_capacity_stats:
                    row.extend([fpr_m[t], fpr_ci[t], rauc_m[t], rauc_ci[t]])
                csv_rows.append(tuple(row))

            # ---- global averages (across tasks) ----
            g_auc = float(np.nanmean(auc_mean))
            g_auc_ci = float(np.nanmean(auc_ci))
            if args.extra_capacity_stats:
                g_fpr = float(np.nanmean(fpr_m))
                g_fpr_ci = float(np.nanmean(fpr_ci))
                g_rauc = float(np.nanmean(rauc_m))
                g_rauc_ci = float(np.nanmean(rauc_ci))
            else:
                g_fpr = g_rauc = float("nan")
                g_fpr_ci = g_rauc_ci = float("nan")
            global_means[method][repeats] = {
                "auc": g_auc,
                "auc_ci": g_auc_ci,
                "fpr": g_fpr,
                "fpr_ci": g_fpr_ci,
                "rauc": g_rauc,
                "rauc_ci": g_rauc_ci,
            }

        # prettify plots ----------------------------------------------------
        axes[0].set_ylabel("Capacity loss ↓")
        axes[0].set_title("AUC-loss")
        axes[0].set_ylim(bottom=0)
        if args.extra_capacity_stats:
            axes[1].set_ylabel("Final‑perf ratio ↑")
            axes[1].set_title("Plateau")
            axes[2].set_ylabel("Raw‑AUC ratio ↑")
            axes[2].set_title("Raw-AUC")
        for ax in axes:
            ax.set_xlabel("Task index")
            ax.grid(True, alpha=0.3)

        # Add a unified legend for all subplots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.15),
                   ncol=len(repeats_list), frameon=True)

        fig.tight_layout(rect=[0, 0.1, 1, 1])
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
            ]
        if args.extra_capacity_stats:
            latex_lines.append("\\begin{tabular}{lccc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Repeats & AUC-loss $\\downarrow$ & FPR $\\uparrow$ & RAUC $\\uparrow$ \\")
            latex_lines.append("\\midrule")
            for rep in repeats_list:
                gm = global_means[method][rep]
                latex_lines.append(f"{rep} & {gm['auc']:.3f} $\\pm$ {gm['auc_ci']:.3f} & {gm['fpr']:.3f} $\\pm$ {gm['fpr_ci']:.3f} & {gm['rauc']:.3f} $\\pm$ {gm['rauc_ci']:.3f} \\")
        else:
            latex_lines.append("\\begin{tabular}{lc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Repeats & AUC-loss $\\downarrow$ \\")
            latex_lines.append("\\midrule")
            for rep in repeats_list:
                gm = global_means[method][rep]
                latex_lines.append(f"{rep} & {gm['auc']:.3f} $\\pm$ {gm['auc_ci']:.3f} \\")
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
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
