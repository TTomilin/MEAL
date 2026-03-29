"""
Coordination forgetting table for the rebuttal.

For every focal task i we measure:
  - het_peak  : mean heterogeneity in the last `window` eval steps while task i
                was being trained (right after specialisation forms)
  - het_final : mean heterogeneity in the last `window` eval steps of the whole
                sequence (after all subsequent tasks have been trained)
  - CF_i = max(0, het_peak - het_final) / (het_peak + eps)   [coordination forgetting]

  Same formula applied to soup gives PF_i (performance forgetting).

  Lead_i = CF_i - PF_i   (positive → het degrades more/earlier than performance)

All three are averaged over tasks, then over seeds, then reported as mean ± 95% CI.

Output: one LaTeX table with rows = methods, columns = levels.
        Sub-columns per level: CF  |  PF  |  Lead

Usage:
    python experiments/results/numerical/coordination_forgetting_table.py \\
        --data_root experiments/results/data \\
        --algo ippo \\
        --methods EWC Online_EWC MAS L2 FT \\
        --levels 1 2 3 \\
        --agents 2 --strategy generate --seq_len 20 \\
        --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from experiments.results.plotting.utils import METHOD_DISPLAY_NAMES


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_series(fp: Path) -> Optional[np.ndarray]:
    if not fp.exists():
        return None
    try:
        if fp.suffix == ".json":
            return np.array(json.loads(fp.read_text()), dtype=float)
        if fp.suffix == ".npz":
            return np.load(fp)["data"].astype(float)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def _tail_mean(series: np.ndarray, window: int) -> float:
    """Mean of the last `window` finite values."""
    s = series[np.isfinite(series)]
    if len(s) == 0:
        return np.nan
    return float(np.mean(s[-window:]))


def _task_end_idx(series_len: int, task_idx: int, seq_len: int) -> int:
    """Index in the eval series corresponding to the end of task_idx's training."""
    chunk = series_len / seq_len
    return min(int(round((task_idx + 1) * chunk)), series_len - 1)


def compute_forgetting(series: np.ndarray, task_idx: int, seq_len: int,
                       window: int) -> float:
    """
    Compute normalised forgetting for one metric series of one focal task.

    peak  = tail mean of series[:end_idx]
    final = tail mean of series[end_idx:]
    forgetting = max(0, peak - final) / (peak + eps)
    """
    end = _task_end_idx(len(series), task_idx, seq_len)
    before = series[:end]
    after  = series[end:]

    peak  = _tail_mean(before, window)
    final = _tail_mean(after,  window)

    if np.isnan(peak) or np.isnan(final) or peak < 1e-8:
        return np.nan
    return float(np.clip((peak - final) / peak, 0.0, 1.0))


def seed_metrics(
    sd: Path,
    seq_len: int,
    window: int,
) -> Tuple[float, float, float]:
    """
    For one seed directory, return (mean_CF, mean_PF, mean_Lead) averaged
    across all tasks for which both soup and het files exist.
    """
    cf_vals, pf_vals = [], []

    for task_idx in range(seq_len):
        het_s  = load_series(sd / f"{task_idx}_heterogeneity.json")
        soup_s = load_series(sd / f"{task_idx}_soup.json")

        if het_s is None or soup_s is None:
            continue
        # Align lengths by truncating to the shorter one
        n = min(len(het_s), len(soup_s))
        if n < 2:
            continue
        het_s  = het_s[:n]
        soup_s = soup_s[:n]

        cf = compute_forgetting(het_s,  task_idx, seq_len, window)
        pf = compute_forgetting(soup_s, task_idx, seq_len, window)
        if not np.isnan(cf) and not np.isnan(pf):
            cf_vals.append(cf)
            pf_vals.append(pf)

    if not cf_vals:
        return np.nan, np.nan, np.nan
    cf_mean   = float(np.mean(cf_vals))
    pf_mean   = float(np.mean(pf_vals))
    lead_mean = cf_mean - pf_mean
    return cf_mean, pf_mean, lead_mean


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def mean_ci(vals: List[float]) -> Tuple[float, float]:
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return np.nan, np.nan
    m = float(np.mean(vals))
    ci = 0.0 if len(vals) == 1 else 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))
    return m, float(ci)


def compute_method_level(
    data_root: Path,
    algo: str,
    method: str,
    level: int,
    agents: int,
    strategy: str,
    seq_len: int,
    seeds: List[int],
    window: int,
) -> dict:
    base = (
        data_root / algo / method
        / f"level_{level}" / f"agents_{agents}"
        / f"{strategy}_{seq_len}"
    )
    cf_seeds, pf_seeds, lead_seeds = [], [], []
    for seed in seeds:
        sd = base / f"seed_{seed}"
        if not sd.exists():
            continue
        cf, pf, lead = seed_metrics(sd, seq_len, window)
        if not np.isnan(cf):
            cf_seeds.append(cf)
            pf_seeds.append(pf)
            lead_seeds.append(lead)

    return {
        "CF":   mean_ci(cf_seeds),
        "PF":   mean_ci(pf_seeds),
        "Lead": mean_ci(lead_seeds),
        "n":    len(cf_seeds),
    }


# ---------------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------------

def build_table(
    data_root: Path,
    algo: str,
    methods: List[str],
    levels: List[int],
    agents: int,
    strategy: str,
    seq_len: int,
    seeds: List[int],
    window: int,
) -> pd.DataFrame:
    rows = []
    for method in methods:
        row = {"Method": METHOD_DISPLAY_NAMES.get(method, method)}
        for level in levels:
            res = compute_method_level(
                data_root, algo, method, level, agents,
                strategy, seq_len, seeds, window,
            )
            for key in ("CF", "PF", "Lead"):
                m, ci = res[key]
                row[f"L{level}_{key}_mean"] = m
                row[f"L{level}_{key}_ci"]   = ci
            row[f"L{level}_n"] = res["n"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(mean: float, ci: float, bold: bool, show_ci: bool) -> str:
    if np.isnan(mean):
        return "--"
    s = f"{mean:.3f}"
    if bold:
        s = rf"\textbf{{{s}}}"
    if show_ci and ci > 0:
        s += rf"{{\scriptsize$\pm{ci:.3f}$}}"
    return s


def print_console(df: pd.DataFrame, levels: List[int]) -> None:
    cols = ["Method"]
    rename = {}
    for level in levels:
        for key in ("CF", "PF", "Lead"):
            col = f"L{level}_{key}_mean"
            cols.append(col)
            rename[col] = f"L{level} {key}"
    print(df[cols].rename(columns=rename).to_string(index=False, float_format="{:.3f}".format))


def print_latex(df: pd.DataFrame, levels: List[int], show_ci: bool) -> None:
    sub_keys = ["CF", "PF", "Lead"]
    n_sub    = len(sub_keys)
    n_lev    = len(levels)
    col_fmt  = "l" + "c" * (n_lev * n_sub)

    # Per (level, key) best value
    # CF/PF: lowest is best;  Lead: highest is best (shows biggest early-warning gap)
    best: dict[tuple, float] = {}
    for level in levels:
        for key in sub_keys:
            col = f"L{level}_{key}_mean"
            vals = df[col].dropna()
            if vals.empty:
                continue
            best[(level, key)] = vals.min() if key in ("CF", "PF") else vals.max()

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        (
            r"\caption{Coordination forgetting (CF), performance forgetting (PF), "
            r"and their gap (Lead $=$ CF $-$ PF) across difficulty levels. "
            r"CF and PF measure normalised post-training degradation of heterogeneity "
            r"and soup score respectively; Lead $> 0$ means role specialisation "
            r"degrades more than performance --- an early-warning signal invisible "
            r"to standard forgetting metrics. "
            r"Bold: best value per column (lowest for CF/PF, highest for Lead).}"
        ),
        r"\label{tab:coord_forgetting}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
    ]

    # Two-row header
    mc_parts = [r"\multirow{2}{*}[-0.7ex]{Method}"]
    for level in levels:
        mc_parts.append(
            rf"\multicolumn{{{n_sub}}}{{c}}{{Level {level}}}"
        )
    lines.append(" & ".join(mc_parts) + r" \\")
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + i * n_sub}-{1 + (i + 1) * n_sub}}}"
        for i in range(n_lev)
    )
    lines.append(cmidrules)
    sub_header = " & ".join(
        r"\textbf{" + k + r"}" for k in sub_keys
    )
    lines.append(r"& " + " & ".join([sub_header] * n_lev) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        cells = [str(row["Method"])]
        for level in levels:
            for key in sub_keys:
                mean = float(row[f"L{level}_{key}_mean"])
                ci   = float(row[f"L{level}_{key}_ci"])
                is_best = (
                    not np.isnan(mean)
                    and np.isclose(mean, best.get((level, key), np.nan))
                )
                cells.append(_fmt(mean, ci, bold=is_best, show_ci=show_ci))
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Coordination forgetting table")
    p.add_argument("--data_root", required=True)
    p.add_argument("--algo",     default="ippo")
    p.add_argument("--methods",  nargs="+", required=True)
    p.add_argument("--levels",   type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--agents",   type=int, default=2)
    p.add_argument("--strategy", default="generate")
    p.add_argument("--seq_len",  type=int, default=20)
    p.add_argument("--seeds",    type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--window",   type=int, default=10,
                   help="Number of tail eval points used to estimate peak / final values")
    p.add_argument("--latex",    action="store_true")
    p.add_argument("--no_ci",    action="store_true")
    args = p.parse_args()

    df = build_table(
        data_root=Path(args.data_root),
        algo=args.algo,
        methods=args.methods,
        levels=args.levels,
        agents=args.agents,
        strategy=args.strategy,
        seq_len=args.seq_len,
        seeds=args.seeds,
        window=args.window,
    )

    print("\nCoordination Forgetting (CF)  |  Performance Forgetting (PF)  |  Lead = CF - PF")
    print("=" * 80)
    print_console(df, args.levels)

    if args.latex:
        print("\nLaTeX:\n" + "-" * 40)
        print_latex(df, args.levels, show_ci=not args.no_ci)
