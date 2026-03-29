"""
Heterogeneity table — rows: CL methods, columns: difficulty levels.

Value: mean Jensen-Shannon divergence between agent action distributions,
averaged across all tasks and all evaluation checkpoints in the sequence,
then averaged across seeds (with 95 % CI).

Usage:
    python experiments/results/numerical/heterogeneity_table.py \
        --data_root data \
        --algo ippo \
        --methods EWC Online_EWC MAS L2 FT \
        --levels 1 2 3 \
        --seq_len 20 \
        --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiments.results.plotting.utils import METHOD_DISPLAY_NAMES


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_series(fp: Path) -> np.ndarray:
    if fp.suffix == ".json":
        return np.array(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == ".npz":
        return np.load(fp)["data"].astype(float)
    raise ValueError(fp)


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def mean_ci(values: List[float]):
    if not values:
        return np.nan, np.nan
    m = float(np.mean(values))
    if len(values) == 1:
        return m, 0.0
    ci = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
    return m, float(ci)


def load_method_het(
    data_root: Path,
    algo: str,
    method: str,
    level: int,
    agents: int,
    strategy: str,
    seq_len: int,
    seeds: List[int],
) -> List[float]:
    """Return one mean-heterogeneity value per seed."""
    base = (
        data_root / algo / method
        / f"level_{level}" / f"agents_{agents}"
        / f"{strategy}_{seq_len}"
    )
    seed_means = []
    for seed in seeds:
        sd = base / f"seed_{seed}"
        if not sd.exists():
            continue
        task_means = []
        for i in range(seq_len):
            fp = sd / f"{i}_heterogeneity.json"
            if fp.exists():
                series = load_series(fp)
                series = series[np.isfinite(series)]
                if len(series):
                    task_means.append(float(np.mean(series)))
        if task_means:
            seed_means.append(float(np.mean(task_means)))
    return seed_means


# ---------------------------------------------------------------------------
# Table building
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
) -> pd.DataFrame:
    rows = []
    for method in methods:
        row = {"Method": METHOD_DISPLAY_NAMES.get(method, method)}
        for level in levels:
            vals = load_method_het(
                data_root, algo, method, level, agents, strategy, seq_len, seeds
            )
            m, ci = mean_ci(vals)
            row[f"level_{level}_mean"] = m
            row[f"level_{level}_ci"]   = ci
            row[f"level_{level}_n"]    = len(vals)
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
    if show_ci and not np.isnan(ci) and ci > 0:
        s += rf"{{\scriptsize$\pm{ci:.3f}$}}"
    return s


def print_console(df: pd.DataFrame, levels: List[int]) -> None:
    level_labels = {l: f"Level {l}" for l in levels}
    display_cols = ["Method"] + [f"level_{l}_mean" for l in levels]
    rename = {f"level_{l}_mean": level_labels[l] for l in levels}
    print(df[display_cols].rename(columns=rename).to_string(index=False, float_format="{:.3f}".format))


def print_latex(
    df: pd.DataFrame,
    levels: List[int],
    show_ci: bool,
) -> None:
    n_lev = len(levels)
    col_fmt = "l" + "c" * n_lev

    # Per-level best (highest heterogeneity)
    best: dict[int, float] = {}
    for level in levels:
        col = f"level_{level}_mean"
        vals = df[col].dropna()
        if not vals.empty:
            best[level] = vals.max()

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        (
            r"\caption{Role-heterogeneity index (mean Jensen-Shannon divergence between "
            r"agent action distributions, averaged over all tasks and evaluation checkpoints). "
            r"0 = homogeneous agents, 1 = fully differentiated roles. "
            r"Bold: highest heterogeneity per level.}"
        ),
        r"\label{tab:heterogeneity}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        r"Method & " + " & ".join(f"Level {l}" for l in levels) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        cells = [str(row["Method"])]
        for level in levels:
            mean = float(row[f"level_{level}_mean"])
            ci   = float(row[f"level_{level}_ci"])
            is_best = (not np.isnan(mean)) and np.isclose(mean, best.get(level, np.nan))
            cells.append(_fmt(mean, ci, bold=is_best, show_ci=show_ci))
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Heterogeneity index table")
    p.add_argument("--data_root", required=True)
    p.add_argument("--algo", default="ippo")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--strategy", default="generate")
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--latex", action="store_true")
    p.add_argument("--no_ci", action="store_true")
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
    )

    print("\nHeterogeneity index (JSD, mean ± 95% CI across seeds)")
    print("=" * 60)
    print_console(df, args.levels)

    if args.latex:
        print("\nLaTeX:\n" + "-" * 40)
        print_latex(df, args.levels, show_ci=not args.no_ci)
