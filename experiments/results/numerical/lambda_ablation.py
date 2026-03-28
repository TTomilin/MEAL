"""
Ablation study over the λ parameter in the curve-based forgetting metric (Eq. 2).

λ controls how strongly early forgetting is penalised via the exponential weight
exp(-λ · t/T).  λ=0 gives uniform weighting; larger λ penalises early forgetting
more heavily.

Produces a LaTeX table: rows = CL methods, columns = λ values, one sub-table per
difficulty level.

Usage example:
    python experiments/results/numerical/lambda_ablation.py \
        --data_root data \
        --algo ippo \
        --methods EWC Online_EWC MAS L2 FT \
        --levels 1 2 3 \
        --lambdas 0.0 0.5 1.0 2.0 4.0 8.0 \
        --seq_len 20 \
        --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiments.results.numerical.results_table import compute_metrics
from experiments.results.plotting.utils import METHOD_DISPLAY_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(mean: float, ci: float, bold: bool, show_ci: bool = True) -> str:
    if np.isnan(mean):
        return "--"
    s = f"{mean:.3f}"
    if bold:
        s = rf"\textbf{{{s}}}"
    if show_ci and not np.isnan(ci) and ci > 0:
        s += rf"{{\scriptsize$\pm{ci:.3f}$}}"
    return s


def compute_forgetting_table(
    data_root: Path,
    algo: str,
    methods: List[str],
    strategy: str,
    seq_len: int,
    seeds: List[int],
    level: int,
    lambdas: List[float],
    agents: int,
) -> pd.DataFrame:
    """Return a DataFrame indexed by method with one (mean, ci) column per λ."""
    rows = []
    for method in methods:
        row = {"Method": METHOD_DISPLAY_NAMES.get(method, method)}
        for lam in lambdas:
            df = compute_metrics(
                data_root=data_root,
                algo=algo,
                methods=[method],
                strategy=strategy,
                seq_len=seq_len,
                seeds=seeds,
                level=level,
                agents=agents,
                lambda_decay=lam,
            )
            if df.empty:
                row[f"lam_{lam}"] = (np.nan, np.nan)
                continue
            r = df.iloc[0]
            row[f"lam_{lam}_mean"] = r["Forgetting"]
            row[f"lam_{lam}_ci"]   = r["Forgetting_CI"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ablation of the λ parameter in the forgetting metric")
    p.add_argument("--data_root", required=True)
    p.add_argument("--algo", default="ippo")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--strategy", default="generate")
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--confidence_intervals", action="store_true", default=True)
    p.add_argument("--no-confidence-intervals", dest="confidence_intervals", action="store_false")
    args = p.parse_args()

    data_root = Path(args.data_root)
    lambdas   = args.lambdas
    levels    = args.levels

    # ── Build one DataFrame per level ───────────────────────────────────────
    level_dfs: dict[int, pd.DataFrame] = {}
    for level in levels:
        level_dfs[level] = compute_forgetting_table(
            data_root=data_root,
            algo=args.algo,
            methods=args.methods,
            strategy=args.strategy,
            seq_len=args.seq_len,
            seeds=args.seeds,
            level=level,
            lambdas=lambdas,
            agents=args.agents,
        )

    # ── Console output ───────────────────────────────────────────────────────
    for level, df in level_dfs.items():
        print(f"\nLevel {level}  —  Forgetting  F(λ)")
        print("=" * 70)
        display_cols = ["Method"] + [f"lam_{lam}_mean" for lam in lambdas]
        rename = {f"lam_{lam}_mean": f"λ={lam}" for lam in lambdas}
        print(df[display_cols].rename(columns=rename).to_string(index=False, float_format="{:.3f}".format))

    # ── LaTeX table ──────────────────────────────────────────────────────────
    n_lam = len(lambdas)
    n_levels = len(levels)
    col_fmt = "l" + "c" * (n_lam * n_levels)

    lam_header = " & ".join(
        rf"$\lambda={lam:.1f}$".rstrip("0").rstrip(".") if lam != int(lam)
        else rf"$\lambda={int(lam)}$"
        for lam in lambdas
    )

    latex_lines = [
        r"\begin{table}",
        r"\centering",
        (
            r"\caption{Ablation of the $\lambda$ parameter in the forgetting metric "
            r"$\mathcal{F}$ (Eq.~2). "
            r"Higher $\lambda$ penalises early forgetting more strongly. "
            r"Bold: lowest forgetting per level and $\lambda$.}"
        ),
        r"\label{tab:lambda_ablation}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
    ]

    # Two-row header: level multicolumns + λ symbols
    if n_levels > 1:
        mc_header = r"\multirow{2}{*}[-0.7ex]{Method}"
        for i, level in enumerate(levels):
            sep = " &" if i < n_levels - 1 else r" \\"
            mc_header += rf" & \multicolumn{{{n_lam}}}{{c}}{{Level {level}}}{sep}"
        latex_lines.append(mc_header)

        cmidrules = " ".join(
            rf"\cmidrule(lr){{{2 + i * n_lam}-{1 + (i + 1) * n_lam}}}"
            for i in range(n_levels)
        )
        latex_lines.append(cmidrules)
        latex_lines.append(r"& " + " & ".join([lam_header] * n_levels) + r" \\")
    else:
        latex_lines.append(r"Method & " + lam_header + r" \\")

    latex_lines.append(r"\midrule")

    # Find per-(level, lambda) best (min forgetting) across methods
    best: dict[tuple, float] = {}
    for level, df in level_dfs.items():
        for lam in lambdas:
            col = f"lam_{lam}_mean"
            vals = df[col].dropna()
            if not vals.empty:
                best[(level, lam)] = vals.min()

    # Data rows — one row per method
    method_display = [METHOD_DISPLAY_NAMES.get(m, m) for m in args.methods]
    for method, display in zip(args.methods, method_display):
        cells = [display]
        for level, df in level_dfs.items():
            mrow = df[df["Method"] == display]
            for lam in lambdas:
                if mrow.empty:
                    cells.append("--")
                    continue
                mean = float(mrow[f"lam_{lam}_mean"].iloc[0])
                ci   = float(mrow[f"lam_{lam}_ci"].iloc[0])
                is_best = (not np.isnan(mean)) and np.isclose(mean, best.get((level, lam), np.nan))
                cells.append(_fmt(mean, ci, bold=is_best, show_ci=args.confidence_intervals))
        latex_lines.append(" & ".join(cells) + r" \\")

    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    print("\nLATEX TABLE:")
    print("-" * 40)
    print("\n".join(latex_lines))
