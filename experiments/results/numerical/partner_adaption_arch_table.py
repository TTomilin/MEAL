#!/usr/bin/env python3
"""LaTeX table comparing partner-adaptation CL methods across architectures.

Rows  : CL methods (default: FT, Online_MAS, Online_EWC)
Columns: for each layout — Multi-head A↑ F↓ | Single-head A↑ F↓

Data path: data/ppo/<method>/<layout_name>/<arch>/partners_<N>/seed_<seed>/

Usage
-----
# All four layouts (default):
python partner_adaption_arch_table.py

# Specific layouts:
python partner_adaption_arch_table.py --layout_names cramped_room coord_ring
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments.results.plotting.utils import METHOD_DISPLAY_NAMES

# -----------------------------------------------------------------------------
# Config defaults
# -----------------------------------------------------------------------------

DEFAULT_LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "counter_circuit"]
DEFAULT_METHODS      = ["FT", "Online_MAS", "Online_EWC"]
DEFAULT_PARTNERS     = 8
DEFAULT_SEEDS        = [1, 2, 3, 4, 5]
ARCHS                = ["multihead", "singlehead"]

LAYOUT_DISPLAY: Dict[str, str] = {
    "cramped_room":      "Cramped Room",
    "asymm_advantages":  "Asymm. Advantages",
    "coord_ring":        "Coordination Ring",
    "counter_circuit":   "Counter Circuit",
}

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def _load(fp: Path) -> np.ndarray:
    if fp.suffix == ".json":
        return np.array(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == ".npz":
        return np.load(fp)["data"].astype(float)
    raise ValueError(fp)


def _mean_ci(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return np.nan, np.nan
    m = float(np.mean(vals))
    ci = 0.0 if len(vals) == 1 else 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
    return m, ci


# -----------------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------------

def _compute_seed(sd: Path, num_partners: int, end_window: int) -> Tuple[float, float]:
    """Return (AP, F) for one seed directory."""
    partner_files = sorted(sd.glob("eval_partner_*_soup.*"))
    if not partner_files:
        return np.nan, np.nan

    curves = []
    for fp in partner_files:
        arr = _load(fp)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        curves.append(arr)

    L = max(len(c) for c in curves)
    mat = np.vstack([np.pad(c, (0, L - len(c)), constant_values=c[-1]) for c in curves])
    # shape: (num_partners, T)

    # Average Performance — mean across partners at the final time step
    ap = float(np.nanmean(mat[:, -1]))

    # Forgetting — per partner: drop from best-ever to window-averaged final
    fw = max(0, L - end_window)
    f_vals = [max(float(np.nanmax(mat[i])) - float(np.nanmean(mat[i, fw:])), 0.0)
              for i in range(mat.shape[0])]
    f = float(np.nanmean(f_vals))

    return ap, f


def compute_metrics(
    data_root: Path,
    layout_name: str,
    methods: List[str],
    num_partners: int,
    seeds: List[int],
    end_window: int,
) -> pd.DataFrame:
    """Return DataFrame with columns: Method, {arch}_AP, {arch}_AP_ci, {arch}_F, {arch}_F_ci."""
    rows = []
    for method in methods:
        row: dict = {"Method": method}
        for arch in ARCHS:
            folder = data_root / "ppo" / method / layout_name / arch / f"partners_{num_partners}"
            ap_seeds, f_seeds = [], []
            for seed in seeds:
                sd = folder / f"seed_{seed}"
                if not sd.exists():
                    print(f"[warn] missing {sd}")
                    continue
                ap, f = _compute_seed(sd, num_partners, end_window)
                if not np.isnan(ap):
                    ap_seeds.append(ap)
                if not np.isnan(f):
                    f_seeds.append(f)
            row[f"{arch}_AP"],    row[f"{arch}_AP_ci"]  = _mean_ci(ap_seeds)
            row[f"{arch}_F"],     row[f"{arch}_F_ci"]   = _mean_ci(f_seeds)
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# LaTeX formatting
# -----------------------------------------------------------------------------

def _fmt(mean: float, ci: float, bold: bool, show_ci: bool) -> str:
    if np.isnan(mean):
        return "--"
    s = f"{mean:.3f}"
    if bold:
        s = rf"\textbf{{{s}}}"
    if show_ci and not np.isnan(ci) and ci > 0:
        s += rf"{{\scriptsize$\pm{ci:.3f}$}}"
    return s


def _display_method(m: str) -> str:
    return METHOD_DISPLAY_NAMES.get(m, m)


def build_table(df: pd.DataFrame, show_ci: bool) -> str:
    """Single-layout table: 2-level header (arch group → metric)."""
    best: dict = {}
    for arch in ARCHS:
        best[f"{arch}_AP"] = df[f"{arch}_AP"].max()
        best[f"{arch}_F"]  = df[f"{arch}_F"].min()

    df_out = pd.DataFrame()
    df_out["Method"] = df["Method"].map(_display_method)
    for arch in ARCHS:
        df_out[f"{arch}_AP"] = df.apply(
            lambda r, a=arch: _fmt(r[f"{a}_AP"], r[f"{a}_AP_ci"],
                                   r[f"{a}_AP"] == best[f"{a}_AP"], show_ci), axis=1)
        df_out[f"{arch}_F"] = df.apply(
            lambda r, a=arch: _fmt(r[f"{a}_F"], r[f"{a}_F_ci"],
                                   r[f"{a}_F"] == best[f"{a}_F"], show_ci), axis=1)

    df_out.columns = ["Method", "mh_AP", "mh_F", "sh_AP", "sh_F"]
    latex = df_out.to_latex(index=False, escape=False, column_format="lcccc")

    header = (
        r"Method & \multicolumn{2}{c}{Multi-head} & \multicolumn{2}{c}{Single-head} \\"
        "\n"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
        "\n"
        r" & $\mathcal{A}\!\uparrow$ & $\mathcal{F}\!\downarrow$"
        r" & $\mathcal{A}\!\uparrow$ & $\mathcal{F}\!\downarrow$ \\"
    )
    lines = latex.split("\n")
    for i, line in enumerate(lines):
        if "mh_AP" in line:
            lines[i] = header
            break
    return "\n".join(lines)


def build_vertical_table(
    layout_dfs: Dict[str, pd.DataFrame],
    layout_names: List[str],
    show_ci: bool,
) -> str:
    """Multi-layout table: layouts as row groups, arch×metric as columns (5 cols total)."""
    A = r"$\mathcal{A}\!\uparrow$"
    F = r"$\mathcal{F}\!\downarrow$"
    num_cols = 5

    header = (
        r"Method & \multicolumn{2}{c}{Multi-head} & \multicolumn{2}{c}{Single-head} \\"
        "\n"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
        "\n"
        rf" & {A} & {F} & {A} & {F} \\"
    )

    body_lines = []
    for layout in layout_names:
        df = layout_dfs[layout]
        display = LAYOUT_DISPLAY.get(layout, layout.replace("_", " ").title())
        best_mh_ap = df["multihead_AP"].max()
        best_mh_f  = df["multihead_F"].min()
        best_sh_ap = df["singlehead_AP"].max()
        best_sh_f  = df["singlehead_F"].min()

        body_lines.append(r"\midrule")
        body_lines.append(rf"\multicolumn{{{num_cols}}}{{c}}{{\textit{{{display}}}}} \\")
        body_lines.append(r"\midrule")

        for _, row in df.iterrows():
            method = _display_method(row["Method"])
            mh_ap = _fmt(row["multihead_AP"], row["multihead_AP_ci"], row["multihead_AP"] == best_mh_ap, show_ci)
            mh_f  = _fmt(row["multihead_F"],  row["multihead_F_ci"],  row["multihead_F"]  == best_mh_f,  show_ci)
            sh_ap = _fmt(row["singlehead_AP"], row["singlehead_AP_ci"], row["singlehead_AP"] == best_sh_ap, show_ci)
            sh_f  = _fmt(row["singlehead_F"],  row["singlehead_F_ci"],  row["singlehead_F"]  == best_sh_f,  show_ci)
            body_lines.append(rf"{method} & {mh_ap} & {mh_f} & {sh_ap} & {sh_f} \\")

    return (
        r"\begin{tabular}{lcccc}" + "\n"
        r"\toprule" + "\n"
        + header + "\n"
        + "\n".join(body_lines) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}"
    )


def build_wide_table(
    layout_dfs: Dict[str, pd.DataFrame],
    layout_names: List[str],
    show_ci: bool,
) -> str:
    """Multi-layout table: 3-level header (layout group → arch group → metric)."""
    methods = layout_dfs[layout_names[0]]["Method"].tolist()
    N = len(layout_names)

    # Merge all layout DataFrames into one wide DataFrame of raw floats
    merged = pd.DataFrame({"Method": methods})
    for layout in layout_names:
        df = layout_dfs[layout]
        for arch in ARCHS:
            for metric in ("AP", "F"):
                col = f"{arch}_{metric}"
                merged[f"{layout}__{col}"]     = df[col].values
                merged[f"{layout}__{col}_ci"]  = df[f"{col}_ci"].values

    # Identify column-wise bests
    best: dict = {}
    for layout in layout_names:
        best[f"{layout}__multihead_AP"]  = merged[f"{layout}__multihead_AP"].max()
        best[f"{layout}__singlehead_AP"] = merged[f"{layout}__singlehead_AP"].max()
        best[f"{layout}__multihead_F"]   = merged[f"{layout}__multihead_F"].min()
        best[f"{layout}__singlehead_F"]  = merged[f"{layout}__singlehead_F"].min()

    # Build formatted cell columns (flat names, header patched later)
    flat_cols = []
    df_out = pd.DataFrame()
    df_out["Method"] = merged["Method"].map(_display_method)
    for layout in layout_names:
        for arch in ARCHS:
            for metric, direction in [("AP", "max"), ("F", "min")]:
                key = f"{layout}__{arch}_{metric}"
                ci_key = f"{key}_ci"
                bv = best[key]
                col_label = f"{layout}__{arch}__{metric}"
                df_out[col_label] = merged.apply(
                    lambda r, k=key, ck=ci_key, b=bv: _fmt(
                        r[k], r[ck],
                        not np.isnan(r[k]) and r[k] == b,
                        show_ci,
                    ), axis=1,
                )
                flat_cols.append(col_label)

    col_fmt = "l" + "c" * (4 * N)
    latex = df_out.to_latex(index=False, escape=False, column_format=col_fmt)

    # --- Build 3-level header ---
    # Level 1: layout groups (each spans 4 cols)
    lvl1_parts, cmidrule1 = [], []
    for k, layout in enumerate(layout_names):
        c_start, c_end = 2 + 4 * k, 5 + 4 * k
        display = LAYOUT_DISPLAY.get(layout, layout.replace("_", " ").title())
        lvl1_parts.append(rf"\multicolumn{{4}}{{c}}{{{display}}}")
        cmidrule1.append(rf"\cmidrule(lr){{{c_start}-{c_end}}}")

    # Level 2: arch groups within each layout (each spans 2 cols)
    lvl2_parts, cmidrule2 = [], []
    for k in range(N):
        c = 2 + 4 * k
        lvl2_parts += [r"\multicolumn{2}{c}{MH}", r"\multicolumn{2}{c}{SH}"]
        cmidrule2 += [rf"\cmidrule(lr){{{c}-{c+1}}}", rf"\cmidrule(lr){{{c+2}-{c+3}}}"]

    # Level 3: metric headers repeated per arch per layout
    A = r"$\mathcal{A}\!\uparrow$"
    F_sym = r"$\mathcal{F}\!\downarrow$"
    lvl3_parts = [A, F_sym] * (2 * N)

    header = (
        f"Method & {' & '.join(lvl1_parts)} \\\\\n"
        f"{' '.join(cmidrule1)}\n"
        f" & {' & '.join(lvl2_parts)} \\\\\n"
        f"{' '.join(cmidrule2)}\n"
        f"Method & {' & '.join(lvl3_parts)} \\\\"
    )

    # Patch the auto-generated header line
    lines = latex.split("\n")
    for i, line in enumerate(lines):
        if flat_cols[0] in line:
            lines[i] = header
            break
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Partner-adaptation arch comparison table (multi- vs single-head).",
    )
    p.add_argument("--data_root",    default="results/data")
    p.add_argument("--layout_names", nargs="+", default=DEFAULT_LAYOUT_NAMES,
                   help="One or more layout names; multiple layouts produce a wide table.")
    p.add_argument("--methods",      nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--num_partners", type=int,  default=DEFAULT_PARTNERS)
    p.add_argument("--seeds",        type=int,  nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--end_window",   type=int,  default=10,
                   help="Final eval points averaged when computing forgetting")
    p.add_argument("--no_ci", action="store_true", help="Suppress ±CI in cells")
    p.add_argument("--wide", action="store_true",
                   help="Use wide 3-level header layout instead of vertical row-group layout")
    args = p.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parents[2] / data_root

    layout_dfs = {}
    for layout in args.layout_names:
        layout_dfs[layout] = compute_metrics(
            data_root=data_root,
            layout_name=layout,
            methods=args.methods,
            num_partners=args.num_partners,
            seeds=args.seeds,
            end_window=args.end_window,
        )

    show_ci = not args.no_ci
    if len(args.layout_names) == 1:
        print(build_table(layout_dfs[args.layout_names[0]], show_ci))
    elif args.wide:
        print(build_wide_table(layout_dfs, args.layout_names, show_ci))
    else:
        print(build_vertical_table(layout_dfs, args.layout_names, show_ci))
