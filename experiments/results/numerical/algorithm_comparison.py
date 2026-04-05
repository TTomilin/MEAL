from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Type alias for confidence intervals
ConfInt = tuple[float, float]


def load_series(fp: Path) -> List[float]:
    """Load a JSON series from file."""
    with open(fp, "r") as f:
        data = json.load(f)
    return [float(x) for x in data]


def _mean_ci(series: List[float]) -> ConfInt:
    """Compute mean and 95% confidence interval."""
    if not series:
        return np.nan, np.nan
    mean = float(np.mean(series))
    if len(series) == 1:
        return mean, 0.0
    ci = 1.96 * np.std(series, ddof=1) / np.sqrt(len(series))
    return mean, float(ci)


def compute_metrics(
    data_root: Path,
    algo: str,
    method: str,
    strategy: str,
    seq_len: int,
    seeds: List[int],
    level: int,
    agents: int,
    end_window_evals: int = 10,
) -> dict:
    """Compute AP and F for a single (algo, method, level) combination."""
    AP_seeds, F_seeds = [], []

    base_folder = (
        data_root
        / algo
        / method
        / f"level_{level}"
        / f"agents_{agents}"
        / f"{strategy}_{seq_len}"
    )

    for seed in seeds:
        sd = base_folder / f"seed_{seed}"
        if not sd.exists():
            print(f"[debug] missing: {sd}")
            continue

        env_files = sorted([f for f in sd.glob("*_soup.*") if "training" not in f.name])
        if len(env_files) != seq_len:
            print(
                f"[warn] expected {seq_len} env files, found {len(env_files)} "
                f"for {algo}/{method}/level_{level}/seed_{seed}"
            )
            continue

        env_series = [load_series(f) for f in env_files]
        L = max(len(s) for s in env_series)
        env_mat = np.vstack([
            np.pad(s, (0, L - len(s)), constant_values=s[-1]) for s in env_series
        ])

        # Average Performance – mean of final value per environment
        AP_seeds.append(float(env_mat.mean(axis=0)[-1]))

        # Forgetting – drop from best-ever to final performance
        f_vals = []
        final_idx = env_mat.shape[1] - 1
        fw_start = max(0, final_idx - end_window_evals + 1)
        for i in range(seq_len):
            final_avg = float(np.nanmean(env_mat[i, fw_start: final_idx + 1]))
            best_perf = float(np.nanmax(env_mat[i, : final_idx + 1]))
            f_vals.append(max(best_perf - final_avg, 0.0))
        F_seeds.append(float(np.nanmean(f_vals)))

    A_mean, A_ci = _mean_ci(AP_seeds)
    F_mean, F_ci = _mean_ci(F_seeds)

    return {
        "AP": A_mean,
        "AP_CI": A_ci,
        "F": F_mean,
        "F_CI": F_ci,
    }


def _fmt(mean: float, ci: float, best: bool) -> str:
    """Format *mean ±CI* for LaTeX; bold the best value."""
    if np.isnan(mean) or np.isinf(mean):
        return "--"
    main = f"{mean:.3f}"
    if best:
        main = rf"\textbf{{{main}}}"
    ci_part = rf"{{\scriptsize$\pm{ci:.3f}$}}" if not np.isnan(ci) and ci > 0 else ""
    return main + ci_part


def build_table(
    data_root: Path,
    algorithms: List[str],
    methods: List[str],
    strategy: str,
    seq_len: int,
    seeds: List[int],
    level: int,
    agents: int,
    end_window_evals: int = 10,
) -> pd.DataFrame:
    """
    Build a DataFrame for one difficulty level.
    Rows = CL methods, columns = algorithm × {AP, F}.
    """
    # Collect raw values first so we can find column-wise bests
    raw: dict[str, dict[str, dict]] = {}
    for method in methods:
        raw[method] = {}
        for algo in algorithms:
            raw[method][algo] = compute_metrics(
                data_root=data_root,
                algo=algo,
                method=method,
                strategy=strategy,
                seq_len=seq_len,
                seeds=seeds,
                level=level,
                agents=agents,
                end_window_evals=end_window_evals,
            )

    # Find best AP (max) and best F (min) per algorithm column
    best_ap: dict[str, float] = {}
    best_f: dict[str, float] = {}
    for algo in algorithms:
        ap_vals = [raw[m][algo]["AP"] for m in methods if not np.isnan(raw[m][algo]["AP"])]
        f_vals  = [raw[m][algo]["F"]  for m in methods if not np.isnan(raw[m][algo]["F"])]
        best_ap[algo] = max(ap_vals) if ap_vals else np.nan
        best_f[algo]  = min(f_vals)  if f_vals  else np.nan

    rows = []
    for method in methods:
        row: dict[str, str] = {"Method": method}
        for algo in algorithms:
            m = raw[method][algo]
            row[f"{algo.upper()}_AP"] = _fmt(m["AP"], m["AP_CI"], m["AP"] == best_ap[algo])
            row[f"{algo.upper()}_F"]  = _fmt(m["F"],  m["F_CI"],  m["F"]  == best_f[algo])
        rows.append(row)

    return pd.DataFrame(rows)


def _latex_table(df: pd.DataFrame, algorithms: List[str], level: int) -> str:
    """Render the DataFrame as a LaTeX table with multi-column algorithm headers."""
    algo_uppers = [a.upper() for a in algorithms]
    n_algos = len(algo_uppers)

    # Build column format: l (method) + for each algo: two c columns
    col_fmt = "l" + "cc" * n_algos

    # Top header: one multicolumn per algorithm spanning AP and F
    top_header_cells = [""]
    for au in algo_uppers:
        top_header_cells.append(rf"\multicolumn{{2}}{{c}}{{{au}}}")
    top_header = " & ".join(top_header_cells) + r" \\"

    # Cmidrules under each algorithm header (1-indexed, skip col 0 = Method)
    cmidrules = []
    for i, _ in enumerate(algo_uppers):
        left  = 2 + i * 2
        right = left + 1
        cmidrules.append(rf"\cmidrule(lr){{{left}-{right}}}")
    cmidrule_line = " ".join(cmidrules)

    # Sub-header: Method | AP↑ F↓ | AP↑ F↓ | ...
    sub_cells = ["Method"]
    for _ in algo_uppers:
        sub_cells += [r"$\mathcal{A}\!\uparrow$", r"$\mathcal{F}\!\downarrow$"]
    sub_header = " & ".join(sub_cells) + r" \\"

    # Data rows
    data_lines = []
    for _, row in df.iterrows():
        cells = [row["Method"]]
        for au in algo_uppers:
            cells.append(row[f"{au}_AP"])
            cells.append(row[f"{au}_F"])
        data_lines.append("    " + " & ".join(cells) + r" \\")

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        rf"  \caption{{Algorithm comparison — Level {level}. "
        r"$\mathcal{A}$: Average Performance (↑); "
        r"$\mathcal{F}$: Forgetting (↓). "
        r"Bold = best per column.}",
        rf"  \label{{tab:algo_comparison_level{level}}}",
        rf"  \begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        f"    {top_header}",
        f"    {cmidrule_line}",
        f"    {sub_header}",
        r"    \midrule",
    ] + data_lines + [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare MARL algorithms × CL methods. One table per difficulty level."
    )
    p.add_argument("--data_root", default="results/data", help="Root directory containing the data")
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["ippo", "mappo", "happo", "vdn", "qmix"],
        help="MARL algorithms to compare (columns)",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["FT", "L2", "EWC", "MAS", "Online_EWC", "Online_MAS", "packnet", "AGEM", "ER_ACE"],
        help="CL methods to compare (rows)",
    )
    p.add_argument("--strategy", default="generate", help="Strategy name")
    p.add_argument("--seq_len", type=int, default=20, help="Sequence length")
    p.add_argument("--num_agents", type=int, default=2, help="Number of agents")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Seeds to include")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3], help="Difficulty levels")
    p.add_argument(
        "--end_window_evals",
        type=int,
        default=10,
        help="Number of final eval points to average for Forgetting",
    )
    args = p.parse_args()

    print(f"Algorithms : {args.algorithms}")
    print(f"CL methods : {args.methods}")
    print(f"Strategy   : {args.strategy}")
    print(f"Seq length : {args.seq_len}")
    print(f"Seeds      : {args.seeds}")
    print(f"Levels     : {args.levels}")

    for level in args.levels:
        print(f"\n{'=' * 70}")
        print(f"  LEVEL {level}")
        print(f"{'=' * 70}")

        df = build_table(
            data_root=Path(args.data_root),
            algorithms=args.algorithms,
            methods=args.methods,
            strategy=args.strategy,
            seq_len=args.seq_len,
            seeds=args.seeds,
            level=level,
            agents=args.num_agents,
            end_window_evals=args.end_window_evals,
        )

        print(df.to_string(index=False))
        print()
        print(_latex_table(df, args.algorithms, level))
