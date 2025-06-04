#!/usr/bin/env python3
"""
Plot *forgetting* for the MARL continual-learning benchmark.

For each environment i in the task sequence:
    F_i = performance immediately after finishing env-i training
          minus performance in the last window of the run.

The script then averages F_i over all but the last environment
(to avoid the trivial zero for the final task) and plots a
bar-chart with confidence intervals.

Directory layout, filename conventions and the --metric switch
are identical to plot_avg.py.
"""

import argparse, json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import sem

# ---------- constants -------------------------------------------------
METHOD_COLORS = {
    'EWC': '#12939A', 'MAS': '#FF6E54', 'AGEM': '#FFA600',
    'L2':  '#58508D', 'PackNet': '#BC5090', 'ReDo': '#003F5C', 'CBP': '#2F4B7C'
}
Z95 = 1.96             # critical value for 95 % CI
# ---------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--algo',      required=True)
    p.add_argument('--methods',   nargs='+', required=True)
    p.add_argument('--strategy',  required=True)
    p.add_argument('--seq_len',   type=int,  required=True)
    p.add_argument('--steps_per_task', type=float, default=8e6)
    p.add_argument('--seeds',     type=int, nargs='+', default=[1,2,3,4,5])
    p.add_argument('--metric',    choices=['success', 'reward'], default='success')
    p.add_argument('--baseline_file',
                   default='practical_reward_baseline_results.yaml',
                   help='Only used for --metric success')
    p.add_argument('--plot_name', default='forgetting')
    return p.parse_args()


# ---------- helpers ---------------------------------------------------

def load_series(fp: Path) -> np.ndarray:
    if fp.suffix == '.json':
        return np.array(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == '.npz':
        return np.load(fp)['data'].astype(float)
    raise ValueError(f'Unsupported file suffix: {fp.suffix!s}')


def time_indices(arr_len: int, total_steps: float, start: float, end: float):
    """return boolean mask for indices whose virtual time ∈ [start, end)"""
    t = np.linspace(0, total_steps, arr_len, endpoint=False)
    return (t >= start) & (t < end)


def collect_env_series(base: Path, algo: str, method: str, strat: str,
                       seq_len: int, seeds: List[int], metric: str,
                       baselines: dict | None):
    """return list[length=seq_len] of arrays for each env, stacked over seeds"""

    folder = base / algo / method / f"{strat}_{seq_len}"
    env_names, per_seed = [], []

    pattern = f"*_reward.*"
    suffix  = f"_reward"

    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            continue
        files = sorted(sd.glob(pattern))
        if not files:
            continue

        if not env_names:
            env_names = [f.name.split('_', 1)[1].rsplit(suffix, 1)[0] for f in files]

        series = [load_series(f) for f in files]

        # success → baseline normalisation
        if metric == 'success':
            norm = []
            for nm, arr in zip(env_names, series):
                b = baselines.get(nm, {}).get('avg_rewards')
                norm.append(arr / b if (b and b != 0) else np.full_like(arr, np.nan))
            series = norm

        per_seed.append(series)

    if not per_seed:
        raise RuntimeError(f'No data for method {method}')

    # shape → (n_seeds, seq_len, variable-len)
    return env_names, per_seed


def forgetting_per_seed(env_series: List[np.ndarray], total_steps: float,
                        steps_per_task: float) -> float:
    """single scalar “average forgetting” for one seed"""

    seq_len = len(env_series)
    forget_values = []

    for i, arr in enumerate(env_series[:-1]):     # skip last env
        if arr.size == 0 or np.all(np.isnan(arr)):
            forget_values.append(np.nan)
            continue

        after_mask = time_indices(len(arr), total_steps,
                                  start=i*steps_per_task,
                                  end  =(i+1)*steps_per_task)
        final_mask = time_indices(len(arr), total_steps,
                                  start=total_steps-steps_per_task,
                                  end  =total_steps)

        after_val = np.nanmean(arr[after_mask])
        final_val = np.nanmean(arr[final_mask])
        forget_values.append(after_val - final_val)

    return np.nanmean(forget_values)


# ---------- main plotting routine ------------------------------------

def main():
    args = parse_args()

    data_root  = Path(__file__).resolve().parent.parent / args.data_root
    total_steps = args.seq_len * args.steps_per_task

    baselines = {}
    if args.metric == 'success':
        with open(Path(__file__).resolve().parent.parent.parent / args.baseline_file) as f:
            baselines = yaml.safe_load(f)

    means, cis = [], []

    for method in args.methods:
        env_names, per_seed = collect_env_series(data_root, args.algo, method,
                                                 args.strategy, args.seq_len,
                                                 args.seeds, args.metric,
                                                 baselines)

        fvals = [forgetting_per_seed(seed_series, total_steps,
                                     args.steps_per_task)
                 for seed_series in per_seed]

        mean = np.nanmean(fvals)
        ci   = Z95 * sem(fvals, nan_policy='omit')
        means.append(mean)
        cis.append(ci)

    # ---------- plot --------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, 1.5*len(means)), 4))
    x = np.arange(len(args.methods))
    colors = [METHOD_COLORS.get(m.upper(), '#333333') for m in args.methods]

    ax.bar(x, means, yerr=cis, color=colors, alpha=0.8, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(args.methods, rotation=45, ha='right')
    ax.set_ylabel('Forgetting' + (' (success-normalised)' if args.metric=='success' else ''))
    ax.set_title(f'Average Forgetting over {args.seq_len-1} tasks')
    ax.axhline(0, color='black', lw=0.8)

    plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / 'plots'
    out.mkdir(exist_ok=True)
    stem = args.plot_name or f"forgetting_{args.metric}"
    plt.savefig(out / f"{stem}.png")
    plt.savefig(out / f"{stem}.pdf")
    plt.show()


if __name__ == '__main__':
    main()
