"""
Data loading utilities for plotting scripts.

This module contains functions for loading and processing data from the repository
structure, including collecting runs and processing time series data.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .common import load_series
from .metrics import sanitize_series


def build_task_matrix(
    series_list: List[np.ndarray],
    sanitize: bool = True,
    sanitize_label_fn: Optional[Callable[[int], str]] = None,
) -> np.ndarray:
    """Pad a list of per-task curves to a common length and stack into shape
    (n_tasks, T). Padding uses each curve's own final value (constant_values=s[-1]),
    matching every results script's convention for extending a shorter curve past its
    last recorded eval.

    If `sanitize`, each curve is NaN/inf-cleaned via metrics.sanitize_series first;
    `sanitize_label_fn(i)` supplies the [warn]-message label for task index i (default:
    "task series {i}").
    """
    if sanitize:
        if sanitize_label_fn is None:
            sanitize_label_fn = lambda i: f"task series {i}"
        series_list = [sanitize_series(s, sanitize_label_fn(i)) for i, s in enumerate(series_list)]
    else:
        series_list = [np.asarray(s, dtype=float) for s in series_list]
    L = max(len(s) for s in series_list)
    return np.vstack([np.pad(s, (0, L - len(s)), constant_values=s[-1]) for s in series_list])


def load_baseline_task_curves(
    baseline_folder: Path,
    seeds: List[int],
    n_tasks: int,
    file_fn: Optional[Callable[[int], str]] = None,
) -> Dict[int, List[Optional[np.ndarray]]]:
    """Load per-task baseline training curves for forward transfer: one list of `n_tasks`
    curves (or None where missing/empty/all-NaN/all-inf) per seed. `file_fn(i)` maps a
    task index to its filename relative to the seed dir; default "{i}_training_soup.json".
    Seeds whose directory doesn't exist are simply absent from the returned dict.
    """
    if file_fn is None:
        file_fn = lambda i: f"{i}_training_soup.json"
    baseline_data: Dict[int, List[Optional[np.ndarray]]] = {}
    for seed in seeds:
        seed_dir = baseline_folder / f"seed_{seed}"
        if not seed_dir.exists():
            continue
        curves: List[Optional[np.ndarray]] = []
        for i in range(n_tasks):
            fp = seed_dir / file_fn(i)
            if not fp.exists():
                curves.append(None)
                continue
            series = load_series(fp)
            if len(series) == 0:
                print(f"[warn] empty baseline data for task {i}, seed {seed}")
                curves.append(None)
            elif np.all(np.isnan(series)):
                print(f"[warn] baseline data contains all NaN for task {i}, seed {seed}")
                curves.append(None)
            elif np.all(np.isinf(series)):
                print(f"[warn] baseline data contains all inf/-inf for task {i}, seed {seed}")
                curves.append(None)
            elif np.all(np.isnan(series) | np.isinf(series)):
                print(f"[warn] baseline data contains all NaN/inf/-inf for task {i}, seed {seed}")
                curves.append(None)
            else:
                curves.append(series)
        baseline_data[seed] = curves
    return baseline_data


def collect_runs(base: Path, algo: str, method: str, strat: str, seq_len: int, seeds: List[int], metric: str,
                 level: int = 1) -> Tuple[
    np.ndarray, List[str]]:
    """
    Collect run data for training plots.

    Args:
        base: Base directory for data
        algo: Algorithm name
        method: Method name
        strat: Strategy name
        seq_len: Sequence length
        seeds: List of seeds to collect
        metric: Metric to collect ('reward', 'soup', etc.)
        level: Difficulty level (default: 1)

    Returns:
        Tuple of (data_array, environment_names)
    """
    folder = base / algo / method / f"level_{level}" / f"{strat}_{seq_len}"
    env_names, per_seed = [], []

    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            continue
        files = sorted(sd.glob(f"*_{metric}.*"))
        if not files:
            continue

        # first pass → env name order
        if not env_names:
            suffix = f"_{metric}"
            env_names = [f.name.split('_', 1)[1].rsplit(suffix, 1)[0]
                         for f in files]

        arrs = [load_series(f) for f in files]
        L = max(map(len, arrs))
        padded = [np.pad(a, (0, L - len(a)), constant_values=np.nan)
                  for a in arrs]

        per_seed.append(np.nanmean(padded, axis=0))

    if not per_seed:
        raise RuntimeError(f'No data for method {method}')

    N = max(map(len, per_seed))
    data = np.vstack([np.pad(a, (0, N - len(a)), constant_values=np.nan)
                      for a in per_seed])
    return data, env_names


def collect_env_curves(base: Path, algo: str, method: str, strat: str, seq_len: int, seeds: List[int],
                       metric: str = "reward", level: int = 1, agents: int = 2) -> Tuple[List[str], List[np.ndarray]]:
    """
    Collect per-environment curves for per-task evaluation plots.

    Args:
        base: Base directory for data
        algo: Algorithm name
        method: Method name
        strat: Strategy name
        seq_len: Sequence length
        seeds: List of seeds to collect
        metric: Metric to collect (default: 'reward')
        level: Difficulty level (default: 1)

    Returns:
        Tuple of (environment_names, curves_per_environment)
    """
    folder = base / algo / f"{method}_old" / f"level_{level}" / f"agents_{agents}" / f"{strat}_{seq_len}"
    env_names, per_env_seed = [], []

    # discover envs
    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            continue
        files = sorted(f for f in sd.glob(f"*_{metric}.*") if "training" not in f.name)
        if not files:
            continue
        suffix = f"_{metric}"
        env_names = [f.name.split('_', 1)[1].rsplit(suffix, 1)[0] for f in files]
        per_env_seed = [[] for _ in env_names]
        break
    if not env_names:
        raise RuntimeError(f'No data for {method}')

    # gather
    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            continue
        for idx, env in enumerate(env_names):
            fp = sd / f"{idx}_{env}"
            if not fp.exists():
                fp = sd / f"{idx}_{env}_{metric}.npz"
            if not fp.exists():
                continue
            arr = load_series(fp)
            per_env_seed[idx].append(arr)

    T_max = max(max(map(len, curves)) for curves in per_env_seed if curves)
    curves = []
    for env_curves in per_env_seed:
        if env_curves:
            stacked = np.vstack([np.pad(a, (0, T_max - len(a)), constant_values=np.nan)
                                 for a in env_curves])
        else:
            stacked = np.full((1, T_max), np.nan)
        curves.append(stacked)

    return env_names, curves


def collect_partner_curves(
    base: Path,
    algo: str,
    method: str,
    layout_name: str,
    arch: str,
    num_partners: int,
    seeds: List[int],
    metric: str = "soup",
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Collect per-partner evaluation curves for partner-adaptation plots.

    Path: base/<algo>/<method>/<layout_name>/<arch>/partners_<num_partners>/seed_<seed>/
      eval_partner_{i}_{metric}.{ext}

    Returns:
        Tuple of (partner_names, curves_per_partner) where each curve is (n_seeds, T).
    """
    folder = base / algo / method / layout_name / arch / f"partners_{num_partners}"
    partner_names: List[str] = [f"partner_{i}" for i in range(num_partners)]
    per_partner: List[List[np.ndarray]] = [[] for _ in range(num_partners)]

    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            continue
        for i in range(num_partners):
            for ext in (".json", ".npz"):
                fp = sd / f"eval_partner_{i}_{metric}{ext}"
                if fp.exists():
                    per_partner[i].append(load_series(fp))
                    break

    if not any(per_partner):
        raise RuntimeError(f"No partner eval data for {method}/{layout_name}/{arch}")

    T_max = max(len(a) for curves in per_partner for a in curves)
    result = []
    for curves in per_partner:
        if curves:
            stacked = np.vstack([np.pad(a, (0, T_max - len(a)), constant_values=np.nan) for a in curves])
        else:
            stacked = np.full((1, T_max), np.nan)
        result.append(stacked)

    return partner_names, result


def collect_br_cumulative_runs(
    base: Path,
    method: str,
    layout_name: str,
    arch: str,
    num_partners: int,
    seeds: List[int],
    metric: str = "soup",
) -> np.ndarray:
    """Collect data for BR partner-adaptation cumulative evaluation plots.

    Path: base/br/<method>/<layout_name>/<arch>/partners_<num_partners>/seed_<seed>/

    Returns array of shape (n_seeds, L) — the mean-across-partners curve per seed.
    """
    folder = base / "ppo" / method / layout_name / arch / f"partners_{num_partners}"
    per_seed = []

    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            print(f"[warn] missing {sd}")
            continue
        files = sorted(sd.glob(f"eval_partner_*_{metric}.*"))
        if not files:
            print(f"[warn] no eval_partner files in {sd}")
            continue

        partner_curves = [load_series(f) for f in files]
        L = max(map(len, partner_curves))
        padded = [np.pad(c, (0, L - len(c)), constant_values=c[-1]) for c in partner_curves]
        mean_curve = np.nanmean(np.vstack(padded), axis=0)
        per_seed.append(mean_curve)

    if not per_seed:
        return np.empty((0, 0))

    N = max(map(len, per_seed))
    per_seed = [np.pad(c, (0, N - len(c)), constant_values=c[-1]) for c in per_seed]
    return np.vstack(per_seed)  # (S, N)


def collect_cumulative_runs(base: Path, algo: str, method: str, strat: str, metric: str, seq_len: int, seeds: List[int],
                            level: int,  agents: int, experiment: str = "") -> np.ndarray:
    """
    Collect run data for cumulative evaluation plots.

    Args:
        base: Base directory for data
        algo: Algorithm name
        method: Method name
        strat: Strategy name
        metric: Metric to collect
        seq_len: Sequence length
        seeds: List of seeds to collect

    Returns:
        Array of shape (n_seeds, L) containing the cumulative-average-so-far curve for every seed
    """
    folder = base / algo / method / f"level_{level}" / f"agents_{agents}" / f"{strat}_{seq_len}" / experiment
    per_seed = []

    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            print(f"[warn] missing data {sd}")
            continue
        env_files = sorted(sd.glob(f"*_{metric}.*"))
        if not env_files:
            continue

        env_curves = [load_series(f) for f in env_files]
        L = max(map(len, env_curves))
        padded = [np.pad(c, (0, L - len(c)), constant_values=c[-1]) for c in env_curves]

        env_mat = np.vstack(padded)  # shape (n_envs, L)

        env_mat = np.nan_to_num(
            env_mat,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # turn NaNs into 0 so they count as "no performance yet"
        env_mat = np.nan_to_num(env_mat, nan=0.0)

        # Check if env_mat is all zeros (poor performance)
        if np.all(env_mat == 0):
            # Still calculate the average, but this will be all zeros
            cum_avg = env_mat.mean(axis=0)
        else:
            # cumulative-average-so-far curve
            cum_avg = env_mat.mean(axis=0)  # fixed denominator = n_envs

        per_seed.append(cum_avg)

    if not per_seed:
        raise RuntimeError(f"No data found for method {method}")

    # pad to same length (unlikely to differ, but be safe)
    N = max(map(len, per_seed))
    per_seed = [np.pad(c, (0, N - len(c)), constant_values=c[-1]) for c in per_seed]
    return np.vstack(per_seed)  # (S, N)
