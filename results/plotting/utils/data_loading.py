"""
Data loading utilities for plotting scripts.

This module contains functions for loading and processing data from the repository
structure, including collecting runs and processing time series data.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

from .common import load_series, forward_fill

def collect_runs(base: Path, algo: str, method: str, arch: str, strat: str,
                seq_len: int, seeds: List[int], metric: str) -> Tuple[np.ndarray, List[str]]:
    """
    Collect run data for training plots.
    
    Args:
        base: Base directory for data
        algo: Algorithm name
        method: Method name
        arch: Architecture name
        strat: Strategy name
        seq_len: Sequence length
        seeds: List of seeds to collect
        metric: Metric to collect ('reward', 'soup', etc.)
        
    Returns:
        Tuple of (data_array, environment_names)
    """
    folder = base / algo / method / arch / f"{strat}_{seq_len}"
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

def collect_env_curves(base: Path, algo: str, method: str, strat: str,
                      seq_len: int, seeds: List[int], metric: str = "reward") -> Tuple[List[str], List[np.ndarray]]:
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
        
    Returns:
        Tuple of (environment_names, curves_per_environment)
    """
    folder = base / algo / method / f"{strat}_{seq_len}"
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
            fp = sd / f"{idx}_{env}_{metric}.json"
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

def collect_cumulative_runs(base: Path, algo: str, method: str, arch: str, strat: str, 
                           metric: str, seq_len: int, seeds: List[int]) -> np.ndarray:
    """
    Collect run data for cumulative evaluation plots.
    
    Args:
        base: Base directory for data
        algo: Algorithm name
        method: Method name
        arch: Architecture name
        strat: Strategy name
        metric: Metric to collect
        seq_len: Sequence length
        seeds: List of seeds to collect
        
    Returns:
        Array of shape (n_seeds, L) containing the cumulative-average-so-far curve for every seed
    """
    folder = base / algo / method / arch / f"{strat}_{seq_len}"
    per_seed = []

    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            continue
        env_files = sorted(sd.glob(f"*_{metric}.*"))
        if not env_files:
            continue

        env_curves = [load_series(f) for f in env_files]
        L = max(map(len, env_curves))
        padded = [np.pad(c, (0, L - len(c)), constant_values=c[-1]) for c in env_curves]

        env_mat = np.vstack(padded)  # shape (n_envs, L)

        # turn NaNs into 0 so they count as "no performance yet"
        env_mat = np.nan_to_num(env_mat, nan=0.0)

        # cumulative-average-so-far curve
        cum_avg = env_mat.mean(axis=0)  # fixed denominator = n_envs
        per_seed.append(cum_avg)

    if not per_seed:
        raise RuntimeError(f"No data found for method {method}")

    # pad to same length (unlikely to differ, but be safe)
    N = max(map(len, per_seed))
    per_seed = [np.pad(c, (0, N - len(c)), constant_values=c[-1]) for c in per_seed]
    return np.vstack(per_seed)  # (S, N)