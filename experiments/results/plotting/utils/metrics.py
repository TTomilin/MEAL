"""
Shared continual-learning metric formulas for the numerical/plotting result scripts.

Two Forgetting formulas exist in this codebase's history, and different scripts used
different ones to produce their published numbers:

  "weighted"   -- weighted, normalized post-training performance drop relative to
                  end-of-training performance, with exponential decay that penalizes
                  earlier forgetting more strongly.

  "peak_final" -- unnormalized, unweighted drop from the best-ever value to a window-
                  averaged final value: max(0, peak - mean(final N evals)).
"""

from typing import List, Optional, Tuple

import numpy as np


def mean_ci(series: List[float]) -> Tuple[float, float]:
    """Mean and 95%-CI half-width of `series`. (nan, nan) if empty; CI is 0 for a single value."""
    series = list(series)
    if not series:
        return float("nan"), float("nan")
    mean = float(np.mean(series))
    if len(series) == 1:
        return mean, 0.0
    ci = 1.96 * float(np.std(series, ddof=1)) / np.sqrt(len(series))
    return mean, ci


def sanitize_series(series: np.ndarray, label: str) -> np.ndarray:
    """Replace NaN/inf entries with 0, printing a [warn] message identifying which case
    applied. The "check has_nan/has_inf, zero-fill, print" block. `label` should
    identify what's being sanitized, e.g. f"env series {i} for {method} seed {seed}".
    """
    series = np.asarray(series, dtype=float)
    has_nan = bool(np.any(np.isnan(series)))
    has_inf = bool(np.any(np.isinf(series)))
    if not has_nan and not has_inf:
        return series
    if np.all(np.isnan(series)) and not has_inf:
        print(f"[warn] {label} contains all NaN values, replacing with zeros")
        return np.zeros_like(series)
    if np.all(np.isinf(series)) and not has_nan:
        print(f"[warn] {label} contains all inf/-inf values, replacing with zeros")
        return np.zeros_like(series)
    if np.all(np.isnan(series) | np.isinf(series)):
        print(f"[warn] {label} contains all NaN/inf/-inf values, replacing with zeros")
        return np.zeros_like(series)
    if has_nan and has_inf:
        print(f"[warn] {label} contains some NaN and inf/-inf values, replacing with zeros")
        return np.where(np.isnan(series) | np.isinf(series), 0.0, series)
    if has_nan:
        print(f"[warn] {label} contains some NaN values, replacing NaN with zeros")
        return np.where(np.isnan(series), 0.0, series)
    print(f"[warn] {label} contains some inf/-inf values, replacing with zeros")
    return np.where(np.isinf(series), 0.0, series)


def task_auc(curve: np.ndarray) -> float:
    """AUC_i = (1/tau) * integral of curve dt via the trapezoidal rule. A task's
    time-averaged performance. Used by the forward-transfer calculation below."""
    curve = np.asarray(curve, dtype=float)
    if len(curve) > 1:
        return float(np.trapz(curve) / len(curve))
    return float(curve[0]) if len(curve) == 1 else 0.0


def forward_transfer_ratio(auc_cl: float, auc_baseline: float, zero_threshold: float = 1e-8) -> Optional[float]:
    """FT_i = (AUC_cl - AUC_baseline) / AUC_baseline, or None if the task should be
    skipped: NaN/inf inputs, a near-zero baseline, or a NaN/inf result. The guard chain
    every forward-transfer loop had copy-pasted inline."""
    if np.isnan(auc_cl) or np.isinf(auc_cl):
        return None
    if np.isnan(auc_baseline) or np.isinf(auc_baseline):
        return None
    if auc_baseline < zero_threshold:
        return None
    ft_i = (auc_cl - auc_baseline) / auc_baseline
    if np.isnan(ft_i) or np.isinf(ft_i):
        return None
    return float(ft_i)


def average_forward_transfer(
    training: np.ndarray, chunk: int, n_tasks: int, baseline_curves: List[Optional[np.ndarray]],
) -> float:
    """Mean forward transfer across `n_tasks`: for each task, AUC of the CL training
    curve's chunk vs. the matching baseline curve's AUC, combined via
    forward_transfer_ratio. NaN if no task yields a usable ratio.

    `training` is the full concatenated CL training curve; `chunk` is the (training-curve)
    step count per task, i.e. task i's slice is training[i*chunk:(i+1)*chunk].
    `baseline_curves` is per-task (index i = task i), e.g. from
    data_loading.load_baseline_task_curves()[seed]; entries may be None if missing.
    """
    ft_vals = []
    for i in range(n_tasks):
        cl_task_curve = training[i * chunk:(i + 1) * chunk]
        auc_cl = task_auc(cl_task_curve)
        if i >= len(baseline_curves) or baseline_curves[i] is None:
            continue
        auc_baseline = task_auc(baseline_curves[i])
        ft_i = forward_transfer_ratio(auc_cl, auc_baseline)
        if ft_i is not None:
            ft_vals.append(ft_i)
    return float(np.nanmean(ft_vals)) if ft_vals else float("nan")


def calculate_curve_based_forgetting(
    task_curve: np.ndarray, training_end_idx: int = None, lambda_decay: float = 2.0
) -> float:
    """
    "weighted" formula. Normalized forgetting where 0 = no forgetting and 1 = complete forgetting.

    Forgetting is normalized such that:
    - 0 means performance never drops below the end-of-training performance
    - 1 means performance drops to 0 right after training finishes and stays there

    Args:
        task_curve: Performance curve for a single task over time
        training_end_idx: Index where training for this task ends. If None, uses the last index.
        lambda_decay: Higher values penalize early forgetting more strongly.

    Returns:
        Normalized forgetting score between 0 and 1
    """
    if len(task_curve) <= 1:
        return 0.0

    # Determine the end-of-training index
    if training_end_idx is None or training_end_idx >= len(task_curve):
        training_end_idx = len(task_curve) - 1

    # performance at the end of training for this task
    end_of_training_performance = task_curve[training_end_idx]

    if end_of_training_performance < 1e-8:
        return 0.0

    # strictly after the training boundary
    if training_end_idx >= len(task_curve) - 1:
        return 0.0

    post_training_curve = task_curve[training_end_idx + 1:]

    # Forgetting = max(0, end_of_training_performance - current_performance)
    forgetting_at_each_step = np.maximum(end_of_training_performance - post_training_curve, 0.0)

    # Normalize forgetting by end_of_training_performance to get values between 0 and 1
    # (1.0 represents complete forgetting: performance drops to 0)
    normalized_forgetting_at_each_step = forgetting_at_each_step / end_of_training_performance

    # Weight forgetting by how early it occurs: weight = exp(-λ * (t / T))
    time_steps = np.arange(len(post_training_curve))
    total_time = len(post_training_curve) - 1

    if total_time > 0:
        normalized_time = time_steps / total_time
        weights = np.exp(-lambda_decay * normalized_time)
    else:
        weights = np.ones(len(post_training_curve))

    weighted_forgetting = normalized_forgetting_at_each_step * weights

    if len(weighted_forgetting) > 0 and np.sum(weights) > 0:
        curve_based_forgetting = np.sum(weighted_forgetting) / np.sum(weights)
    else:
        curve_based_forgetting = 0.0

    return float(np.clip(curve_based_forgetting, 0.0, 1.0))


def calculate_peak_final_forgetting(task_curve: np.ndarray, end_window_evals: int = 10) -> float:
    """
    "peak_final" formula. Unnormalized, unweighted drop from the best-ever value in the
    curve to a window-averaged final value: max(0, peak - mean(final `end_window_evals`
    points)).

    Args:
        task_curve: Performance curve for a single task over time.
        end_window_evals: Number of trailing points averaged for the "final" value.

    Returns:
        Non-negative forgetting score (unbounded above; not normalized to [0, 1]).
    """
    if len(task_curve) == 0:
        return 0.0
    peak = float(np.nanmax(task_curve))
    window = task_curve[-end_window_evals:] if end_window_evals > 0 else task_curve
    final_avg = float(np.nanmean(window))
    return max(peak - final_avg, 0.0)


def compute_forgetting(
    task_curve: np.ndarray,
    formula: str,
    training_end_idx: int = None,
    end_window_evals: int = 10,
    lambda_decay: float = 2.0,
) -> float:
    """Dispatch to calculate_curve_based_forgetting ("weighted") or
    calculate_peak_final_forgetting ("peak_final") by name. Lets call sites take a
    --forgetting-formula CLI flag and stay a one-line branch instead of an if/else."""
    if formula == "weighted":
        return calculate_curve_based_forgetting(task_curve, training_end_idx, lambda_decay=lambda_decay)
    elif formula == "peak_final":
        return calculate_peak_final_forgetting(task_curve, end_window_evals=end_window_evals)
    else:
        raise ValueError(f"Unknown forgetting formula: {formula!r} (expected 'weighted' or 'peak_final')")


def training_end_idx_for_task(task_idx: int, task_curve_len: int, n_train: int, seq_len: int) -> int:
    """
    Map task `task_idx`'s training-boundary step (in the training curve's own step units)
    onto an index within that task's own evaluation curve (`task_curve_len` points, which
    may use a different sampling rate/length than the training curve).

    `n_train` is the length of the full training curve (all tasks concatenated); `seq_len`
    is the number of tasks, so `n_train / seq_len` is (approximately) how many training-curve
    points belong to one task. Boundary for task `task_idx` (0-indexed) is right after its
    own training window, i.e. after `(task_idx + 1)` tasks' worth of training steps.

    Only meaningful for the "weighted" formula; "peak_final" doesn't use a training boundary.
    """
    chunk = n_train / seq_len if seq_len > 0 else 0
    training_end_step = (task_idx + 1) * chunk
    if n_train > 0:
        idx = int((training_end_step / n_train) * task_curve_len)
    else:
        idx = task_curve_len - 1
    return min(idx, task_curve_len - 1)
