"""
Shared continual-learning metric formulas for the numerical/plotting result scripts.

Two Forgetting formulas exist in this codebase's history, and different scripts used
different ones to produce their published numbers:

  "weighted"   -- weighted, normalized post-training performance drop relative to
                  end-of-training performance, with exponential decay that penalizes
                  earlier forgetting more strongly. This is what results_table.py uses
                  today, though even its own formula changed shape more than once
                  across the project's history.

  "peak_final" -- unnormalized, unweighted drop from the best-ever value to a window-
                  averaged final value: max(0, peak - mean(final N evals)). This is what
                  every *other* results script has used since each file's very first
                  commit -- confirmed via git log, not assumed.

Every results script now takes a --forgetting-formula {weighted,peak_final} flag so
either can be reproduced; each script's *default* is set to whichever formula git
history shows actually produced its results. See each script's own CLI help for its
specific default.

levels_vs_tasks.py's "Cumulative Forgetting" (100-task long-sequence analysis) is
intentionally a third thing: the per-seed *cumulative sum* of the "weighted" per-task
formula across tasks 1..x, not a separate formula in its own right -- it has no
peak_final variant in its history, so it isn't part of this dual-formula setup.
"""

import numpy as np


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
    points)). Ignores where training actually ended for this task -- "peak" is the max
    over the *entire* curve, not just its training portion.

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
    calculate_peak_final_forgetting ("peak_final") by name -- lets call sites take a
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
