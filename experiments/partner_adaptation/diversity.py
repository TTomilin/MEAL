"""Pure-NumPy partner diversity metrics, Vendi selection, and specialist health checks."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MIN_VENDI_GAIN = 0.15
DEFAULT_MAX_OFF_DIAG = 0.7
DEFAULT_STILL_LEARNING_RATIO = 0.15
DEFAULT_WEAK_MARGIN = 1.05
DEFECT_FLAGS = frozenset({"ZERO_OWN_SCORE", "NON_FINITE_OWN_SCORE", "WEAK_SPECIALIST", "STILL_LEARNING"})


def relative_xp(xp: np.ndarray) -> np.ndarray:
    xp = np.asarray(xp, dtype=np.float64)
    diag = np.diag(xp)
    with np.errstate(invalid="ignore", divide="ignore"):
        return xp / np.where(diag > 0.0, diag, np.nan)


def response_kernel(xp: np.ndarray, *, clip: bool = True) -> np.ndarray:
    """Column-cosine PSD kernel over relative cross-play response profiles."""
    rel = np.nan_to_num(relative_xp(xp), nan=0.0, posinf=0.0, neginf=0.0)
    if clip:
        rel = np.clip(rel, 0.0, 1.0)
    norms = np.linalg.norm(rel, axis=0, keepdims=True)
    unit = np.divide(rel, norms, out=np.zeros_like(rel), where=norms > 0.0)
    return unit.T @ unit


def behaviour_kernel(visits_diag: np.ndarray) -> np.ndarray:
    """Bhattacharyya PSD kernel over per-partner state-visitation distributions (own specialist)."""
    occ = np.asarray(visits_diag, dtype=np.float64)
    if occ.ndim != 2:
        raise ValueError(f"visits_diag must be 2D (n_partners, n_cells), got shape {occ.shape}")
    totals = occ.sum(axis=1, keepdims=True)
    probs = np.divide(occ, totals, out=np.zeros_like(occ), where=totals > 0.0)
    root = np.sqrt(np.clip(probs, 0.0, None))
    return root @ root.T


def vendi_score(kernel: np.ndarray) -> float:
    """Effective number of distinct items exp(H(eig(K / tr(K)))) for a PSD kernel."""
    K = np.asarray(kernel, dtype=np.float64)
    if K.size == 0:
        return 0.0
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"kernel must be square, got shape {K.shape}")
    sym = 0.5 * (K + K.T)
    tr = float(np.trace(sym))
    if tr <= 1e-12:
        return 0.0
    eigvals = np.linalg.eigvalsh(sym / tr)
    pos = eigvals[eigvals > 1e-12]
    if pos.size == 0:
        return 0.0
    pos = pos / pos.sum()
    return float(np.exp(-np.sum(pos * np.log(pos))))


def logdet_diversity(kernel: np.ndarray, eps: float = 1e-6) -> float:
    """Regularized log-determinant volume of a PSD similarity kernel."""
    K = np.asarray(kernel, dtype=np.float64)
    if K.size == 0:
        return 0.0
    sym = 0.5 * (K + K.T) + float(eps) * np.eye(K.shape[0], dtype=np.float64)
    sign, logdet = np.linalg.slogdet(sym)
    return float(logdet) if sign > 0 else float("nan")


def vendi_bootstrap_ci(xp: np.ndarray, xp_std: np.ndarray, *, episodes: int = 20,
                       n_boot: int = 500, seed: int = 0) -> Dict[str, float]:
    """Parametric bootstrap 95% CI on response Vendi using per-cell eval std."""
    xp = np.asarray(xp, dtype=np.float64)
    xp_std = np.asarray(xp_std, dtype=np.float64)
    if xp.shape != xp_std.shape:
        raise ValueError(f"xp shape {xp.shape} != xp_std shape {xp_std.shape}")
    if episodes < 1 or n_boot < 1:
        raise ValueError("episodes and n_boot must be >= 1")
    valid = np.diag(xp) > 0.0
    if int(valid.sum()) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_boot": int(n_boot), "episodes": int(episodes)}
    sub = xp[np.ix_(valid, valid)]
    sub_std = xp_std[np.ix_(valid, valid)]
    if not np.isfinite(sub_std).all() or (sub_std < 0.0).any():
        raise ValueError("xp_std must be finite and non-negative on valid partners")
    se = sub_std / np.sqrt(float(episodes))
    rng = np.random.default_rng(int(seed))
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        perturbed = np.maximum(sub + rng.normal(0.0, se), 0.0)
        samples[b] = vendi_score(response_kernel(perturbed))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "mean": float(samples.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
        "episodes": int(episodes),
    }


def _cap_matches(cap_kind: str, kind: str) -> bool:
    return kind != "planner" if cap_kind == "learned" else kind == cap_kind


def _max_and_mean_off_diag(sub_xp: np.ndarray) -> Tuple[float, float]:
    k = sub_xp.shape[0]
    if k <= 1:
        return 0.0, 0.0
    rel = relative_xp(sub_xp)
    mask = ~np.eye(k, dtype=bool)
    vals = rel[mask]
    if not np.isfinite(vals).all():
        return float("inf"), float("inf")
    return float(np.max(vals)), float(np.mean(vals))


def greedy_vendi_curve(
    xp: np.ndarray,
    candidates: Iterable[int],
    *,
    max_k: Optional[int] = None,
    min_gain: float = DEFAULT_MIN_VENDI_GAIN,
    max_off_diag: float = DEFAULT_MAX_OFF_DIAG,
    kinds: Optional[Sequence[str]] = None,
    kind_caps: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Greedy self-contained submatrix Vendi selection and elbow detection.

    Starts from the highest own-score eligible candidate (subject to `kind_caps` if given), then
    repeatedly adds the candidate maximizing `vendi_score(response_kernel(XP[S, S]))` while preferring
    candidates that keep `max_off_diag < threshold`. Records the full trajectory up to `max_k` (or all
    candidates) and marks `recommended_indices` at the first elbow where adding another partner either
    breaches `max_off_diag` or adds less than `min_gain` effective conventions.
    """
    xp = np.asarray(xp, dtype=np.float64)
    if xp.ndim != 2 or xp.shape[0] != xp.shape[1]:
        raise ValueError(f"xp must be square, got shape {xp.shape}")
    n_total = xp.shape[0]
    caps = {str(k): int(v) for k, v in (kind_caps or {}).items()}
    if caps and kinds is None:
        raise ValueError("kind_caps requires kinds")
    if kinds is not None and len(kinds) != n_total:
        raise ValueError(f"len(kinds)={len(kinds)} != xp.shape[0]={n_total}")
    bad_caps = {k: v for k, v in caps.items() if v < 0}
    if bad_caps:
        raise ValueError(f"kind_caps must be non-negative, got {bad_caps}")
    raw_cands = [int(i) for i in sorted(set(candidates))]
    bad_idx = [i for i in raw_cands if not 0 <= i < n_total]
    if bad_idx:
        raise ValueError(f"candidate indices out of range for size {n_total}: {bad_idx}")
    cands = [i for i in raw_cands if xp[i, i] > 0.0]
    if not cands or (max_k is not None and max_k <= 0):
        return {"steps": [], "recommended_k": 0, "recommended_indices": [],
                "min_gain": float(min_gain), "max_off_diag": float(max_off_diag)}

    counts: Dict[str, int] = {}

    def allowed_by_cap(i: int) -> bool:
        if not caps or kinds is None:
            return True
        return not any(_cap_matches(cap_kind, kinds[i]) and counts.get(cap_kind, 0) >= limit
                       for cap_kind, limit in caps.items())

    def record_kind(i: int) -> None:
        if not caps or kinds is None:
            return
        for cap_kind in caps:
            if _cap_matches(cap_kind, kinds[i]):
                counts[cap_kind] = counts.get(cap_kind, 0) + 1

    eligible = [i for i in cands if allowed_by_cap(i)]
    if not eligible:
        return {"steps": [], "recommended_k": 0, "recommended_indices": [],
                "min_gain": float(min_gain), "max_off_diag": float(max_off_diag)}

    first = max(eligible, key=lambda i: (xp[i, i], -i))
    selected = [first]
    record_kind(first)
    prev_vendi = 1.0
    steps: List[Dict[str, Any]] = [{
        "k": 1,
        "added_index": first,
        "own_score": float(xp[first, first]),
        "vendi": 1.0,
        "delta_vendi": 1.0,
        "efficiency": 1.0,
        "max_off_diag": 0.0,
        "mean_off_diag": 0.0,
        "accepted": True,
    }]
    elbow_hit = False
    recommended = [first]
    limit_k = len(cands) if max_k is None else min(len(cands), int(max_k))

    while len(selected) < limit_k:
        rem = [i for i in cands if i not in selected and allowed_by_cap(i)]
        if not rem:
            break
        best_choice = None
        best_key = None
        best_metrics = None
        for i in rem:
            sub_idx = selected + [i]
            sub = xp[np.ix_(sub_idx, sub_idx)]
            v = vendi_score(response_kernel(sub))
            max_off, mean_off = _max_and_mean_off_diag(sub)
            key = (max_off < max_off_diag, v, float(xp[i, i]), -i)
            if best_key is None or key > best_key:
                best_key = key
                best_choice = i
                best_metrics = (v, max_off, mean_off)
        assert best_choice is not None and best_metrics is not None
        v, max_off, mean_off = best_metrics
        dv = v - prev_vendi
        accepted = (not elbow_hit) and (max_off < max_off_diag) and (dv >= min_gain)
        if not accepted:
            elbow_hit = True
        else:
            recommended.append(best_choice)
        selected.append(best_choice)
        record_kind(best_choice)
        k = len(selected)
        steps.append({
            "k": k,
            "added_index": best_choice,
            "own_score": float(xp[best_choice, best_choice]),
            "vendi": float(v),
            "delta_vendi": float(dv),
            "efficiency": float(v / k),
            "max_off_diag": float(max_off),
            "mean_off_diag": float(mean_off),
            "accepted": accepted,
        })
        prev_vendi = v

    return {
        "steps": steps,
        "recommended_k": len(recommended),
        "recommended_indices": recommended,
        "min_gain": float(min_gain),
        "max_off_diag": float(max_off_diag),
    }


def select_by_vendi(
    xp: np.ndarray,
    candidates: Iterable[int],
    *,
    max_k: Optional[int] = None,
    min_gain: float = DEFAULT_MIN_VENDI_GAIN,
    max_off_diag: float = DEFAULT_MAX_OFF_DIAG,
    kinds: Optional[Sequence[str]] = None,
    kind_caps: Optional[Mapping[str, int]] = None,
) -> List[int]:
    """Return sorted bank indices of the Vendi-selected subset up to the elbow."""
    curve = greedy_vendi_curve(
        xp, candidates, max_k=max_k, min_gain=min_gain, max_off_diag=max_off_diag,
        kinds=kinds, kind_caps=kind_caps,
    )
    return sorted(curve["recommended_indices"])


def diagnose_specialists(
    xp: np.ndarray,
    labels: Sequence[str],
    learnability_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    min_own_score: float = 0.5,
    still_learning_ratio: float = DEFAULT_STILL_LEARNING_RATIO,
    weak_margin: float = DEFAULT_WEAK_MARGIN,
) -> Dict[str, Any]:
    """Detect corrupted/unconverged specialists that distort cross-play and Vendi metrics."""
    xp = np.asarray(xp, dtype=np.float64)
    n = len(labels)
    if xp.shape != (n, n):
        raise ValueError(f"xp shape {xp.shape} does not match len(labels)={n}")
    learn_by_label = {str(r["label"]): r for r in (learnability_rows or [])}
    partners = []
    defect_indices = []
    for j in range(n):
        own = float(xp[j, j])
        own_finite = bool(np.isfinite(own))
        finite_others = [i for i in range(n) if i != j and np.isfinite(xp[i, j])]
        if finite_others:
            best_i = max(finite_others, key=lambda i: float(xp[i, j]))
            best_other = float(xp[best_i, j])
            best_other_label = labels[best_i]
        else:
            best_other = 0.0
            best_other_label = ""
        flags: List[str] = []
        if not own_finite:
            flags.append("NON_FINITE_OWN_SCORE")
        elif own <= 0.0:
            flags.append("ZERO_OWN_SCORE")
        elif own < min_own_score:
            flags.append("LOW_OWN_SCORE")
        if finite_others and (
            ((not own_finite or own <= 0.0) and best_other > 0.0)
            or (own_finite and own > 0.0 and best_other > weak_margin * own)
        ):
            flags.append("WEAK_SPECIALIST")
        lr = learn_by_label.get(labels[j])
        if lr is not None:
            final_raw, gain_raw = lr.get("final_train_soups", ""), lr.get("last_quarter_gain", "")
            if final_raw not in ("", None) and gain_raw not in ("", None):
                final_val, gain_val = float(final_raw), float(gain_raw)
                if np.isfinite(final_val) and np.isfinite(gain_val) and final_val > 0.0 and gain_val > still_learning_ratio * final_val:
                    flags.append("STILL_LEARNING")
        has_defect = bool(DEFECT_FLAGS & set(flags))
        if has_defect:
            defect_indices.append(j)
        partners.append({
            "index": j,
            "label": labels[j],
            "own_score": own,
            "best_other_score": best_other,
            "best_other_specialist": best_other_label,
            "flags": flags,
            "defect": has_defect,
        })
    return {
        "healthy": len(defect_indices) == 0,
        "n_defects": len(defect_indices),
        "defect_indices": defect_indices,
        "partners": partners,
    }


def _subset_diversity(xp: np.ndarray, indices: Sequence[int], *,
                      xp_std: Optional[np.ndarray] = None, episodes: int = 20,
                      visits: Optional[np.ndarray] = None) -> Dict[str, Any]:
    idx = [int(i) for i in indices]
    n_total = len(idx)
    if n_total == 0:
        return {"n_total": 0, "n_valid": 0, "vendi_resp": 0.0, "vendi_efficiency": 0.0, "logdet_resp": 0.0}
    sub = xp[np.ix_(idx, idx)]
    valid_mask = np.diag(sub) > 0.0
    valid_local = np.flatnonzero(valid_mask)
    n_valid = int(valid_local.size)
    if n_valid == 0:
        return {"n_total": n_total, "n_valid": 0, "vendi_resp": 0.0, "vendi_efficiency": 0.0, "logdet_resp": 0.0}
    sub_valid = sub[np.ix_(valid_local, valid_local)]
    K_resp = response_kernel(sub_valid)
    v_resp = vendi_score(K_resp)
    out: Dict[str, Any] = {
        "n_total": n_total,
        "n_valid": n_valid,
        "vendi_resp": v_resp,
        "vendi_efficiency": float(v_resp / n_valid),
        "logdet_resp": logdet_diversity(K_resp),
        "mean_own_score": float(np.diag(sub_valid).mean()),
    }
    if xp_std is not None:
        sub_std = xp_std[np.ix_(idx, idx)]
        out["vendi_resp_ci"] = vendi_bootstrap_ci(sub, sub_std, episodes=episodes)
    if visits is not None:
        diag_vis = np.asarray([visits[idx[k], idx[k]] for k in valid_local], dtype=np.float64)
        K_beh = behaviour_kernel(diag_vis)
        v_beh = vendi_score(K_beh)
        out["vendi_beh"] = v_beh
        out["vendi_beh_efficiency"] = float(v_beh / n_valid)
    return out


def diversity_summary(
    xp: np.ndarray,
    labels: Sequence[str],
    kinds: Sequence[str],
    indices: Optional[Sequence[int]] = None,
    *,
    xp_std: Optional[np.ndarray] = None,
    episodes: int = 20,
    visits: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute response (+ optional behaviour & bootstrap CI) Vendi summary overall and per kind."""
    idx = list(range(len(labels))) if indices is None else [int(i) for i in indices]
    out = _subset_diversity(xp, idx, xp_std=xp_std, episodes=episodes, visits=visits)
    by_kind: Dict[str, Any] = {}
    for kind in sorted({kinds[i] for i in idx}):
        kind_idx = [i for i in idx if kinds[i] == kind]
        by_kind[kind] = _subset_diversity(xp, kind_idx, visits=visits)
    out["by_kind"] = by_kind
    return out
