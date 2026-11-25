from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from typing import Dict, List, Mapping, Sequence

DELIVERY_REWARD = 20.0
MIN_NET_SOUP_RETURN = 10.0

EVENT_SHAPING_CHOICES: Dict[str, Sequence[float]] = {
    "PLACEMENT_IN_POT_REW": (0.0, 1.0, 3.0),
    "ONION_PICKUP_REWARD": (-1.0, 0.0, 1.0),
    "PLATE_PICKUP_REWARD": (-1.0, 0.0, 3.0),
    "SOUP_PICKUP_REWARD": (0.0, 2.0, 5.0),
    "COUNTER_DROP_REWARD": (-1.0, 0.0, 1.0),
    "COUNTER_PICKUP_REWARD": (-1.0, 0.0, 1.0),
}
EVENT_SHAPING_KEYS = tuple(sorted(EVENT_SHAPING_CHOICES))
NON_NEGATIVE_MILESTONES = ("PLACEMENT_IN_POT_REW", "SOUP_PICKUP_REWARD")


def net_soup_cycle_return(weights: Mapping[str, float]) -> float:
    """Net undiscounted return of a minimal 3-onion soup cycle under permanent shaping."""
    return (
        DELIVERY_REWARD
        + 3.0 * float(weights.get("ONION_PICKUP_REWARD", 0.0))
        + 3.0 * float(weights.get("PLACEMENT_IN_POT_REW", 0.0))
        + float(weights.get("PLATE_PICKUP_REWARD", 0.0))
        + float(weights.get("SOUP_PICKUP_REWARD", 0.0))
    )


def validate_shaping_weights(weights: Mapping[str, float],
                             min_net_soup_return: float = MIN_NET_SOUP_RETURN) -> None:
    unknown = sorted(set(weights) - set(EVENT_SHAPING_KEYS))
    if unknown:
        raise ValueError(f"unknown event shaping key(s): {unknown}; allowed: {EVENT_SHAPING_KEYS}")
    for key, val in weights.items():
        if not math.isfinite(float(val)):
            raise ValueError(f"{key}={val!r} must be finite")
    if not any(float(v) != 0.0 for v in weights.values()):
        raise ValueError("event shaping weights must not be all zero")
    for key in NON_NEGATIVE_MILESTONES:
        if float(weights.get(key, 0.0)) < 0.0:
            raise ValueError(f"{key}={weights[key]} must be >= 0 so cooking is never penalized below idling")
    counter_cycle = float(weights.get("COUNTER_DROP_REWARD", 0.0)) + float(weights.get("COUNTER_PICKUP_REWARD", 0.0))
    if counter_cycle > 0.0:
        raise ValueError(
            f"COUNTER_DROP_REWARD + COUNTER_PICKUP_REWARD = {counter_cycle:.2f} > 0: "
            f"repeatedly dropping and picking up an item on a counter farms unbounded reward without cooking")
    net = net_soup_cycle_return(weights)
    if net < min_net_soup_return:
        raise ValueError(
            f"net soup-cycle return {net:.2f} < {min_net_soup_return:.2f}: "
            f"permanent negative shaping makes doing nothing dominate cooking")


def is_viable_shaping(weights: Mapping[str, float],
                      min_net_soup_return: float = MIN_NET_SOUP_RETURN) -> bool:
    try:
        validate_shaping_weights(weights, min_net_soup_return=min_net_soup_return)
        return True
    except ValueError:
        return False


def sample_event_shaping(n_settings: int, setting_seed: int = 0,
                         choices: Mapping[str, Sequence[float]] = EVENT_SHAPING_CHOICES,
                         min_net_soup_return: float = MIN_NET_SOUP_RETURN) -> List[Dict[str, float]]:
    """Deterministically sample `n_settings` unique, cooking-viable shaping weight dicts."""
    if n_settings < 1:
        raise ValueError(f"n_settings must be >= 1, got {n_settings}")
    keys = sorted(choices)
    pool = []
    for combo in itertools.product(*(choices[k] for k in keys)):
        w = {k: float(v) for k, v in zip(keys, combo)}
        if is_viable_shaping(w, min_net_soup_return=min_net_soup_return):
            pool.append(w)
    if n_settings > len(pool):
        raise ValueError(f"requested {n_settings} unique settings, but only {len(pool)} viable combinations exist")
    rng = random.Random(int(setting_seed))
    return rng.sample(pool, n_settings)


def hydra_overrides(weights: Mapping[str, float]) -> List[str]:
    validate_shaping_weights(weights)
    def _fmt(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else f"{float(v):g}"
    return [f"task.ENV_KWARGS.reward_shaping_params.{n}=[{_fmt(w)},{_fmt(w)}]"
            for n, w in sorted(weights.items())]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Sample viable event-reward shaping weights for jax-aht IPPO.")
    ap.add_argument("--k", type=int, required=True, help="1-based setting index")
    ap.add_argument("--n-settings", type=int, required=True, help="total number of unique settings")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not 1 <= args.k <= args.n_settings:
        raise SystemExit(f"--k={args.k} must be in [1, {args.n_settings}]")
    weights = sample_event_shaping(args.n_settings, setting_seed=args.seed)[args.k - 1]
    print(json.dumps({"weights": weights, "overrides": hydra_overrides(weights)}))


if __name__ == "__main__":
    main()
