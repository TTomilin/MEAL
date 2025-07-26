#!/usr/bin/env python3
"""Download evaluation curves *per environment* for the MARL continual-learning
benchmark and store them one metric per file.

Optimized logic:
1. Discover available evaluation keys per run via `run.history(samples=1)`.
2. Fetch each key's full time series separately, only once.
3. Skip keys whose output files already exist (unless `--overwrite`).
4. Write files in
   `data/<algo>/<cl_method>/<experiment>/<strategy>_<seq_len>/seed_<seed>/`.

Enhanced with reward settings support:
- Filter by reward settings (sparse_rewards, individual_rewards)
- Store data in appropriate folders with reward setting prefixes
- Backward compatible with existing experiments
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import List

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import wandb
from wandb.apis.public import Run

from results.download.common import cli, want, experiment_suffix

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
EVAL_PREFIX = "Evaluation/Soup_Scaled/"
KEY_PATTERN = re.compile(rf"^{re.escape(EVAL_PREFIX)}(\d+)__(.+)_(\d+)$")
TRAINING_KEY = "Soup/scaled"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def discover_eval_keys(run: Run) -> List[str]:
    """Retrieve & sort eval keys, plus the one training key if present."""
    df = run.history(samples=500)
    # only exact eval keys
    keys = [k for k in df.columns if KEY_PATTERN.match(k)]
    # include training series, if logged
    if TRAINING_KEY in df.columns:
        keys.append(TRAINING_KEY)

    # sort eval ones by idx, leave training last
    def idx_of(key: str) -> int:
        m = KEY_PATTERN.match(key)
        return int(m.group(1)) if m else 10 ** 6

    return sorted(keys, key=idx_of)


def fetch_full_series(run: Run, key: str) -> List[float]:
    """Fetch every recorded value for a single key via scan_history."""
    vals: List[float] = []
    for row in run.scan_history(keys=[key], page_size=10000):
        v = row.get(key)
        if v is not None:
            vals.append(v)
    return vals


def store_array(arr: List[float], path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        with path.open("w") as f:
            json.dump(arr, f)
    else:
        np.savez_compressed(path.with_suffix('.npz'), data=np.asarray(arr, dtype=np.float32))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    args = cli()
    api = wandb.Api()
    base_workspace = Path(__file__).resolve().parent.parent
    ext = 'json' if args.format == 'json' else 'npz'

    for run in api.runs(args.project):
        if not want(run, args):
            continue

        cfg = run.config
        algo = cfg.get("alg_name")
        cl_method = cfg.get("cl_method", "UNKNOWN_CL")

        if algo == 'ippo_cbp':
            algo = 'ippo'
            cl_method = 'CBP'
        if cl_method == 'EWC' and cfg.get("ewc_mode") == "online":
            cl_method = "Online_EWC"

        strategy = cfg.get("strategy")
        seq_len = cfg.get("seq_length")
        seed = max(cfg.get("seed", 1), 1)
        experiment = experiment_suffix(cfg)

        # Get reward setting info for logging
        sparse_rewards = cfg.get("sparse_rewards", False)
        individual_rewards = cfg.get("individual_rewards", False)
        reward_setting = "default"
        if sparse_rewards:
            reward_setting = "sparse_rewards"
        elif individual_rewards:
            reward_setting = "individual_rewards"

        # Don't skip 'main' experiments if they have reward settings or if explicitly requested
        if experiment == 'main' and not args.include_reward_experiments:
            # Only skip if it's truly a default main experiment (no reward settings)
            if reward_setting == "default":
                continue

        # find eval keys as W&B actually logged them
        eval_keys = discover_eval_keys(run)
        if not eval_keys:
            print(f"[warn] {run.name} has no Scaled_returns/ keys")
            continue

        exp_path = f"{strategy}_{seq_len}"

        # Handle repeat_sequence parameter
        if args.repeat_sequence is not None:
            repeat_sequence = cfg.get("repeat_sequence")
            if repeat_sequence is not None:
                exp_path += f"_rep_{repeat_sequence}"
                # effective_seq_len = seq_len * args.repeat_sequence
                print(f"[info] {run.name} using repeat_sequence={args.repeat_sequence}, seq_len={seq_len}")

        out_base = (base_workspace / args.output / algo / cl_method /
                    experiment / exp_path / f"seed_{seed}")

        print(f"[info] Processing {run.name} with setting: {experiment}")
        print(f"[info] Output path: {out_base}")

        # iterate keys, skipping existing files unless overwrite
        for key in discover_eval_keys(run):
            # choose filename
            if key == TRAINING_KEY:
                filename = f"training_soup.{ext}"
            else:
                idx, name, _ = KEY_PATTERN.match(key).groups()
                filename = f"{idx}_{name}_soup.{ext}"

            out = out_base / filename
            if out.exists() and not args.overwrite:
                print(f"→ {out} exists, skip")
                continue

            series = fetch_full_series(run, key)
            if not series:
                print(f"→ {out} no data, skip")
                continue

            print(f"→ writing {out}")
            store_array(series, out, args.format)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
