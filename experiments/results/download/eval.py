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

Enhanced with partner generalization support:
- Download training returns from Train/Ego_returned_episode_returns
- Download eval returns from Eval/EgoReturn_Partner{partner_idx}
- Calculate max soup from layout configuration and normalize returns
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import wandb
from wandb.apis.public import Run

from experiments.results.download.common import cli, want, experiment_suffix

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
EVAL_PREFIX = "Evaluation/Soup_Scaled/"
KEY_PATTERN = re.compile(rf"^{re.escape(EVAL_PREFIX)}(\d+)__(.+)_(\d+)$")
TRAINING_KEY = "Soup/scaled"
DORMANT_RATIO_KEY = "Neural_Activity/dormant_ratio"
PARTNER_EVAL_PATTERN = re.compile(r"^Eval/EgoReturn_Partner(\d+)$")
TRAINING_RETURNS_KEY = "Train/Ego_returned_episode_returns"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def discover_eval_keys(run: Run, include_dormant_ratio: bool = False) -> List[str]:
    """Retrieve & sort eval keys, plus the one training key if present."""
    df = run.history(samples=500)
    # only exact eval keys
    keys = [k for k in df.columns if KEY_PATTERN.match(k)]
    # include training series, if logged
    if TRAINING_KEY in df.columns:
        keys.append(TRAINING_KEY)

    # include dormant ratio, if requested and logged
    if include_dormant_ratio and DORMANT_RATIO_KEY in df.columns:
        keys.append(DORMANT_RATIO_KEY)

    # sort eval ones by idx, leave training and dormant ratio last
    def idx_of(key: str) -> int:
        m = KEY_PATTERN.match(key)
        if m:
            return int(m.group(1))
        elif key == TRAINING_KEY:
            return 10 ** 6
        elif key == DORMANT_RATIO_KEY:
            return 10 ** 6 + 1
        else:
            return 10 ** 6 + 2

    return sorted(keys, key=idx_of)


def fetch_full_series(run: Run, key: str) -> List[float]:
    """Fetch every recorded value for a single key via scan_history."""
    vals: List[float] = []
    page_size = 1e6 if run.config['seq_length'] > 20 else 1e4
    for row in run.scan_history(keys=[key], page_size=page_size):
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


def get_layout_from_config(cfg: dict) -> tuple[str, dict]:
    """Extract layout name and layout dict from run config.

    Returns:
        tuple: (layout_name, layout_dict) where layout_dict is the actual layout configuration
    """
    # Import here to avoid circular imports
    from meal.env.layouts.presets import easy_layouts_legacy, medium_layouts_legacy, hard_layouts_legacy, \
        overcooked_layouts

    layout_name = cfg.get("layout_name", "")

    if layout_name:
        # Layout name was explicitly specified
        # Try to find it in the available layouts
        if layout_name in easy_layouts_legacy:
            return layout_name, easy_layouts_legacy[layout_name]
        elif layout_name in medium_layouts_legacy:
            return layout_name, medium_layouts_legacy[layout_name]
        elif layout_name in hard_layouts_legacy:
            return layout_name, hard_layouts_legacy[layout_name]
        elif layout_name in overcooked_layouts:
            return layout_name, overcooked_layouts[layout_name]
        else:
            print(f"Warning: Layout '{layout_name}' not found in predefined layouts")
            return layout_name, None
    else:
        # Layout was specified by index, try to infer from other config parameters
        layout_difficulty = cfg.get("layout_difficulty", "easy")
        layout_idx = cfg.get("layout_idx", 0)

        if layout_difficulty == "easy":
            layout_names = list(easy_layouts_legacy.keys())
            if layout_idx < len(layout_names):
                layout_name = layout_names[layout_idx]
                return layout_name, easy_layouts_legacy[layout_name]
        elif layout_difficulty == "medium":
            layout_names = list(medium_layouts_legacy.keys())
            if layout_idx < len(layout_names):
                layout_name = layout_names[layout_idx]
                return layout_name, medium_layouts_legacy[layout_name]
        elif layout_difficulty == "hard":
            layout_names = list(hard_layouts_legacy.keys())
            if layout_idx < len(layout_names):
                layout_name = layout_names[layout_idx]
                return layout_name, hard_layouts_legacy[layout_name]

        # Fallback: use layout_idx as name
        return f"layout_{layout_idx}", None


def calculate_max_soup(layout_dict: dict, max_steps: int = 400, n_agents: int = 2) -> float:
    """Calculate max soup for a layout using the estimate_max_soup function.

    Args:
        layout_dict: The layout configuration dictionary
        max_steps: Maximum episode steps (default 400)
        n_agents: Number of agents (default 2)

    Returns:
        float: Maximum soup count for the layout
    """
    if layout_dict is None:
        print("Warning: Layout dict is None, using default max soup of 1.0")
        return 1.0

    try:
        # Import here to avoid circular imports
        from meal.env.utils.max_soup_calculator import calculate_max_soup
        return calculate_max_soup(layout_dict, max_steps, n_agents=n_agents)
    except Exception as e:
        print(f"Warning: Failed to calculate max soup: {e}, using default of 1.0")
        return 1.0


def discover_partner_keys(run: Run) -> List[str]:
    """Discover partner generalization keys in the run.

    Returns:
        List of keys including training returns and partner eval returns
    """
    df = run.history(samples=500)
    keys = []

    # Add training returns key if present
    if TRAINING_RETURNS_KEY in df.columns:
        keys.append(TRAINING_RETURNS_KEY)

    # Add partner evaluation keys
    for col in df.columns:
        if PARTNER_EVAL_PATTERN.match(col):
            keys.append(col)

    # Sort partner eval keys by partner index
    def partner_idx(key: str) -> int:
        if key == TRAINING_RETURNS_KEY:
            return -1  # Training comes first
        match = PARTNER_EVAL_PATTERN.match(key)
        return int(match.group(1)) if match else 999

    return sorted(keys, key=partner_idx)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    args = cli()
    api = wandb.Api(timeout=180)
    base_workspace = Path(__file__).resolve().parent.parent
    ext = 'json' if args.format == 'json' else 'npz'

    for run in api.runs(args.project):
        if not want(run, args):
            continue

        cfg = run.config
        algo = cfg.get("alg_name")
        cl_method = cfg.get("cl_method", "UNKNOWN_CL")

        if algo is None:
            algo = 'ppo'  # Default to PPO if not specified
        elif algo == 'ippo_cbp':
            algo = 'ippo'
            cl_method = 'CBP'
        if cl_method == 'EWC' and cfg.get("ewc_mode") == "online":
            cl_method = "Online_EWC"

        # Handle partial observability experiments
        if cfg.get("env_name") == "overcooked_po":
            cl_method = f"{cl_method}_partial"

        strategy = cfg.get("strategy")
        seq_len = cfg.get("seq_length")
        seed = max(cfg.get("seed", 1), 1)
        num_agents = cfg.get("num_agents", 1)  # Default to 1 agent if not specified
        experiment = experiment_suffix(cfg)

        # Get reward setting info for logging
        sparse_rewards = cfg.get("sparse_rewards", False)
        individual_rewards = cfg.get("individual_rewards", False)
        reward_setting = "default"
        if sparse_rewards:
            reward_setting = "sparse_rewards"
        elif individual_rewards:
            reward_setting = "individual_rewards"

        # Check if this is a partner generalization run
        partner_keys = discover_partner_keys(run)
        is_partner_run = len(partner_keys) > 0

        if is_partner_run:
            # Handle partner generalization run
            print(f"[info] Processing partner generalization run: {run.name}")

            # Get layout information and calculate max soup
            layout_name, layout_dict = get_layout_from_config(cfg)
            max_soup = calculate_max_soup(layout_dict)

            print(f"[info] Layout: {layout_name}, Max soup: {max_soup}")

            # Create storage path for partner runs with 8 hardcode partners for now
            out_base = (base_workspace / args.output / algo / cl_method / "partners_8" / f"seed_{seed}")

            print(f"[info] Partner run output path: {out_base}")

            # Process partner generalization keys
            for key in partner_keys:
                if key == TRAINING_RETURNS_KEY:
                    filename = f"training_soup.{ext}"
                else:
                    # Extract partner index from key
                    match = PARTNER_EVAL_PATTERN.match(key)
                    if match:
                        partner_idx = match.group(1)
                        filename = f"eval_partner_{partner_idx}_soup.{ext}"
                    else:
                        continue

                out = out_base / filename
                if out.exists() and not args.overwrite:
                    print(f"→ {out} exists, skip")
                    continue

                # Fetch and normalize the series
                series = fetch_full_series(run, key)
                if not series:
                    print(f"→ {out} no data, skip")
                    continue

                # Normalize by max soup
                normalized_series = [val / max_soup for val in series]

                print(f"→ writing {out} (normalized by max_soup={max_soup})")
                store_array(normalized_series, out, args.format)

            continue  # Skip regular eval key processing for partner runs

        # find eval keys as W&B actually logged them
        eval_keys = discover_eval_keys(run, include_dormant_ratio=args.include_dormant_ratio)
        if not eval_keys:
            print(f"[warn] {run.name} has no Scaled_returns/ keys and no partner keys")
            continue

        exp_path = f"{strategy}_{seq_len}"

        # Handle repeat_sequence parameter
        if args.repeat_sequence is not None:
            repeat_sequence = cfg.get("repeat_sequence")
            if repeat_sequence is not None:
                exp_path += f"_rep_{repeat_sequence}"
                # effective_seq_len = seq_len * args.repeat_sequence
                print(f"[info] {run.name} using repeat_sequence={args.repeat_sequence}, seq_len={seq_len}")

        agents_string = f"agents_{num_agents}" if args.num_agents else ""

        out_base = (
                    base_workspace / args.output / algo / cl_method / experiment / agents_string / exp_path / f"seed_{seed}")

        print(f"[info] Processing {run.name} with setting: {experiment}")
        print(f"[info] Output path: {out_base}")

        # iterate keys, skipping existing files unless overwrite
        for key in discover_eval_keys(run, include_dormant_ratio=args.include_dormant_ratio):
            # choose filename
            if key == TRAINING_KEY:
                filename = f"training_soup.{ext}"
            elif key == DORMANT_RATIO_KEY:
                filename = f"dormant_ratio.{ext}"
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
