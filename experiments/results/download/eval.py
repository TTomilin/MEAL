#!/usr/bin/env python3
"""Download evaluation curves *per environment* for the MARL continual-learning
benchmark and store them one metric per file.

Supported environments (detected via config.env_name):
  overcooked  – Evaluation/Soup_Scaled/{idx}_{name}_{seed}   → *_soup.{ext}
  jaxnav      – Evaluation/Success/{idx}_{name}               → *_success.{ext}
  mpe         – Evaluation/CoverageFraction/{idx}_{name}      → *_coverage_fraction.{ext}
  smax        – Evaluation/WinRate/{idx}_{name}               → *_win_rate.{ext}

New environments are added by inserting one entry into ENV_CONFIGS.

Optimized logic:
1. Discover available evaluation keys per run via run.history(samples=1).
2. Fetch each key's full time series separately, only once.
3. Skip keys whose output files already exist (unless --overwrite).
4. Write files in
   data/<algo>/<cl_method>/<env_folder>/<agents>/<strategy>_<seq_len>/<experiment>/seed_<seed>/.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import wandb
from wandb.apis.public import Run

from experiments.results.download.common import (
    cli, experiment_suffix, unwrap_wandb_config, build_filters, difficulty_string,
)

# ---------------------------------------------------------------------------
# Per-environment configuration
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    eval_pattern: re.Pattern   # must have groups: (idx, name[, ...])
    metric_name:  str          # used as suffix in eval filenames
    training_key: Optional[str]
    training_filename: Optional[str]


ENV_CONFIGS: dict[str, EnvConfig] = {
    "overcooked": EnvConfig(
        eval_pattern=re.compile(r"^Evaluation/Soup_Scaled/(\d+)_(.+)_(\d+)$"),
        metric_name="soup",
        training_key="Soup/scaled",
        training_filename="training_soup",
    ),
    "jaxnav": EnvConfig(
        eval_pattern=re.compile(r"^Evaluation/Success/(\d+)_(.+)$"),
        metric_name="success",
        training_key="Return",
        training_filename="training_return",
    ),
    "mpe": EnvConfig(
        eval_pattern=re.compile(r"^Evaluation/CoverageFraction/(\d+)_(.+)$"),
        metric_name="coverage_fraction",
        training_key="coverage_fraction",
        training_filename="training_coverage_fraction",
    ),
    "smax": EnvConfig(
        eval_pattern=re.compile(r"^Evaluation/Returns/(\d+)_(.+)$"),
        metric_name="return",
        training_key="kill_fraction",
        training_filename="training_kill_fraction",
    ),
}

DORMANT_RATIO_KEY = "Neural_Activity/dormant_ratio"
PARTNER_EVAL_PATTERN = re.compile(r"^Eval/EgoReturn_Partner(\d+)$")
TRAINING_RETURNS_KEY = "Train/Ego_returned_episode_returns"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_eval_keys(
    run: Run,
    env_cfg: EnvConfig,
    include_dormant_ratio: bool = False,
    extra_metric_prefixes: List[str] | None = None,
) -> List[str]:
    """Return sorted eval keys + optional training/dormant-ratio/extra keys."""
    df = run.history(samples=500)
    keys = [k for k in df.columns if env_cfg.eval_pattern.match(k)]

    if env_cfg.training_key and env_cfg.training_key in df.columns:
        keys.append(env_cfg.training_key)

    if include_dormant_ratio and DORMANT_RATIO_KEY in df.columns:
        keys.append(DORMANT_RATIO_KEY)

    for prefix in (extra_metric_prefixes or []):
        prefix_slash = prefix.rstrip("/") + "/"
        keys += [k for k in df.columns if k.startswith(prefix_slash)]

    def sort_key(k: str) -> int:
        m = env_cfg.eval_pattern.match(k)
        if m:
            return int(m.group(1))
        if k == env_cfg.training_key:
            return 10 ** 6
        if k == DORMANT_RATIO_KEY:
            return 10 ** 6 + 1
        return 10 ** 6 + 2

    return sorted(keys, key=sort_key)


def eval_filename(key: str, env_cfg: EnvConfig, ext: str) -> str:
    """Map a W&B key to the output filename — just {idx}_{metric}.{ext}."""
    m = env_cfg.eval_pattern.match(key)
    idx = m.group(1)
    return f"{idx}_{env_cfg.metric_name}.{ext}"


def extra_metric_filename(key: str, prefix: str, ext: str) -> str:
    """Map an extra-metric W&B key to an output filename — just {idx}_{suffix}.{ext}.

    e.g. key='Evaluation/Heterogeneity/17_medium_gen_17', prefix='Evaluation/Heterogeneity'
         → '17_heterogeneity.{ext}'
    """
    suffix_label = prefix.rstrip("/").split("/")[-1].lower()
    rest = key[len(prefix.rstrip("/")) :].lstrip("/")
    idx = rest.split("_")[0]
    return f"{idx}_{suffix_label}.{ext}"


def fetch_full_series(run: Run, key: str, cfg: dict) -> List[float]:
    """Fetch every recorded value for a single key via scan_history."""
    page_size = 1e6 if cfg.get("seq_length", 0) > 20 else 1e4
    vals: List[float] = []
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
        np.savez_compressed(path.with_suffix(".npz"), data=np.asarray(arr, dtype=np.float32))


def discover_partner_keys(run: Run) -> List[str]:
    df = run.history(samples=500)
    keys = [TRAINING_RETURNS_KEY] if TRAINING_RETURNS_KEY in df.columns else []
    keys += [c for c in df.columns if PARTNER_EVAL_PATTERN.match(c)]

    def partner_idx(k: str) -> int:
        if k == TRAINING_RETURNS_KEY:
            return -1
        m = PARTNER_EVAL_PATTERN.match(k)
        return int(m.group(1)) if m else 999

    return sorted(keys, key=partner_idx)


def get_layout_from_config(cfg: dict):
    from meal.env.layouts.presets import (
        easy_layouts_legacy, medium_layouts_legacy,
        hard_layouts_legacy, overcooked_layouts,
    )
    layout_name = cfg.get("layout_name", "")
    for store in (easy_layouts_legacy, medium_layouts_legacy,
                  hard_layouts_legacy, overcooked_layouts):
        if layout_name in store:
            return layout_name, store[layout_name]
    if layout_name:
        return layout_name, None

    layout_difficulty = cfg.get("layout_difficulty", "easy")
    layout_idx = cfg.get("layout_idx", 0)
    stores = {"easy": easy_layouts_legacy, "medium": medium_layouts_legacy,
              "hard": hard_layouts_legacy}
    store = stores.get(layout_difficulty, easy_layouts_legacy)
    names = list(store.keys())
    if layout_idx < len(names):
        return names[layout_idx], store[names[layout_idx]]
    return f"layout_{layout_idx}", None


def calculate_max_soup(layout_dict, max_steps: int = 400, n_agents: int = 2) -> float:
    if layout_dict is None:
        return 1.0
    try:
        from meal.env.utils.max_soup_calculator import calculate_max_soup as _calc
        return _calc(layout_dict, max_steps, n_agents=n_agents)
    except Exception as e:
        print(f"Warning: Failed to calculate max soup: {e}, using default of 1.0")
        return 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = cli()
    api = wandb.Api(timeout=180)
    base_workspace = Path(__file__).resolve().parent.parent
    ext = "json" if args.format == "json" else "npz"

    filters = build_filters(args)
    runs = api.runs(args.project, filters=filters, per_page=200)

    for run in runs:
        cfg = run.config
        if not isinstance(cfg, dict):
            cfg = unwrap_wandb_config(json.loads(cfg))

        algo = cfg.get("alg_name") or "ppo"
        if algo == "ippo_cbp":
            algo = "ippo"

        cl_method = cfg.get("cl_method", "UNKNOWN_CL")
        if cl_method == "EWC" and cfg.get("importance_mode") == "online":
            cl_method = "Online_EWC"
        elif cl_method == "MAS" and cfg.get("importance_mode") == "online":
            cl_method = "Online_MAS"

        env_name = cfg.get("env_name", "overcooked")
        if env_name == "overcooked_po":
            cl_method = f"{cl_method}_partial"
            env_name = "overcooked"

        strategy   = cfg.get("strategy")
        seq_len    = cfg.get("seq_length")
        seed       = max(cfg.get("seed", 1), 1)
        num_agents = cfg.get("num_allies") if env_name == "smax" else cfg.get("num_agents", 1)
        level_string = difficulty_string(cfg)
        experiment   = experiment_suffix(cfg)

        # ----------------------------------------------------------------
        # Partner-generalization runs (overcooked only)
        # ----------------------------------------------------------------
        partner_keys = discover_partner_keys(run)
        if partner_keys:
            print(f"[info] Processing partner generalization run: {run.name}")
            layout_name, layout_dict = get_layout_from_config(cfg)
            max_soup = calculate_max_soup(layout_dict)
            print(f"[info] Layout: {layout_name}, Max soup: {max_soup}")

            out_base = base_workspace / args.output / algo / cl_method / "partners_8" / f"seed_{seed}"
            for key in partner_keys:
                if key == TRAINING_RETURNS_KEY:
                    filename = f"training_soup.{ext}"
                else:
                    m = PARTNER_EVAL_PATTERN.match(key)
                    if not m:
                        continue
                    filename = f"eval_partner_{m.group(1)}_soup.{ext}"

                out = out_base / filename
                if out.exists() and not args.overwrite:
                    print(f"→ {out} exists, skip")
                    continue

                series = fetch_full_series(run, key, cfg)
                if not series:
                    print(f"→ {out} no data, skip")
                    continue

                normalized = [v / max_soup for v in series]
                print(f"→ writing {out} (max_soup={max_soup})")
                store_array(normalized, out, args.format)
            continue

        # ----------------------------------------------------------------
        # Regular eval-key runs
        # ----------------------------------------------------------------
        env_cfg = ENV_CONFIGS.get(env_name)
        if env_cfg is None:
            print(f"[warn] {run.name}: unknown env_name={env_name!r}, skipping")
            continue

        eval_keys = discover_eval_keys(
            run, env_cfg,
            include_dormant_ratio=args.include_dormant_ratio,
            extra_metric_prefixes=args.extra_metrics,
        )
        if not eval_keys:
            print(f"[warn] {run.name} has no matched eval keys")
            continue

        sequence = f"{strategy}_{seq_len}"
        if args.repeat_sequence is not None:
            repeat_sequence = cfg.get("repeat_sequence")
            if repeat_sequence is not None:
                sequence += f"_rep_{repeat_sequence}"

        agents_string = f"agents_{num_agents}" if num_agents else ""
        out_base = (
            base_workspace / args.output / algo / cl_method
            / level_string / agents_string / sequence / experiment / f"seed_{seed}"
        )

        print(f"[info] {run.name}  env={env_name}  →  {out_base}")

        # Build a quick lookup: key → prefix for extra metrics
        extra_key_to_prefix = {}
        for prefix in args.extra_metrics:
            prefix_slash = prefix.rstrip("/") + "/"
            for k in eval_keys:
                if k.startswith(prefix_slash):
                    extra_key_to_prefix[k] = prefix

        for key in eval_keys:
            # Determine filename
            if key == env_cfg.training_key:
                filename = f"{env_cfg.training_filename}.{ext}"
            elif key == DORMANT_RATIO_KEY:
                filename = f"dormant_ratio.{ext}"
            elif key in extra_key_to_prefix:
                filename = extra_metric_filename(key, extra_key_to_prefix[key], ext)
            else:
                filename = eval_filename(key, env_cfg, ext)

            out = out_base / filename
            if out.exists() and not args.overwrite:
                print(f"→ {out} exists, skip")
                continue

            series = fetch_full_series(run, key, cfg)
            if not series:
                print(f"→ {out} no data, skip")
                continue

            print(f"→ writing {out}")
            store_array(series, out, args.format)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
