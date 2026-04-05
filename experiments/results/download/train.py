#!/usr/bin/env python3
"""
Download training-time performance curves from W&B runs.

For overcooked:  fetches the dense ``Soup/scaled`` training signal.
For SMAX / MPE:  no dense training key is logged, so the periodic
                 evaluation metric for the trained task is used instead.

Folder structure (mirrors eval.py exactly):
  data/<algo>/<cl_method>/<env_folder>/<agents>/<strategy>_<seq_len>/<experiment>/seed_<seed>/
    training_<metric>.{ext}        – full CL run  (overcooked only)
    {task_idx}_<metric>.{ext}      – single-task / forward-transfer run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import wandb
from wandb.apis.public import Run

from experiments.results.download.common import cli, build_filters, experiment_suffix, difficulty_string, unwrap_wandb_config
from experiments.results.download.eval import ENV_CONFIGS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_full_series(run: Run, key: str) -> List[float]:
    vals: List[float] = []
    for row in run.scan_history(keys=[key], page_size=10_000):
        v = row.get(key)
        if v is not None:
            vals.append(v)
    return vals


def store(arr: List[float], path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        with path.open("w") as f:
            json.dump(arr, f)
    else:
        np.savez_compressed(path.with_suffix(".npz"), data=np.asarray(arr, dtype=np.float32))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = cli()
    api = wandb.Api(timeout=180)
    base = Path(__file__).resolve().parent.parent
    ext = "json" if args.format == "json" else "npz"

    filters = build_filters(args)
    for run in api.runs(args.project, filters=filters, per_page=200):
        cfg = run.config
        if not isinstance(cfg, dict):
            cfg = unwrap_wandb_config(json.loads(cfg))
        algo = cfg.get("alg_name") or "ippo"
        if algo == "ippo_cbp":
            algo = "ippo"

        cl_method = cfg.get("cl_method", "UNKNOWN_CL")
        if cl_method.lower() == "ft":
            cl_method = "single"
        if cl_method == "EWC" and cfg.get("importance_mode") == "online":
            cl_method = "Online_EWC"
        elif cl_method == "MAS" and cfg.get("importance_mode") == "online":
            cl_method = "Online_MAS"

        env_name = cfg.get("env_name", "overcooked")
        env_cfg = ENV_CONFIGS.get(env_name)
        if env_cfg is None:
            print(f"[warn] {run.name}: unknown env_name={env_name!r}, skipping")
            continue

        strategy   = cfg.get("strategy")
        seq_len    = cfg.get("seq_length")
        seed       = max(cfg.get("seed", 1), 1)
        task_idx   = cfg.get("single_task_idx")   # None for full CL runs

        num_agents = cfg.get("num_allies") if env_name == "smax" else cfg.get("num_agents", 1)
        level_str  = difficulty_string(cfg)
        experiment = experiment_suffix(cfg)

        sequence = f"{strategy}_{seq_len}"
        rep = cfg.get("repeat_sequence", 1)
        if rep != 1:
            sequence += f"_rep_{rep}"

        agents_str = f"agents_{num_agents}" if num_agents else ""
        out_dir = (
            base / args.output / algo / cl_method
            / level_str / agents_str / sequence / experiment / f"seed_{seed}"
        )

        # ----------------------------------------------------------------
        # Determine which W&B key to fetch and what to call the output file
        # ----------------------------------------------------------------
        if env_cfg.training_key is None:
            print(f"[skip] {run.name}: {env_name} has no training key configured")
            continue

        key = env_cfg.training_key
        if task_idx is not None:
            filename = f"{task_idx}_{env_cfg.training_filename}.{ext}"
        else:
            filename = f"{env_cfg.training_filename}.{ext}"

        out_file = out_dir / filename
        if out_file.exists() and not args.overwrite:
            print(f"→ {out_file} exists, skip")
            continue

        series = fetch_full_series(run, key)
        if not series:
            print(f"[warn] {run.name}: key '{key}' has no data, skipping")
            continue

        print(f"→ writing {out_file}  (key={key})")
        store(series, out_file, ext)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
