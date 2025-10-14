#!/usr/bin/env python3
"""
Download only the training curve (e.g.\ ``Soup/scaled``) from W&B runs.

It re-uses the same CLI, filters and folder structure as
``download_evaluation.py`` but:

* ignores every key that starts with ``Evaluation/``;
* never skips runs whose ``experiment_suffix`` is "main".

Saved file:
  data/<algo>/<cl_method>/<experiment>/<strategy>_<seq_len>[_rep_N]/seed_<seed>/training_soup.{json|npz}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import wandb
from wandb.apis.public import Run

from experiments.results.download.common import cli, want, experiment_suffix

# --------------------------------------------------------------------------- #
TRAIN_KEY = "Soup/scaled"


def fetch_full_series(run: Run, key: str) -> List[float]:
    vals: List[float] = []
    for row in run.scan_history(keys=[key], page_size=10000):
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
        np.savez_compressed(path.with_suffix(".npz"),
                            data=np.asarray(arr, dtype=np.float32))


# --------------------------------------------------------------------------- #
def main() -> None:
    args = cli()  # same flags as before
    api = wandb.Api()
    base = Path(__file__).resolve().parent.parent
    ext = "json" if args.format == "json" else "npz"

    for run in api.runs(args.project):
        if not want(run, args):
            continue

        cfg = run.config
        algo = cfg.get("alg_name")
        cl_method = cfg.get("cl_method", "UNKNOWN_CL")
        if cl_method == "ft":
            cl_method = "single"

        seq_idx = cfg.get("single_task_idx")

        if algo == "ippo_cbp":
            algo, cl_method = "ippo", "CBP"
        if cl_method == "EWC" and cfg.get("ewc_mode") == "online":
            cl_method = "Online_EWC"

        strategy = cfg.get("strategy")
        seq_len = cfg.get("seq_length")
        seed = max(cfg.get("seed", 1), 1)
        experiment = experiment_suffix(cfg)  # keep *all* experiments

        exp_path = f"{strategy}_{seq_len}"
        rep = cfg.get("repeat_sequence", 1)
        if rep != 1:
            exp_path += f"_rep_{rep}"

        out_dir = (base / args.output / algo / cl_method /
                   experiment / exp_path / f"seed_{seed}")
        if seq_idx is not None:               # single-task baseline → env-specific file
            filename = f"{seq_idx}_training_soup.{ext}"
        else:                                 # regular continual run
            filename = f"training_soup.{ext}"
        out_file = out_dir / filename

        if out_file.exists() and not args.overwrite:
            print(f"→ {out_file} exists, skip")
            continue

        # make sure the run actually logged TRAIN_KEY
        df = run.history(samples=500)
        if TRAIN_KEY not in df.columns:
            print(f"[warn] {run.name} has no '{TRAIN_KEY}'")
            continue

        series = fetch_full_series(run, TRAIN_KEY)
        if not series:
            print(f"→ {out_file} no data, skip")
            continue

        print(f"→ writing {out_file}")
        store(series, out_file, args.format)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
