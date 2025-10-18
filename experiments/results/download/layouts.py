#!/usr/bin/env python3
"""
Fetch the *exact* Overcooked layouts that were used in each W&B run and
save them to disk so they can be replayed later.

Location on disk:
    <OUT>/<algo>/<cl_method>/<experiment>/<strategy>_<seq_len>[_rep_N]/seed_<seed>/layouts.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb

# --------------------------------------------------------------------- helpers
# you already have these two in results/download/common.py — re-use them
from experiments.results.download.common import want, experiment_suffix, cli


# ------------------------------------------------------------------------ MAIN
def main() -> None:
    args = cli()
    api = wandb.Api()
    out_root = Path(args.output).expanduser()

    # override output folder and add layout-specific flag
    args.output = args.output or "layouts"
    if not hasattr(args, "format"):
        args.format = "json"

    for run in api.runs(args.project):
        if not want(run, args):
            continue

        cfg = run.config
        algo = cfg.get("alg_name")
        cl_method = cfg.get("cl_method", "UNKNOWN_CL")

        # unify the naming quirks
        if algo == "ippo_cbp":
            algo, cl_method = "ippo", "CBP"
        if cl_method == "EWC" and cfg.get("ewc_mode") == "online":
            cl_method = "Online_EWC"

        strategy = cfg.get("strategy")
        seq_len = cfg.get("seq_length")
        seed = max(cfg.get("seed", 1), 1)
        experiment = experiment_suffix(cfg)

        exp_folder = f"{strategy}_{seq_len}"
        rep = cfg.get("repeat_sequence", 1)
        if rep != 1:
            exp_folder += f"_rep_{rep}"

        out_dir = (out_root / algo / cl_method / experiment / exp_folder / f"seed_{seed}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"layouts.{args.format}"

        if out_file.exists() and not args.overwrite:
            print(f"→ {out_file} exists, skip")
            continue

        env_kwargs = cfg.get("env_kwargs")  # ← logged by the trainer
        if not env_kwargs:
            print(f"[warn] {run.name} has no 'env_kwargs' in its config")
            continue

        print(f"→ writing {out_file}")
        if args.format == "json":
            with out_file.open("w") as f:
                json.dump(env_kwargs, f, indent=2)
        else:  # compressed numpy archive
            import numpy as np
            np.savez_compressed(out_file.with_suffix('.npz'),
                                data=np.asarray(env_kwargs, dtype=object))
        print(f"[saved] {out_file.resolve()}")  # ← full path log


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
