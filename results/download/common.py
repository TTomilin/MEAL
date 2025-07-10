from __future__ import annotations

import argparse

from wandb.apis.public import Run

FORBIDDEN_TAGS = {"TEST", "LOCAL"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--output", default="data", help="Base folder for output")
    p.add_argument("--format", choices=["json", "npz"], default="json", help="Output file format")
    p.add_argument("--seq_length", type=int, default=[])
    p.add_argument("--repeat_sequence", type=int, default=None, help="Repeat sequence value to multiply with seq_length")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--wall_density", type=float, default=0.15, help="Wall density for the environment")
    p.add_argument("--strategy", choices=["ordered", "random", "generate"], default=None)
    p.add_argument("--algos", nargs="+", default=[], help="Filter by alg_name")
    p.add_argument("--cl_methods", nargs="+", default=[], help="Filter by cl_method")
    p.add_argument("--wandb_tags", nargs="+", default=[], help="Require at least one tag")
    p.add_argument("--include_runs", nargs="+", default=[], help="Include runs by substring")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# FILTER
# ---------------------------------------------------------------------------
def want(run: Run, args: argparse.Namespace) -> bool:
    cfg = run.config
    if any(tok in run.name for tok in args.include_runs): return True
    if run.state != "finished": return False
    if args.seeds and cfg.get("seed") not in args.seeds: return False
    if args.algos and cfg.get("alg_name") not in args.algos: return False
    if args.cl_methods and cfg.get("cl_method") not in args.cl_methods: return False
    if args.seq_length and cfg.get("seq_length") != args.seq_length: return False
    if args.strategy and cfg.get("strategy") != args.strategy: return False
    if args.wall_density and cfg.get("wall_density") != args.wall_density: return False
    if 'tags' in cfg:
        tags = set(cfg['tags'])
        if args.wandb_tags and not tags.intersection(args.wandb_tags):
            return False
        if tags.intersection(FORBIDDEN_TAGS) and not tags.intersection(args.wandb_tags):
            return False
    return True


def experiment_suffix(cfg: dict) -> str:
    """Return folder name encoding ablation settings."""
    suffixes = []
    if not cfg.get("evaluation", True):
        return "plasticity"

    ##### ----- Temporary hardcoded levels for wall density ----- #####
    if cfg.get("wall_density") == 0.25:
        return "level_2"
    if cfg.get("wall_density") == 0.35:
        return "level_3"
    ##### ----- Temporary hardcoded levels for wall density ----- #####

    if not cfg.get("use_multihead", True):
        suffixes.append("no_multihead")
    if not cfg.get("use_task_id", True):
        suffixes.append("no_task_id")
    if cfg.get("regularize_critic"):
        suffixes.append("reg_critic")
    if not cfg.get("use_layer_norm", True):
        suffixes.append("no_layer_norm")
    if cfg.get("use_cnn"):
        suffixes.append("cnn")
    return "-".join(suffixes) if suffixes else "main"
