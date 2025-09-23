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
    p.add_argument("--wall_density", type=float, default=None, help="Wall density for the environment")
    p.add_argument("--difficulty", type=str, nargs="+", default=[], help="Difficulty levels for the environment")
    p.add_argument("--strategy", choices=["ordered", "random", "generate", "curriculum"], default=None)
    p.add_argument("--algos", nargs="+", default=[], help="Filter by alg_name")
    p.add_argument("--cl_methods", nargs="+", default=[], help="Filter by cl_method")
    p.add_argument("--wandb_tags", nargs="+", default=[], help="Require at least one tag")
    p.add_argument("--include_runs", nargs="+", default=[], help="Include runs by substring")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    # Reward settings arguments
    p.add_argument("--reward_settings", nargs="+", choices=["default", "sparse", "individual"], 
                   default=[], help="Filter by reward settings (sparse_rewards, individual_rewards)")
    p.add_argument("--include_reward_experiments", action="store_true", 
                   help="Include experiments with reward settings (sparse/individual)")

    # Complementary restrictions arguments
    p.add_argument("--complementary_restrictions", action="store_true", 
                   help="Filter by complementary restrictions experiments")

    # Number of agents parameter
    p.add_argument("--num_agents", type=int, default=None, help="Filter by number of agents")

    # Neural activity parameters
    p.add_argument("--include_dormant_ratio", action="store_true", 
                   help="Also download Neural_Activity/dormant_ratio data")

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
    if args.difficulty and cfg.get("difficulty") not in args.difficulty: return False
    if args.wall_density and cfg.get("wall_density") != args.wall_density: return False
    if args.num_agents and cfg.get("num_agents") != args.num_agents: return False

    # Filter by reward settings
    if args.reward_settings:
        sparse_rewards = cfg.get("sparse_rewards", False)
        individual_rewards = cfg.get("individual_rewards", False)

        current_setting = "default"
        if sparse_rewards:
            current_setting = "sparse"
        elif individual_rewards:
            current_setting = "individual"

        if current_setting not in args.reward_settings:
            return False

    # Filter by complementary restrictions
    if args.complementary_restrictions:
        complementary_restrictions = cfg.get("complementary_restrictions", False)
        if not complementary_restrictions:
            return False

    if 'tags' in cfg:
        tags = set(cfg['tags'])
        if args.wandb_tags and not tags.intersection(args.wandb_tags):
            return False
        if tags.intersection(FORBIDDEN_TAGS) and not tags.intersection(args.wandb_tags):
            return False
    return True


def experiment_suffix(cfg: dict) -> str:
    """Return folder name encoding ablation settings. Returns a single suffix."""

    if cfg.get("complementary_restrictions", False):
        return "complementary_restrictions"
    if cfg.get("sparse_rewards", False):
        return "sparse_rewards"
    if cfg.get("individual_rewards", False):
        return "individual_rewards"
    if not cfg.get("use_multihead", True) and cfg.get("cl_method") != "AGEM":
        return "no_multihead"
    if not cfg.get("use_task_id", True):
        return "no_task_id"
    if cfg.get("regularize_critic"):
        return "reg_critic"
    if not cfg.get("use_layer_norm", True):
        return "no_layer_norm"
    if cfg.get("use_cnn"):
        return "cnn"
    if cfg.get("difficulty") == 'easy':
        return "level_1"
    if cfg.get("difficulty") == 'medium':
        return "level_2"
    if cfg.get("difficulty") == 'hard':
        return "level_3"
    return "main"
