from __future__ import annotations

import argparse
import json

from wandb.apis.public import Run


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
def build_filters(args: argparse.Namespace) -> dict:
    """Server-side filters for wandb.Api().runs()."""
    # base: only finished runs
    f: dict = {"state": "finished"}

    # config.* filters mirroring the old `want` logic

    if args.seeds:
        f["config.seed"] = {"$in": args.seeds}

    if args.algos:
        # in this project it's `alg_name`
        f["config.alg_name"] = {"$in": args.algos}

    if args.cl_methods:
        f["config.cl_method"] = {"$in": args.cl_methods}

    if args.seq_length:
        # you used it as a single int in `want`, so same here
        f["config.seq_length"] = args.seq_length

    if args.strategy:
        f["config.strategy"] = args.strategy

    if args.difficulty:
        f["config.difficulty"] = {"$in": args.difficulty}

    if args.wall_density is not None:
        f["config.wall_density"] = args.wall_density

    if args.num_agents is not None:
        f["config.num_agents"] = args.num_agents

    # reward_settings logic: default / sparse / individual
    if args.reward_settings:
        ors = []

        # "default": neither sparse_rewards nor individual_rewards
        if "default" in args.reward_settings:
            ors.append({
                "config.sparse_rewards": {"$in": [False, None]},
                "config.individual_rewards": {"$in": [False, None]},
            })

        if "sparse" in args.reward_settings:
            ors.append({"config.sparse_rewards": True})

        if "individual" in args.reward_settings:
            ors.append({"config.individual_rewards": True})

        if len(ors) == 1:
            # just merge a single case into f
            f.update(ors[0])
        else:
            # combine base filter with OR over reward settings
            f = {"$and": [f, {"$or": ors}]}

    # complementary_restrictions flag
    if args.complementary_restrictions:
        f["config.complementary_restrictions"] = True

    # wandb tags: these are run-level tags, not config tags
    if args.wandb_tags:
        f["tags"] = {"$in": args.wandb_tags}

    # include specific runs by (partial) name: keep old semantics using regex
    # old `want` did: any(tok in run.name for tok in args.include_runs)
    if args.include_runs:
        ors = [{"display_name": {"$regex": tok}} for tok in args.include_runs]
        # $or coexists with the ANDed topline filters
        f = {"$or": [f, *ors]}

    return f


def want(run: Run, args: argparse.Namespace) -> bool:
    cfg = run.config
    if not isinstance(cfg, dict):
        try:
            cfg = unwrap_wandb_config(json.loads(cfg))
        except Exception as e:
            print(f"Could not parse config for run {run.name}: {e}")
            return False
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
    return True


def unwrap_wandb_config(cfg_like):
    if isinstance(cfg_like, dict):
        # single-level wandb wrapper
        if set(cfg_like.keys()) == {"value"}:
            return unwrap_wandb_config(cfg_like["value"])
        # otherwise recurse into all dict values
        return {k: unwrap_wandb_config(v) for k, v in cfg_like.items()}
    elif isinstance(cfg_like, list):
        return [unwrap_wandb_config(v) for v in cfg_like]
    else:
        return cfg_like


def experiment_suffix(cfg: dict) -> str:
    """Return folder name encoding ablation settings. Returns a single suffix."""
    if cfg.get("big_network", False):
        return "big_network"
    if cfg.get("separated_agents", False):
        return "separated_agents"
    if cfg.get("sticky_actions", False):
        return "sticky_actions"
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
    return ""

def difficulty_string(cfg: dict) -> str:
    if cfg.get("difficulty") == 'easy':
        return "level_1"
    if cfg.get("difficulty") == 'medium':
        return "level_2"
    if cfg.get("difficulty") == 'hard':
        return "level_3"
    if cfg.get("difficulty") == 'extreme':
        return "level_4"
    return "UNDEFINED"