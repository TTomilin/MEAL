'''Main entry point for running teammate generation algorithms.

One run trains one BRDiv population per ``num_seeds`` entry (``seed`` is the generation seed). Outputs go to a
directory that names the layout, population size and generation seed, and contain:

  params_seed{i}_agent{j}.pt   one pickle of {"actor_params": ...} per member (i = index inside this run)
  config.pckl                  the dataclass as given (legacy format)
  generation.json              resolved settings, interaction accounting and the explicit member list
                               -> input of ``python -m experiments.partner_adaptation.partner_bank``

There is deliberately no generic ``params.pt``: it was a copy of whichever member was written last.
'''
import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import tyro
import wandb

from meal.env.overcooked.layouts.presets import overcooked_layouts
from experiments.partner_adaptation.partner_generation.BRDiv import run_brdiv
from experiments.partner_adaptation.partner_generation.utils import frozendict_from_layout_repr

MEMBER_FILE_FMT = "params_seed{i}_agent{j}.pt"


@dataclass
class TrainConfig:
    # Wandb and other logging
    project: str = "MEAL"
    mode: str = "online"  # Literal["online", "offline", "disabled"]
    group: str = "overcooked"
    entity: str = ""
    checkpoint_path: str = "checkpoints"
    checkpoint_freq: int = 50  # Checkpoint every N updates

    # MEAL
    # Pregenerated MEAL layouts that we are interested in.
    layouts_path: str = "meal/env/layouts/"

    # Overcooked
    env_name: str = "overcooked"
    layout_difficulty: str = "easy"
    layout_idx: int = 0
    layout_name: str = ""  # If specified, overrides layout_idx and names the output directory

    rew_shaping_horizon: int = 2.5e8
    num_agents: int = 2

    # teammate generation
    alg = "brdiv"

    # Actor-Critic
    activation: str = "tanh"
    fc_dim_size: int = 256
    gru_hidden_dim: int = 256

    partner_pop_size: int = 3
    xp_loss_weights: float = 1
    num_checkpoints: int = 5
    num_seeds: int = 1

    seed: int = 0  # generation seed: population i of this run uses split(PRNGKey(seed), num_seeds)[i]

    # Training
    lr: float = 1e-3
    anneal_lr: bool = False
    num_envs_xp: int = 32
    num_envs_sp: int = 32
    num_steps: int = 400
    # Interaction budget per seed in *agent* transitions (every environment step has `num_agents` of them).
    # See interaction_counts for what is actually trained.
    total_timesteps: int = 2.5e8
    update_epochs: int = 8
    num_minibatches: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    # Eval
    num_eval_episodes: int = 20

    log_train_out: bool = True

    def __post_init__(self):
        ### MEAL ###

        if self.layout_difficulty == "medium":
            self.layouts_path = self.layouts_path + "gen_20_medium.json"
        elif self.layout_difficulty == "easy":
            self.layouts_path = self.layouts_path + "gen_20_easy.json"
        elif self.layout_difficulty == "hard":
            self.layouts_path = self.layouts_path + "gen_20_hard.json"

        ### BRDiv ###
        self.num_envs = self.num_envs_xp + self.num_envs_sp
        self.num_game_agents = self.num_agents

        self.num_actors = 2 * self.num_envs
        self.num_controlled_actors = self.num_actors

        self.num_conf_actors = self.num_envs
        self.num_br_actors = self.num_envs

        #############
        self.num_updates = int(self.total_timesteps //
                               (self.num_agents * self.num_steps * self.num_envs))
        self.minibatch_size = self.num_actors * \
                              self.num_steps // self.num_minibatches
        self.minibatch_size_ego = ((
                                           self.num_game_agents - 1) * self.num_actors * self.num_steps) // self.num_minibatches
        self.minibatch_size_br = (
                                         self.num_actors * self.num_steps) // self.num_minibatches

        print("Number of updates: ", self.num_updates)


def read_layouts(config):
    with open(config.layouts_path, "r") as f:
        layouts = json.load(f)
    return layouts


def get_run_string(config: TrainConfig):
    layout = config.layout_name or f"{config.layout_difficulty}_{config.layout_idx}"
    return f"FF_BRDIV_IPPO_Overcooked_{layout}"


def interaction_counts(config: TrainConfig) -> dict:
    """What a run actually trains on, per seed and in total.

    ``total_timesteps`` is a budget of agent transitions; updates are whole, so the trained amount is the
    largest multiple of one update (``num_steps * num_envs`` joint transitions) that fits. Evaluation episodes
    (checkpoint evaluation, ``num_eval_episodes``) are not training interactions and are not counted.
    """
    joint_per_update = config.num_steps * config.num_envs
    joint = config.num_updates * joint_per_update
    agent = joint * config.num_agents
    return {
        "requested_agent_transitions_per_seed": config.total_timesteps,
        "num_updates": config.num_updates,
        "joint_env_transitions_per_update": joint_per_update,
        "joint_env_transitions_per_seed": joint,
        "agent_transitions_per_seed": agent,
        "unused_requested_agent_transitions_per_seed": config.total_timesteps - agent,
        "num_seeds": config.num_seeds,
        "joint_env_transitions_total": joint * config.num_seeds,
        "agent_transitions_total": agent * config.num_seeds,
    }


def resolve_layout(config: TrainConfig):
    """Returns (env kwargs, resolved layout name, source description)."""
    if config.layout_name != "":
        if config.layout_name not in overcooked_layouts:
            raise ValueError(f"unknown layout '{config.layout_name}'; available: {sorted(overcooked_layouts)}")
        return {"layout": overcooked_layouts[config.layout_name]}, config.layout_name, "preset"
    layouts = read_layouts(config)
    layout = frozendict_from_layout_repr(layouts[config.layout_idx]["layout"])
    return {"layout": layout}, f"{config.layout_difficulty}_{config.layout_idx}", f"{config.layouts_path}"


def resolve_save_dir(config: TrainConfig, fallback_name: str) -> str:
    """Deterministic when a layout name is given, so a run is identified by layout, size and seed."""
    if config.layout_name:
        name = f"brdiv_{config.layout_name}_pop{config.partner_pop_size}_gseed{config.seed}"
    else:
        name = fallback_name
    return os.path.join(config.checkpoint_path, name)


def write_generation_record(save_dir, config, layout_name, layout_source, status, members=None):
    record = {
        "format_version": 1,
        "status": status,
        **asdict(config),
        "resolved_layout_name": layout_name,
        "layout_source": layout_source,
        "interactions": interaction_counts(config),
        "members": members or [],
    }
    with open(os.path.join(save_dir, "generation.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)


def save_population_members(save_dir, partner_params, num_seeds, partner_pop_size):
    """Write one pickle per member; returns the explicit member list recorded in generation.json."""
    members = []
    for i in range(num_seeds):
        for j in range(partner_pop_size):
            params = jax.tree.map(lambda x: x[i, j], partner_params)
            name = MEMBER_FILE_FMT.format(i=i, j=j)
            with open(os.path.join(save_dir, name), "wb") as f:
                pickle.dump({"actor_params": params}, f)
            members.append({"seed_index": i, "member_index": j, "file": name})
    return members


def generate_population(config: TrainConfig) -> str:
    """Trains the population(s) described by ``config``; returns the output directory."""
    if config.num_updates < 1:
        minimum = config.num_agents * config.num_steps * config.num_envs
        raise ValueError(f"total_timesteps={config.total_timesteps} is less than one update "
                         f"({minimum} agent transitions with these settings)")
    env_kwargs, layout_name, layout_source = resolve_layout(config)

    tags = ["FF", "BRDIV", "IPPO", layout_name]
    run_string = f"{get_run_string(config)}_SEED_{config.seed}"
    run = wandb.init(
        project=config.project,
        group=config.group,
        mode=config.mode,
        config=asdict(config),
        save_code=True,
        tags=tags,
    )
    if run.sweep_id is not None:
        run.name = run.sweep_id + "___" + run_string
    else:
        run.name = run.name + "___" + run_string
    print("XPID ID name:")
    print(run.name)
    print("-------------")

    save_dir = resolve_save_dir(config, run.name)
    existing = [p for p in Path(save_dir).glob("params_seed*_agent*.pt")] if os.path.isdir(save_dir) else []
    if existing:
        raise FileExistsError(f"{save_dir} already contains {len(existing)} member checkpoints; "
                              f"refusing to mix runs (choose another checkpoint_path or remove it)")
    config.save_dir = save_dir
    # Make sure we can write the checkpoint later _before_ we wait 1 day for training!
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/config.pckl", 'wb') as f:
        pickle.dump(asdict(config), f)
    write_generation_record(save_dir, config, layout_name, layout_source, status="started")
    print(f"Saved to {save_dir}")

    config.layout = env_kwargs.copy()  # These are env kwargs
    print(config.layout)
    print(json.dumps(interaction_counts(config), indent=2))

    # train partner population
    if config.alg == "brdiv":
        partner_params, partner_population = run_brdiv(config)
    else:
        raise NotImplementedError("Selected method not implemented.")

    print("Saving partner params ...")
    members = save_population_members(save_dir, partner_params, config.num_seeds, config.partner_pop_size)
    write_generation_record(save_dir, config, layout_name, layout_source, status="complete", members=members)
    return save_dir


def run_training():
    generate_population(tyro.cli(TrainConfig))


if __name__ == '__main__':
    run_training()
