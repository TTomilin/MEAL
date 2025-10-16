'''Main entry point for running teammate generation algorithms.'''
import json
import os
import pickle
from dataclasses import asdict, dataclass

import jax
import tyro
import wandb

from meal.env.layouts.presets import overcooked_layouts
from experiments.partner_adaptation.partner_generation.BRDiv import run_brdiv
from experiments.partner_adaptation.partner_generation.utils import frozendict_from_layout_repr


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
    layouts_path: str = "jax_marl/environments/overcooked/"

    # Overcooked
    env_name: str = "overcooked"
    layout_difficulty: str = "easy"
    layout_idx: int = 0
    layout_name: str = ""  # If specified, overrides layout_idx

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

    seed: int = 0

    # Training
    lr: float = 1e-3
    anneal_lr: bool = False
    num_envs_xp: int = 32
    num_envs_sp: int = 32
    num_steps: int = 400
    total_timesteps: int = 2.5e8
    update_epochs: int = 15
    num_minibatches: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    ent_coef: float = 0.05
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0

    # Eval
    num_eval_episodes: int = 20

    log_train_out: bool = True

    def __post_init__(self):
        ### MEAL ###

        if self.layout_difficulty == "medium":
            self.layouts_path = self.layouts_path + "layouts_20_medium.json"
        elif self.layout_difficulty == "easy":
            self.layouts_path = self.layouts_path + "layouts_20_easy.json"
        elif self.layout_difficulty == "hard":
            self.layouts_path = self.layouts_path + "layouts_20_easy.json"

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
    return f"FF_BRDIV_IPPO_Overcooked_{config.layout_difficulty}_{config.layout_idx}"


def run_training():
    config = tyro.cli(TrainConfig)
    tags = [
        "FF",
        "BRDIV",
        "IPPO",
        str(config.layout_difficulty),
        str(config.layout_idx),
    ]

    group_string = get_run_string(config)
    run_string = f"{group_string}_SEED_{config.seed}"

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

    if config.checkpoint_path is not None:
        save_dir = os.path.join(config.checkpoint_path, run.name)
        config.save_dir = save_dir
        # Make sure we can write the checkpoint later _before_ we wait 1 day for training!
        os.makedirs(save_dir, exist_ok=True)
        config_dict = asdict(config)
        with open(f"{save_dir}/config.pckl", 'wb') as f:
            pickle.dump(config_dict, f)

        print(f"Saved to {save_dir}/config.pckl")

    if config.layout_name != "":
        layout_dict = {"layout": overcooked_layouts[config.layout_name]}
    else:
        layouts = read_layouts(config)
        layout_dict = {"layout": frozendict_from_layout_repr(
            layouts[config.layout_idx]["layout"])}

    config.layout = layout_dict.copy()  # These are env kwargs

    print(config.layout)

    # train partner population
    if config.alg == "brdiv":
        partner_params, partner_population = run_brdiv(config)
    else:
        raise NotImplementedError("Selected method not implemented.")

    print("Saving partner params ...")
    for i in range(config.num_seeds):
        for j in range(config.partner_pop_size):
            params = jax.tree.map(lambda x: x[i, j], partner_params)

            path = f"{save_dir}/"
            os.makedirs(path, exist_ok=True)
            payload = {"actor_params": params}
            pickle.dump(payload, open(
                path + f"params_seed{i}_agent{j}.pt", "wb"))
            pickle.dump(payload, open(
                path + f"params.pt", "wb"))


if __name__ == '__main__':
    run_training()
