'''Main entry point for running teammate generation algorithms.'''
import json
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import wandb
from dotenv import load_dotenv
from gym import make

from experiments.model.cnn import ActorCritic as CNNActorCritic
from experiments.model.mlp import ActorCritic as MLPActorCritic
# Import utility functions from baselines
from experiments.utils import record_gif_of_episode
# Import continual learning methods
from experiments.continual.agem import AGEM, init_agem_memory
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from meal.env.layouts.presets import easy_layouts_legacy
from meal.env.utils.max_soup_calculator import calculate_max_soup
from meal.visualization.visualizer import OvercookedVisualizer
from meal import make_env
from meal.wrappers.logging import LogWrapper
from experiments.partner_adaptation.partner_agents.agent_interface import ActorWithConditionalCriticPolicy, MLPActorCriticPolicyCL
from experiments.partner_adaptation.partner_agents.overcooked.agent_policy_wrappers import OvercookedIndependentPolicyWrapper, \
    OvercookedOnionPolicyWrapper, OvercookedPlatePolicyWrapper, OvercookedRandomPolicyWrapper, \
    OvercookedStaticPolicyWrapper
from experiments.partner_adaptation.partner_generation.utils import frozendict_from_layout_repr
from experiments.partner_adaptation.train_br import DummyPolicyPopulation, HeuristicPolicyPopulation, run_br_training


@dataclass
class TrainConfig:
    # Wandb and other logging
    project: str = "MEAL"
    mode: str = "online"  # Literal["online", "offline", "disabled"]
    group: str = "overcooked"
    entity: str = ""
    checkpoint_path: str = "checkpoints"
    checkpoint_freq: int = 50  # Checkpoint every N updates
    save_dir: str = ""  # Set programmatically based on wandb run name

    # MEAL
    # Pregenerated MEAL layouts that we are interested in.
    layouts_path: str = "jax_marl/environments/overcooked/"

    # Overcooked
    env_name: str = "overcooked"
    layout_difficulty: str = "easy"
    layout_idx: int = 0
    layout_name: str = ""  # If specified, overrides layout_idx

    rew_shaping_horizon: int = 1e7
    num_agents: int = 2

    # best_response
    alg = "br"

    # Actor-Critic
    fc_dim_size: int = 256
    gru_hidden_dim: int = 256

    seed: int = 0
    num_checkpoints: int = 20

    # Training
    lr: float = 1e-3
    anneal_lr: bool = False
    num_envs: int = 512
    num_steps: int = 400
    total_timesteps: int = 5e7
    update_epochs: int = 15
    num_minibatches: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.05
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    # ═══════════════════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    activation: str = "relu"
    use_cnn: bool = False
    use_layer_norm: bool = True
    big_network: bool = False

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTINUAL LEARNING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    cl_method: Optional[str] = None
    reg_coef: Optional[float] = None
    use_task_id: bool = True
    use_multihead: bool = True
    shared_backbone: bool = False
    normalize_importance: bool = False
    regularize_critic: bool = False
    regularize_heads: bool = False

    # EWC specific parameters
    ewc_mode: str = "online"
    ewc_decay: float = 0.9

    # AGEM specific parameters
    agem_memory_size: int = 1000
    agem_sample_size: int = 64
    agem_gradient_scale: float = 1.0

    # Importance computation parameters
    importance_episodes: int = 10
    importance_steps: int = 100

    # Eval
    num_eval_episodes: int = 20
    record_gif: bool = False  # Record and upload gifs after each partner training
    gif_len: int = 100  # Maximum steps for gif recording

    log_train_out: bool = True

    # Partner/Task Configuration
    num_population_partners: int = 3  # Number of population partners to train against
    num_heuristic_partners: int = 5  # Number of heuristic partners to train against

    def __post_init__(self):
        ### MEAL ###

        if self.layout_difficulty == "medium":
            self.layouts_path = self.layouts_path + "layouts_20_medium.json"
        elif self.layout_difficulty == "easy":
            self.layouts_path = self.layouts_path + "layouts_20_easy.json"
        elif self.layout_difficulty == "hard":
            self.layouts_path = self.layouts_path + "layouts_20_easy.json"

        self.num_actors = 2 * self.num_envs
        self.num_controlled_actors = self.num_envs
        self.num_uncontrolled_actors = self.num_envs
        self.num_updates = self.total_timesteps // self.num_envs // self.num_steps

        # Hardcoded for Overcooked
        self.num_actions = 6

        self.minibatch_size = (
                                      self.num_controlled_actors * self.num_steps) // self.num_minibatches

        #############
        print("Number of updates: ", self.num_updates)
        # TODO we probably don't need this many checkpoints
        self.num_checkpoints = max(1, int(self.num_updates))


def read_layouts(config):
    with open(config.layouts_path, "r") as f:
        layouts = json.load(f)
    return layouts


def get_run_string(config: TrainConfig):
    return f"FF_BRDIV_PPO_{config.layout_difficulty}_{config.layout_idx}"


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

    # Create a unique run name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    run_name = f"{run_string}_{timestamp}"

    # Initialize WandB with unified parameters like baselines/PPO_CL.py
    load_dotenv()
    wandb_tags = tags if tags is not None else []
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    run = wandb.init(
        project=config.project,
        config=asdict(config),
        sync_tensorboard=True,
        mode=config.mode,
        tags=wandb_tags,
        group=config.group,
        name=run_name,
        id=run_name,
        save_code=True,
    )

    print("XPID ID name:")
    print(run.name)
    print("-------------")

    # Use a shorter name for checkpoint directory (keep wandb id as is)
    checkpoint_dir_name = run_string

    # Initialize continual learning method if specified
    cl = None
    if config.cl_method is not None:
        # Set default regularization coefficient based on the CL method if not specified
        if config.reg_coef is None:
            if config.cl_method.lower() == "ewc":
                config.reg_coef = 1e11
            elif config.cl_method.lower() == "mas":
                config.reg_coef = 1e9
            elif config.cl_method.lower() == "l2":
                config.reg_coef = 1e7

        # Initialize the continual learning method
        method_map = dict(
            ewc=EWC(mode=config.ewc_mode, decay=config.ewc_decay),
            mas=MAS(),
            l2=L2(),
            ft=FT(),
            agem=AGEM(memory_size=config.agem_memory_size, sample_size=config.agem_sample_size)
        )

        if config.cl_method.lower() in method_map:
            cl = method_map[config.cl_method.lower()]
            print(f"Initialized continual learning method: {config.cl_method.upper()}")
        else:
            raise ValueError(f"Unknown continual learning method: {config.cl_method}")

    if config.checkpoint_path is not None:
        save_dir = os.path.join(config.checkpoint_path, checkpoint_dir_name)
        config.save_dir = save_dir
        # Make sure we can write the checkpoint later _before_ we wait 1 day for training!
        os.makedirs(save_dir, exist_ok=True)
        config_dict = asdict(config)
        with open(f"{save_dir}/config.pckl", 'wb') as f:
            pickle.dump(config_dict, f)

        print(f"Saved to {save_dir}/config.pckl")

    if config.layout_name != "":
        layout_dict = {"layout": easy_layouts_legacy[config.layout_name]}
    else:
        layouts = read_layouts(config)
        layout_dict = {"layout": frozendict_from_layout_repr(
            layouts[config.layout_idx]["layout"]), "random_agent_start": True}

    config.layout = layout_dict.copy()  # These are env kwargs
    env = make_env(config.env_name, **config.layout)
    env = LogWrapper(env)

    # Calculate max soup for the layout
    layout_name = config.layout_name if config.layout_name != "" else f"layout_{config.layout_idx}"
    max_soup_dict = {layout_name: calculate_max_soup(config.layout["layout"], env.max_steps, n_agents=env.num_agents)}

    # Initialize visualizer for gif recording
    visualizer = OvercookedVisualizer(num_agents=env.num_agents)

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng, 2)

    # TODO: Fix this to work with difficulty and seed also if layout_name is not given
    pop_dir = f"{Path(__file__).resolve().parent}/partner_agents/BRDiv_population/{config.layout_name}"
    with open(os.path.join(pop_dir, "config.pckl"), "rb") as f:
        partner_agent_config = pickle.load(f)  # has 'partner_pop_size'
    pop_params = []
    for p in sorted(os.listdir(pop_dir)):
        if "param" in p:
            with open(os.path.join(pop_dir, p), "rb") as f:
                pop_params.append(pickle.load(f)["actor_params"])

    pop_size = partner_agent_config["partner_pop_size"]
    partner_policy = ActorWithConditionalCriticPolicy(
        6, obs_dim=np.prod(env.observation_space().shape), pop_size=pop_size)

    # train partner population
    if config.alg == "br":
        # Initialize ego agent
        ac_cls = CNNActorCritic if config.use_cnn else MLPActorCritic

        seq_length = config.num_population_partners + config.num_heuristic_partners

        ego_network = ac_cls(
            len(env.action_set), config.activation, seq_length, config.use_multihead,
            config.shared_backbone, config.big_network, config.use_task_id,
            config.regularize_heads, config.use_layer_norm)

        obs_dim = env.observation_space().shape
        if not config.use_cnn:
            obs_dim = np.prod(obs_dim)

        ego_policy = MLPActorCriticPolicyCL(ego_network, obs_dim)

        # Initialize the network
        rng, network_rng = jax.random.split(rng)

        ego_params = ego_policy.init_params(network_rng)

        # Initialize continual learning state if CL method is specified
        cl_state = None
        if cl is not None:
            cl_state = cl.init_state(ego_params, config.regularize_critic, config.regularize_heads)

            # Initialize AGEM memory if using AGEM
            if config.cl_method.lower() == "agem":
                obs_dim_agem = env.observation_space().shape
                if not config.use_cnn:
                    obs_dim_agem = (np.prod(obs_dim_agem),)
                cl_state = init_agem_memory(config.agem_memory_size, obs_dim_agem)

            print(f"Initialized CL state for method: {config.cl_method.upper()}")

        indp = OvercookedIndependentPolicyWrapper(
            layout=config.layout["layout"], p_onion_on_counter=0.5, p_plate_on_counter=0.5)
        onin = OvercookedOnionPolicyWrapper(layout=config.layout["layout"])
        plate = OvercookedPlatePolicyWrapper(layout=config.layout["layout"])
        rndm = OvercookedRandomPolicyWrapper(layout=config.layout["layout"])
        static = OvercookedStaticPolicyWrapper(layout=config.layout["layout"])

        fake_params = jax.tree.map(
            lambda x: x[jnp.newaxis, ...], ego_params)

        # Build evaluation partner list based on configuration
        eval_partner = []
        partner_idx = 0

        # Add population partners
        for i in range(min(config.num_population_partners, len(pop_params))):
            eval_partner.append((
                DummyPolicyPopulation(policy_cls=partner_policy),
                jax.tree.map(lambda x: x[jnp.newaxis, ...], pop_params[i]),
                partner_idx
            ))
            partner_idx += 1

        # Add heuristic partners
        heuristic_policies = [indp, onin, plate, rndm, static]

        for i in range(min(config.num_heuristic_partners, len(heuristic_policies))):
            eval_partner.append((
                HeuristicPolicyPopulation(policy_cls=heuristic_policies[i]),
                fake_params,
                partner_idx
            ))
            partner_idx += 1

        # Train ego agent against partners in a configurable schedule
        heuristic_policies = [indp, onin, plate, rndm, static]
        heuristic_names = ["Independent_Policy", "Onion_Policy", "Plate_Policy", "Random_Policy", "Static_Policy"]

        # Train against population partners
        for i in range(min(config.num_population_partners, len(pop_params))):
            ego_params, cl_state = run_br_training(
                config, env, partner_agent_config, ego_policy,
                ego_params, partner_policy, pop_params[i], env_id_idx=i, eval_partner=eval_partner,
                max_soup_dict=max_soup_dict, layout_names=[layout_name], cl=cl, cl_state=cl_state)
            # TODO when using vmap over seeds, do the following
            # ego_params = jax.tree.map(lambda x: x[0, ...], ego_params) # take the first params set from the batch dimension

            # Record gif after training with population partner
            if hasattr(config, 'record_gif') and config.record_gif:
                from flax.training.train_state import TrainState
                import optax
                temp_train_state = TrainState.create(
                    apply_fn=ego_policy.network.apply,
                    params=ego_params,
                    tx=optax.adam(1e-4)  # dummy optimizer
                )
                states = record_gif_of_episode(config, temp_train_state, env, ego_policy.network, env_idx=i,
                                               max_steps=config.gif_len)
                partner_name = f"BRDiv_Partner_{i}"
                visualizer.animate(states, agent_view_size=5, task_idx=i, task_name=partner_name,
                                   exp_dir=f"gifs/{run.name}")

        # Train against heuristic partners if enabled
        for i in range(min(config.num_heuristic_partners, len(heuristic_policies))):
            env_id_idx = config.num_population_partners + i
            partner_policy_obj = heuristic_policies[i]
            partner_name = heuristic_names[i]

            ego_params, cl_state = run_br_training(
                config, env, partner_agent_config, ego_policy,
                ego_params, partner_policy_obj, None, env_id_idx=env_id_idx, eval_partner=eval_partner,
                max_soup_dict=max_soup_dict, layout_names=[layout_name], cl=cl, cl_state=cl_state)
            ego_params = jax.tree.map(  # take the first params set from the batch dimension
                lambda x: x[0, ...], ego_params)

            # Record gif after training with heuristic partner
            if hasattr(config, 'record_gif') and config.record_gif:
                temp_train_state = TrainState.create(
                    apply_fn=ego_policy.network.apply, params=ego_params, tx=optax.adam(1e-4))
                states = record_gif_of_episode(config, temp_train_state, env, ego_policy.network, env_idx=env_id_idx,
                                               max_steps=config.gif_len)
                visualizer.animate(states, agent_view_size=5, task_idx=env_id_idx, task_name=partner_name,
                                   exp_dir=f"gifs/{run.name}")
    else:
        raise NotImplementedError("Selected method not implemented.")

    if config.checkpoint_path is not None:
        path = f"{save_dir}/"
        os.makedirs(path, exist_ok=True)
        payload = {"actor_params": ego_params}
        pickle.dump(payload, open(path + f"params_seed{config.seed}.pt", "wb"))


if __name__ == '__main__':
    run_training()
