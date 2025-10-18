import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Optional, Sequence, List

import flashbax as fbx
import flax
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from flax.core.frozen_dict import freeze, unfreeze
from tensorboardX import SummaryWriter

from experiments.continual.agem import AGEM
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from experiments.experimental.utils_vdn import (
    Timestep,
    CustomTrainState,
    eps_greedy_exploration,
    batchify as vdn_batchify,
    unbatchify as vdn_unbatchify
)
from experiments.model.q_mlp import QNetwork
from experiments.utils import batchify, unbatchify
from meal import make_env
from meal import create_sequence
from meal.env.utils.max_soup_calculator import calculate_max_soup
from meal.wrappers.jaxmarl import (
    CTRolloutManager,
)
from meal.wrappers.logging import LogWrapper


@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING / VDN PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    alg_name: str = "vdn"
    total_timesteps: float = 1e7
    steps_per_task: float = 1e7
    num_envs: int = 16
    num_steps: int = 128
    hidden_size: int = 64
    eps_start: float = 1.0
    eps_finish: float = 0.05
    eps_decay: float = 0.3
    max_grad_norm: int = 1
    num_epochs: int = 8
    lr: float = 0.00007
    lr_linear_decay: bool = True
    anneal_lr: bool = False
    lambda_: float = 0.5
    gamma: float = 0.99
    tau: float = 1
    buffer_size: int = 1e5
    buffer_batch_size: int = 128
    learning_starts: int = 1e3
    target_update_interval: int = 10

    # Reward shaping
    reward_shaping: bool = True
    reward_shaping_horizon: float = 2.5e6
    rew_shaping_horizon: float = 2.5e6

    # Reward distribution settings
    sparse_rewards: bool = False  # Only shared reward for soup delivery
    individual_rewards: bool = False  # Only respective agent gets reward for their actions

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

    # Regularization method specific parameters
    importance_episodes: int = 5
    importance_steps: int = 500

    # EWC specific parameters
    ewc_mode: str = "multi"  # "online", "last" or "multi"
    ewc_decay: float = 0.9  # Only for online EWC

    # AGEM specific parameters
    agem_memory_size: int = 100000
    agem_sample_size: int = 1024
    agem_gradient_scale: float = 1.0

    # ═══════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    env_name: str = "overcooked"
    seq_length: int = 10
    repeat_sequence: int = 1
    strategy: str = "generate"
    layouts: Optional[Sequence[str]] = field(default_factory=lambda: [])
    env_kwargs: Optional[Sequence[dict]] = None
    difficulty: Optional[str] = None
    single_task_idx: Optional[int] = None
    layout_file: Optional[str] = None
    random_reset: bool = True

    # Random layout generator parameters
    height_min: int = 6  # minimum layout height
    height_max: int = 7  # maximum layout height
    width_min: int = 6  # minimum layout width
    width_max: int = 7  # maximum layout width
    wall_density: float = 0.15  # fraction of internal tiles that are untraversable

    # Agent restriction parameters
    complementary_restrictions: bool = False  # One agent can't pick up onions, other can't pick up plates

    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATION PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    evaluation: bool = True
    eval_forward_transfer: bool = False
    test_during_training: bool = True
    test_interval: float = 0.1  # fraction
    test_num_steps: int = 400
    test_num_envs: int = 32
    eval_num_steps: int = 1000
    eval_num_episodes: int = 5
    record_gif: bool = False
    gif_len: int = 300
    log_interval: int = 75

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGGING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    wandb_mode: str = "online"
    entity: Optional[str] = ""
    project: str = "MEAL"
    tags: List[str] = field(default_factory=list)
    wandb_log_all_seeds: bool = False

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    seed: int = 30
    num_seeds: int = 1

    # Legacy parameters for backward compatibility
    layout_name: Optional[Sequence[str]] = None
    network_name: str = "cnn"
    cl_method_name: str = "none"
    group: str = "none"

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPUTED DURING RUNTIME
    # ═══════════════════════════════════════════════════════════════════════════
    num_updates: int = 0
    num_actors: int = 0
    minibatch_size: int = 0


###################################################
############### MAIN FUNCTION #####################
###################################################

def main():
    # set the device to the first available GPU
    jax.config.update("jax_platform_name", "gpu")

    # print the device that is being used
    print("Device: ", jax.devices())

    config = tyro.cli(Config)

    # Validate reward settings
    if config.sparse_rewards and config.individual_rewards:
        raise ValueError(
            "Cannot enable both sparse_rewards and individual_rewards simultaneously. "
            "Please choose only one reward setting."
        )

    if config.single_task_idx is not None:  # single-task baseline
        config.cl_method = "ft"
    if config.cl_method is None:
        raise ValueError(
            "cl_method is required. Please specify a continual learning method (e.g., ewc, mas, l2, ft, agem).")

    difficulty = config.difficulty
    seq_length = config.seq_length
    strategy = config.strategy
    seed = config.seed

    # Set default regularization coefficient based on the CL method if not specified
    if config.reg_coef is None:
        if config.cl_method.lower() == "ewc":
            config.reg_coef = 1e11
        elif config.cl_method.lower() == "mas":
            config.reg_coef = 1e9
        elif config.cl_method.lower() == "l2":
            config.reg_coef = 1e7

    method_map = dict(ewc=EWC(mode=config.ewc_mode, decay=config.ewc_decay),
                      mas=MAS(),
                      l2=L2(),
                      ft=FT(),
                      agem=AGEM(memory_size=config.agem_memory_size, sample_size=config.agem_sample_size))

    cl = method_map[config.cl_method.lower()]

    # generate a sequence of tasks
    config.env_kwargs, layout_names = create_sequence(
        sequence_length=seq_length,
        strategy=strategy,
        layout_names=config.layouts,
        seed=seed,
        height_rng=(config.height_min, config.height_max),
        width_rng=(config.width_min, config.width_max),
        wall_density=config.wall_density,
        layout_file=config.layout_file,
        complementary_restrictions=config.complementary_restrictions,
    )

    # Add random_reset parameter to all environments
    for env_args in config.env_kwargs:
        env_args["random_reset"] = config.random_reset

    # ── optional single-task baseline ─────────────────────────────────────────
    if config.single_task_idx is not None:
        idx = config.single_task_idx
        config.env_kwargs = [config.env_kwargs[idx]]
        layout_names = [layout_names[idx]]
        config.seq_length = 1

    # repeat the base sequence `repeat_sequence` times
    config.env_kwargs = config.env_kwargs * config.repeat_sequence
    layout_names = layout_names * config.repeat_sequence

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    network = "cnn" if config.use_cnn else "mlp"
    run_name = f'{config.alg_name}_{config.cl_method}_{difficulty}_{network}_seq{seq_length}_{strategy}_seed_{seed}_{timestamp}'
    exp_dir = os.path.join("../runs", run_name)

    # Initialize WandB
    wandb_tags = config.tags if config.tags is not None else []
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project=config.project,
        config=config,
        sync_tensorboard=True,
        mode=config.wandb_mode,
        tags=wandb_tags,
        group=config.cl_method.upper(),
        name=run_name,
        id=run_name,
    )

    # Set up Tensorboard
    writer = SummaryWriter(exp_dir)

    # add the hyperparameters to the tensorboard
    rows = []
    for key, value in vars(config).items():
        value_str = str(value).replace("\n", "<br>")
        value_str = value_str.replace("|", "\\|")  # escape pipe chars if needed
        rows.append(f"|{key}|{value_str}|")

    table_body = "\n".join(rows)
    markdown = f"|param|value|\n|-|-|\n{table_body}"
    writer.add_text("hyperparameters", markdown)

    def get_view_params():
        '''
        Gets view parameters for environments when difficulty is specified
        '''
        params = {"random_reset": config.random_reset}
        if config.env_name == "overcooked_po" and difficulty:
            params.update({
                "view_ahead": config.view_ahead,
                "view_sides": config.view_sides,
                "view_behind": config.view_behind
            })
        return params

    def create_environments():
        '''
        Creates environments, with padding for regular Overcooked but not for PO environments
        since PO environments have local observations that don't need padding.
        returns the environment layouts and agent restrictions
        '''
        agent_restrictions_list = []
        for env_args in config.env_kwargs:
            # Extract agent restrictions from env_args
            agent_restrictions_list.append(env_args.get('agent_restrictions', {}))

        # For PO environments, no padding is needed since observations are local
        # PO environments naturally have consistent observation spaces based on view parameters
        if config.env_name == "overcooked_po":
            # Return the original layouts without modification
            env_layouts = []
            for env_args in config.env_kwargs:
                temp_env = make_env(config.env_name, **env_args)
                env_layouts.append(temp_env.layout)
            return env_layouts, agent_restrictions_list

        # For regular environments, apply padding as before
        # Create environments first
        envs = []
        for env_args in config.env_kwargs:
            env = make_env(config.env_name, **env_args)
            envs.append(env)

        # find the environment with the largest observation space
        max_width, max_height = 0, 0
        for env in envs:
            max_width = max(max_width, env.layout["width"])
            max_height = max(max_height, env.layout["height"])

        # pad the observation space of all environments to be the same size by adding extra walls to the outside
        padded_envs = []
        for env in envs:
            # unfreeze the environment so that we can apply padding
            env = unfreeze(env.layout)

            # calculate the padding needed
            width_diff = max_width - env["width"]
            height_diff = max_height - env["height"]

            # determine the padding needed on each side
            left = width_diff // 2
            right = width_diff - left
            top = height_diff // 2
            bottom = height_diff - top

            width = env["width"]

            # Adjust the indices of the observation space to match the padded observation space
            def adjust_indices(indices):
                '''
                adjusts the indices of the observation space
                @param indices: the indices to adjust
                returns the adjusted indices
                '''
                indices = jnp.asarray(indices)
                rows, cols = jnp.divmod(indices, width)
                return (rows + top) * (width + left + right) + (cols + left)

            # adjust the indices of the observation space to account for the new walls
            env["wall_idx"] = adjust_indices(env["wall_idx"])
            env["agent_idx"] = adjust_indices(env["agent_idx"])
            env["goal_idx"] = adjust_indices(env["goal_idx"])
            env["plate_pile_idx"] = adjust_indices(env["plate_pile_idx"])
            env["onion_pile_idx"] = adjust_indices(env["onion_pile_idx"])
            env["pot_idx"] = adjust_indices(env["pot_idx"])

            # pad the observation space with walls
            padded_wall_idx = list(env["wall_idx"])  # Existing walls

            # Top and bottom padding
            for y in range(top):
                for x in range(max_width):
                    padded_wall_idx.append(y * max_width + x)  # Top row walls

            for y in range(max_height - bottom, max_height):
                for x in range(max_width):
                    padded_wall_idx.append(y * max_width + x)  # Bottom row walls

            # Left and right padding
            for y in range(top, max_height - bottom):
                for x in range(left):
                    padded_wall_idx.append(y * max_width + x)  # Left column walls

                for x in range(max_width - right, max_width):
                    padded_wall_idx.append(y * max_width + x)  # Right column walls

            env["wall_idx"] = jnp.array(padded_wall_idx)

            # set the height and width of the environment to the new padded height and width
            env["height"] = max_height
            env["width"] = max_width

            padded_envs.append(freeze(env))  # Freeze the environment to prevent further modifications

        return padded_envs, agent_restrictions_list

    @jax.jit
    def evaluate_model(rng, train_state):
        '''
        Evaluates the current model on all environments in the sequence
        @param rng: the random number generator
        @param train_state: the current training state
        returns the metrics: the average returns of the evaluated episodes
        '''

        def evaluate_on_environment(test_env, rng, train_state):
            '''
            Evaluates the current model on a single environment
            @param rng: the random number generator
            @param train_state: the current training state
            returns the metrics: the average returns of the evaluated episodes
            '''

            def evaluation_step(step_state, unused):
                last_obs, env_state, rng = step_state

                rng, rng_a, rng_s = jax.random.split(rng, 3)

                # prepare the observations for the network
                obs_batch = batchify(last_obs, test_env.agents, test_env.num_agents * test_env.batch_size,
                                     not config.use_cnn)

                # Reshape observations for each agent: (num_agents, num_envs, obs_dim)
                obs_batch = obs_batch.reshape(test_env.num_agents, test_env.batch_size, -1)

                # Compute Q-values for all agents
                q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.params,
                    obs_batch,
                )  # (num_agents, num_envs, num_actions)

                actions = jnp.argmax(q_vals, axis=-1)
                actions = vdn_unbatchify(actions, test_env.agents)
                new_obs, new_env_state, rewards, dones, infos = test_env.batch_step(
                    rng_s, env_state, actions
                )
                step_state = (new_obs, new_env_state, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            rng, _rng = jax.random.split(rng)
            step_state, (rewards, dones, infos) = jax.lax.scan(
                evaluation_step,
                (init_obs, env_state, _rng),
                None,
                config.test_num_steps,
            )
            metrics = jnp.nanmean(
                jnp.where(
                    infos["returned_episode"],
                    infos["returned_episode_returns"],
                    jnp.nan,
                )
            )
            return metrics

        evaluation_returns = []

        for env in test_envs:
            all_returns = jax.vmap(lambda k: evaluate_on_environment(env, k, train_state))(
                jax.random.split(rng, config.eval_num_episodes)
            )

            mean_returns = jnp.mean(all_returns)
            evaluation_returns.append(mean_returns)

        return evaluation_returns

    env_layouts, agent_restrictions_list = create_environments()

    envs = []
    env_names = []
    max_soup_dict = {}
    for i, env_layout in enumerate(env_layouts):
        # Create the environment with agent restrictions
        agent_restrictions = agent_restrictions_list[i]
        view_params = get_view_params()
        env = make_env(config.env_name, layout=env_layout, layout_name=layout_names[i], task_id=i,
                   agent_restrictions=agent_restrictions, **view_params)
        env = LogWrapper(env, replace_info=False)
        env_name = env.layout_name
        envs.append(env)
        env_names.append(env_name)
        max_soup_dict[env_name] = calculate_max_soup(env_layout, env.max_steps, n_agents=env.num_agents)

    # Create train and test environments
    train_envs = []
    test_envs = []
    for env in envs:
        train_env = CTRolloutManager(
            env, batch_size=config.num_envs, preprocess_obs=False
        )
        test_env = CTRolloutManager(
            env, batch_size=config.test_num_envs, preprocess_obs=False
        )
        train_envs.append(train_env)
        test_envs.append(test_env)

    # set extra config parameters based on the environment
    temp_env = train_envs[0]
    config.num_actors = temp_env.num_agents * config.num_envs
    config.num_updates = config.steps_per_task // config.num_steps // config.num_envs
    config.minibatch_size = (config.num_actors * config.num_steps) // config.buffer_batch_size

    def linear_schedule(count):
        '''
        Linearly decays the learning rate depending on the number of minibatches and number of epochs
        returns the learning rate
        '''
        frac = 1.0 - (count // (config.buffer_batch_size * config.num_epochs)) / config.num_updates
        return config.lr * frac

    eps_scheduler = optax.linear_schedule(
        config.eps_start,
        config.eps_finish,
        config.eps_decay * config.num_updates,
    )

    # Initialize the network
    rng = jax.random.PRNGKey(config.seed)
    init_env = train_envs[0]

    network = QNetwork(
        action_dim=init_env.max_action_space,
        hidden_size=config.hidden_size,
        encoder_type=config.network_name,
        activation=config.activation,
        big_network=config.big_network,
        use_layer_norm=config.use_layer_norm,
        use_multihead=config.use_multihead,
        use_task_id=config.use_task_id,
        num_tasks=seq_length
    )

    rng, agent_rng = jax.random.split(rng)

    obs_dim = init_env.observation_space().shape
    if not config.use_cnn:
        obs_dim = np.prod(obs_dim)

    init_x = jnp.zeros((1, *obs_dim)) if config.use_cnn else jnp.zeros((1, obs_dim,))
    init_network_params = network.init(agent_rng, init_x)

    lr_scheduler = optax.linear_schedule(
        config.lr,
        1e-10,
        (config.num_epochs) * config.num_updates,
    )

    lr = lr_scheduler if config.lr_linear_decay else config.lr

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.radam(learning_rate=lr),
    )

    train_state = CustomTrainState.create(
        apply_fn=jax.jit(network.apply),
        params=init_network_params,
        target_network_params=init_network_params,
        tx=tx,
    )

    # Create the replay buffer
    buffer = fbx.make_flat_buffer(
        max_length=int(config.buffer_size),
        min_length=int(config.buffer_batch_size),
        sample_batch_size=int(config.buffer_batch_size),
        add_sequences=False,
        add_batch_size=int(config.num_envs * config.num_steps),
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    rng, init_rng = jax.random.split(rng)

    init_obs, init_env_state = init_env.batch_reset(init_rng)
    init_actions = {
        agent: init_env.batch_sample(init_rng, agent) for agent in env.agents
    }
    init_obs, _, init_rewards, init_dones, init_infos = init_env.batch_step(
        init_rng, init_env_state, init_actions
    )
    init_avail_actions = init_env.get_valid_actions(init_env_state.env_state)
    init_timestep = Timestep(
        obs=init_obs,
        actions=init_actions,
        avail_actions=init_avail_actions,
        rewards=init_rewards,
        dones=init_dones,
    )
    init_timestep_unbatched = jax.tree.map(
        lambda x: x[0], init_timestep
    )  # remove the NUM_ENV dim
    buffer_state = buffer.init(init_timestep_unbatched)

    @partial(jax.jit, static_argnums=(2, 4))
    def train_on_environment(rng, train_state, env, cl_state, env_idx):
        '''
        Trains the agent on a single environment using VDN with continual learning
        @param rng: the random number generator
        @param train_state: the current training state
        @param env: the environment to train on
        @param cl_state: the continual learning state
        @param env_idx: the environment index
        returns the updated rng, training state, and continual learning state
        '''

        print(f"Training on environment: {env_idx} - {env.layout_name}")

        # Create train_env from the base env
        train_env = CTRolloutManager(env, batch_size=config.num_envs, preprocess_obs=False)

        # for each new environment, we want to start with a fresh buffer
        buffer_state = buffer.init(init_timestep_unbatched)

        # reset the learning rate
        lr = lr_scheduler if config.lr_linear_decay else config.lr
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.radam(learning_rate=lr),
        )
        new_optimizer = tx.init(train_state.params)
        train_state = train_state.replace(tx=tx, opt_state=new_optimizer, n_updates=0)

        reward_shaping_horizon = config.total_timesteps / 2
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.,
            end_value=0.,
            transition_steps=reward_shaping_horizon
        )

        def _update_step(runner_state, unused):

            train_state, buffer_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                '''
                steps the environment for a single step
                @param carry: the current state of the environment
                returns the new state of the environment and the timestep
                '''
                last_obs, env_state, rng = carry

                rng, rng_action, rng_step = jax.random.split(rng, 3)

                # prepare the observations for the network
                obs_batch = batchify(last_obs, env.agents, config.num_actors,
                                     not config.use_cnn)

                # Reshape observations for each agent: (num_agents, num_envs, obs_dim)
                obs_batch = obs_batch.reshape(env.num_agents, config.num_envs, -1)

                # Compute Q-values for all agents
                q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.params,
                    obs_batch,
                )  # (num_agents, num_envs, num_actions)

                # retrieve the valid actions
                avail_actions = train_env.get_valid_actions(env_state.env_state)

                # perform epsilon-greedy exploration
                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_action, env.num_agents)
                new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, vdn_batchify(avail_actions, env.agents)
                )
                actions = unbatchify(new_action, env.agents, config.num_envs, env.num_agents)

                new_obs, new_env_state, rewards, dones, infos = train_env.batch_step(
                    rng_step, env_state, actions
                )

                # add shaped reward
                shaped_reward = infos.pop("shaped_reward")
                shaped_reward["__all__"] = vdn_batchify(shaped_reward, env.agents).sum(axis=0)
                rewards = jax.tree.map(
                    lambda x, y: x + y * rew_shaping_anneal(train_state.timesteps),
                    rewards,
                    shaped_reward,
                )

                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    avail_actions=avail_actions,
                    rewards=rewards,
                    dones=dones,
                )
                return (new_obs, new_env_state, rng), (timestep, infos)

            # step the env
            rng, _rng = jax.random.split(rng)
            carry, (timesteps, infos) = jax.lax.scan(
                f=_step_env,
                init=(*expl_state, _rng),
                xs=None,
                length=config.num_steps,
            )
            expl_state = carry[:2]

            # update the steps count of the train state
            train_state = train_state.replace(
                timesteps=train_state.timesteps
                          + config.num_steps * config.num_envs
            )

            # prepare the timesteps for the buffer
            timesteps = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), timesteps
            )  # (num_envs*num_steps, ...)

            # add the timesteps to the buffer
            buffer_state = buffer.add(buffer_state, timesteps)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):

                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state,
                                          _rng).experience  # collects a minibatch of size buffer_batch_size

                batched_sample = batchify(minibatch.second.obs, env.agents, config.num_actors,
                                          not config.use_cnn)
                # Reshape observations for each agent: (num_agents, batch_size, obs_dim)
                batched_sample = batched_sample.reshape(env.num_agents, config.buffer_batch_size, -1)
                q_next_target = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.target_network_params, batched_sample)

                q_next_target = jnp.max(q_next_target, axis=-1)

                vdn_target = minibatch.first.rewards["__all__"] + (
                        1 - minibatch.first.dones["__all__"]
                ) * config.gamma * jnp.sum(
                    q_next_target, axis=0
                )  # sum over agents

                def _loss_fn(params):
                    batched_obs = batchify(minibatch.first.obs, env.agents, config.num_actors,
                                           not config.use_cnn)
                    # Reshape observations for each agent: (num_agents, batch_size, obs_dim)
                    batched_obs = batched_obs.reshape(env.num_agents, config.buffer_batch_size, -1)
                    q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                        params, batched_obs)

                    # get logits of the chosen actions
                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals,
                        vdn_batchify(minibatch.first.actions, env.agents)[..., jnp.newaxis],
                        axis=-1,
                    ).squeeze()  # (num_agents, batch_size, )

                    chosen_action_q_vals = jnp.sum(chosen_action_q_vals, axis=0)
                    loss = jnp.mean((chosen_action_q_vals - vdn_target) ** 2)

                    return loss, chosen_action_q_vals.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)

            # Check if learning should happen
            can_train = buffer.can_sample(buffer_state)
            has_enough_timesteps = train_state.timesteps > config.learning_starts
            is_learn_time = can_train & has_enough_timesteps

            # Define learning and no-op functions
            def perform_learning(train_state, rng):
                return jax.lax.scan(
                    f=_learn_phase,
                    init=(train_state, rng),
                    xs=None,
                    length=config.num_epochs
                )

            def do_nothing(train_state, rng):
                return (train_state, rng), (jnp.zeros(config.num_epochs), jnp.zeros(config.num_epochs))

            # Conditionally execute learning
            (train_state, rng), (loss, qvals) = jax.lax.cond(
                is_learn_time,
                perform_learning,
                do_nothing,
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config.target_update_interval == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config.tau,
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,

            )

            def compute_action_gap(q_vals):
                '''
                Computes the action gap
                @param q_vals: the Q-values
                returns the action gap
                '''
                top_2_q_vals, _ = jax.lax.top_k(q_vals, 2)
                top_q = top_2_q_vals[0]
                second_q = top_2_q_vals[1]
                return top_q - second_q

            # UPDATE METRICS
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "General/env_step": train_state.timesteps,
                "General/update_steps": train_state.n_updates,
                "General/grad_steps": train_state.grad_steps,
                "General/learning_rate": lr_scheduler(train_state.n_updates),
                "Losses/loss": loss.mean(),
                "Values/qvals": qvals.mean(),
                "General/epsilon": eps_scheduler(train_state.n_updates),
                "General/action_gap": compute_action_gap(qvals),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            def evaluate_and_log(rng, update_steps, test_metrics):
                '''
                Evaluates the model and logs the metrics
                @param rng: the random number generator
                @param update_steps: the number of update steps
                returns the metrics
                '''
                rng, eval_rng = jax.random.split(rng)

                def log_metrics(metrics, update_steps, env_counter):
                    # evaluate the model
                    test_metrics = evaluate_model(eval_rng, train_state)
                    for i, eval_metric in enumerate(test_metrics):
                        metrics[f"Evaluation/evaluation_{config.layout_name[i]}"] = eval_metric

                    # log the metrics
                    def callback(args):
                        metrics, update_steps, env_counter = args
                        real_step = (int(env_counter) - 1) * config.num_updates + int(update_steps)
                        for k, v in metrics.items():
                            writer.add_scalar(k, v, real_step)

                    jax.experimental.io_callback(callback, None, (metrics, update_steps, env_counter))
                    return None

                def do_not_log(metrics, update_steps, env_counter):
                    return None

                # conditionally evaluate and log the metrics
                jax.lax.cond((train_state.n_updates % int(config.num_updates * config.test_interval)) == 0,
                             log_metrics,
                             do_not_log,
                             metrics, update_steps, env_counter)

            # Evaluate the model and log metrics
            evaluate_and_log(rng, train_state.n_updates, test_metrics)

            runner_state = (train_state, buffer_state, expl_state, test_metrics, rng)

            return runner_state, None

        rng, eval_rng = jax.random.split(rng)
        test_metrics = evaluate_model(eval_rng, train_state)

        rng, reset_rng = jax.random.split(rng)
        obs, env_state = train_env.batch_reset(reset_rng)
        expl_state = (obs, env_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, expl_state, test_metrics, _rng)

        runner_state, metrics = jax.lax.scan(
            f=_update_step,
            init=runner_state,
            xs=None,
            length=config.num_updates
        )

        # Extract final train_state from runner_state
        final_train_state = runner_state[0]

        # Update continual learning state after training on this environment
        cl_state = cl.update_state(cl_state, final_train_state.params, env_idx)

        return rng, final_train_state, cl_state

    def loop_over_envs(rng, train_state, cl_state, envs):
        '''
        Loops over the environments and trains the network with continual learning
        @param rng: random number generator
        @param train_state: the current state of the training
        @param cl_state: the continual learning state
        @param envs: the environments to train on
        returns the final train state and cl state
        '''
        # split the random number generator for training on the environments
        rng, *env_rngs = jax.random.split(rng, len(envs) + 1)

        for env_idx, (env_rng, env) in enumerate(zip(env_rngs, envs)):
            print(f"Training on environment {env_idx + 1}/{len(envs)}: {env.layout_name}")

            # Train on this environment
            rng, train_state, cl_state = train_on_environment(env_rng, train_state, env, cl_state, env_idx)

            # save the model
            path = f"checkpoints/overcooked/{run_name}/model_env_{env_idx + 1}"
            save_params(path, train_state, config.env_kwargs[env_idx] if config.env_kwargs else None,
                        env.layout_name, config)

        return train_state, cl_state

    def save_params(path, train_state, env_kwargs=None, layout_name=None, config=None):
        '''
        Saves the parameters of the network along with environment and config information
        @param path: the path to save the parameters
        @param train_state: the current state of the training
        @param env_kwargs: environment kwargs for this task
        @param layout_name: name of the layout
        @param config: configuration object
        returns None
        '''
        import json

        def convert_frozen_dict(obj):
            """
            Recursively converts FrozenDict objects to regular dictionaries for JSON serialization.
            """
            if isinstance(obj, flax.core.frozen_dict.FrozenDict):
                return {k: convert_frozen_dict(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: convert_frozen_dict(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_frozen_dict(item) for item in obj]
            elif isinstance(obj, jnp.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model parameters
        with open(path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    {"params": train_state.params}
                )
            )

        # Save environment kwargs and config if provided
        if env_kwargs is not None:
            env_kwargs_path = path + "_env_kwargs.json"
            with open(env_kwargs_path, "w") as f:
                json.dump(convert_frozen_dict(env_kwargs), f, indent=2)

        if layout_name is not None:
            layout_path = path + "_layout_name.txt"
            with open(layout_path, "w") as f:
                f.write(layout_name)

        if config is not None:
            config_path = path + "_config.json"
            config_dict = {k: v for k, v in vars(config).items()
                           if not k.startswith('_') and not callable(v)}
            with open(config_path, "w") as f:
                json.dump(convert_frozen_dict(config_dict), f, indent=2)

        print('model saved to', path)

    # Initialize continual learning state
    cl_state = cl.init_state(train_state.params, config.regularize_critic, config.regularize_heads)

    # train the network
    rng, train_rng = jax.random.split(rng)

    final_train_state, final_cl_state = loop_over_envs(train_rng, train_state, cl_state, train_envs)


def record_gif_of_episode(config, train_state, env, network, env_idx=0, max_steps=300):
    rng = jax.random.PRNGKey(0)
    rng, env_rng = jax.random.split(rng)
    obs, state = env.reset(env_rng)
    done = False
    step_count = 0
    states = [state]

    while not done and step_count < max_steps:
        obs_dict = {}
        for agent_id, obs_v in obs.items():
            # Determine the expected raw shape for this agent.
            expected_shape = env.observation_space().shape
            # If the observation is unbatched, add a batch dimension.
            if obs_v.ndim == len(expected_shape):
                obs_b = jnp.expand_dims(obs_v, axis=0)  # now (1, ...)
            else:
                obs_b = obs_v
            if not config.use_cnn:
                # Flatten the nonbatch dimensions.
                obs_b = jnp.reshape(obs_b, (obs_b.shape[0], -1))
            obs_dict[agent_id] = obs_b

        actions = {}
        act_keys = jax.random.split(rng, env.num_agents)
        for i, agent_id in enumerate(env.agents):
            pi, _ = network.apply(train_state.params, obs_dict[agent_id], env_idx=env_idx)
            actions[agent_id] = jnp.squeeze(pi.sample(seed=act_keys[i]), axis=0)

        # Compute Q-values for all agents
        obs_batch = batchify(obs, env.agents, env.num_agents, not config.use_cnn)
        # Reshape observations for each agent: (num_agents, 1, obs_dim)
        obs_batch = obs_batch.reshape(env.num_agents, 1, -1)
        q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
            train_state.params,
            obs_batch,
        )  # (num_agents, 1, num_actions)

        # perform epsilon-greedy exploration
        eps = eps_scheduler(train_state.n_updates)
        _rngs = jax.random.split(rng_action, env.num_agents)
        new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
            _rngs, q_vals, eps, batchify(avail_actions, env.agents)
        )
        actions = unbatchify(new_action, env.agents)

        rng, key_step = jax.random.split(rng)
        next_obs, next_state, reward, done_info, info = env.step(key_step, state, actions)
        done = done_info["__all__"]

        obs, state = next_obs, next_state
        step_count += 1
        states.append(state)

    return states


if __name__ == "__main__":
    main()
