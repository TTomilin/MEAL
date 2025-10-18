import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Sequence, Optional, List

import flax
import numpy as np
import optax
import tyro
import wandb
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState
from jax._src.flatten_util import ravel_pytree

from experiments.model.cnn import ActorCritic as CNNActorCritic
from experiments.model.mlp import ActorCritic as MLPActorCritic
from experiments.utils import *
from experiments.continual.agem import AGEM, init_agem_memory, sample_memory, compute_memory_gradient, agem_project, \
    update_agem_memory
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from meal.env.utils.max_soup_calculator import calculate_max_soup
from meal import make_env
from meal.wrappers.logging import LogWrapper


@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING / PPO PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    alg_name: str = "ippo"
    lr: float = 1e-3
    anneal_lr: bool = False
    num_envs: int = 512
    num_steps: int = 400
    steps_per_task: float = 5e7
    update_epochs: int = 4
    num_minibatches: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.05
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    # Reward shaping
    reward_shaping: bool = True
    reward_shaping_horizon: float = 2.5e6

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
    env_name: str = "overcooked_n_agent"
    num_agents: int = 2  # number of agents in the environment
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
    eval_num_steps: int = 1000
    eval_num_episodes: int = 5
    record_gif: bool = False
    gif_len: int = 300
    log_interval: int = 5

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGGING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    wandb_mode: str = "online"
    entity: Optional[str] = ""
    project: str = "MEAL"
    tags: List[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    seed: int = 30
    num_seeds: int = 1

    # ═══════════════════════════════════════════════════════════════════════════
    # RUNTIME COMPUTED PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    num_actors: int = 0
    num_updates: int = 0
    minibatch_size: int = 0


############################
######  MAIN FUNCTION  #####
############################


def main():
    # set the device to the first available GPU
    jax.config.update("jax_platform_name", "gpu")

    # print the device that is being used
    print("Device: ", jax.devices())

    cfg = tyro.cli(Config)

    # Validate reward settings
    if cfg.sparse_rewards and cfg.individual_rewards:
        raise ValueError(
            "Cannot enable both sparse_rewards and individual_rewards simultaneously. "
            "Please choose only one reward setting."
        )

    if cfg.single_task_idx is not None:  # single-task baseline
        cfg.cl_method = "ft"
    if cfg.cl_method is None:
        raise ValueError(
            "cl_method is required. Please specify a continual learning method (e.g., ewc, mas, l2, ft, agem).")

    difficulty = cfg.difficulty
    seq_length = cfg.seq_length
    strategy = cfg.strategy
    seed = cfg.seed

    # Set default regularization coefficient based on the CL method if not specified
    if cfg.reg_coef is None:
        if cfg.cl_method.lower() == "ewc":
            cfg.reg_coef = 1e11
        elif cfg.cl_method.lower() == "mas":
            cfg.reg_coef = 1e9
        elif cfg.cl_method.lower() == "l2":
            cfg.reg_coef = 1e7

    method_map = dict(ewc=EWC(mode=cfg.ewc_mode, decay=cfg.ewc_decay),
                      mas=MAS(),
                      l2=L2(),
                      ft=FT(),
                      agem=AGEM(memory_size=cfg.agem_memory_size, sample_size=cfg.agem_sample_size))

    cl = method_map[cfg.cl_method.lower()]

    # generate a sequence of tasks
    cfg.env_kwargs, layout_names = create_sequence(
        num_agents=cfg.num_agents,
        sequence_length=seq_length,
        strategy=strategy,
        layout_names=cfg.layouts,
        seed=seed,
        height_rng=(cfg.height_min, cfg.height_max),
        width_rng=(cfg.width_min, cfg.width_max),
        wall_density=cfg.wall_density,
        layout_file=cfg.layout_file,
    )

    # Add view parameters for PO environments when difficulty is specified
    if cfg.env_name == "overcooked_po" and difficulty:
        for env_args in cfg.env_kwargs:
            env_args["view_ahead"] = cfg.view_ahead
            env_args["view_sides"] = cfg.view_sides
            env_args["view_behind"] = cfg.view_behind

    # Add random_reset parameter to all environments
    for env_args in cfg.env_kwargs:
        env_args["random_reset"] = cfg.random_reset

    # ── optional single-task baseline ─────────────────────────────────────────
    if cfg.single_task_idx is not None:
        idx = cfg.single_task_idx
        cfg.env_kwargs = [cfg.env_kwargs[idx]]
        layout_names = [layout_names[idx]]
        cfg.seq_length = 1

    # repeat the base sequence `repeat_sequence` times
    cfg.env_kwargs = cfg.env_kwargs * cfg.repeat_sequence
    layout_names = layout_names * cfg.repeat_sequence

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    network = "cnn" if cfg.use_cnn else "mlp"
    run_name = f'{cfg.alg_name}_{cfg.cl_method}_{difficulty}_{cfg.num_agents}agents_{network}_seq{seq_length}_{strategy}_seed_{seed}_{timestamp}'
    exp_dir = os.path.join("../runs", run_name)

    # Initialize WandB
    wandb_tags = cfg.tags if cfg.tags is not None else []
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project=cfg.project,
        config=cfg,
        sync_tensorboard=True,
        mode=cfg.wandb_mode,
        tags=wandb_tags,
        group=cfg.cl_method.upper(),
        name=run_name,
        id=run_name,
    )

    # Set up Tensorboard
    writer = SummaryWriter(exp_dir)
    # add the hyperparameters to the tensorboard
    rows = []
    for key, value in vars(cfg).items():
        value_str = str(value).replace("\n", "<br>")
        value_str = value_str.replace("|", "\\|")  # escape pipe chars if needed
        rows.append(f"|{key}|{value_str}|")

    table_body = "\n".join(rows)
    markdown = f"|param|value|\n|-|-|\n{table_body}"
    writer.add_text("hyperparameters", markdown)

    def get_view_params():
        '''
        Get view parameters for overcooked_po environments from config.
        Returns a dictionary with view parameters if applicable, empty dict otherwise.
        '''
        if cfg.env_name == "overcooked_po" and cfg.difficulty:
            return {
                "random_reset": cfg.random_reset,
                "view_ahead": cfg.view_ahead,
                "view_sides": cfg.view_sides,
                "view_behind": cfg.view_behind
            }
        return {}

    def create_environments():
        '''
        Creates environments, with padding for regular Overcooked but not for PO environments
        since PO environments have local observations that don't need padding.
        returns the environment layouts and agent restrictions
        '''
        agent_restrictions_list = []
        for env_args in cfg.env_kwargs:
            # Extract agent restrictions from env_args
            agent_restrictions_list.append(env_args.get('agent_restrictions', {}))

        # For PO environments, no padding is needed since observations are local
        # PO environments naturally have consistent observation spaces based on view parameters
        if cfg.env_name == "overcooked_po":
            # Return the original layouts without modification
            env_layouts = []
            for env_args in cfg.env_kwargs:
                temp_env = make_env(cfg.env_name, **env_args, num_agents=cfg.num_agents)
                env_layouts.append(temp_env.layout)
            return env_layouts, agent_restrictions_list

        # For regular environments, apply padding as before
        # Create environments first
        envs = []
        for env_args in cfg.env_kwargs:
            env = make_env(cfg.env_name, **env_args, num_agents=cfg.num_agents)
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

    def eval_current_env(train_state, key, env, eval_steps: int, n_episodes: int, env_idx: int):
        """Pure-JAX evaluation on the *current env* (no Python loops).
           Returns (avg_reward, total_soups) across n_episodes."""
        num_agents_local = env.num_agents
        agents_local = env.agents

        def one_episode(k):
            def ep_init(carry_key):
                key0, = (carry_key,)
                k1, k_reset = jax.random.split(key0)
                obs0, st0 = env.reset(k_reset)
                return (k1, st0, obs0, jnp.array(0.0), jnp.array(0.0))  # key, state, obs, total_reward, total_soup

            def ep_step(carry, _):
                key_c, st, obs, tot_r, tot_soups = carry
                # split keys: one per agent + env step
                keys = jax.random.split(key_c, num_agents_local + 1)
                key_next = keys[0]
                agent_keys = keys[1:]

                # build batched obs for each agent: (1, obs_dim)
                batched_obs = jax.tree_util.tree_map(
                    lambda v: jnp.reshape(v, (1, -1)) if not cfg.use_cnn else jnp.expand_dims(v, 0),
                    obs
                )

                def act_for_agent(k, a_id):
                    pi, val, _ = network.apply(train_state.params, batched_obs[agents_local[a_id]], env_idx=env_idx)
                    a = jnp.squeeze(pi.sample(seed=k), axis=0)
                    return a

                acts = {agents_local[i]: act_for_agent(agent_keys[i], i) for i in range(num_agents_local)}

                obs1, st1, rew, done, info = env.step(key_next, st, acts)

                r_step = sum(rew[a] for a in agents_local)  # scalar reward this step
                soups_step = sum(info["soups"][a] for a in agents_local)  # soups delivered this step

                return (key_next, st1, obs1, tot_r + r_step, tot_soups + soups_step), None

            carry0 = ep_init(k)
            carryT, _ = jax.lax.scan(ep_step, carry0, xs=None, length=eval_steps)
            # carryT: (..., total_reward, total_soup)
            return carryT[3], carryT[4]  # total_reward, total_soup

        keys = jax.random.split(key, n_episodes)
        ep_rewards, ep_soups = jax.vmap(one_episode)(keys)  # (N,), (N,)
        return ep_rewards.mean(), ep_soups.sum()

    env_layouts, agent_restrictions_list = create_environments()

    envs = []
    env_names = []
    max_soup_dict = {}
    for i, env_layout in enumerate(env_layouts):
        # Create the environment with agent restrictions
        agent_restrictions = agent_restrictions_list[i]
        view_params = get_view_params()
        env = make_env(cfg.env_name, layout=env_layout, layout_name=layout_names[i], task_id=i,
                   agent_restrictions=agent_restrictions, num_agents=cfg.num_agents, **view_params)
        env = LogWrapper(env, replace_info=False)
        env_name = env.layout_name
        envs.append(env)
        env_names.append(env_name)
        max_soup_dict[env_name] = calculate_max_soup(env_layout, env.max_steps, n_agents=env.num_agents)

    # set extra config parameters based on the environment
    temp_env = envs[0]
    num_agents = temp_env.num_agents
    agents = temp_env.agents

    cfg.num_actors = num_agents * cfg.num_envs
    cfg.num_updates = cfg.steps_per_task // cfg.num_steps // cfg.num_envs
    cfg.minibatch_size = (cfg.num_actors * cfg.num_steps) // cfg.num_minibatches

    def linear_schedule(count):
        '''
        Linearly decays the learning rate depending on the number of minibatches and number of epochs
        returns the learning rate
        '''
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / cfg.num_updates
        return cfg.lr * frac

    ac_cls = CNNActorCritic if cfg.use_cnn else MLPActorCritic

    network = ac_cls(temp_env.action_space().n, cfg.activation, seq_length, cfg.use_multihead,
                     cfg.shared_backbone, cfg.big_network, cfg.use_task_id, cfg.regularize_heads,
                     cfg.use_layer_norm)

    # Get the correct observation dimension by simulating the batchify process
    # This ensures the network is initialized with the same shape it will receive during training
    rng = jax.random.PRNGKey(cfg.seed)
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, cfg.num_envs)
    temp_obs, _ = jax.vmap(temp_env.reset, in_axes=(0,))(reset_rngs)
    temp_obs_batch = batchify(temp_obs, temp_env.agents, cfg.num_actors, not cfg.use_cnn)
    obs_dim = temp_obs_batch.shape[1]  # Get the actual dimension after batchify

    # Initialize the network
    rng, network_rng = jax.random.split(rng)
    init_x = jnp.zeros((1, obs_dim))
    network_params = network.init(network_rng, init_x)

    # Initialize the optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5)
    )

    # jit the apply function
    network.apply = jax.jit(network.apply)

    # Initialize the training state
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx
    )

    @partial(jax.jit, static_argnums=(2, 4))
    def train_on_environment(rng, train_state, env, cl_state, env_idx):
        '''
        Trains the network using IPPO
        @param rng: random number generator
        returns the runner state and the metrics
        '''

        print(f"Training on environment: {env.task_id} - {env.layout_name}")

        # reset the learning rate and the optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=linear_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5)
        )
        new_optimizer = tx.init(train_state.params)
        train_state = train_state.replace(tx=tx, opt_state=new_optimizer)

        # Initialize and reset the environment
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, cfg.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        reward_shaping_horizon = cfg.steps_per_task / 2
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.,
            end_value=0.,
            transition_steps=reward_shaping_horizon
        )

        # TRAIN
        def _update_step(runner_state, _):
            '''
            perform a single update step in the training loop
            @param runner_state: the carry state that contains all important training information
            returns the updated runner state and the metrics
            '''

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                '''
                selects an action based on the policy, calculates the log probability of the action,
                and performs the selected action in the environment
                @param runner_state: the current state of the runner
                returns the updated runner state and the transition
                '''
                # Unpack the runner state
                train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state

                # split the random number generator for action selection
                rng, _rng = jax.random.split(rng)

                # prepare the observations for the network
                obs_batch = batchify(last_obs, env.agents, cfg.num_actors, not cfg.use_cnn)  # (num_actors, obs_dim)

                # apply the policy network to the observations to get the suggested actions and their values
                pi, value, _ = network.apply(train_state.params, obs_batch, env_idx=env_idx)

                # Sample and action from the policy
                action = pi.sample(seed=_rng)

                log_prob = pi.log_prob(action)

                # format the actions to be compatible with the environment
                env_act = unbatchify(action, env.agents, cfg.num_envs, env.num_agents)
                # env_act is a dict with static keys (agent ids)
                env_act = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,)), env_act)

                # STEP ENV
                # split the random number generator for stepping the environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, cfg.num_envs)

                # simultaniously step all environments with the selected actions (parallelized over the number of environments with vmap)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                current_timestep = update_step * cfg.num_steps * cfg.num_envs

                # Apply different reward settings based on configuration
                anneal = rew_shaping_anneal(current_timestep)  # scalar
                mode_index = 0 if cfg.sparse_rewards else 1 if cfg.individual_rewards else 2

                def case_sparse(_):
                    # passthrough env delivery rewards
                    return reward

                def case_individual(_):
                    # r + shaped*anneal
                    return jax.tree_util.tree_map(lambda r, s: r + s * anneal, reward, info["shaped_reward"])

                def case_shared(_):
                    # sum deliveries across agents → give to all; then add shaped*anneal
                    r_sum = jnp.sum(jnp.stack([reward[a] for a in env.agents], axis=0), axis=0)
                    return {a: r_sum + info["shaped_reward"][a] * anneal for a in env.agents}

                reward = jax.lax.switch(mode_index, [case_sparse, case_individual, case_shared], operand=None)

                transition = Transition(
                    batchify(done, env.agents, cfg.num_actors, not cfg.use_cnn).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, cfg.num_actors).squeeze(),
                    log_prob,
                    obs_batch
                )

                # Increment steps_for_env by the number of parallel envs
                steps_for_env = steps_for_env + cfg.num_envs

                runner_state = (train_state, env_state, obsv, update_step, steps_for_env, rng, cl_state)
                return runner_state, (transition, info)

            # Apply the _env_step function a series of times, while keeping track of the runner state
            runner_state, (traj_batch, info) = jax.lax.scan(
                f=_env_step,
                init=runner_state,
                xs=None,
                length=cfg.num_steps
            )

            # unpack the runner state that is returned after the scan function
            train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state

            # create a batch of the observations that is compatible with the network
            last_obs_batch = batchify(last_obs, env.agents, cfg.num_actors, not cfg.use_cnn)

            # apply the network to the batch of observations to get the value of the last state
            _, last_val, _ = network.apply(train_state.params, last_obs_batch, env_idx=env_idx)

            def _calculate_gae(traj_batch, last_val):
                '''
                calculates the generalized advantage estimate (GAE) for the trajectory batch
                @param traj_batch: the trajectory batch
                @param last_val: the value of the last state
                returns the advantages and the targets
                '''

                def _get_advantages(gae_and_next_value, transition):
                    '''
                    calculates the advantage for a single transition
                    @param gae_and_next_value: the GAE and value of the next state
                    @param transition: the transition to calculate the advantage for
                    returns the updated GAE and the advantage
                    '''
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + cfg.gamma * next_value * (1 - done) - value  # calculate the temporal difference
                    gae = (
                            delta
                            + cfg.gamma * cfg.gae_lambda * (1 - done) * gae
                    )  # calculate the GAE (used instead of the standard advantage estimate in PPO)

                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    f=_get_advantages,
                    init=(jnp.zeros_like(last_val), last_val),
                    xs=traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            # calculate the generalized advantage estimate (GAE) for the trajectory batch
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                '''
                performs a single update epoch in the training loop
                @param update_state: the current state of the update
                returns the updated update_state and the total loss
                '''

                def _update_minbatch(carry, batch_info):
                    '''
                    performs a single update minibatch in the training loop
                    @param carry: the current state of the training and cl_state
                    @param batch_info: the information of the batch
                    returns the updated train_state, cl_state and the total loss
                    '''
                    train_state, cl_state = carry
                    # unpack the batch information
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        '''
                        calculates the loss of the network
                        @param params: the parameters of the network
                        @param traj_batch: the trajectory batch
                        @param gae: the generalized advantage estimate
                        @param targets: the targets
                        @param network: the network
                        returns the total loss and the value loss, actor loss, and entropy
                        '''
                        # apply the network to the observations in the trajectory batch
                        pi, value, _ = network.apply(params, traj_batch.obs, env_idx=env_idx)
                        log_prob = pi.log_prob(traj_batch.action)

                        # calculate critic loss
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-cfg.clip_eps,
                                                                                                cfg.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

                        # Calculate actor loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor_unclipped = ratio * gae
                        loss_actor_clipped = (
                                jnp.clip(
                                    ratio,
                                    1.0 - cfg.clip_eps,
                                    1.0 + cfg.clip_eps,
                                )
                                * gae
                        )

                        loss_actor = -jnp.minimum(loss_actor_unclipped, loss_actor_clipped)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # CL penalty (for regularization-based methods)
                        cl_penalty = cl.penalty(params, cl_state, cfg.reg_coef)

                        total_loss = (loss_actor
                                      + cfg.vf_coef * value_loss
                                      - cfg.ent_coef * entropy
                                      + cl_penalty)
                        return total_loss, (value_loss, loss_actor, entropy, cl_penalty)

                    # returns a function with the same parameters as loss_fn that calculates the gradient of the loss function
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    # call the grad_fn function to get the total loss and the gradients
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                    # For AGEM, we need to project the gradients
                    agem_stats = {}

                    def apply_agem_projection():
                        # Sample from memory
                        rng_1, sample_rng = jax.random.split(rng)
                        # Pick a random sample from AGEM memory
                        mem_obs, mem_actions, mem_log_probs, mem_advs, mem_targets, mem_values = sample_memory(
                            cl_state, cfg.agem_sample_size, sample_rng
                        )

                        # Compute memory gradient
                        grads_mem, grads_stats = compute_memory_gradient(
                            network, train_state.params,
                            cfg.clip_eps, cfg.vf_coef, cfg.ent_coef,
                            mem_obs, mem_actions, mem_advs, mem_log_probs,
                            mem_targets, mem_values,
                            env_idx=env_idx
                        )

                        # scale memory gradient by batch-size ratio
                        # ppo_bs = config.num_actors * config.num_steps
                        # mem_bs = config.agem_sample_size
                        g_ppo, _ = ravel_pytree(grads)  # grads  = fresh PPO grads
                        g_mem, _ = ravel_pytree(grads_mem)  # grads_mem = memory grads
                        norm_ppo = jnp.linalg.norm(g_ppo) + 1e-12
                        norm_mem = jnp.linalg.norm(g_mem) + 1e-12
                        scale = norm_ppo / norm_mem * cfg.agem_gradient_scale
                        grads_mem_scaled = jax.tree_util.tree_map(lambda g: g * scale, grads_mem)

                        # Project new grads
                        projected_grads, proj_stats = agem_project(grads, grads_mem_scaled)

                        # Combine stats for logging
                        combined_stats = {**grads_stats, **proj_stats}

                        scaled_norm = jnp.linalg.norm(ravel_pytree(grads_mem_scaled)[0])
                        combined_stats["agem/mem_grad_norm_scaled"] = scaled_norm

                        # Add memory buffer fullness percentage
                        total_used = jnp.sum(cl_state.sizes)
                        total_capacity = cl_state.max_tasks * cl_state.max_size_per_task
                        memory_fullness_pct = (total_used / total_capacity) * 100.0
                        combined_stats["agem/memory_fullness_pct"] = memory_fullness_pct

                        return projected_grads, combined_stats

                    def no_agem_projection():
                        # Return empty stats with the same structure as apply_agem_projection
                        empty_stats = {
                            "agem/agem_alpha": jnp.array(0.0),
                            "agem/agem_dot_g": jnp.array(0.0),
                            "agem/agem_final_grad_norm": jnp.array(0.0),
                            "agem/agem_is_proj": jnp.array(False),
                            "agem/agem_mem_grad_norm": jnp.array(0.0),
                            "agem/agem_ppo_grad_norm": jnp.array(0.0),
                            "agem/agem_projected_grad_norm": jnp.array(0.0),
                            "agem/mem_grad_norm_scaled": jnp.array(0.0),
                            "agem/memory_fullness_pct": jnp.array(0.0),
                            "agem/ppo_actor_loss": jnp.array(0.0),
                            "agem/ppo_entropy": jnp.array(0.0),
                            "agem/ppo_total_loss": jnp.array(0.0),
                            "agem/ppo_value_loss": jnp.array(0.0)
                        }
                        return grads, empty_stats

                    # Use JAX-compatible conditional logic
                    is_agem = (cfg.cl_method.lower() == "agem")  # Python bool; bake into trace

                    def do_proj(_):
                        return apply_agem_projection()

                    def skip_proj(_):
                        return no_agem_projection()

                    # Combine static+dynamic predicate into one JAX bool
                    pred = (jnp.array(is_agem, dtype=jnp.bool_) &
                            (jnp.sum(cl_state.sizes) > 0))

                    grads, agem_stats = jax.lax.cond(pred, do_proj, skip_proj, operand=None)

                    loss_information = total_loss, grads, agem_stats

                    # apply the gradients to the network
                    train_state = train_state.apply_gradients(grads=grads)

                    # Of course we also need to add the network to the carry here
                    return (train_state, cl_state), loss_information

                train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state

                # set the batch size and check if it is correct
                batch_size = cfg.minibatch_size * cfg.num_minibatches
                assert (
                        batch_size == cfg.num_steps * cfg.num_actors
                ), "batch size must be equal to number of steps * number of actors"

                # create a batch of the trajectory, advantages, and targets
                batch = (traj_batch, advantages, targets)

                # reshape the batch to be compatible with the network
                batch = jax.tree_util.tree_map(
                    f=(lambda x: x.reshape((batch_size,) + x.shape[2:])), tree=batch
                )
                # split the random number generator for shuffling the batch
                rng, _rng = jax.random.split(rng)

                # creates random sequences of numbers from 0 to batch_size, one for each vmap
                permutation = jax.random.permutation(_rng, batch_size)

                # shuffle the batch
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )  # outputs a tuple of the batch, advantages, and targets shuffled

                minibatches = jax.tree_util.tree_map(
                    f=(lambda x: jnp.reshape(x, [cfg.num_minibatches, -1] + list(x.shape[1:]))), tree=shuffled_batch,
                )

                (train_state, cl_state), loss_information = jax.lax.scan(
                    f=_update_minbatch,
                    init=(train_state, cl_state),
                    xs=minibatches
                )

                # Handle different return formats based on CL method
                total_loss, grads, agem_stats = loss_information
                # Create a dictionary to store all loss information
                loss_dict = {
                    "total_loss": total_loss
                }
                if cfg.cl_method.lower() == "agem":
                    loss_dict["agem_stats"] = agem_stats

                update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state)
                return update_state, loss_dict

            # create a tuple to be passed into the jax.lax.scan function
            update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state)

            update_state, loss_info = jax.lax.scan(
                f=_update_epoch,
                init=update_state,
                xs=None,
                length=cfg.update_epochs
            )

            # unpack update_state
            train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state
            current_timestep = update_step * cfg.num_steps * cfg.num_envs
            metrics = jax.tree_util.tree_map(lambda x: x.mean(), info)

            if cfg.cl_method.lower() == "agem" and cl_state is not None:
                rng, mem_rng = jax.random.split(rng)
                perm = jax.random.permutation(mem_rng, advantages.shape[0])  # length = traj_len
                idx = perm[: cfg.agem_sample_size]

                obs_for_mem = traj_batch.obs[idx].reshape(-1, traj_batch.obs.shape[-1])
                acts_for_mem = traj_batch.action[idx].reshape(-1)
                logp_for_mem = traj_batch.log_prob[idx].reshape(-1)
                adv_for_mem = advantages[idx].reshape(-1)
                tgt_for_mem = targets[idx].reshape(-1)
                val_for_mem = traj_batch.value[idx].reshape(-1)

                cl_state = update_agem_memory(
                    cl_state, env_idx,
                    obs_for_mem, acts_for_mem, logp_for_mem,
                    adv_for_mem, tgt_for_mem, val_for_mem
                )

            # General section
            # Update the step counter
            update_step += 1

            metrics["General/env_index"] = env_idx
            metrics["General/update_step"] = update_step
            metrics["General/steps_for_env"] = steps_for_env
            metrics["General/env_step"] = update_step * cfg.num_steps * cfg.num_envs
            lr_now = jax.lax.select(
                jnp.array(bool(cfg.anneal_lr)),
                linear_schedule(update_step * cfg.num_minibatches * cfg.update_epochs),
                jnp.array(cfg.lr, dtype=jnp.float32),
            )
            metrics["General/learning_rate"] = lr_now

            # Losses section
            # Extract total_loss and components from loss_info
            loss_dict = loss_info
            total_loss = loss_dict["total_loss"]
            # Unpack the components of total_loss
            value_loss, loss_actor, entropy, reg_loss = total_loss[1]
            total_loss = total_loss[0]  # The actual scalar loss value

            metrics["Losses/total_loss"] = total_loss.mean()
            metrics["Losses/value_loss"] = value_loss.mean()
            metrics["Losses/actor_loss"] = loss_actor.mean()
            metrics["Losses/entropy"] = entropy.mean()
            metrics["Losses/reg_loss"] = reg_loss.mean()

            # Add AGEM stats to metrics if they exist
            if "agem_stats" in loss_dict:
                agem_stats = loss_dict["agem_stats"]
                for k, v in agem_stats.items():
                    if v.size > 0:  # Only add if there are values
                        metrics[k] = v.mean()

            # ── Soup section ─────────────────────────────────────────────
            # info["soups"][agent] has shape (T, E) with {0,1} (deliveries at each step)
            # Stack in a fixed order (env.agents is static for a given env):
            soups_stack = jnp.stack([info["soups"][a] for a in env.agents], axis=0)  # (N_agents, T, E)

            # Sum over time and agents → soups per env for this fragment
            soups_per_env = soups_stack.sum(axis=(0, 1))  # (E,)

            # Average per env (this is the right quantity to compare to a per-env bound)
            avg_per_env = soups_per_env.mean()  # scalar

            # Total across all parallel envs (raw count)
            total_all_envs = soups_per_env.sum()  # scalar

            episode_frac = cfg.num_steps / env.max_steps  # usually 1.0 if num_steps == max_steps
            per_env_upper_bound = max_soup_dict[env_names[env_idx]] * episode_frac

            metrics["Soup/avg_per_env"] = avg_per_env
            metrics["Soup/scaled"] = avg_per_env / per_env_upper_bound
            metrics["Soup/total_all_envs"] = total_all_envs

            # don't keep the heavy arrays around
            metrics.pop("soups", None)

            # Rewards section
            # Agent-agnostic reward logging
            for agent in env.agents:
                metrics[f"General/shaped_reward_{agent}"] = metrics["shaped_reward"][agent]
                metrics[f"General/shaped_reward_annealed_{agent}"] = metrics[
                                                                         f"General/shaped_reward_{agent}"] * rew_shaping_anneal(
                    current_timestep)
            metrics.pop('shaped_reward', None)

            # Advantages and Targets section
            metrics["Advantage_Targets/advantages"] = advantages.mean()
            metrics["Advantage_Targets/targets"] = targets.mean()

            # Dormant neuron ratio - calculate from current batch
            obs_batch = batchify(last_obs, env.agents, cfg.num_actors, not cfg.use_cnn)
            _, _, current_dormant_ratio = network.apply(train_state.params, obs_batch, env_idx=env_idx)
            metrics["Neural_Activity/dormant_ratio"] = current_dormant_ratio

            def evaluate_and_log(metrics_in, train_state_in, rng_in, update_step_in):
                def do_eval(mets, ts, key, upd):
                    key, k_eval = jax.random.split(key)
                    avg_r, total_soups = eval_current_env(ts, k_eval, env, cfg.eval_num_steps, cfg.eval_num_episodes,
                                                          env_idx)

                    # Scale soups by episode length (to make it comparable with max soup bound)
                    eval_episode_frac = cfg.eval_num_steps / env.max_steps
                    scaled_eval_soups = total_soups * eval_episode_frac

                    mets = {
                        **mets,
                        "Eval/avg_return": avg_r,
                        "Eval/total_soups": total_soups,
                        "Eval/scaled_soups": scaled_eval_soups,
                    }

                    # single io_callback to write everything
                    def cb(args):
                        mets_cb, upd_cb, env_counter = args
                        real_step = (int(env_counter) - 1) * cfg.num_updates + int(upd_cb)
                        for k, v in mets_cb.items():
                            writer.add_scalar(k, float(v), real_step)

                    jax.experimental.io_callback(cb, None, (mets, upd, env_idx + 1))
                    return key

                def skip_eval(mets, ts, key, upd):
                    # still write the non-eval metrics
                    def cb(args):
                        mets_cb, upd_cb, env_counter = args
                        real_step = (int(env_counter) - 1) * cfg.num_updates + int(upd_cb)
                        for k, v in mets_cb.items():
                            writer.add_scalar(k, float(v), real_step)

                    jax.experimental.io_callback(cb, None, (mets, upd, env_idx + 1))
                    return key

                return jax.lax.cond(
                    (update_step_in % cfg.log_interval) == 0,
                    do_eval, skip_eval,
                    metrics_in, train_state_in, rng_in, update_step_in
                )

            # Evaluate the model and log the metrics
            rng = evaluate_and_log(metrics, train_state, rng, update_step)

            runner_state = (train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state)

            return runner_state, metrics

        rng, train_rng = jax.random.split(rng)

        # initialize a carrier that keeps track of the states and observations of the agents
        runner_state = (train_state, env_state, obsv, 0, 0, train_rng, cl_state)

        # apply the _update_step function a series of times, while keeping track of the state
        runner_state, metrics = jax.lax.scan(
            f=_update_step,
            init=runner_state,
            xs=None,
            length=cfg.num_updates
        )

        # Return the runner state after the training loop, and the metrics arrays
        return runner_state, metrics

    def loop_over_envs(rng, train_state, cl_state, envs):
        '''
        Loops over the environments and trains the network
        @param rng: random number generator
        @param train_state: the current state of the training
        @param envs: the environments
        returns the runner state and the metrics
        '''
        # split the random number generator for training on the environments
        rng, *env_rngs = jax.random.split(rng, len(envs) + 1)

        visualizer = None
        for i, (rng, env) in enumerate(zip(env_rngs, envs)):
            # --- Train on environment i using the *current* ewc_state ---
            runner_state, metrics = train_on_environment(rng, train_state, env, cl_state, i)
            train_state = runner_state[0]
            cl_state = runner_state[6]

            importance = cl.compute_importance(train_state.params, env, network, i, rng, cfg.use_cnn,
                                               cfg.importance_episodes, cfg.importance_steps,
                                               cfg.normalize_importance)

            cl_state = cl.update_state(cl_state, train_state.params, importance)

            if cfg.record_gif:
                if visualizer is None:
                    visualizer = create_visualizer(temp_env.num_agents, cfg.env_name)
                # Generate & log a GIF after finishing task i
                env_name = layout_names[i]
                states = record_gif_of_episode(cfg, train_state, env, network, i, cfg.gif_len)
                # Pass environment instance to PO visualizer for view highlighting
                if cfg.env_name == "overcooked_po":
                    visualizer.animate(states, agent_view_size=5, task_idx=i, task_name=env_name, exp_dir=exp_dir,
                                       env=env)
                else:
                    visualizer.animate(states, agent_view_size=5, task_idx=i, task_name=env_name, exp_dir=exp_dir)

            # save the model
            repo_root = Path(__file__).resolve().parent.parent
            path = f"{repo_root}/checkpoints/overcooked/{cfg.cl_method}/{run_name}/model_env_{i + 1}"
            save_params(path, train_state, env_kwargs=env.layout, layout_name=env.layout_name, config=cfg)

            if cfg.single_task_idx is not None:
                break  # stop after the first env

    def save_params(path, train_state, env_kwargs=None, layout_name=None, config=None):
        '''
        Saves the parameters of the network along with environment configuration
        @param path: the path to save the parameters
        @param train_state: the current state of the training
        @param env_kwargs: the environment kwargs used to create the environment
        @param layout_name: the name of the layout
        @param config: the configuration used for training
        returns None
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model parameters
        with open(path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    {"params": train_state.params}
                )
            )

        # Save configuration and layout information
        if env_kwargs is not None or layout_name is not None or config is not None:
            # Define a recursive function to convert FrozenDict to regular dict
            def convert_frozen_dict(obj):
                if isinstance(obj, flax.core.frozen_dict.FrozenDict):
                    return {k: convert_frozen_dict(v) for k, v in unfreeze(obj).items()}
                elif isinstance(obj, dict):
                    return {k: convert_frozen_dict(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_frozen_dict(item) for item in obj]
                elif isinstance(obj, jax.Array):
                    # Convert JAX arrays to native Python types
                    array_obj = np.array(obj)
                    # Handle scalar values
                    if array_obj.size == 1:
                        return array_obj.item()
                    # Handle arrays
                    return array_obj.tolist()
                else:
                    return obj

            # Convert env_kwargs to regular dict
            env_kwargs = convert_frozen_dict(env_kwargs)

            config_data = {
                "env_kwargs": env_kwargs,
                "layout_name": layout_name
            }

            # Add relevant configuration parameters
            if config is not None:
                config_dict = {
                    "use_cnn": config.use_cnn,
                    "num_tasks": config.seq_length,
                    "use_multihead": config.use_multihead,
                    "shared_backbone": config.shared_backbone,
                    "big_network": config.big_network,
                    "use_task_id": config.use_task_id,
                    "regularize_heads": config.regularize_heads,
                    "use_layer_norm": config.use_layer_norm,
                    "activation": config.activation,
                    "strategy": config.strategy,
                    "seed": config.seed,
                    "height_min": config.height_min,
                    "height_max": config.height_max,
                    "width_min": config.width_min,
                    "width_max": config.width_max,
                    "wall_density": config.wall_density
                }
                # Convert any FrozenDict objects in the config
                config_dict = convert_frozen_dict(config_dict)
                config_data.update(config_dict)

            # Apply the conversion to the entire config_data to ensure all nested FrozenDict objects are converted
            config_data = convert_frozen_dict(config_data)

            config_path = f"{path}_config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

        print('model saved to', path)

    rng, train_rng = jax.random.split(rng)
    cl_state = cl.init_state(train_state.params, cfg.regularize_critic, cfg.regularize_heads)

    # Initialize AGEM memory if using AGEM and this is the first environment
    if cfg.cl_method.lower() == "agem":
        # Get observation dimension
        obs_dim = envs[0].observation_space().shape
        if not cfg.use_cnn:
            obs_dim = (np.prod(obs_dim),)
        # Initialize memory buffer
        cl_state = init_agem_memory(cfg.agem_memory_size, obs_dim, max_tasks=cfg.seq_length)

    # apply the loop_over_envs function to the environments
    loop_over_envs(train_rng, train_state, cl_state, envs)


if __name__ == "__main__":
    print("Running main...")
    main()
