import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Sequence, Any, Optional, List

import flax
import numpy as np
import optax
import tyro
import wandb
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState
from jax._src.flatten_util import ravel_pytree

from experiments.model.decoupled_mlp import Actor, Critic
from experiments.utils import *
from experiments.continual.agem import AGEM, init_agem_memory, sample_memory, compute_memory_gradient, agem_project, \
    update_agem_memory
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from meal.env.utils.max_soup_calculator import calculate_max_soup
from meal.visualization.visualizer import OvercookedVisualizer
from meal.visualization.visualizer_po import OvercookedVisualizerPO
from meal import make_env
from meal.wrappers.logging import LogWrapper


@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING / PPO PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    alg_name: str = "mappo"
    lr: float = 3e-4
    anneal_lr: bool = False
    num_envs: int = 16
    num_steps: int = 128
    steps_per_task: float = 1e7
    update_epochs: int = 8
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.957
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

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
    use_agent_id: bool = True
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
    env_name: str = "overcooked_po"
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


def create_global_state_for_critic(obs_dict, agent_list, num_envs, use_cnn=False):
    """
    Create global state for MAPPO critic by concatenating all agents' observations.
    For MAPPO, the critic should receive concatenated observations from all agents,
    not batched observations.

    Args:
        obs_dict: Dictionary of observations for each agent
        agent_list: List of agent names
        num_envs: Number of parallel environments
        use_cnn: Whether to use CNN mode (preserves spatial dimensions)

    Returns:
        Global state with appropriate shape for the critic network:
        - For MLP: (num_envs, total_obs_dim) where total_obs_dim is flattened
        - For CNN: (num_envs, height, width, total_channels) where channels are concatenated
    """
    # Stack observations from all agents: (num_agents, num_envs, ...)
    agent_obs = jnp.stack([obs_dict[agent] for agent in agent_list])

    if use_cnn:
        # For CNN mode: preserve spatial dimensions and concatenate along channel dimension
        # Expected shape: (num_agents, num_envs, height, width, channels)
        # Transpose to: (num_envs, num_agents, height, width, channels)
        agent_obs = jnp.transpose(agent_obs, (1, 0, 2, 3, 4))

        # Concatenate along the channel dimension for each spatial position
        # Shape: (num_envs, height, width, num_agents * channels)
        global_state = jnp.concatenate([agent_obs[:, i] for i in range(agent_obs.shape[1])], axis=-1)
    else:
        # For MLP mode: flatten all dimensions except the first two
        # Original shape: (num_agents, num_envs, ...) where ... can be multiple dimensions
        # Reshape to: (num_agents, num_envs, flattened_obs_dim)
        agent_obs = agent_obs.reshape(agent_obs.shape[0], agent_obs.shape[1], -1)

        # Transpose to (num_envs, num_agents, flattened_obs_dim)
        agent_obs = jnp.transpose(agent_obs, (1, 0, 2))

        # Concatenate along the last dimension to create global state
        # Shape: (num_envs, num_agents * flattened_obs_dim)
        global_state = agent_obs.reshape(num_envs, -1)

    return global_state


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

    # Add view parameters for PO environments when difficulty is specified
    if config.env_name == "overcooked_po" and difficulty:
        for env_args in config.env_kwargs:
            env_args["view_ahead"] = config.view_ahead
            env_args["view_sides"] = config.view_sides
            env_args["view_behind"] = config.view_behind

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
    exp_dir = os.path.join("runs", run_name)

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
        Get view parameters for overcooked_po environments from config.
        Returns a dictionary with view parameters if applicable, empty dict otherwise.
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

    @partial(jax.jit)
    def evaluate_model(train_state, key):
        '''
        Evaluates the model by running 10 episodes on all environments and returns the average reward
        @param train_state: the current state of the training
        @param config: the configuration of the training
        returns the average reward
        '''

        def run_episode_while(env, key_r, max_steps=1000):
            """
            Run a single episode using jax.lax.while_loop
            """

            class EvalState(NamedTuple):
                key: Any
                state: Any
                obs: Any
                done: bool
                total_reward: float
                soup: float
                step_count: int

            def cond_fun(state: EvalState):
                '''
                Checks if the episode is done or if the maximum number of steps has been reached
                @param state: the current state of the loop
                returns a boolean indicating whether the loop should continue
                '''
                return jnp.logical_and(jnp.logical_not(state.done), state.step_count < max_steps)

            def body_fun(state: EvalState):
                '''
                Performs a single step in the environment
                @param state: the current state of the loop
                returns the updated state
                '''

                key, state_env, obs, _, total_reward, total_soup, step_count = state
                key, key_a0, key_a1, key_s = jax.random.split(key, 4)

                # ***Create a batched copy for the network only.***
                # For each agent, expand dims to get shape (1, H, W, C) then flatten to (1, -1)
                batched_obs = {}
                for agent, v in obs.items():
                    v_b = jnp.expand_dims(v, axis=0)  # now (1, H, W, C)
                    if not config.use_cnn:
                        v_b = jnp.reshape(v_b, (v_b.shape[0], -1))  # flatten
                    batched_obs[agent] = v_b

                def select_action(train_state, rng, obs):
                    '''
                    Selects an action based on the policy network
                    @param params: the parameters of the network
                    @param rng: random number generator
                    @param obs: the observation
                    returns the action
                    '''
                    network_apply = train_state.apply_fn
                    params = train_state.params

                    # For evaluation, we only need the actor network for action selection
                    pi, _ = network_apply(params, obs, env_idx=eval_idx, network_type='actor')
                    value = None  # We don't need value during evaluation

                    action = jnp.squeeze(pi.sample(seed=rng), axis=0)
                    return action, value

                # Get action distributions
                action_a1, _ = select_action(train_state, key_a0, batched_obs["agent_0"])
                action_a2, _ = select_action(train_state, key_a1, batched_obs["agent_1"])

                # Sample actions
                actions = {
                    "agent_0": action_a1,
                    "agent_1": action_a2
                }

                # Environment step
                next_obs, next_state, reward, done_step, info = env.step(key_s, state_env, actions)
                done = done_step["__all__"]
                reward = reward["agent_0"]  # Common reward
                soups_this_step = info["soups"]["agent_0"] + info["soups"]["agent_1"]
                total_reward += reward
                total_soup += soups_this_step
                step_count += 1

                return EvalState(key, next_state, next_obs, done, total_reward, total_soup, step_count)

            # Initialize
            key, key_s = jax.random.split(key_r)
            obs, state = env.reset(key_s)
            init_state = EvalState(key, state, obs, False, 0.0, 0.0, 0)

            # Run while loop
            final_state = jax.lax.while_loop(
                cond_fun=cond_fun,
                body_fun=body_fun,
                init_val=init_state
            )

            return final_state.total_reward, final_state.soup

        # Loop through all environments
        all_avg_rewards = []
        all_avg_soups = []

        envs, agent_restrictions_list = create_environments()

        for eval_idx, env in enumerate(envs):
            # Create the environment with agent restrictions
            agent_restrictions = agent_restrictions_list[eval_idx]
            view_params = get_view_params()
            env = make_env(config.env_name, layout=env, agent_restrictions=agent_restrictions, **view_params)

            # Run k episodes
            all_rewards, all_soups = jax.vmap(lambda k: run_episode_while(env, k, config.eval_num_steps))(
                jax.random.split(key, config.eval_num_episodes)
            )

            avg_reward = jnp.mean(all_rewards)
            avg_soups = jnp.sum(all_soups)
            all_avg_rewards.append(avg_reward)
            all_avg_soups.append(avg_soups)

        return all_avg_rewards, all_avg_soups

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

    # set extra config parameters based on the environment
    temp_env = envs[0]
    config.num_actors = temp_env.num_agents * config.num_envs
    config.num_updates = config.steps_per_task // config.num_steps // config.num_envs
    config.minibatch_size = (config.num_actors * config.num_steps) // config.num_minibatches

    def linear_schedule(count):
        '''
        Linearly decays the learning rate depending on the number of minibatches and number of epochs
        returns the learning rate
        '''
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    # MAPPO always uses decoupled Actor and Critic networks
    obs_dim = temp_env.observation_space().shape
    if not config.use_cnn:
        local_obs_dim = np.prod(obs_dim)  # Individual agent observation dimension
        global_obs_dim = local_obs_dim * temp_env.num_agents  # Global state dimension
    else:
        local_obs_dim = obs_dim  # Keep original shape for CNN
        global_obs_dim = (obs_dim[0], obs_dim[1], obs_dim[2] * temp_env.num_agents)  # Stack channels for CNN

    # Create separate actor and critic networks
    actor_network = Actor(
        action_dim=temp_env.action_space().n,
        activation=config.activation,
        num_tasks=seq_length,
        use_multihead=config.use_multihead,
        use_task_id=config.use_task_id,
        use_cnn=config.use_cnn,
        use_layer_norm=config.use_layer_norm,
        use_agent_id=config.use_agent_id,
        num_agents=temp_env.num_agents,
        num_envs=config.num_envs
    )

    critic_network = Critic(
        activation=config.activation,
        num_tasks=seq_length,
        use_multihead=config.use_multihead,
        use_task_id=config.use_task_id,
        use_cnn=config.use_cnn,
        use_layer_norm=config.use_layer_norm
    )

    # Initialize both networks
    rng = jax.random.PRNGKey(seed)
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)

    # Actor uses local observations
    if config.use_cnn:
        actor_init_x = jnp.zeros((1, *local_obs_dim))
    else:
        actor_init_x = jnp.zeros((1, local_obs_dim))
    actor_params = actor_network.init(actor_rng, actor_init_x, env_idx=0)

    # Critic uses global state (concatenated observations from all agents)
    if config.use_cnn:
        critic_init_x = jnp.zeros((1, *global_obs_dim))
    else:
        critic_init_x = jnp.zeros((1, global_obs_dim))
    critic_params = critic_network.init(critic_rng, critic_init_x, env_idx=0)

    # Combine parameters into a single structure for compatibility
    network_params = {
        'actor': actor_params,
        'critic': critic_params
    }

    # Create a wrapper network object for compatibility with functions expecting unified interface
    class DecoupledNetworkWrapper:
        """
        Wrapper class to provide unified network interface for decoupled actor/critic networks.
        This allows EWC and other functions to work with decoupled networks.
        """

        def __init__(self, actor_net, critic_net):
            self.actor_network = actor_net
            self.critic_network = critic_net

        def apply(self, params, obs, *, env_idx=0):
            """
            Apply method that mimics unified network behavior.
            For EWC and similar functions, we primarily need the actor network's policy output.
            """
            # Use actor network for policy (which is what EWC needs for Fisher computation)
            if isinstance(params, dict) and 'actor' in params:
                # Decoupled parameters structure
                pi = self.actor_network.apply(params['actor'], obs, env_idx=env_idx)
                # For compatibility, return None for value (EWC doesn't need it)
                return pi, None
            else:
                # Fallback for unified parameters structure
                pi = self.actor_network.apply(params, obs, env_idx=env_idx)
                return pi, None

    network = DecoupledNetworkWrapper(actor_network, critic_network)

    # Initialize the optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
    )

    # JIT compile the separate networks
    actor_network.apply = jax.jit(actor_network.apply)
    critic_network.apply = jax.jit(critic_network.apply)

    # Create a combined apply function for compatibility
    def combined_apply_fn(params, obs, *, env_idx=0, network_type='both'):
        if network_type == 'actor':
            return actor_network.apply(params['actor'], obs, env_idx=env_idx), None
        elif network_type == 'critic':
            return None, critic_network.apply(params['critic'], obs, env_idx=env_idx)
        else:  # both
            pi = actor_network.apply(params['actor'], obs, env_idx=env_idx)
            value = critic_network.apply(params['critic'], obs, env_idx=env_idx)
            return pi, value

    # Initialize the training state
    train_state = TrainState.create(
        apply_fn=combined_apply_fn,
        params=network_params,
        tx=tx
    )

    @partial(jax.jit, static_argnums=(2, 4))
    def train_on_environment(rng, train_state, env, cl_state, env_idx):
        '''
        Trains the network using MAPPO
        @param rng: random number generator
        returns the runner state and the metrics
        '''

        print(f"Training on environment: {env.task_id} - {env.layout_name}")

        # reset the learning rate and the optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
        )
        new_optimizer = tx.init(train_state.params)
        train_state = train_state.replace(tx=tx, opt_state=new_optimizer)

        # Initialize and reset the environment
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        reward_shaping_horizon = config.steps_per_task / 2
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
                obs_batch = batchify(last_obs, env.agents, config.num_actors, not config.use_cnn)
                # print("obs_shape", obs_batch.shape)

                # For MAPPO: Create global state for centralized critic
                # The critic should receive concatenated observations from all agents
                global_state = create_global_state_for_critic(last_obs, env.agents, config.num_envs, config.use_cnn)

                # MAPPO: Actor uses local observations, Critic uses global state
                pi = train_state.apply_fn(train_state.params, obs_batch, env_idx=env_idx, network_type='actor')[0]
                # Critic outputs one value per environment (not per agent)
                value_per_env = \
                train_state.apply_fn(train_state.params, global_state, env_idx=env_idx, network_type='critic')[1]
                # Tile the values to match the batch structure (one value per agent, but same value for agents in same env)
                value = jnp.repeat(value_per_env, len(env.agents), axis=0)

                # Store the global state batch for use in loss computation (repeat for each agent)
                global_state_batch = jnp.repeat(global_state, len(env.agents), axis=0)

                # Sample and action from the policy
                action = pi.sample(seed=_rng)

                log_prob = pi.log_prob(action)

                # format the actions to be compatible with the environment
                env_act = unbatchify(action, env.agents, config.num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                # split the random number generator for stepping the environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.num_envs)

                # simultaniously step all environments with the selected actions (parallelized over the number of environments with vmap)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                current_timestep = update_step * config.num_steps * config.num_envs

                # Apply different reward settings based on configuration
                if config.sparse_rewards:
                    # Sparse rewards: only delivery rewards (no shaped rewards)
                    # reward already contains individual delivery rewards from environment
                    pass
                elif config.individual_rewards:
                    # Individual rewards: delivery rewards + individual shaped rewards
                    # Environment now provides individual delivery rewards directly
                    reward = jax.tree_util.tree_map(lambda x, y:
                                                    x + y * rew_shaping_anneal(current_timestep),
                                                    reward,
                                                    info["shaped_reward"]
                                                    )
                else:
                    # Default behavior: shared delivery rewards + individual shaped rewards
                    # Convert individual delivery rewards to shared rewards (both agents get total)
                    total_delivery_reward = reward["agent_0"] + reward["agent_1"]
                    shared_delivery_rewards = {"agent_0": total_delivery_reward, "agent_1": total_delivery_reward}

                    reward = jax.tree_util.tree_map(lambda x, y:
                                                    x + y * rew_shaping_anneal(current_timestep),
                                                    shared_delivery_rewards,
                                                    info["shaped_reward"]
                                                    )

                transition = Transition_MAPPO(
                    batchify(done, env.agents, config.num_actors, not config.use_cnn).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config.num_actors).squeeze(),
                    log_prob,
                    obs_batch,  # Use local observations for actor
                    global_state_batch  # Store real global state for critic
                )

                # Increment steps_for_env by the number of parallel envs
                steps_for_env = steps_for_env + config.num_envs

                runner_state = (train_state, env_state, obsv, update_step, steps_for_env, rng, cl_state)
                return runner_state, (transition, info)

            # Apply the _env_step function a series of times, while keeping track of the runner state
            runner_state, (traj_batch, info) = jax.lax.scan(
                f=_env_step,
                init=runner_state,
                xs=None,
                length=config.num_steps
            )

            # unpack the runner state that is returned after the scan function
            train_state, env_state, last_obs, update_step, steps_for_env, rng, cl_state = runner_state

            # create a batch of the observations that is compatible with the network
            last_obs_batch = batchify(last_obs, env.agents, config.num_actors, not config.use_cnn)

            # Compute last value for MAPPO: Critic uses global state, outputs one value per environment
            last_global_state = create_global_state_for_critic(last_obs, env.agents, config.num_envs, config.use_cnn)
            _, last_val_per_env = train_state.apply_fn(train_state.params, last_global_state, env_idx=env_idx,
                                                       network_type='critic')
            # Tile the last values to match the batch structure (one value per agent, but same value for agents in same env)
            last_val = jnp.repeat(last_val_per_env, len(env.agents), axis=0)

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
                    delta = reward + config.gamma * next_value * (1 - done) - value  # calculate the temporal difference
                    gae = (
                            delta
                            + config.gamma * config.gae_lambda * (1 - done) * gae
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
            def _update_epoch(update_state, unused):
                '''
                performs a single update epoch in the training loop
                @param update_state: the current state of the update
                returns the updated update_state and the total loss
                '''

                def _update_minbatch(carry, batch_info):
                    '''
                    performs a single update minibatch in the training loop
                    @param train_state: the current state of the training
                    @param batch_info: the information of the batch
                    returns the updated train_state and the total loss
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
                        returns the total loss and the value loss, actor loss, and entropy
                        '''
                        # MAPPO: Actor uses local observations, Critic uses stored global state
                        local_obs = traj_batch.obs  # Local observations for actor
                        global_state = traj_batch.global_state  # Real global state for critic (no reconstruction needed!)

                        # Apply networks separately
                        pi = actor_network.apply(params['actor'], local_obs, env_idx=env_idx)
                        value = critic_network.apply(params['critic'], global_state, env_idx=env_idx)
                        log_prob = pi.log_prob(traj_batch.action)

                        # calculate critic loss
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config.clip_eps,
                                                                                                config.clip_eps)
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
                                    1.0 - config.clip_eps,
                                    1.0 + config.clip_eps,
                                )
                                * gae
                        )

                        loss_actor = -jnp.minimum(loss_actor_unclipped,
                                                  loss_actor_clipped)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # CL penalty (for regularization-based methods)
                        cl_penalty = cl.penalty(params, cl_state, config.reg_coef)

                        total_loss = (loss_actor
                                      + config.vf_coef * value_loss
                                      - config.ent_coef * entropy
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
                            cl_state, config.agem_sample_size, sample_rng
                        )

                        # Compute memory gradient
                        grads_mem, grads_stats = compute_memory_gradient(
                            network, train_state.params,
                            config.clip_eps, config.vf_coef, config.ent_coef,
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
                        scale = norm_ppo / norm_mem * config.agem_gradient_scale
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
                    if config.cl_method.lower() == "agem" and cl_state is not None:
                        grads, agem_stats = jax.lax.cond(
                            jnp.sum(cl_state.sizes) > 0,
                            lambda: apply_agem_projection(),
                            lambda: no_agem_projection()
                        )

                    loss_information = total_loss, grads, agem_stats

                    # apply the gradients to the network
                    train_state = train_state.apply_gradients(grads=grads)

                    # Of course we also need to add the network to the carry here
                    return (train_state, cl_state), loss_information

                train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state

                # set the batch size and check if it is correct
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                        batch_size == config.num_steps * config.num_actors
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
                    f=(lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:]))), tree=shuffled_batch,
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
                if config.cl_method.lower() == "agem":
                    loss_dict["agem_stats"] = agem_stats

                avg_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)

                update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state)
                return update_state, loss_dict

            # create a tuple to be passed into the jax.lax.scan function
            update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state)

            update_state, loss_info = jax.lax.scan(
                f=_update_epoch,
                init=update_state,
                xs=None,
                length=config.update_epochs
            )

            # unpack update_state
            train_state, traj_batch, advantages, targets, steps_for_env, rng, cl_state = update_state
            current_timestep = update_step * config.num_steps * config.num_envs
            metrics = jax.tree_util.tree_map(lambda x: x.mean(), info)

            if config.cl_method.lower() == "agem" and cl_state is not None:
                rng, mem_rng = jax.random.split(rng)
                perm = jax.random.permutation(mem_rng, advantages.shape[0])  # length = traj_len
                idx = perm[: config.agem_sample_size]

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
            metrics["General/env_step"] = update_step * config.num_steps * config.num_envs
            if config.anneal_lr:
                metrics["General/learning_rate"] = linear_schedule(
                    update_step * config.num_minibatches * config.update_epochs)
            else:
                metrics["General/learning_rate"] = config.lr

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

            # Soup section
            agent_0_soup = info["soups"]["agent_0"].sum()
            agent_1_soup = info["soups"]["agent_1"].sum()
            soup_delivered = agent_0_soup + agent_1_soup
            episode_frac = config.num_steps / env.max_steps
            metrics["Soup/agent_0_soup"] = agent_0_soup
            metrics["Soup/agent_1_soup"] = agent_1_soup
            metrics["Soup/total"] = soup_delivered
            metrics["Soup/scaled"] = soup_delivered / (max_soup_dict[env_names[env_idx]] * episode_frac)
            metrics.pop('soups', None)

            # Rewards section
            metrics["General/shaped_reward_agent0"] = metrics["shaped_reward"]["agent_0"]
            metrics["General/shaped_reward_agent1"] = metrics["shaped_reward"]["agent_1"]
            metrics.pop('shaped_reward', None)
            metrics["General/shaped_reward_annealed_agent0"] = metrics[
                                                                   "General/shaped_reward_agent0"] * rew_shaping_anneal(
                current_timestep)
            metrics["General/shaped_reward_annealed_agent1"] = metrics[
                                                                   "General/shaped_reward_agent1"] * rew_shaping_anneal(
                current_timestep)

            # Advantages and Targets section
            metrics["Advantage_Targets/advantages"] = advantages.mean()
            metrics["Advantage_Targets/targets"] = targets.mean()

            def evaluate_and_log(rng, update_step):
                rng, eval_rng = jax.random.split(rng)
                train_state_eval = jax.tree_util.tree_map(lambda x: x.copy(), train_state)

                def log_metrics(metrics, update_step):
                    if config.evaluation:
                        avg_rewards, avg_soups = evaluate_model(train_state_eval, eval_rng)
                        episode_frac = config.eval_num_steps / env.max_steps
                        avg_soups = [soup * episode_frac for soup, env_name in zip(avg_soups, env_names)]
                        metrics = add_eval_metrics(avg_rewards, avg_soups, env_names, max_soup_dict, metrics)

                    def callback(args):
                        metrics, update_step, env_counter = args
                        real_step = (int(env_counter) - 1) * config.num_updates + int(update_step)
                        for key, value in metrics.items():
                            writer.add_scalar(key, value, real_step)

                    jax.experimental.io_callback(callback, None, (metrics, update_step, env_idx + 1))
                    return None

                def do_not_log(metrics, update_step):
                    return None

                jax.lax.cond((update_step % config.log_interval) == 0, log_metrics, do_not_log, metrics, update_step)

            # Evaluate the model and log the metrics
            evaluate_and_log(rng=rng, update_step=update_step)

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
            length=config.num_updates
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

        # Create appropriate visualizer based on environment type
        if config.env_name == "overcooked_po":
            visualizer = OvercookedVisualizerPO(num_agents=temp_env.num_agents)
        else:
            visualizer = OvercookedVisualizer(num_agents=temp_env.num_agents)

        evaluation_matrix = None
        if config.eval_forward_transfer:
            evaluation_matrix = jnp.zeros(((len(envs) + 1), len(envs)))
            rng, eval_rng = jax.random.split(rng)
            evaluations = evaluate_model(train_state, eval_rng)
            evaluation_matrix = evaluation_matrix.at[0, :].set(evaluations)

        for task_idx, (rng, env) in enumerate(zip(env_rngs, envs)):
            # --- Train on environment i using the *current* ewc_state ---
            runner_state, metrics = train_on_environment(rng, train_state, env, cl_state, task_idx)
            train_state = runner_state[0]
            cl_state = runner_state[6]

            importance = cl.compute_importance(train_state.params, env, network, task_idx, rng, config.use_cnn,
                                               config.importance_episodes, config.importance_steps,
                                               config.normalize_importance)

            cl_state = cl.update_state(cl_state, train_state.params, importance)

            if config.record_gif:
                # Generate & log a GIF after finishing task i
                env_name = f"{task_idx}__{env.layout_name}"
                states = record_gif_of_episode(config, train_state, env, network, env_idx=task_idx, max_steps=config.gif_len)
                file_path = f"{exp_dir}/task_{task_idx}_{env_name}.gif"
                # Pass environment instance to PO visualizer for view highlighting
                if env_name == "overcooked_po":
                    visualizer.animate(states, out_path=file_path, env=env)
                else:
                    visualizer.animate(states, out_path=file_path)

            if config.eval_forward_transfer:
                # Evaluate at the end of training to get the average performance of the task right after training
                evaluations = evaluate_model(train_state, rng)
                evaluation_matrix = evaluation_matrix.at[task_idx, :].set(evaluations)

            # save the model
            repo_root = Path(__file__).resolve().parent.parent
            path = f"{repo_root}/checkpoints/overcooked/{config.cl_method}/{run_name}/model_env_{task_idx + 1}"
            save_params(path, train_state, env_kwargs=env.layout, layout_name=env.layout_name, config=config)

            if config.eval_forward_transfer:
                # calculate the forward transfer and backward transfer
                pass

            if config.single_task_idx is not None:
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

    # Run the model
    rng, train_rng = jax.random.split(rng)
    cl_state = cl.init_state(train_state.params, config.regularize_critic, config.regularize_heads)

    # Initialize AGEM memory if using AGEM and this is the first environment
    if config.cl_method.lower() == "agem":
        # Get observation dimension
        obs_dim = envs[0].observation_space().shape
        if not config.use_cnn:
            obs_dim = (np.prod(obs_dim),)
        # Initialize memory buffer
        cl_state = init_agem_memory(config.agem_memory_size, obs_dim, max_tasks=config.seq_length)

    # apply the loop_over_envs function to the environments
    loop_over_envs(train_rng, train_state, cl_state, envs)


if __name__ == "__main__":
    print("Running main...")
    main()
