import json
from dataclasses import dataclass, field
from functools import partial
from typing import Sequence, Any, Optional, List

import flax
import flax.linen as nn
import numpy as np
import optax
import tyro
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState

from experiments.model.decoupled_mlp import Actor, Critic
from experiments.utils import *
from experiments.continual.packnet import Packnet, PacknetState
from meal.env.utils.max_soup_calculator import calculate_max_soup
from meal.visualization.visualizer import OvercookedVisualizer
from meal import make_env
from meal.wrappers.logging import LogWrapper


@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING / PPO PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    alg_name: str = "ippo"
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
    cl_method: Optional[str] = "packnet"
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

    # Packnet specific parameters
    train_epochs: int = 8
    finetune_epochs: int = 2
    finetune_lr: float = 1e-4
    finetune_timesteps: int = 1e6
    network_architecture: str = "mlp_decoupled"

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
    group: Optional[str] = None

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
    finetune_updates: int = 0


############################
##### MAIN FUNCTION    #####
############################


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

    # Validate that we're using Packnet
    if config.cl_method is None or config.cl_method.lower() != "packnet":
        raise ValueError("This script only supports Packnet. Please set cl_method to 'packnet'.")

    difficulty = config.difficulty
    seq_length = config.seq_length
    strategy = config.strategy
    seed = config.seed

    # Initialize Packnet
    cl = Packnet(seq_length=config.seq_length,
                 prune_instructions=0.4,
                 train_finetune_split=(config.train_epochs, config.finetune_epochs),
                 prunable_layers=[nn.Dense])

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

    # generate the run name
    network = config.network_architecture
    run_name = create_run_name(config, network)
    exp_dir = os.path.join("../runs", run_name)

    # Initialize WandB and Tensorboard 
    writer = initialize_logging_setup(config, run_name, exp_dir)

    # Log hyperparameters table
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
        if config.env_name == "overcooked_po" and difficulty:
            return {
                "view_ahead": config.view_ahead,
                "view_sides": config.view_sides,
                "view_behind": config.view_behind
            }
        return {}

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
                    @param train_state: the training state (tuple of actor and critic for Packnet)
                    @param rng: random number generator
                    @param obs: the observation
                    returns the action and value
                    '''
                    # For Packnet, train_state is a tuple of (actor_train_state, critic_train_state)
                    if isinstance(train_state, tuple):
                        actor_train_state, critic_train_state = train_state
                        pi = actor.apply(actor_train_state.params, obs, env_idx=eval_idx)
                        value = critic.apply(critic_train_state.params, obs, env_idx=eval_idx)
                    else:
                        # For unified architecture
                        network_apply = train_state.apply_fn
                        params = train_state.params
                        pi, value, _ = network_apply(params, obs, env_idx=eval_idx)

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
            keys = jax.random.split(key, config.eval_num_episodes)
            all_rewards, all_soups = jax.vmap(lambda k: run_episode_while(env, k, config.eval_num_steps))(keys)

            avg_reward = jnp.mean(all_rewards)
            avg_soup = jnp.mean(all_soups)
            all_avg_rewards.append(avg_reward)
            all_avg_soups.append(avg_soup)

        return all_avg_rewards, all_avg_soups

    # create all environments
    envs_layouts, agent_restrictions_list = create_environments()

    envs = []
    max_soup_dict = {}
    for i, env_layout in enumerate(envs_layouts):
        agent_restrictions = agent_restrictions_list[i]
        view_params = get_view_params()
        env = make_env(config.env_name, layout=env_layout, agent_restrictions=agent_restrictions, **view_params)
        env = LogWrapper(env, replace_info=False)
        envs.append(env)

        # Calculate maximum soup for this environment
        env_name = layout_names[i] if i < len(layout_names) else f"env_{i}"
        max_soup_dict[env_name] = calculate_max_soup(env_layout, env.max_steps, n_agents=env.num_agents)

    # set extra config parameters based on the environment
    temp_env = envs[0]
    config.num_actors = temp_env.num_agents * config.num_envs
    config.num_updates = config.steps_per_task // config.num_steps // config.num_envs
    print(f"num_updates: {config.num_updates}")
    config.finetune_updates = config.finetune_timesteps // config.num_steps // config.num_envs
    print(f"finetune_updates: {config.finetune_updates}")
    config.minibatch_size = (config.num_actors * config.num_steps) // config.num_minibatches

    def linear_schedule(count):
        '''
        Linearly decays the learning rate depending on the number of minibatches and number of epochs
        returns the learning rate
        '''
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    # Initialize Packnet architecture
    rng = jax.random.PRNGKey(config.seed)

    # Use decoupled architecture for Packnet
    actor = Actor(
        action_dim=temp_env.action_space().n,
        activation=config.activation,
        num_tasks=config.seq_length,
        use_multihead=config.use_multihead,
        use_task_id=config.use_task_id
    )
    critic = Critic(
        activation=config.activation,
        num_tasks=config.seq_length,
        use_multihead=config.use_multihead,
        use_task_id=config.use_task_id
    )

    rng, actor_rng, critic_rng = jax.random.split(rng, 3)
    init_x = jnp.zeros(temp_env.observation_space().shape).flatten()
    init_x = jnp.expand_dims(init_x, axis=0)  # Add batch dimension
    actor_params = actor.init(actor_rng, init_x, env_idx=0)
    critic_params = critic.init(critic_rng, init_x, env_idx=0)

    # Initialize the optimizer
    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
    )

    # jit the apply function
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # calculate sparsity
    sparsity = cl.compute_sparsity(actor_params["params"])
    print(f"Sparsity: {sparsity}")

    # Initialize the Packnet state
    packnet_state = PacknetState(
        masks=cl.init_mask_tree(actor_params["params"]),
        current_task=0,
        train_mode=True
    )

    # Initialize the training state      
    actor_train_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=actor_tx
    )
    critic_train_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=critic_tx
    )

    train_states = (actor_train_state, critic_train_state)

    # def get_shape(x):
    #     return x.shape if hasattr(x, "shape") else type(x)

    # # This returns a nested structure with each array replaced by its shape.
    # shapes = jax.tree_util.tree_map(get_shape, train_state.params)
    # print(shapes)

    @partial(jax.jit, static_argnums=(3))
    def train_on_environment(rng, train_states, packnet_state, env, env_counter):
        '''
        Trains the network using IPPO
        @param rng: random number generator 
        returns the runner state and the metrics
        '''
        print("Training on environment")
        actor_train_state, critic_train_state = train_states

        # reset the learning rate and the optimizer
        actor_tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
        )
        new_actor_optimizer = actor_tx.init(actor_train_state.params)
        new_critic_optimizer = critic_tx.init(critic_train_state.params)
        actor_train_state = actor_train_state.replace(tx=actor_tx, opt_state=new_actor_optimizer)
        critic_train_state = critic_train_state.replace(tx=critic_tx, opt_state=new_critic_optimizer)

        # Initialize and reset the environment 
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.,
            end_value=0.,
            transition_steps=config.reward_shaping_horizon
        )

        # TRAIN 
        def _update_step(runner_state, unused):
            '''
            perform a single update step in the training loop
            @param runner_state: the carry state that contains all important training information
            returns the updated runner state and the metrics 
            '''

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                '''
                selects an action based on the policy, calculates the log probability of the action, 
                and performs the selected action in the environment
                @param runner_state: the current state of the runner
                returns the updated runner state and the transition
                '''
                # Unpack the runner state
                train_states, env_state, packnet_state, last_obs, update_step, grads, rng = runner_state
                actor_train_state, critic_train_state = train_states
                # split the random number generator for action selection
                rng, _rng = jax.random.split(rng)

                # prepare the observations for the network
                obs_batch = batchify(last_obs, env.agents, config.num_actors)
                # print("obs_shape", obs_batch.shape)

                # apply the policy network to the observations to get the suggested actions and their values
                pi = actor.apply(actor_train_state.params, obs_batch, env_idx=packnet_state.current_task)
                value = critic.apply(critic_train_state.params, obs_batch, env_idx=packnet_state.current_task)

                # sample the actions from the policy distribution 
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

                # REWARD SHAPING IN NEW VERSION
                # add the reward of one of the agents to the info dictionary
                info["reward"] = reward["agent_0"]

                current_timestep = update_step * config.num_steps * config.num_envs

                # add the shaped reward to the normal reward 
                reward = jax.tree_util.tree_map(lambda x, y:
                                                x + y * rew_shaping_anneal(current_timestep),
                                                reward,
                                                info["shaped_reward"]
                                                )

                transition = Transition(
                    batchify(done, env.agents, config.num_actors).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config.num_actors).squeeze(),
                    log_prob,
                    obs_batch
                )

                runner_state = (train_states, env_state, packnet_state, obsv, update_step, grads, rng)
                return runner_state, (transition, info)

            # Apply the _env_step function a series of times, while keeping track of the runner state
            runner_state, (traj_batch, info) = jax.lax.scan(
                f=_env_step,
                init=runner_state,
                xs=None,
                length=config.num_steps
            )

            # unpack the runner state that is returned after the scan function
            train_states, env_state, packnet_state, last_obs, update_step, grads, rng = runner_state
            actor_train_state, critic_train_state = train_states
            # create a batch of the observations that is compatible with the network
            last_obs_batch = batchify(last_obs, env.agents, config.num_actors)

            # apply the network to the batch of observations to get the value of the last state
            last_val = critic.apply(critic_train_state.params, last_obs_batch, env_idx=packnet_state.current_task)

            # @profile
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

                # iteratively apply the _get_advantages function to calculate the advantage for each step in the trajectory batch
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
            # @profile
            def _update_epoch(update_state, unused):
                '''
                performs a single update epoch in the training loop
                @param update_state: the current state of the update
                returns the updated update_state and the total loss
                '''

                def _update_minbatch(train_states, batch_info):
                    '''
                    performs a single update minibatch in the training loop
                    @param train_state: the current state of the training
                    @param batch_info: the information of the batch
                    returns the updated train_state and the total loss
                    '''
                    # unpack the parameters
                    actor_train_state, critic_train_state = train_states
                    traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, traj_batch, gae):
                        '''
                        calculates the loss of the actor network
                        @param actor_params: the parameters of the actor network
                        @param traj_batch: the trajectory batch
                        @param gae: the generalized advantage estimate
                        returns the actor loss
                        '''
                        # Rerun the network
                        pi = actor.apply(actor_params, traj_batch.obs, env_idx=packnet_state.current_task)

                        # Calculate the log probability 
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)

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

                        loss_actor = -jnp.minimum(loss_actor_clipped, loss_actor_unclipped)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config.clip_eps)

                        actor_loss = (
                                loss_actor
                                - config.ent_coef * entropy
                        )
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, traj_batch, targets):
                        '''
                        calculates the loss of the critic network
                        @param critic_params: the parameters of the critic network
                        @param traj_batch: the trajectory batch
                        @param targets: the targets
                        returns the critic loss
                        '''
                        # Rerun the network
                        value = critic.apply(critic_params, traj_batch.obs, env_idx=packnet_state.current_task)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config.vf_coef * value_loss
                        return critic_loss, (value_loss)

                    # returns a function with the same parameters as loss_fn that calculates the gradient of the loss function
                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)

                    actor_loss, actor_grads = actor_grad_fn(actor_train_state.params, traj_batch, advantages)
                    critic_loss, critic_grads = critic_grad_fn(critic_train_state.params, traj_batch, targets)

                    # Mask gradients for frozen weights BEFORE optimizer step (proper PackNet)
                    actor_grads = cl.mask_gradients(packnet_state, actor_grads)
                    # Note: We don't mask critic gradients as PackNet typically only applies to actor

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_information = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                        "actor_grads": actor_grads,
                        "critic_grads": critic_grads,
                    }

                    return (actor_train_state, critic_train_state), loss_information

                # unpack the update_state (because of the scan function)
                train_states, packnet_state, traj_batch, advantages, targets, rng = update_state

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

                train_states, loss_information = jax.lax.scan(
                    f=_update_minbatch,
                    init=train_states,
                    xs=minibatches
                )

                update_state = (train_states, packnet_state, traj_batch, advantages, targets, rng)
                return update_state, loss_information

            # create a tuple to be passed into the jax.lax.scan function
            update_state = (train_states, packnet_state, traj_batch, advantages, targets, rng)

            update_state, loss_info = jax.lax.scan(
                f=_update_epoch,
                init=update_state,
                xs=None,
                length=config.update_epochs
            )

            # unpack update_state
            train_states, packnet_state, traj_batch, advantages, targets, rng = update_state

            # set the metric to be the information of the last update epoch
            metric = info

            # calculate the current timestep
            current_timestep = update_step * config.num_steps * config.num_envs
            update_step = update_step + 1

            def evaluate_and_log(rng, update_step, train_states):
                rng, eval_rng = jax.random.split(rng)
                # Unpack the train states
                actor_train_state, critic_train_state = train_states

                actor_train_state_eval = jax.tree_util.tree_map(lambda x: x.copy(), actor_train_state)
                grads_eval = jax.tree_util.tree_map(lambda x: x.copy(), loss_info["actor_grads"])

                def log_metrics(metric, update_step):
                    # average the metric
                    metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
                    sparsity_actor = cl.compute_sparsity(actor_train_state_eval.params["params"])
                    sparsity_grads = cl.compute_sparsity(grads_eval["params"])
                    # add the sparsity and mask compliance to the metric dictionary
                    metric["PackNet/sparsity_actor"] = sparsity_actor
                    metric["PackNet/sparsity_grads"] = sparsity_grads
                    metric["PackNet/current_task"] = packnet_state.current_task
                    metric["PackNet/train_mode"] = packnet_state.train_mode

                    prev_mask = cl.combine_masks(packnet_state.masks, packnet_state.current_task)  # frozen weights
                    cur_mask = cl.get_mask(packnet_state.masks, packnet_state.current_task)  # active weights

                    # Flatten the mask trees and compute mean
                    prev_mask_flat, _ = jax.tree_util.tree_flatten(prev_mask)
                    cur_mask_flat, _ = jax.tree_util.tree_flatten(cur_mask)

                    # Concatenate all mask arrays and compute mean
                    prev_mask_concat = jnp.concatenate([mask.flatten() for mask in prev_mask_flat])
                    cur_mask_concat = jnp.concatenate([mask.flatten() for mask in cur_mask_flat])

                    metric["PackNet/frozen_frac"] = jnp.mean(prev_mask_concat)
                    metric["PackNet/active_frac"] = jnp.mean(cur_mask_concat)

                    # add the general metrics to the metric dictionary
                    metric["General/update_step"] = update_step
                    metric["General/env_step"] = update_step * config.num_steps * config.num_envs
                    if config.anneal_lr:
                        metric["General/learning_rate"] = linear_schedule(
                            update_step * config.num_minibatches * config.update_epochs)
                    else:
                        metric["General/learning_rate"] = config.lr

                    # Losses section
                    metric["Losses/total_loss"] = loss_info["total_loss"].mean()
                    metric["Losses/value_loss"] = loss_info["value_loss"].mean()
                    metric["Losses/actor_loss"] = loss_info["actor_loss"].mean()
                    metric["Losses/entropy"] = loss_info["entropy"].mean()

                    # Rewards section
                    if "shaped_reward" in metric and isinstance(metric["shaped_reward"], dict):
                        metric["General/shaped_reward_agent0"] = metric["shaped_reward"]["agent_0"]
                        metric["General/shaped_reward_agent1"] = metric["shaped_reward"]["agent_1"]
                        metric.pop("shaped_reward", None)
                        metric["General/shaped_reward_annealed_agent0"] = metric[
                                                                              "General/shaped_reward_agent0"] * rew_shaping_anneal(
                            current_timestep)
                        metric["General/shaped_reward_annealed_agent1"] = metric[
                                                                              "General/shaped_reward_agent1"] * rew_shaping_anneal(
                            current_timestep)

                    # Advantages and Targets section
                    metric["Advantage_Targets/advantages"] = advantages.mean()
                    metric["Advantage_Targets/targets"] = targets.mean()

                    # Evaluation section
                    if config.evaluation:
                        # For Packnet, we need to pass both actor and critic train states
                        if config.cl_method.lower() == "packnet":
                            # Create a tuple with both actor and critic train states for evaluation
                            critic_train_state_eval = jax.tree_util.tree_map(lambda x: x.copy(), critic_train_state)
                            eval_train_states = (actor_train_state_eval, critic_train_state_eval)
                            avg_rewards, avg_soups = evaluate_model(eval_train_states, eval_rng)
                        else:
                            avg_rewards, avg_soups = evaluate_model(actor_train_state_eval, eval_rng)

                        # Apply episode fraction scaling to avg_soups like in IPPO_CL.py
                        episode_frac = config.eval_num_steps / env.max_steps
                        avg_soups = [soup * episode_frac for soup in avg_soups]

                        # Use basic logging since layout_names and max_soup_dict cannot be passed to JIT function
                        for i, (reward, soup) in enumerate(zip(avg_rewards, avg_soups)):
                            metric[f"Evaluation/env_{i}_soup_scaled"] = soup
                            metric[f"Evaluation/env_{i}_soup"] = soup
                    # Extract parameters 
                    actor_params = jax.tree_util.tree_map(lambda x: x, actor_train_state_eval.params["params"])
                    actor_grads = jax.tree_util.tree_map(lambda x: x, grads_eval["params"])

                    def callback(args):
                        metric, update_step, env_counter, actor_params, actor_grads = args
                        real_step = (int(env_counter) - 1) * config.num_updates + int(update_step)

                        # Convert all metric values to Python scalars for TensorboardX
                        def convert_to_scalar(value):
                            if isinstance(value, dict):
                                # Skip dictionaries - they shouldn't be logged as scalars
                                return None
                            elif hasattr(value, 'item'):
                                # JAX arrays and numpy arrays
                                return float(value.item())
                            elif hasattr(value, '__float__'):
                                # Other numeric types
                                return float(value)
                            else:
                                # Already a Python scalar
                                return value

                        for key, value in metric.items():
                            scalar_value = convert_to_scalar(value)
                            if scalar_value is not None:
                                writer.add_scalar(key, scalar_value, real_step)
                        for layer, layer_dict in actor_params.items():
                            for layer_name, param_array in layer_dict.items():
                                writer.add_histogram(
                                    tag=f"weights/{layer}/{layer_name}",
                                    values=jnp.array(param_array),
                                    global_step=real_step,
                                    bins=100)
                                writer.add_histogram(
                                    tag=f"grads/{layer}/{layer_name}",
                                    values=jnp.array(actor_grads[layer][layer_name]),
                                    global_step=real_step,
                                    bins=100)

                    jax.experimental.io_callback(callback, None,
                                                 (metric, update_step, env_counter, actor_params, actor_grads))
                    return None

                def do_not_log(metric, update_step):
                    return None

                jax.lax.cond((update_step % config.log_interval) == 0, log_metrics, do_not_log, metric, update_step)

            # Evaluate the model and log the metrics
            evaluate_and_log(rng=rng, update_step=update_step, train_states=train_states)

            # unpack the loss information
            actor_grads = loss_info["actor_grads"]
            actor_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=(0, 1)), actor_grads)
            sparsity = cl.compute_sparsity(actor_grads["params"])

            rng = update_state[-1]
            runner_state = (train_states, env_state, packnet_state, last_obs, update_step, actor_grads, rng)

            return runner_state, metric

        rng, train_rng = jax.random.split(rng)

        # initialize a carrier that keeps track of the states and observations of the agents
        actor_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), actor_train_state.params)
        runner_state = (train_states, env_state, packnet_state, obsv, 0, actor_grads, train_rng)

        # apply the _update_step function a series of times, while keeping track of the state 
        runner_state, metric = jax.lax.scan(
            f=_update_step,
            init=runner_state,
            xs=None,
            length=config.num_updates
        )

        # unpack the runner state
        train_states, env_state, packnet_state, last_obs, update_step, actor_grads, rng = runner_state
        actor_train_state, critic_train_state = train_states

        # # Prune the model and update the parameters
        new_actor_params, packnet_state = cl.on_train_end(actor_train_state.params["params"], packnet_state)

        # check the sparsity of the new params
        sparsity = cl.compute_sparsity(new_actor_params["params"])
        jax.debug.print(
            "Sparsity after pruning: {sparsity}", sparsity=sparsity)

        # update the actor train state with the new parameters
        actor_train_state = actor_train_state.replace(params=new_actor_params)
        train_states = (actor_train_state, critic_train_state)

        rng, finetune_rng = jax.random.split(rng)

        # Create a new runner state for the finetuning phase
        runner_state = (train_states, env_state, packnet_state, last_obs, update_step, actor_grads, finetune_rng)

        runner_state, metric = jax.lax.scan(
            f=_update_step,
            init=runner_state,
            xs=None,
            length=config.finetune_updates
        )

        # check the sparsity after finetuning
        actor_train_state = runner_state[0][0]
        sparsity = cl.compute_sparsity(actor_train_state.params["params"])
        jax.debug.print(
            "Sparsity after finetuning: {sparsity}", sparsity=sparsity)

        # handle the end of the finetune phase 
        packnet_state = cl.on_finetune_end(packnet_state)

        # add the packnet_state to the new runner state
        runner_state = (train_states, env_state, packnet_state, last_obs, update_step, actor_grads, finetune_rng)

        return runner_state, metric

    def loop_over_envs(rng, train_states, envs, cl_state, layout_names):
        '''
        Loops over the environments and trains the network using Packnet
        @param rng: random number generator
        @param train_states: the current state of the training (tuple of actor and critic)
        @param envs: the environments
        @param cl_state: the Packnet state
        @param layout_names: names of the layouts
        returns the runner state and the metrics
        '''
        actor_train_state, critic_train_state = train_states

        # split the random number generator for training on the environments
        rng, *env_rngs = jax.random.split(rng, len(envs) + 1)

        visualizer = OvercookedVisualizer()

        # counter for the environment 
        env_counter = 1

        if config.evaluation:
            evaluation_matrix = jnp.zeros(((len(envs) + 1), len(envs)))
            rng, eval_rng = jax.random.split(rng)
            avg_rewards, avg_soups = evaluate_model(train_states, eval_rng)
            evaluation_matrix = evaluation_matrix.at[0, :].set(avg_rewards)

        for i, (env_rng, env) in enumerate(zip(env_rngs, envs)):
            # Call the train_on_environment function for Packnet
            runner_state, metrics = train_on_environment(env_rng, train_states, cl_state, env, env_counter)
            # unpack the runner state
            train_states, env_state, cl_state, last_obs, update_step, grads, rng = runner_state
            current_train_state = train_states[0]  # actor_train_state

            # Generate & log a GIF after finishing task i
            env_name = layout_names[i] if i < len(layout_names) else f"env_{i}"
            states = record_gif_of_episode(config, current_train_state, env, actor, env_idx=i,
                                           max_steps=config.gif_len)
            visualizer.animate(states, agent_view_size=5, task_idx=i, task_name=env_name, exp_dir=exp_dir)

            if config.evaluation:
                # Evaluate at the end of training to get the average performance of the task right after training
                avg_rewards, avg_soups = evaluate_model(train_states, rng)
                evaluation_matrix = evaluation_matrix.at[env_counter, :].set(avg_rewards)

            # save the model
            path = f"checkpoints/overcooked/{run_name}/model_env_{env_counter}"
            save_params(path, train_states, env_kwargs=env.layout, layout_name=env_name, config=config)

            # update the environment counter
            env_counter += 1

        return runner_state

    def save_params(path, train_state, env_kwargs=None, layout_name=None, config=None):
        '''
        Saves the parameters of the Packnet network along with environment configuration
        @param path: the path to save the parameters
        @param train_state: the current state of the training (tuple of actor and critic)
        @param env_kwargs: the environment kwargs used to create the environment
        @param layout_name: the name of the layout
        @param config: the configuration used for training
        returns None
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Packnet case: separate actor and critic
        actor_train_state, critic_train_state = train_state
        params_to_save = {
            "actor_params": actor_train_state.params,
            "critic_params": critic_train_state.params
        }

        # Save model parameters
        with open(path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(params_to_save)
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
                    "wall_density": config.wall_density,
                    "cl_method": config.cl_method,
                    "network_architecture": getattr(config, 'network_architecture', 'mlp')
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

    # Initialize Packnet state
    cl_state = packnet_state

    # Apply the loop_over_envs function to the environments
    runner_state = loop_over_envs(train_rng, train_states, envs, cl_state, layout_names)


if __name__ == "__main__":
    print("Running main...")
    main()
