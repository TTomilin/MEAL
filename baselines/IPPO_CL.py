import os
from pathlib import Path

from cl_methods.AGEM import AGEM, init_agem_memory, sample_memory, compute_memory_gradient, agem_project, \
    update_agem_memory
from cl_methods.FT import FT
from cl_methods.L2 import L2
from cl_methods.MAS import MAS

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
from typing import Sequence, Any, Optional, List

import flax
import optax
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState

from jax_marl.registration import make
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment.overcooked_upper_bound import estimate_max_soup
from architectures.shared_mlp import ActorCritic as MLPActorCritic
from architectures.cnn import ActorCritic as CNNActorCritic
from baselines.utils import *
from cl_methods.EWC import EWC

import wandb
from functools import partial
from dataclasses import dataclass, field
import tyro
from tensorboardX import SummaryWriter


@dataclass
class Config:
    reg_coef: float = 1e7
    lr: float = 3e-4
    num_envs: int = 16
    num_steps: int = 128
    total_timesteps: float = 1e7
    update_epochs: int = 8
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.957
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # reward shaping
    reward_shaping: bool = True
    reward_shaping_horizon: float = 2.5e6

    explore_fraction: float = 0.0
    activation: str = "relu"
    env_name: str = "overcooked"
    alg_name: str = "ippo"
    cl_method: str = None
    use_cnn: bool = False
    use_task_id: bool = False
    use_multihead: bool = False
    shared_backbone: bool = False
    normalize_importance: bool = False
    regularize_critic: bool = False
    regularize_heads: bool = True
    big_network: bool = False
    use_layer_norm: bool = True

    # Reg method specific
    importance_episodes: int = 5
    importance_steps: int = 500

    # EWC specific
    ewc_mode: str = "online"  # "online", "last" or "multi"
    ewc_decay: float = 0.9  # Only for online EWC

    # AGEM specific
    agem_memory_size: int = 1000
    agem_sample_size: int = 128

    # Environment
    seq_length: int = 10
    repeat_sequence: int = 1
    strategy: str = "generate"
    layouts: Optional[Sequence[str]] = field(default_factory=lambda: [])
    env_kwargs: Optional[Sequence[dict]] = None
    evaluation: bool = True
    eval_forward_transfer: bool = False
    record_gif: bool = True
    log_interval: int = 75
    eval_num_steps: int = 1000
    eval_num_episodes: int = 5
    gif_len: int = 300

    # ─── random‐layout generator knobs ───────────────────────────────────────
    height_min: int = 6  # minimum layout height
    height_max: int = 8  # maximum layout height
    width_min: int = 6  # minimum layout width
    width_max: int = 8  # maximum layout width
    wall_density: float = 0.25  # fraction of internal tiles that are untraversable

    anneal_lr: bool = False
    seed: int = 30
    num_seeds: int = 1

    # Wandb settings
    wandb_mode: str = "online"
    entity: Optional[str] = ""
    project: str = "COOX"
    tags: List[str] = field(default_factory=list)

    # to be computed during runtime
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

    config = tyro.cli(Config)

    method_map = dict(ewc=EWC(mode=config.ewc_mode, decay=config.ewc_decay),
                      mas=MAS(),
                      l2=L2(),
                      ft=FT(),
                      agem=AGEM(memory_size=config.agem_memory_size, sample_size=config.agem_sample_size))

    cl = method_map[config.cl_method.lower()]

    # generate a sequence of tasks
    config.env_kwargs, layout_names = generate_sequence(
        sequence_length=config.seq_length,
        strategy=config.strategy,
        layout_names=config.layouts,
        seed=config.seed,
        height_rng=(config.height_min, config.height_max),
        width_rng=(config.width_min, config.width_max),
        wall_density=config.wall_density,
    )

    # repeat the base sequence `repeat_sequence` times
    config.env_kwargs = config.env_kwargs * config.repeat_sequence
    layout_names = layout_names * config.repeat_sequence

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    network = "cnn" if config.use_cnn else "mlp"
    run_name = f'{config.alg_name}_{config.cl_method}_{network}_seq{config.seq_length}_{config.strategy}_seed_{config.seed}_{timestamp}'
    exp_dir = os.path.join("runs", run_name)

    # Initialize WandB
    load_dotenv()
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

    # pad the observation space
    def pad_observation_space():
        '''
        Pads the observation space of the environment to be compatible with the network
        @param envs: the environment
        returns the padded observation space
        '''
        envs = []
        for env_args in config.env_kwargs:
            # Create the environment
            env = make(config.env_name, **env_args)
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
                adjusted_indices = []

                for idx in indices:
                    # Compute the row and column of the index
                    row = idx // width
                    col = idx % width

                    # Shift the row and column by the padding
                    new_row = row + top
                    new_col = col + left

                    # Compute the new index
                    new_idx = new_row * (width + left + right) + new_col
                    adjusted_indices.append(new_idx)

                return jnp.array(adjusted_indices)

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

        return padded_envs

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
                    pi, value = network_apply(params, obs, env_idx=eval_idx)
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

        envs = pad_observation_space()

        for eval_idx, env in enumerate(envs):
            env = make(config.env_name, layout=env)  # Create the environment

            # Run k episodes
            all_rewards, all_soups = jax.vmap(lambda k: run_episode_while(env, k, config.eval_num_steps))(
                jax.random.split(key, config.eval_num_episodes)
            )

            avg_reward = jnp.mean(all_rewards)
            avg_soups = jnp.sum(all_soups)
            all_avg_rewards.append(avg_reward)
            all_avg_soups.append(avg_soups)

        return all_avg_rewards, all_avg_soups

    padded_envs = pad_observation_space()

    envs = []
    env_names = []
    max_soup_dict = {}
    for i, env_layout in enumerate(padded_envs):
        env = make(config.env_name, layout=env_layout, layout_name=layout_names[i], task_id=i)
        env = LogWrapper(env, replace_info=False)
        env_name = env.layout_name
        envs.append(env)
        env_names.append(env_name)
        max_soup_dict[env_name] = estimate_max_soup(env_layout, env.max_steps, n_agents=env.num_agents)

    # set extra config parameters based on the environment
    temp_env = envs[0]
    config.num_actors = temp_env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_steps // config.num_envs
    config.minibatch_size = (config.num_actors * config.num_steps) // config.num_minibatches

    def linear_schedule(count):
        '''
        Linearly decays the learning rate depending on the number of minibatches and number of epochs
        returns the learning rate
        '''
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    ac_cls = CNNActorCritic if config.use_cnn else MLPActorCritic

    network = ac_cls(temp_env.action_space().n, config.activation, config.seq_length, config.use_multihead,
                     config.shared_backbone, config.big_network, config.use_task_id, config.regularize_heads,
                     config.use_layer_norm)

    obs_dim = temp_env.observation_space().shape
    if not config.use_cnn:
        obs_dim = np.prod(obs_dim)

    # Initialize the network
    rng = jax.random.PRNGKey(config.seed)
    rng, network_rng = jax.random.split(rng)
    init_x = jnp.zeros((1, *obs_dim)) if config.use_cnn else jnp.zeros((1, obs_dim,))
    network_params = network.init(network_rng, init_x)

    # Initialize the optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
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

        # Initialize AGEM memory if using AGEM and this is the first environment
        agem_mem = None
        if config.cl_method.lower() == "agem" and env_idx > 0:
            # For AGEM, we need to initialize the memory buffer if it doesn't exist
            if cl_state is None:
                # Get observation dimension
                obs_dim = env.observation_space().shape
                if not config.use_cnn:
                    obs_dim = (np.prod(obs_dim),)
                else:
                    obs_dim = obs_dim

                # Initialize memory buffer
                agem_mem = init_agem_memory(config.agem_memory_size, obs_dim[0])

        # How many steps to explore the environment with random actions
        exploration_steps = int(config.explore_fraction * config.total_timesteps)

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

        reward_shaping_horizon = config.total_timesteps / 2
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
                train_state, env_state, last_obs, update_step, steps_for_env, rng = runner_state

                # split the random number generator for action selection
                rng, _rng = jax.random.split(rng)

                # prepare the observations for the network
                obs_batch = batchify(last_obs, env.agents, config.num_actors,
                                     not config.use_cnn)  # (num_actors, obs_dim)
                # print("obs_shape", obs_batch.shape)

                # apply the policy network to the observations to get the suggested actions and their values
                pi, value = network.apply(train_state.params, obs_batch, env_idx=env_idx)

                # Decide whether to explore randomly or use the policy
                policy_action = pi.sample(seed=_rng)
                random_action = jax.random.randint(_rng, (config.num_actors,), 0, env.action_space().n)
                explore = (steps_for_env < exploration_steps)

                # Expand bool to match the shape of action arrays:
                mask = jnp.repeat(jnp.array([explore]), config.num_actors)
                action = jnp.where(mask, random_action, policy_action)

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
                    batchify(done, env.agents, config.num_actors, not config.use_cnn).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config.num_actors).squeeze(),
                    log_prob,
                    obs_batch
                )

                # Increment steps_for_env by the number of parallel envs
                steps_for_env = steps_for_env + config.num_envs
                info["explore"] = jnp.ones((config.num_envs,), dtype=jnp.float32) * jnp.float32(explore)

                runner_state = (train_state, env_state, obsv, update_step, steps_for_env, rng)
                return runner_state, (transition, info)

            # Apply the _env_step function a series of times, while keeping track of the runner state
            runner_state, (traj_batch, info) = jax.lax.scan(
                f=_env_step,
                init=runner_state,
                xs=None,
                length=config.num_steps
            )

            # unpack the runner state that is returned after the scan function
            train_state, env_state, last_obs, update_step, steps_for_env, rng = runner_state

            # create a batch of the observations that is compatible with the network
            last_obs_batch = batchify(last_obs, env.agents, config.num_actors, not config.use_cnn)

            # apply the network to the batch of observations to get the value of the last state
            _, last_val = network.apply(train_state.params, last_obs_batch, env_idx=env_idx)

            # this returns the value network for the last observation batch

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

                def _update_minbatch(train_state, batch_info):
                    '''
                    performs a single update minibatch in the training loop
                    @param train_state: the current state of the training
                    @param batch_info: the information of the batch
                    returns the updated train_state and the total loss
                    '''
                    # unpack the batch information
                    traj_batch, advantages, targets = batch_info

                    # @profile
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
                        pi, value = network.apply(params, traj_batch.obs, env_idx=env_idx)
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

                    # For AGEM, we need to project the gradients if this is not the first environment
                    agem_stats = {}
                    if config.cl_method.lower() == "agem" and env_idx > 0 and agem_mem is not None:
                        # Sample from memory
                        rng_1, sample_rng = jax.random.split(rng)
                        # Pick a random sample from AGEM memory
                        mem_obs, mem_actions, mem_log_probs, mem_advs, mem_targets, mem_values = sample_memory(
                            agem_mem, config.agem_sample_size, sample_rng
                        )

                        # Compute memory gradient
                        grads_mem, grads_stats = compute_memory_gradient(
                            network, train_state.params,
                            config.clip_eps, config.vf_coef, config.ent_coef,
                            mem_obs, mem_actions, mem_advs, mem_log_probs,
                            mem_targets, mem_values,
                            env_idx=env_idx
                        )

                        # Project new grads
                        grads, proj_stats = agem_project(grads, grads_mem)

                        # Combine stats for logging
                        agem_stats = {**grads_stats, **proj_stats}

                    loss_information = total_loss, grads, agem_stats

                    # apply the gradients to the network
                    train_state = train_state.apply_gradients(grads=grads)

                    # Of course we also need to add the network to the carry here
                    return train_state, loss_information

                train_state, traj_batch, advantages, targets, steps_for_env, rng = update_state

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

                train_state, loss_information = jax.lax.scan(
                    f=_update_minbatch,
                    init=train_state,
                    xs=minibatches
                )

                # Handle different return formats based on CL method
                total_loss, grads, agem_stats = loss_information
                # Create a dictionary to store all loss information
                loss_dict = {
                    "total_loss": total_loss
                }
                if config.cl_method.lower() == "agem" and env_idx > 0:
                    loss_dict["agem_stats"] = agem_stats

                avg_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)
                update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng)
                return update_state, loss_dict

            # create a tuple to be passed into the jax.lax.scan function
            update_state = (train_state, traj_batch, advantages, targets, steps_for_env, rng)

            update_state, loss_info = jax.lax.scan(
                f=_update_epoch,
                init=update_state,
                xs=None,
                length=config.update_epochs
            )

            # unpack update_state
            train_state, traj_batch, advantages, targets, steps_for_env, rng = update_state
            current_timestep = update_step * config.num_steps * config.num_envs
            metrics = jax.tree_util.tree_map(lambda x: x.mean(), info)

            # General section
            # Update the step counter
            update_step += 1
            mean_explore = jnp.mean(info["explore"])

            metrics["General/env_index"] = env_idx
            metrics["General/explore"] = mean_explore
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
                        metrics[k] = jnp.mean(v, axis=0)

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

            rng = update_state[-1]
            runner_state = (train_state, env_state, last_obs, update_step, steps_for_env, rng)

            return runner_state, metrics

        rng, train_rng = jax.random.split(rng)

        # initialize a carrier that keeps track of the states and observations of the agents
        runner_state = (train_state, env_state, obsv, 0, 0, train_rng)

        # apply the _update_step function a series of times, while keeping track of the state
        runner_state, metrics = jax.lax.scan(
            f=_update_step,
            init=runner_state,
            xs=None,
            length=config.num_updates
        )

        # Update AGEM memory if using AGEM and this is not the first environment
        if config.cl_method.lower() == "agem" and env_idx > 0 and agem_mem is not None:
            # Sample a batch of data from the current environment
            # We'll use the last batch from the training loop
            train_state, env_state, last_obs, update_step, steps_for_env, rng = runner_state

            # Prepare the data for updating the memory
            obs_batch = batchify(last_obs, env.agents, config.num_actors, not config.use_cnn)

            # Get values and log_probs for the current observations
            pi, value = network.apply(train_state.params, obs_batch, env_idx=env_idx)

            # Sample actions from the policy
            rng, _rng = jax.random.split(rng)
            actions = pi.sample(seed=_rng)
            log_probs = pi.log_prob(actions)

            # Use dummy advantages and targets (these will be replaced in the next environment)
            advantages = jnp.zeros_like(log_probs)
            targets = jnp.zeros_like(log_probs)

            # Update the memory buffer
            agem_mem = update_agem_memory(
                agem_mem,
                obs_batch, actions, log_probs,
                advantages, targets, value,
                config.agem_memory_size
            )

            # Store the updated memory in cl_state
            cl_state = agem_mem

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

        visualizer = OvercookedVisualizer(num_agents=temp_env.num_agents)

        evaluation_matrix = None
        if config.eval_forward_transfer:
            evaluation_matrix = jnp.zeros(((len(envs) + 1), len(envs)))
            rng, eval_rng = jax.random.split(rng)
            evaluations = evaluate_model(train_state, eval_rng)
            evaluation_matrix = evaluation_matrix.at[0, :].set(evaluations)

        for i, (rng, env) in enumerate(zip(env_rngs, envs)):
            # --- Train on environment i using the *current* ewc_state ---
            runner_state, metrics = train_on_environment(rng, train_state, env, cl_state, i)
            train_state = runner_state[0]

            importance = cl.compute_importance(train_state.params, env, network, i, rng, config.use_cnn,
                                               config.importance_episodes, config.importance_steps,
                                               config.normalize_importance)

            cl_state = cl.update_state(cl_state, train_state.params, importance)

            if config.record_gif:
                # Generate & log a GIF after finishing task i
                env_name = f"{i}__{env.layout_name}"
                states = record_gif_of_episode(config, train_state, env, network, env_idx=i, max_steps=config.gif_len)
                visualizer.animate(states, agent_view_size=5, task_idx=i, task_name=env_name, exp_dir=exp_dir)

            if config.eval_forward_transfer:
                # Evaluate at the end of training to get the average performance of the task right after training
                evaluations = evaluate_model(train_state, rng)
                evaluation_matrix = evaluation_matrix.at[i, :].set(evaluations)

            # save the model
            repo_root = Path(__file__).resolve().parent.parent
            path = f"{repo_root}/checkpoints/overcooked/{config.cl_method}/{run_name}/model_env_{i + 1}"
            save_params(path, train_state)

            if config.eval_forward_transfer:
                # calculate the forward transfer and backward transfer
                show_heatmap_bwt(evaluation_matrix, run_name)
                show_heatmap_fwt(evaluation_matrix, run_name)

    def save_params(path, train_state):
        '''
        Saves the parameters of the network
        @param path: the path to save the parameters
        @param train_state: the current state of the training
        returns None
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    {"params": train_state.params}
                )
            )
        print('model saved to', path)

    # Run the model
    rng, train_rng = jax.random.split(rng)
    cl_state = cl.init_state(train_state.params, config.regularize_critic, config.regularize_heads)

    # apply the loop_over_envs function to the environments
    loop_over_envs(train_rng, train_state, cl_state, envs)


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

        rng, key_step = jax.random.split(rng)
        next_obs, next_state, reward, done_info, info = env.step(key_step, state, actions)
        done = done_info["__all__"]

        obs, state = next_obs, next_state
        step_count += 1
        states.append(state)

    return states


if __name__ == "__main__":
    print("Running main...")
    main()
