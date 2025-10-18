import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Sequence, Any, Optional, List, Literal

import flax
import numpy as np
import optax
import tyro
import wandb
from flax.core.frozen_dict import unfreeze
from flax.training.train_state import TrainState
from jax._src.flatten_util import ravel_pytree

from experiments.continual.agem import AGEM, init_agem_memory, sample_memory, compute_memory_gradient, agem_project, \
    update_agem_memory
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from experiments.model.cnn import ActorCritic as CNNActorCritic
from experiments.model.mlp import ActorCritic as MLPActorCritic
from experiments.utils import *
from meal import make_sequence
from meal.env.utils.max_soup_calculator import calculate_max_soup
from meal.wrappers.logging import LogWrapper


@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING / PPO PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    alg_name: Literal["ippo","mappo"] = "ippo"
    lr: float = 1e-3
    anneal_lr: bool = False
    num_envs: int = 2048
    num_steps: int = 400
    steps_per_task: float = 1e8
    update_epochs: int = 8
    num_minibatches: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
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
    env_name: str = "overcooked"
    num_agents: int = 2  # number of agents in the environment
    seq_length: int = 10
    repeat_sequence: int = 1
    strategy: str = "generate"
    layouts: Optional[Sequence[str]] = field(default_factory=lambda: [])
    env_kwargs: Optional[Sequence[dict]] = None
    difficulty: Optional[str] = None
    single_task_idx: Optional[int] = None
    random_reset: bool = False
    random_agent_start: bool = True

    # Random layout generator parameters
    height_min: int = 6  # minimum layout height
    height_max: int = 7  # maximum layout height
    width_min: int = 6  # minimum layout width
    width_max: int = 7  # maximum layout width
    wall_density: float = 0.15  # fraction of internal tiles that are untraversable

    # Partial Observability parameters (default: easy difficulty)
    view_ahead: int = 1
    view_sides: int = 1
    view_behind: int = 0

    # Agent restriction parameters
    complementary_restrictions: bool = False  # One agent can't pick up onions, other can't pick up plates

    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATION PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    evaluation: bool = True
    eval_num_episodes: int = 5
    record_gif: bool = True
    gif_len: int = 250
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

    # Create environments using the improved make_sequence function
    envs = make_sequence(
        sequence_length=seq_length,
        strategy=strategy,
        env_id=cfg.env_name,
        seed=seed,
        num_agents=cfg.num_agents,
        max_steps=cfg.num_steps,
        random_reset=cfg.random_reset,
        layout_names=cfg.layouts,
        difficulty=cfg.difficulty,
        height_rng=(cfg.height_min, cfg.height_max),
        width_rng=(cfg.width_min, cfg.width_max),
        wall_density=cfg.wall_density,
        repeat_sequence=cfg.repeat_sequence,
        random_agent_start=cfg.random_agent_start
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    network = "cnn" if cfg.use_cnn else "mlp"
    run_name = f'{cfg.alg_name}_{cfg.cl_method}_{difficulty}_{cfg.num_agents}agents_{network}_seq{len(envs)}_{strategy}_seed_{seed}_{timestamp}'
    exp_dir = os.path.join("runs", run_name)

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

    @partial(jax.jit)
    def evaluate_model(train_state, key):
        '''
        Evaluates the model by running 10 episodes on all environments and returns the average reward
        @param train_state: the current state of the training
        @param config: the configuration of the training
        returns the average reward
        '''

        def run_episode_while(env, key_r):
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
                return jnp.logical_and(jnp.logical_not(state.done), state.step_count < env.max_steps)

            def body_fun(state: EvalState):
                '''
                Performs a single step in the environment
                @param state: the current state of the loop
                returns the updated state
                '''

                key, state_env, obs, _, total_reward, total_soup, step_count = state
                subkeys = jax.random.split(key, num_agents + 2)
                key, *agent_keys, key_s = subkeys

                # ***Create a batched copy for the network only.***
                # For each agent, expand dims to get shape (1, H, W, C) then flatten to (1, -1)
                batched_obs = {}
                for agent, v in obs.items():
                    v_b = jnp.expand_dims(v, axis=0)  # now (1, H, W, C)
                    if not cfg.use_cnn:
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
                    pi, value, _ = network_apply(params, obs, env_idx=eval_idx)
                    action = jnp.squeeze(pi.sample(seed=rng), axis=0)
                    return action, value

                # Get action distributions
                actions = {}
                for key_agent, agent in zip(agent_keys, agents):
                    act, _ = select_action(train_state, key_agent, batched_obs[agent])
                    actions[agent] = act

                # Environment step
                next_obs, next_state, reward, done_step, info = env.step(key_s, state_env, actions)
                done = done_step["__all__"]
                reward = sum(reward[agent] for agent in agents)  # Sum of all agent rewards
                soups_this_step = sum(info["soups"][agent] for agent in agents)
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

        # Use the already created environments
        for eval_idx, env in enumerate(envs):
            # Environments are already created with correct parameters
            pass

            # Run k episodes
            all_rewards, all_soups = jax.vmap(
                lambda k: run_episode_while(env, k)
            )(jax.random.split(key, cfg.eval_num_episodes))

            all_avg_rewards.append(jnp.mean(all_rewards))  # reward can be plain mean
            all_avg_soups.append(jnp.mean(all_soups))

        return all_avg_rewards, all_avg_soups

    # Wrap environments with LogWrapper and calculate max soup
    env_names = []
    max_soup_dict = {}
    for i, env in enumerate(envs):
        env = LogWrapper(env, replace_info=False)
        env_name = env.layout_name
        envs[i] = env  # Update the environment in the list
        env_names.append(env_name)
        max_soup_dict[env_name] = calculate_max_soup(env.layout, env.max_steps, n_agents=env.num_agents)

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

    network = ac_cls(temp_env.action_space().n, cfg.activation, len(envs), cfg.use_multihead,
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
                env_act = {k: v.flatten() for k, v in env_act.items()}

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
                if cfg.sparse_rewards:
                    # Sparse rewards: only delivery rewards (no shaped rewards)
                    # reward already contains individual delivery rewards from environment
                    pass
                elif cfg.individual_rewards:
                    # Individual rewards: delivery rewards + individual shaped rewards
                    # Environment now provides individual delivery rewards directly
                    reward = jax.tree_util.tree_map(lambda x, y:
                                                    x + y * rew_shaping_anneal(current_timestep),
                                                    reward, info["shaped_reward"])
                else:
                    # Default behavior: shared delivery rewards + individual shaped rewards
                    # Convert individual delivery rewards to shared rewards (all agents get total)
                    total_delivery_reward = sum(reward[agent] for agent in env.agents)
                    shared_delivery_rewards = {agent: total_delivery_reward for agent in env.agents}

                    reward = jax.tree_util.tree_map(lambda x, y:
                                                    x + y * rew_shaping_anneal(current_timestep),
                                                    shared_delivery_rewards, info["shaped_reward"])

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
                    if cfg.cl_method.lower() == "agem" and cl_state is not None:
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
            if cfg.anneal_lr:
                metrics["General/learning_rate"] = linear_schedule(
                    update_step * cfg.num_minibatches * cfg.update_epochs)
            else:
                metrics["General/learning_rate"] = cfg.lr

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

            # Soup Kitchen
            T, E = cfg.num_steps, cfg.num_envs
            A = env.num_agents
            max_per_episode = max_soup_dict[env_names[env_idx]]

            # soups_tea: (T, E, A)
            soups_tea = jnp.stack([info["soups"][a] for a in env.agents], axis=-1)

            # total soups per env in this window
            soups_per_env = soups_tea.sum(axis=(0, 2))  # (E,)

            # done flags per env from first agent
            done_tea = traj_batch.done.reshape(T, E, A)
            done_te = done_tea[..., 0]  # (T, E)
            episodes_per_env = done_te.sum(axis=0)  # (E,)

            # ------- TRUE per-episode average (only finished episodes) -------
            mask = episodes_per_env > 0
            true_avg_per_ep_env = jnp.where(mask, soups_per_env / jnp.maximum(episodes_per_env, 1), 0.0)
            # mean over envs that finished
            num_finished_envs = jnp.maximum(mask.sum(), 1)
            metrics["Soup/total"] = true_avg_per_ep_env.sum() / num_finished_envs

            # scaled (true) vs capacity
            metrics["Soup/scaled"] = jnp.where(max_per_episode > 0,
                                               (true_avg_per_ep_env / max_per_episode).sum() / num_finished_envs, 0.0)

            # per-agent soup
            for ai, agent in enumerate(env.agents):
                soups_te = soups_tea[:, :, ai].sum(axis=0)  # (E,)
                per_agent = jnp.where(mask, soups_te / jnp.maximum(episodes_per_env, 1), 0.0)
                metrics[f"Soup/{agent}"] = per_agent.sum() / num_finished_envs

            # ------- RATE-based estimate (if no finished episodes) -------
            # per-episode estimate: scale soups-per-window by (episode length / window steps)
            episode_len = env.max_steps
            per_ep_est_env = soups_per_env * (episode_len / T)  # (E,)

            metrics["Soup/total_unf"] = per_ep_est_env.mean()
            metrics["Soup/scaled_unf"] = jnp.where(max_per_episode > 0,
                                                   (per_ep_est_env / max_per_episode).mean(), 0.0)

            metrics.pop('soups', None)

            # Rewards section
            # Agent-agnostic reward logging
            for agent in env.agents:
                metrics[f"General/shaped_reward_{agent}"] = metrics["shaped_reward"][agent]
                metrics[f"General/shaped_reward_annealed_{agent}"] = (metrics[f"General/shaped_reward_{agent}"] *
                                                                      rew_shaping_anneal(current_timestep))

            metrics.pop('shaped_reward', None)

            # Advantages and Targets section
            metrics["Advantage_Targets/advantages"] = advantages.mean()
            metrics["Advantage_Targets/targets"] = targets.mean()

            # Dormant neuron ratio - calculate from current batch
            obs_batch = batchify(last_obs, env.agents, cfg.num_actors, not cfg.use_cnn)
            _, _, current_dormant_ratio = network.apply(train_state.params, obs_batch, env_idx=env_idx)
            metrics["Neural_Activity/dormant_ratio"] = current_dormant_ratio

            def evaluate_and_log(rng, update_step):
                rng, eval_rng = jax.random.split(rng)
                train_state_eval = jax.tree_util.tree_map(lambda x: x.copy(), train_state)

                def log_metrics(metrics, update_step):
                    if cfg.evaluation:
                        avg_rewards, avg_soups = evaluate_model(train_state_eval, eval_rng)
                        metrics = add_eval_metrics(avg_rewards, avg_soups, env_names, max_soup_dict, metrics)

                    def callback(args):
                        metrics, update_step, env_counter = args
                        real_step = (int(env_counter) - 1) * cfg.num_updates + int(update_step)
                        for key, value in metrics.items():
                            writer.add_scalar(key, value, real_step)

                    jax.experimental.io_callback(callback, None, (metrics, update_step, env_idx + 1))
                    return None

                def do_not_log(metrics, update_step):
                    return None

                jax.lax.cond((update_step % cfg.log_interval) == 0, log_metrics, do_not_log, metrics, update_step)

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
        for task_idx, (rng, env) in enumerate(zip(env_rngs, envs)):
            # --- Train on environment i using the *current* ewc_state ---
            runner_state, metrics = train_on_environment(rng, train_state, env, cl_state, task_idx)
            train_state = runner_state[0]
            cl_state = runner_state[6]

            importance = cl.compute_importance(train_state.params, env, network, task_idx, rng, cfg.use_cnn,
                                               cfg.importance_episodes, cfg.importance_steps,
                                               cfg.normalize_importance)

            cl_state = cl.update_state(cl_state, train_state.params, importance)

            if cfg.record_gif:
                if visualizer is None:
                    visualizer = create_visualizer(temp_env.num_agents, cfg.env_name)
                # Generate & log a GIF after finishing task i
                env_name = env.layout_name
                states = record_gif_of_episode(cfg, train_state, env, network, task_idx, cfg.gif_len)
                file_path = f"{exp_dir}/task_{task_idx}_{env_name}.gif"
                visualizer.animate(states, out_path=file_path, task_idx=task_idx, env=env)

            # save the model
            repo_root = Path(__file__).resolve().parent.parent
            path = f"{repo_root}/checkpoints/overcooked/{cfg.cl_method}/{run_name}/model_env_{task_idx + 1}"
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
                    "num_tasks": len(envs),
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
    cl_state = cl.init_state(train_state.params, cfg.regularize_critic, cfg.regularize_heads)

    # Initialize AGEM memory if using AGEM and this is the first environment
    if cfg.cl_method.lower() == "agem":
        # Get observation dimension
        obs_dim = envs[0].observation_space().shape
        if not cfg.use_cnn:
            obs_dim = (np.prod(obs_dim),)
        # Initialize memory buffer
        cl_state = init_agem_memory(cfg.agem_memory_size, obs_dim, max_tasks=len(envs))

    # apply the loop_over_envs function to the environments
    loop_over_envs(train_rng, train_state, cl_state, envs)


if __name__ == "__main__":
    print("Running main...")
    main()
