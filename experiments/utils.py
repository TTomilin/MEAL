import os
import uuid
import optax
from datetime import datetime
from typing import NamedTuple, Union, Tuple
import flax.linen as nn

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.typing import FrozenVariableDict
from tensorboardX import SummaryWriter
from flax.training.train_state import TrainState
from optax._src.base import GradientTransformation

from experiments.continual.base import RegCLState, CLMethod, CLState
from experiments.continual.packnet import PacknetState, Packnet
from experiments.model.mlp import ActorCritic as MLPActorCritic
from experiments.model.cnn import ActorCritic as CNNActorCritic
from experiments.model.decoupled_mlp import Actor, Critic


class Transition(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray  # whether the episode is done
    action: jnp.ndarray  # the action taken
    value: jnp.ndarray  # the value of the state
    reward: jnp.ndarray  # the reward received
    log_prob: jnp.ndarray  # the log probability of the action
    obs: jnp.ndarray  # the observation


class Transition_MAPPO(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray  # whether the episode is done
    action: jnp.ndarray  # the action taken
    value: jnp.ndarray  # the value of the state
    reward: jnp.ndarray  # the reward received
    log_prob: jnp.ndarray  # the log probability of the action
    obs: jnp.ndarray  # the observation
    global_state: jnp.ndarray  # the global state for centralized critic


def batchify(x, agent_list, num_actors, flatten=True):
    '''
    converts the observations of a batch of agents into an array of size (num_actors, -1) that can be used by the network
    @param flatten: for MLP architectures
    @param x: dictionary of observations (multi-agent) or direct array (single-agent)
    @param agent_list: list of agents
    @param num_actors: number of actors
    returns the batchified observations
    '''
    x = jnp.stack([x[a] for a in agent_list])
    batched = jnp.concatenate(x, axis=0)
    if flatten:
        batched = batched.reshape((num_actors, -1))
    return batched


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    '''
    converts the array of size (num_actors, -1) into a dictionary of observations for all agents
    @param unflatten: for MLP architectures
    @param x: array of observations
    @param agent_list: list of agents
    @param num_envs: number of environments
    @param num_actors: number of actors
    returns the unbatchified observations (dict for multi-agent, direct array for single-agent)
    '''
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def sample_discrete_action(key, action_space):
    """Samples a discrete action based on the action space provided."""
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)


def make_task_onehot(task_idx: int, num_tasks: int) -> jnp.ndarray:
    """
    Returns a one-hot vector of length `num_tasks` with a 1 at `task_idx`.
    """
    return jnp.eye(num_tasks, dtype=jnp.float32)[task_idx]


def copy_params(params):
    return jax.tree_util.tree_map(lambda x: x.copy(), params)


def add_eval_metrics(avg_rewards, avg_soups, layout_names, max_soup_dict, metrics):
    for i, layout_name in enumerate(layout_names):
        metrics[f"Evaluation/Returns/{i}_{layout_name}"] = avg_rewards[i]
        metrics[f"Evaluation/Soup/{i}_{layout_name}"] = avg_soups[i]
        metrics[f"Evaluation/Soup_Scaled/{i}_{layout_name}"] = avg_soups[i] / max_soup_dict[i]
    return metrics


def build_reg_weights(params, regularize_critic: bool, regularize_heads: bool) -> FrozenDict:
    def _mark(path, x):
        path_str = "/".join(map(str, path)).lower()
        if not regularize_heads and ("actor_head" in path_str or "critic_head" in path_str):
            return jnp.zeros_like(x)
        if not regularize_critic and "critic" in path_str:
            return jnp.zeros_like(x)
        return jnp.ones_like(x)

    return jax.tree_util.tree_map_with_path(_mark, params)


def init_cl_state(params: Union[FrozenVariableDict, Tuple[FrozenVariableDict, FrozenVariableDict]], regularize_critic: bool, 
                  regularize_heads: bool, cl: CLMethod) -> CLState:
    if isinstance(cl, Packnet):
        actor_params, _ = params # unpack params
        return PacknetState(
            masks=cl.init_mask_tree(actor_params),
            current_task=0,
            train_mode=True
        )
    else:
        mask = build_reg_weights(params, regularize_critic, regularize_heads)
        return RegCLState(
            old_params=jax.tree.map(lambda x: x.copy(), params),
            importance=jax.tree.map(jnp.zeros_like, params),
            mask=mask
        )
    
def create_network(env, cfg, cl) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
    # TODO: implement CNN-support for Packnet
    if isinstance(cl, Packnet):
        actor = Actor(
            action_dim=env.action_space().n,
            activation=cfg.activation,
            num_tasks=cfg.seq_length,
            use_multihead=cfg.use_multihead,
            use_task_id=cfg.use_task_id
        )
        critic = Critic(
            activation=cfg.activation,
            num_tasks=cfg.seq_length,
            use_multihead=cfg.use_multihead,
            use_task_id=cfg.use_task_id
        )
        return (actor, critic) # return tuple if using Packnet
    else:
        ac_cls = CNNActorCritic if cfg.use_cnn else MLPActorCritic
        actor_critic = ac_cls(env.action_space().n, cfg.activation, cfg.seq_length, cfg.use_multihead,
                            cfg.shared_backbone, cfg.big_network, cfg.use_task_id, cfg.regularize_heads,
                            cfg.use_layer_norm)
        return actor_critic
    
def init_network(env, network, cfg, cl) -> Union[FrozenVariableDict, Tuple[FrozenVariableDict, FrozenVariableDict]]:
    if isinstance(cl, Packnet):
        actor, critic = network # unpack actor, critic from network
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        init_x = jnp.zeros(env.observation_space().shape).flatten()
        init_x = jnp.expand_dims(init_x, axis=0)  # Add batch dimension
        actor_params = actor.init(actor_rng, init_x, env_idx=0)
        critic_params = critic.init(critic_rng, init_x, env_idx=0)

        return (actor_params, critic_params) # return tuple of parameters
    else:
        # Get the correct observation dimension by simulating the batchify process
        # This ensures the network is initialized with the same shape it will receive during training
        rng = jax.random.PRNGKey(cfg.seed)
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, cfg.num_envs)
        temp_obs, _ = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
        temp_obs_batch = batchify(temp_obs, env.agents, cfg.num_actors, not cfg.use_cnn)
        obs_dim = temp_obs_batch.shape[1]  # Get the actual dimension after batchify
        # Initialize the network
        rng, network_rng = jax.random.split(rng)
        init_x = jnp.zeros((1, obs_dim))
        network_params = network.init(network_rng, init_x)
        
        return network_params

def init_optimizer(cfg, schedule, cl) -> Union[GradientTransformation, Tuple[GradientTransformation, GradientTransformation]]:
    if isinstance(cl, Packnet):
        # Initialize the optimizers
        actor_tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=schedule if cfg.anneal_lr else cfg.lr, eps=1e-5)
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=schedule if cfg.anneal_lr else cfg.lr, eps=1e-5)
        )

        return (actor_tx, critic_tx) # if packnet is used, optimizer constitutes a tuple
    else:
        # Initialize the optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=schedule if cfg.anneal_lr else cfg.lr, eps=1e-5)
        )
        return tx

def init_train_state(network, network_params, optimizer, cl) -> Union[TrainState, Tuple[TrainState, TrainState]]:
    if isinstance(cl, Packnet):
        actor, critic = network
        actor_params, critic_params = network_params
        actor_tx, critic_tx = optimizer
        # jit the apply function
        actor.apply = jax.jit(actor.apply)
        critic.apply = jax.jit(critic.apply)
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
        return (actor_train_state, critic_train_state) # if packnet is used, train state constitutes a tuple
    else:
        # jit the apply function
        network.apply = jax.jit(network.apply)
        # Initialize the training state
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=optimizer
        )
        return train_state
    
def reset_optimizer(train_state, tx, cfg, cl):
    if cfg.reset_optimizer:
        if isinstance(cl, Packnet):
            actor_train_state, critic_train_state = train_state
            actor_tx, critic_tx = tx
            
            new_actor_optimizer = actor_train_state.tx.init(actor_train_state.params)
            actor_train_state.replace(tx=actor_tx, opt_state=new_actor_optimizer)
            new_critic_optimizer = critic_train_state.tx.init(critic_train_state.params)
            critic_train_state.replace(tx=critic_tx, opt_state=new_critic_optimizer)

            return (actor_train_state, critic_train_state), (new_actor_optimizer, new_critic_optimizer) # return tuples in case of packnet
        else:
            new_optimizer = train_state.tx.init(train_state.params)
            train_state = train_state.replace(tx=tx, opt_state=new_optimizer)
            return train_state, new_optimizer
    else:
        return train_state, tx

# ---------------------------------------------------------------
# util: build a (2, …) batch without Python branches
# ---------------------------------------------------------------
def _prep_obs(raw_obs, use_cnn: bool) -> jnp.ndarray:
    """
    Stack per‐agent observations into a single array of shape
    (num_agents, …).

    If use_cnn=False, each obs is flattened to a 1D float32 vector first.

    Handles both single-agent (direct array) and multi-agent (dictionary) observations.
    """

    def _single(obs: jnp.ndarray) -> jnp.ndarray:
        # flatten & cast when not using CNN
        if not use_cnn:
            obs = obs.reshape(-1).astype(jnp.float32)
        # introduce a leading "agent" axis
        return obs[None]

    # Handle both single-agent (direct array) and multi-agent (dictionary) cases
    if isinstance(raw_obs, dict):
        # Multi-agent case: raw_obs is a dictionary
        # Sort the keys so that the agent‐ordering is deterministic
        agent_ids = sorted(raw_obs.keys())

        # Build a list of (1, …) arrays, one per agent
        per_agent = [_single(raw_obs[a]) for a in agent_ids]
    else:
        # Single-agent case: raw_obs is a direct array
        per_agent = [_single(raw_obs)]

    # Concatenate along the new leading axis → (num_agents, …)
    return jnp.concatenate(per_agent, axis=0)


def create_run_name(config, network_architecture):
    """
    Generates a unique run name based on the config, current timestamp and a UUID.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = uuid.uuid4()
    difficulty_str = f"_{config.difficulty}" if config.difficulty else ""
    run_name = f'{config.alg_name}_{config.cl_method}{difficulty_str}_{network_architecture}_\
        seq{config.seq_length}_{config.strategy}_seed_{config.seed}_{timestamp}_{unique_id}'
    return run_name


def initialize_logging_setup(config, run_name, exp_dir):
    """
    Initializes WandB and TensorBoard logging setup.
    """
    if config.use_wandb:
        import wandb
        # Initialize WandB
        wandb_tags = config.tags if config.tags is not None else []
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(
            project=config.project,
            config=config,
            sync_tensorboard=True,
            mode=config.wandb_mode,
            name=run_name,
            id=run_name,
            tags=wandb_tags,
            group=config.group
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

    return writer


def rollout_for_video(rng, config, train_state, env, network, env_idx=0, max_steps=300):
    """
    Records a rollout of an episode by running the trained network on the environment.

    Args:
        rng: JAX random key
        config: Configuration object containing use_cnn flag
        train_state: Training state containing network parameters
        env: Environment to run the episode on
        network: Network to use for action selection
        env_idx: Environment/task index for multi-task networks (default: 0)
        max_steps: Maximum number of steps to record (default: 300)

    Returns:
        List of environment states for visualization
    """
    rng, env_rng = jax.random.split(rng)
    obs, state = env.reset(env_rng)
    done = False
    step_count = 0
    states = [env.unwrap_env_state(state)]

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
            pi, _, _ = network.apply(train_state.params, obs_dict[agent_id], env_idx=env_idx)
            actions[agent_id] = jnp.squeeze(pi.sample(seed=act_keys[i]), axis=0)

        rng, key_step = jax.random.split(rng)
        next_obs, next_state, reward, done_info, info = env.step(key_step, state, actions)
        done = done_info["__all__"]

        obs, state = next_obs, next_state
        step_count += 1
        states.append(env.unwrap_env_state(state))

    return states


def create_visualizer(num_agents, env_name):
    from meal.visualization.visualizer import OvercookedVisualizer
    from meal.visualization.visualizer_po import OvercookedVisualizerPO
    # Create appropriate visualizer based on environment type
    return OvercookedVisualizerPO(num_agents) if env_name == "overcooked_po" else OvercookedVisualizer(num_agents)
