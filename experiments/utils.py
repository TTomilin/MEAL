import os
import uuid
from datetime import datetime
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from tensorboardX import SummaryWriter


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
        metrics[f"Evaluation/Returns/{i}__{layout_name}"] = avg_rewards[i]
        metrics[f"Evaluation/Soup/{i}__{layout_name}"] = avg_soups[i]
        metrics[f"Evaluation/Soup_Scaled/{i}__{layout_name}"] = avg_soups[i] / max_soup_dict[layout_name]
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


def record_gif_of_episode(config, train_state, env, network, env_idx=0, max_steps=300):
    """
    Records a GIF of an episode by running the trained network on the environment.

    This is the centralized version from IPPO_CL.py that works reliably across all baselines.

    Args:
        config: Configuration object containing use_cnn flag
        train_state: Training state containing network parameters
        env: Environment to run the episode on
        network: Network to use for action selection
        env_idx: Environment/task index for multi-task networks (default: 0)
        max_steps: Maximum number of steps to record (default: 300)

    Returns:
        List of environment states for visualization
    """
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
            pi, _, _ = network.apply(train_state.params, obs_dict[agent_id], env_idx=env_idx)
            actions[agent_id] = jnp.squeeze(pi.sample(seed=act_keys[i]), axis=0)

        rng, key_step = jax.random.split(rng)
        next_obs, next_state, reward, done_info, info = env.step(key_step, state, actions)
        done = done_info["__all__"]

        obs, state = next_obs, next_state
        step_count += 1
        states.append(state)

    return states


def create_visualizer(num_agents, env_name):
    from meal.visualization.visualizer import OvercookedVisualizer
    from meal.visualization.visualizer_po import OvercookedVisualizerPO
    # Create appropriate visualizer based on environment type
    return OvercookedVisualizerPO(num_agents) if env_name == "overcooked_po" else OvercookedVisualizer(num_agents)
