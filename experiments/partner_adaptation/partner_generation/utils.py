import re
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

KEYS_WITH_ARRAYS = [
    "wall_idx",
    "agent_idx",
    "goal_idx",
    "plate_pile_idx",
    "onion_pile_idx",
    "pot_idx",
]


def get_metric_names(env_name):
    if "overcooked-v1" in env_name:
        return ("base_return", "returned_episode_returns")
    else:
        return ("returned_episode_returns",)


@partial(jax.jit, static_argnames=['stats'])
def get_stats(metrics, stats: tuple):
    '''
    Computes mean and std of metrics of interest for each seed and update, 
    using only the final steps of episodes. Note that each rollout contains multiple episodes.

    metrics is a pytree where each leaf has shape 
        (..., rollout_length, num_envs)
    stats is a tuple of strings, each corresponding to a metric of interest in metrics
    '''
    # Get mask for final steps of episodes
    mask = metrics["returned_episode"]

    # Initialize output dictionary
    all_stats = {}
    # convert to list to correctly iterate if the tuple only has a single element
    stats = list(stats)
    for stat_name in stats:
        # Get the metric array
        # Shape: (..., rollout_length, num_envs)
        metric_data = metrics[stat_name]

        # Compute means and stds for each seed and update
        # Use masked operations to only consider final episode steps
        means = jnp.where(mask, metric_data, 0).sum(
            axis=(-2, -1)) / mask.sum(axis=(-2, -1))
        # For std, first compute masked values
        masked_vals = jnp.where(mask, metric_data, 0)
        squared_diff = (masked_vals - means[..., None, None]) ** 2
        variance = jnp.where(mask, squared_diff, 0).sum(
            axis=(-2, -1)) / mask.sum(axis=(-2, -1))
        stds = jnp.sqrt(variance)
        # Stack means and stds
        all_stats[stat_name] = jnp.stack([means, stds], axis=-1)

    return all_stats


def _parse_int_list(block: str):
    # Grab everything between the outermost square brackets
    m = re.search(r"\[\s*(.*?)\s*\]", block, flags=re.DOTALL)
    if not m:
        return []
    # Split on commas, allow newlines/spaces, keep signs
    items = [x.strip() for x in m.group(1).replace(
        "\n", " ").split(",") if x.strip() != ""]
    return [int(x) for x in items]


def frozendict_from_layout_repr(s: str) -> FrozenDict:
    """
    Parse strings like:
      FrozenDict({
          wall_idx: Array([ 0,  0,  1, ...], dtype=int32),
          agent_idx: Array([ 9, 32], dtype=int32),
          ...
          height: 6,
          width: 7,
      })
    and return a FrozenDict with jnp.int32 arrays for the index keys
    and Python ints for height/width.
    """
    data = {}

    # Arrays
    for key in KEYS_WITH_ARRAYS:
        # Match e.g. "wall_idx: Array([ ... ], dtype=int32)"
        pattern = rf"{key}\s*:\s*Array\(\s*\[(.*?)\]\s*,\s*dtype=int32\s*\)"
        m = re.search(pattern, s, flags=re.DOTALL)
        if m:
            # Reuse the helper on the whole Array(...) block so we don't miss edge spacing
            # Reconstruct the slice to benefit from the same bracket extraction
            block = "[" + m.group(1) + "]"
            ints = _parse_int_list(block)
            data[key] = jnp.array(ints, dtype=jnp.int32)
        else:
            # If an array key is missing, default to empty int32 array
            data[key] = jnp.array([], dtype=jnp.int32)

    # Scalars
    for key in ["height", "width"]:
        m = re.search(rf"{key}\s*:\s*(\d+)", s)
        if not m:
            raise ValueError(f"Missing '{key}' in layout string.")
        data[key] = int(m.group(1))

    return FrozenDict(data)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_info(x: dict, agent_list, num_actors):
    '''Handle special case that info has both per-agent and global information'''
    x = jnp.stack([x[a] for a in x if a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _create_minibatches(traj_batch, advantages, targets, init_hstate, num_actors, num_minibatches, perm_rng):
    """Create minibatches for PPO updates, where each leaf has shape 
        (num_minibatches, rollout_len, num_actors / num_minibatches, ...) 
    This function ensures that the rollout (time) dimension is kept separate from the minibatch and num_actors 
    dimensions, so that the minibatches are compatible with recurrent ActorCritics.
    """
    # Create batch containing trajectory, advantages, and targets
    batch = (
        init_hstate,  # shape (1, num_actors, hidden_dim)
        # pytree: obs is shape (rollout_len, num_actors, feat_shape)
        traj_batch,
        advantages,  # shape (rollout_len, num_actors)
        targets  # shape (rollout_len, num_actors)
    )

    permutation = jax.random.permutation(perm_rng, num_actors)

    # each leaf of shuffled batch has shape (rollout_len, num_actors, feat_shape)
    # except for init_hstate which has shape (1, num_actors, hidden_dim)
    shuffled_batch = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=1), batch
    )
    # each leaf has shape (num_minibatches, rollout_len, num_actors/num_minibatches, feat_shape)
    # except for init_hstate which has shape (num_minibatches, 1, num_actors/num_minibatches, hidden_dim)
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(
            jnp.reshape(
                x,
                [x.shape[0], num_minibatches, -1]
                + list(x.shape[2:]),
            ), 1, 0, ),
        shuffled_batch,
    )

    return minibatches


def _create_minibatches_no_time(traj_batch, advantages, targets, init_hstate, num_actors, num_minibatches, batch_size,
                                perm_rng):
    # reshape the batch to be compatible with the network
    batch = (init_hstate, traj_batch, advantages, targets)
    batch = jax.tree_util.tree_map(
        f=(lambda x: x.reshape((batch_size,) + x.shape[2:])), tree=batch
    )
    # split the random number generator for shuffling the batch
    rng, _rng = jax.random.split(perm_rng)

    # creates random sequences of numbers from 0 to batch_size, one for each vmap
    permutation = jax.random.permutation(_rng, batch_size)

    # shuffle the batch
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )  # outputs a tuple of the batch, advantages, and targets shuffled

    minibatches = jax.tree_util.tree_map(
        f=(lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:]))), tree=shuffled_batch,
    )
    return minibatches
