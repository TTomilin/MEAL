import jax
import jax.numpy as jnp
from typing import List, Any
import chex
from flax.training.train_state import TrainState


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    avail_actions: dict
    rewards: dict
    dones: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def batchify(x: dict, agent_list: List[str]):
    '''
    stack the observations of all agents into a single array
    @param x: the observations
    @param env: the environment
    returns the batchified observations
    '''
    return jnp.stack([x[agent] for agent in agent_list], axis=0)

def unbatchify(x: jnp.ndarray, agent_list: List[str]):
    '''
    unstack the observations of all agents into a dictionary
    @param x: the batchified observations
    @param env: the environment
    returns the unbatchified observations
    '''
    return {agent: x[i] for i, agent in enumerate(agent_list)}    

def get_greedy_actions(q_vals, valid_actions):
    '''
    Get the greedy actions from the Q-values
    @param q_vals: the Q-values
    @param valid_actions: the valid actions
    returns the greedy actions
    '''
    unavail_actions = 1 - valid_actions
    q_vals = q_vals - (unavail_actions * 1e10) #subtract a large number from the Q-values of unavailable actions
    return jnp.argmax(q_vals, axis=-1) #returns the index of the best action (with the highest q-value)

def eps_greedy_exploration(rng, q_vals, eps, valid_actions):
    '''
    Function that performs epsilon-greedy exploration
    @param rng: the random number generator
    @param q_vals: the Q-values
    @param eps: the epsilon value
    @param valid_actions: the valid actions
    returns the chosen actions
    '''
    rng_a, rng_e = jax.random.split(
        rng
    )  # a key for sampling random actions and one for picking

    greedy_actions = get_greedy_actions(q_vals, valid_actions)

    # pick random actions from the valid actions
    def get_random_actions(rng, val_action):
        return jax.random.choice(
            rng,
            jnp.arange(val_action.shape[-1]),
            p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
        )

    _rngs = jax.random.split(rng_a, valid_actions.shape[0]) #the first dimension is the number of agents
    random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

    chosen_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape) < eps,  # pick the actions that should be random
        random_actions,
        greedy_actions,
    )
    return chosen_actions