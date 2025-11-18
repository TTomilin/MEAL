from functools import partial
from typing import Tuple, Union

import chex
import jax
from flax import struct
from jax import Array, numpy as jnp

from meal.env import State, MultiAgentEnv
from meal.wrappers.jaxmarl import JaxMARLWrapper


@struct.dataclass
class RandomizedActionsState:
    env_state: State


class RandomizedActions(JaxMARLWrapper):
    """With probability p_replace, replace chosen action by a random action.

    Intended for discrete action spaces, where actions are integers.
    `n_actions` can be:
      - int: same number of actions for all agents
      - dict[str, int]: per-agent number of actions (matching the action dict)
    """

    def __init__(self, env: MultiAgentEnv, p_replace: float = 0.1):
        super().__init__(env)
        self.p_replace = p_replace
        self.n_actions = len(self._env.action_set)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, RandomizedActionsState]:
        obs, env_state = self._env.reset(key)
        state = RandomizedActionsState(env_state=env_state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: RandomizedActionsState,
        action,
    ):
        key, subkey = jax.random.split(key)

        def sample_uniform(x):
            return jax.random.uniform(subkey, shape=x.shape)

        def sample_random(x):
            # Here we assume a global int n_actions
            return jax.random.randint(subkey, x.shape, 0, self.n_actions)

        # Decide whether to replace with random action per element
        u = jax.tree_util.tree_map(sample_uniform, action)
        replace_mask = jax.tree_util.tree_map(
            lambda uu: uu < self.p_replace, u
        )
        random_action = jax.tree_util.tree_map(sample_random, action)

        effective_action = jax.tree_util.tree_map(
            lambda m, a_rand, a: jnp.where(m, a_rand, a),
            replace_mask, random_action, action,
        )

        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, effective_action
        )

        info = {**info, "applied_action": effective_action}

        new_state = RandomizedActionsState(env_state=env_state)
        return obs, new_state, reward, done, info
