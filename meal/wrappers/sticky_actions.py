from functools import partial
from typing import Tuple

import chex
import jax
from flax import struct
from jax import Array, numpy as jnp

from meal.env import State, MultiAgentEnv
from meal.wrappers.jaxmarl import JaxMARLWrapper


@struct.dataclass
class StickyActionsState:
    env_state: State
    last_action: dict[str, Array]


class StickyActions(JaxMARLWrapper):
    """Repeats the old action with probability p.
    """

    def __init__(self, env: MultiAgentEnv, p: float = 0.1):
        super().__init__(env)
        self.p = p

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, StickyActionsState]:
        obs, env_state = self._env.reset(key)
        state = StickyActionsState(
            env_state,
            {agent: jnp.array(0) for agent in self._env.agents}
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: StickyActionsState,
            action: Array,
    ) -> Tuple[chex.Array, StickyActionsState, float, bool, dict]:
        key, subkey = jax.random.split(key)
        last_action = state.last_action

        def sample_mask(x):
            return jax.random.bernoulli(subkey, self.p, shape=x.shape)

        sticky_mask = jax.tree_util.tree_map(sample_mask, action)

        effective_action = jax.tree_util.tree_map(
            lambda m, a, la: jnp.where(m, la, a),
            sticky_mask, action, last_action
        )

        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, effective_action
        )

        info = {**info, "applied_action": effective_action}

        new_state = StickyActionsState(
            env_state=env_state,
            last_action=effective_action,
        )

        return obs, new_state, reward, done, info
