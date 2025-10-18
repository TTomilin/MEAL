from functools import partial
from typing import Tuple, Union

import chex
import jax
from flax import struct
from jax import Array, numpy as jnp

from meal.env import State, MultiAgentEnv
from meal.wrappers.jaxmarl import JaxMARLWrapper


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: Array
    episode_lengths: Array
    returned_episode_returns: Array
    returned_episode_lengths: Array


class LogWrapper(JaxMARLWrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: LogEnvState,
            action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:

        # perform the step
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action)
        # normalize done to vector (A,)
        if isinstance(done, dict):
            # ignore per-agent flags there; use "__all__" for episode boundary
            ep_done_scalar = jnp.asarray(done["__all__"], jnp.bool_)
            ep_done_vec = jnp.full((self._env.num_agents,), ep_done_scalar)
        else:
            done_arr = jnp.asarray(done, jnp.bool_)
            ep_done_vec = done_arr if done_arr.ndim > 0 else jnp.full((self._env.num_agents,), done_arr)

        new_episode_return = state.episode_returns + self._batchify_floats(reward)  # reward of the current step
        new_episode_length = state.episode_lengths + 1  # length of the current episode

        keep = (1 - ep_done_vec).astype(new_episode_return.dtype)  # (A,)
        add = ep_done_vec.astype(new_episode_length.dtype)

        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * keep,
            episode_lengths=(state.episode_lengths + 1) * keep,
            returned_episode_returns=state.returned_episode_returns * keep + new_episode_return * add,
            returned_episode_lengths=state.returned_episode_lengths * keep + new_episode_length * add,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = ep_done_vec
        return obs, state, reward, done, info
