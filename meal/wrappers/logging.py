from functools import partial
from typing import Optional, Tuple, Union

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
    episode_soups: Array
    returned_episode_returns: Array
    returned_episode_lengths: Array
    returned_episode_soups: Array
    # Optional per-agent diagnostics, present only with `LogWrapper(..., agent_stats=True)`; None otherwise.
    episode_shaped: Optional[Array] = None             # (num_agents,) raw shaped reward so far
    episode_actions: Optional[Array] = None            # (num_agents, num_actions) executed-action counts so far
    returned_episode_shaped: Optional[Array] = None
    returned_episode_actions: Optional[Array] = None


class LogWrapper(JaxMARLWrapper):
    """Log the episode returns, lengths, and soup deliveries.
    NOTE for now for envs where agents terminate at the same time.

    With `agent_stats=True` (opt-in; Overcooked-style envs only) it also accumulates, per agent, the raw shaped
    reward (`info["shaped_reward"]`, never annealed) and a count of every executed action, indexed by the
    environment's own action value (`env.action_set.take(index)`, i.e. `Actions` for Overcooked). Both are
    reported as `info["returned_episode_shaped"]` and `info["returned_episode_actions"]` and, like the other
    `returned_*` values, are only meaningful where `info["returned_episode"]` is True. The counts include the
    action of the terminal step and are reset together with the episode (before the env auto-resets).
    """

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False, agent_stats: bool = False):
        super().__init__(env)
        self.replace_info = replace_info
        self.agent_stats = agent_stats
        if agent_stats and not hasattr(env, "action_set"):
            raise ValueError("agent_stats needs an environment with an `action_set` (Overcooked)")

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        extra = {}
        if self.agent_stats:
            shaped = jnp.zeros((self._env.num_agents,))
            actions = jnp.zeros((self._env.num_agents, self._env.num_actions))
            extra = dict(episode_shaped=shaped, episode_actions=actions,
                         returned_episode_shaped=shaped, returned_episode_actions=actions)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            **extra,
        )
        return obs, state

    def _executed_action_counts(self, action):
        """One-hot (num_agents, num_actions) of the actions this step executes, via the env's action mapping."""
        env = self._env
        if isinstance(action, dict):
            indices = jnp.array([action[a].flatten()[0] for a in env.agents], dtype=jnp.uint8)
        else:
            indices = action
        return jax.nn.one_hot(env.action_set.take(indices=indices), env.num_actions, dtype=jnp.float32)

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

        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1

        # Per-agent soup deliveries this step (0.0 if env doesn't provide soups)
        if "soups" in info:
            step_soups = jnp.stack([info["soups"][a] for a in self._env.agents])  # (num_agents,)
            new_episode_soups = state.episode_soups + step_soups
        else:
            new_episode_soups = state.episode_soups

        extra = {}
        if self.agent_stats:
            new_shaped = state.episode_shaped + jnp.stack([info["shaped_reward"][a] for a in self._env.agents])
            new_actions = state.episode_actions + self._executed_action_counts(action)

        keep = (1 - ep_done_vec).astype(new_episode_return.dtype)
        add = ep_done_vec.astype(new_episode_length.dtype)
        if self.agent_stats:
            extra = dict(
                episode_shaped=new_shaped * keep,
                episode_actions=new_actions * keep[:, None],
                returned_episode_shaped=state.returned_episode_shaped * keep + new_shaped * add,
                returned_episode_actions=state.returned_episode_actions * keep[:, None] + new_actions * add[:, None])

        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * keep,
            episode_lengths=(state.episode_lengths + 1) * keep,
            episode_soups=new_episode_soups * keep,
            returned_episode_returns=state.returned_episode_returns * keep + new_episode_return * add,
            returned_episode_lengths=state.returned_episode_lengths * keep + new_episode_length * add,
            returned_episode_soups=state.returned_episode_soups * keep + new_episode_soups * add,
            **extra,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode_soups"] = state.returned_episode_soups
        info["returned_episode"] = ep_done_vec
        if self.agent_stats:
            info["returned_episode_shaped"] = state.returned_episode_shaped
            info["returned_episode_actions"] = state.returned_episode_actions
        return obs, state, reward, done, info

    def unwrap_env_state(self, state: LogEnvState):
        """Unwrap the LogEnvState to return the raw env state."""
        return state.env_state
