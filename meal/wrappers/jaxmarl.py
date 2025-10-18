""" Wrappers for use with jaxmarl baselines. """
from functools import partial

import jax
import jax.numpy as jnp

from meal.env.multi_agent_env import MultiAgentEnv


class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_floats(self, x):
        # reward can be dict, array (A,), or scalar
        if isinstance(x, dict):
            return jnp.stack([x[a] for a in self._env.agents])  # (A,)
        x = jnp.asarray(x)
        return x if x.ndim > 0 else x[None]  # (A,) or (1,)


class CTRolloutManager(JaxMARLWrapper):
    """
    Rollout Manager for Centralized Training of with Parameters Sharing. Used by JaxMARL Q-Learning Baselines.
    - Batchify multiple environments (the number of parallel envs is defined by batch_size in __init__).
    - Adds a global state (obs["__all__"]) and a global reward (rewards["__all__"]) in the env.step returns.
    - Pads the observations of the agents in order to have all the same length.
    - Adds an agent id (one hot encoded) to the observation vectors.

    By default:
    - global_state is the concatenation of all agents' observations.
    - global_reward is the sum of all agents' rewards.
    """

    def __init__(self, env: MultiAgentEnv, batch_size: int, preprocess_obs: bool = True):

        super().__init__(env)

        self.batch_size = batch_size
        self.training_agents = self.agents
        self.preprocess_obs = preprocess_obs

        if len(env.observation_spaces) == 0:
            self.observation_spaces = {agent: self.observation_space() for agent in self.agents}
        if len(env.action_spaces) == 0:
            self.action_spaces = {agent: env.action_space() for agent in self.agents}

        # batched action sampling
        self.batch_samplers = {agent: jax.jit(jax.vmap(self.action_space(agent).sample, in_axes=0)) for agent in
                               self.agents}

        # assumes the observations are flattened vectors
        self.max_obs_length = max(list(map(lambda x: get_space_dim(x), self.observation_spaces.values())))
        self.max_action_space = max(list(map(lambda x: get_space_dim(x), self.action_spaces.values())))
        self.obs_size = self.max_obs_length
        if self.preprocess_obs:
            self.obs_size += len(self.agents)

        # agents ids
        self.agents_one_hot = {a: oh for a, oh in zip(self.agents, jnp.eye(len(self.agents)))}
        # valid actions
        self.valid_actions = {a: jnp.arange(u.n) for a, u in self.action_spaces.items()}
        self.valid_actions_oh = {a: jnp.concatenate((jnp.ones(u.n), jnp.zeros(self.max_action_space - u.n))) for a, u in
                                 self.action_spaces.items()}

        self.global_state = lambda obs, state: jnp.concatenate([obs[agent].flatten() for agent in self.agents], axis=-1)
        self.global_reward = lambda rewards: rewards[self.training_agents[0]]

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, key):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_reset, in_axes=0)(keys)

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, key, states, actions):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_step, in_axes=(0, 0, 0))(keys, states, actions)

    @partial(jax.jit, static_argnums=0)
    def wrapped_reset(self, key):
        obs_, state = self._env.reset(key)
        if self.preprocess_obs:
            obs = jax.tree_util.tree_map(self._preprocess_obs, {agent: obs_[agent] for agent in self.agents},
                                         self.agents_one_hot)
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def wrapped_step(self, key, state, actions):
        obs_, state, reward, done, infos = self._env.step(key, state, actions)
        if self.preprocess_obs:
            obs = jax.tree_util.tree_map(self._preprocess_obs, {agent: obs_[agent] for agent in self.agents},
                                         self.agents_one_hot)
            obs = jax.tree_util.tree_map(lambda d, o: jnp.where(d, 0., o),
                                         {agent: done[agent] for agent in self.agents},
                                         obs)  # ensure that the obs are 0s for done agents
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        reward["__all__"] = self.global_reward(reward)
        return obs, state, reward, done, infos

    def batch_sample(self, key, agent):
        return self.batch_samplers[agent](jax.random.split(key, self.batch_size)).astype(int)

    @partial(jax.jit, static_argnums=0)
    def get_valid_actions(self, state):
        # default is to return the same valid actions one hot encoded for each env 
        return {agent: jnp.tile(actions, self.batch_size).reshape(self.batch_size, -1) for agent, actions in
                self.valid_actions_oh.items()}

    @partial(jax.jit, static_argnums=0)
    def _preprocess_obs(self, arr, extra_features):
        # flatten
        arr = arr.flatten()
        # pad the observation vectors to the maximum length
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, max(0, self.max_obs_length - arr.shape[-1]))]
        arr = jnp.pad(arr, pad_width, mode='constant', constant_values=0)
        # concatenate the extra features
        arr = jnp.concatenate((arr, extra_features), axis=-1)
        return arr
