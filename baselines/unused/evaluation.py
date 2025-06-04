import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import distrax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from functools import partial

import jax_marl
from jax_marl.environments.overcooked_environment import overcooked_layouts
from baselines.utils import pad_observation_space

@partial(jax.jit, static_argnums=(1,2))
def evaluate_model(train_state, network, config, key):
    '''
    Evaluates the model by running 10 episodes on all environments and returns the average reward
    @param train_state: the current state of the training
    @param config: the configuration of the training
    returns the average reward
    '''

    def run_episode_while(env, key_r, network, network_params, max_steps=1000):
        """
        Run a single episode using jax.lax.while_loop 
        """
        class LoopState(NamedTuple):
            key: Any
            state: Any
            obs: Any
            done: bool
            total_reward: float
            step_count: int

        def loop_cond(state: LoopState):
            '''
            Checks if the episode is done or if the maximum number of steps has been reached
            @param state: the current state of the loop
            returns a boolean indicating whether the loop should continue
            '''
            return jnp.logical_and(~state.done, state.step_count < max_steps)

        def loop_body(state: LoopState):
            '''
            Performs a single step in the environment
            @param state: the current state of the loop
            returns the updated state
            '''
            key, state_env, obs, _, total_reward, step_count = state
            key, key_a0, key_a1, key_s = jax.random.split(key, 4)

            # Flatten observations
            flat_obs = {k: v.flatten() for k, v in obs.items()}

            # Get action distributions
            pi_0, _ = network.apply(network_params, flat_obs["agent_0"])
            pi_1, _ = network.apply(network_params, flat_obs["agent_1"])

            # Sample actions
            actions = {
                "agent_0": pi_0.sample(seed=key_a0),
                "agent_1": pi_1.sample(seed=key_a1)
            }

            # Environment step
            next_obs, next_state, reward, done_step, info = env.step(key_s, state_env, actions)
            done = done_step["__all__"]
            reward = reward["agent_0"]  
            total_reward += reward
            step_count += 1

            return LoopState(key, next_state, next_obs, done, total_reward, step_count)

        # Initialize
        key, key_s = jax.random.split(key_r)
        obs, state = env.reset(key_s)
        init_state = LoopState(key, state, obs, False, 0.0, 0)

        # Run while loop
        final_state = jax.lax.while_loop(
            cond_fun=loop_cond,
            body_fun=loop_body,
            init_val=init_state
        )

        return final_state.total_reward

    # Loop through all environments
    all_avg_rewards = []

    envs = pad_observation_space(config)

    for env in envs:
        env = jax_marl.make(config["ENV_NAME"], layout=env)  # Create the environment

        # Initialize the network
        # key, key_a = jax.random.split(key)
        # init_x = jnp.zeros(env.observation_space().shape).flatten()  # initializes and flattens observation space

        # network.init(key_a, init_x)  # initializes the network with the observation space
        network_params = train_state.params

        # Run k episodes
        all_rewards = jax.vmap(lambda k: run_episode_while(env, k, network, network_params, 500))(
            jax.random.split(key, 5)
        )
        
        avg_reward = jnp.mean(all_rewards)
        all_avg_rewards.append(avg_reward)

    return all_avg_rewards