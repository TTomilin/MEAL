from functools import partial
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp

from experiments.utils import batchify, unbatchify


@partial(jax.jit)
def evaluate_model(envs, cfg, num_agents, agents, train_state, key):
    '''
    Evaluates the model by running 10 episodes on all environments and returns the average reward
    @param train_state: the current state of the training
    @param config: the configuration of the training
    returns the average reward
    '''

    def run_episode_while(env, key_r):
        """
        Run a single episode using jax.lax.while_loop
        """

        class EvalState(NamedTuple):
            key: Any
            state: Any
            obs: Any
            done: bool
            total_reward: float
            soup: float
            step_count: int

        def cond_fun(state: EvalState):
            '''
            Checks if the episode is done or if the maximum number of steps has been reached
            @param state: the current state of the loop
            returns a boolean indicating whether the loop should continue
            '''
            return jnp.logical_and(jnp.logical_not(state.done), state.step_count < env.max_steps)

        def body_fun(state: EvalState):
            '''
            Performs a single step in the environment
            @param state: the current state of the loop
            returns the updated state
            '''

            key, state_env, obs, _, total_reward, total_soup, step_count = state
            subkeys = jax.random.split(key, num_agents + 2)
            key, *agent_keys, key_s = subkeys

            # ***Create a batched copy for the network only.***
            # For each agent, expand dims to get shape (1, H, W, C) then flatten to (1, -1)
            batched_obs = {}
            for agent, v in obs.items():
                v_b = jnp.expand_dims(v, axis=0)  # now (1, H, W, C)
                if not cfg.use_cnn:
                    v_b = jnp.reshape(v_b, (v_b.shape[0], -1))  # flatten
                batched_obs[agent] = v_b

            def select_action(train_state, rng, obs):
                '''
                Selects an action based on the policy network
                @param params: the parameters of the network
                @param rng: random number generator
                @param obs: the observation
                returns the action
                '''
                network_apply = train_state.apply_fn
                params = train_state.params
                pi, value, _ = network_apply(params, obs, env_idx=eval_idx)
                action = jnp.squeeze(pi.sample(seed=rng), axis=0)
                return action, value

            # Get action distributions
            actions = {}
            for key_agent, agent in zip(agent_keys, agents):
                act, _ = select_action(train_state, key_agent, batched_obs[agent])
                actions[agent] = act

            # Environment step
            next_obs, next_state, reward, done_step, info = env.step(key_s, state_env, actions)
            done = done_step["__all__"]
            reward = sum(reward[agent] for agent in agents)  # Sum of all agent rewards
            soups_this_step = sum(info["soups"][agent] for agent in agents)
            total_reward += reward
            total_soup += soups_this_step
            step_count += 1

            return EvalState(key, next_state, next_obs, done, total_reward, total_soup, step_count)

        # Initialize
        key, key_s = jax.random.split(key_r)
        obs, state = env.reset(key_s)
        init_state = EvalState(key, state, obs, False, 0.0, 0.0, 0)

        # Run while loop
        final_state = jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=init_state
        )

        return final_state.total_reward, final_state.soup

    # Loop through all environments
    all_avg_rewards = []
    all_avg_soups = []

    for eval_idx, env in enumerate(envs):
        # Run k episodes
        all_rewards, all_soups = jax.vmap(
            lambda k: run_episode_while(env, k)
        )(jax.random.split(key, cfg.eval_num_episodes))

        all_avg_rewards.append(jnp.mean(all_rewards))
        all_avg_soups.append(jnp.mean(all_soups))

    return all_avg_rewards, all_avg_soups


def make_eval_fn(reset_switch, step_switch, network, agents, num_envs: int, num_steps: int, use_cnn: bool):
    """
    Returns a jitted evaluate_single_env(rng, params, env_idx) that *closes over*
    the static callables and constants so we don't pass non-arrays to jit.
    """
    @jax.jit
    def evaluate_env(rng, params, env_idx):
        # Reset a batch of envs for shape parity with train
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, num_envs)
        obs, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

        # Accumulators across the horizon per parallel env
        agent_ret = {a: jnp.zeros((num_envs,), jnp.float32) for a in agents}
        soups_env = jnp.zeros((num_envs,), jnp.float32)

        def one_step(carry, _):
            env_state, obs, agent_ret, soups, rng = carry

            # policy forward (greedy) on batched obs
            obs_batch = batchify(obs, agents, len(agents) * num_envs, not use_cnn)  # (num_actors, obs_dim)
            pi, _, _ = network.apply(params, obs_batch, env_idx=env_idx)
            action = pi.mode()  # deterministic eval

            # unbatch to dict of (num_envs,)
            env_act = unbatchify(action, agents, num_envs, len(agents))
            env_act = {k: v.flatten() for k, v in env_act.items()}

            # env step
            rng, sub = jax.random.split(rng)
            step_rng = jax.random.split(sub, num_envs)
            obs2, env_state2, reward, done, info = jax.vmap(
                lambda k, s, a: step_switch(k, s, a, jnp.int32(env_idx))
            )(step_rng, env_state, env_act)

            # accumulate per-agent rewards and soups
            agent_ret = {a: agent_ret[a] + reward[a] for a in agents}
            soups = soups + sum(info["soups"][a] for a in agents)

            return (env_state2, obs2, agent_ret, soups, rng), None

        (env_state, obs, agent_ret, soups_env, rng), _ = jax.lax.scan(
            one_step,
            (env_state, obs, agent_ret, soups_env, rng),
            xs=None,
            length=num_steps
        )

        avg_rewards = {a: agent_ret[a].mean() for a in agents}
        avg_soups = soups_env.mean()
        return avg_rewards, avg_soups

    return evaluate_env


def evaluate_all_envs(rng, params, num_envs, evaluate_env):
    avg_rewards_list = []
    avg_soups_list = []
    for env_idx in range(num_envs):
        rng, env_rng = jax.random.split(rng)
        avg_rewards, avg_soups = evaluate_env(env_rng, params, jnp.int32(env_idx))
        avg_rewards_list.append(avg_rewards)
        avg_soups_list.append(avg_soups)
    return avg_rewards_list, avg_soups_list
