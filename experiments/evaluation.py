import jax
import jax.numpy as jnp

from experiments.utils import batchify, unbatchify
from experiments.continual.packnet import Packnet
from functools import partial
from typing import Any, NamedTuple

def make_eval_fn(reset_switch, step_switch, actor, critic, agents, cl, cl_state, eval_envs, num_envs: int, num_steps: int, use_cnn: bool):
    """
    Returns a jitted evaluate_single_env(rng, params, env_idx) that *closes over*
    the static callables and constants so we don't pass non-arrays to jit.
    """

    @partial(jax.jit)
    def evaluate_model_packnet(rng, actor_params, critic_params):
        '''
        Evaluates the model by running 10 episodes on all environments and returns the average reward
        @param train_state: the current state of the training
        @param config: the configuration of the training
        returns the average reward
        '''

        def run_episode_while(env, key_r, env_idx: int, max_steps=1000):
            """
            Run a single episode using jax.lax.while_loop
            """
            mask = cl.combine_masks(cl_state.masks, env_idx+1)

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
                return jnp.logical_and(jnp.logical_not(state.done), state.step_count < max_steps)

            def body_fun(state: EvalState):
                '''
                Performs a single step in the environment
                @param state: the current state of the loop
                returns the updated state
                '''

                key, state_env, obs, _, total_reward, total_soup, step_count = state
                key, key_a0, key_a1, key_s = jax.random.split(key, 4)

                # ***Create a batched copy for the network only.***
                # For each agent, expand dims to get shape (1, H, W, C) then flatten to (1, -1)
                batched_obs = {}
                for agent, v in obs.items():
                    v_b = jnp.expand_dims(v, axis=0)  # now (1, H, W, C)
                    if not use_cnn:
                        v_b = jnp.reshape(v_b, (v_b.shape[0], -1))  # flatten
                    batched_obs[agent] = v_b

                def select_action(actor_params, critic_params, rng, obs):
                    '''
                    Selects an action based on the policy network
                    @param train_state: the training state (tuple of actor and critic for Packnet)
                    @param rng: random number generator
                    @param obs: the observation
                    returns the action and value
                    '''
                    # For Packnet, train_state is a tuple of (actor_train_state, critic_train_state)
                    masked_params = cl.apply_mask(actor_params["params"], mask, env_idx)
                    pi, _ = actor.apply(masked_params, obs, env_idx=env_idx)
                    value, _ = critic.apply(critic_params, obs, env_idx=env_idx)

                    action = jnp.squeeze(pi.sample(seed=rng), axis=0)
                    return action, value

                # Get action distributions
                action_a1, _ = select_action(actor_params, critic_params, key_a0, batched_obs["agent_0"])
                action_a2, _ = select_action(actor_params, critic_params, key_a1, batched_obs["agent_1"])

                # Sample actions
                actions = {
                    "agent_0": action_a1,
                    "agent_1": action_a2
                }

                # Environment step
                next_obs, next_state, reward, done_step, info = env.step(key_s, state_env, actions)
                done = done_step["__all__"]
                reward = reward["agent_0"]  # Common reward
                soups_this_step = info["soups"]["agent_0"] + info["soups"]["agent_1"]
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

        for env_idx, env in enumerate(eval_envs):
            # Run k episodes
            EVAL_NUM_EPISODES = 5 # make configurable later
            keys = jax.random.split(rng, EVAL_NUM_EPISODES)
            all_rewards, all_soups = jax.vmap(lambda k: run_episode_while(env, k, env_idx, num_steps))(keys)

            avg_reward = jnp.mean(all_rewards)
            avg_soup = jnp.mean(all_soups)
            all_avg_rewards.append(avg_reward)
            all_avg_soups.append(avg_soup)

        return all_avg_rewards, all_avg_soups

    @jax.jit
    def evaluate_env(rng, actor_params, env_idx):
        # Reset a batch of envs for shape parity with train
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, num_envs)
        obs, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

        # Accumulators across the horizon per parallel env
        total_rewards = jnp.zeros((num_envs,), jnp.float32)
        total_soups = jnp.zeros((num_envs,), jnp.float32)

        # set mask if needed:
        combined_mask = None
        if isinstance(cl, Packnet):
            combined_mask = cl.combine_masks(cl_state.masks, env_idx+1)

        def one_step(carry, _):
            env_state, obs, rewards, soups, rng = carry

            # policy forward (greedy) on batched obs
            obs_batch = batchify(obs, agents, len(agents) * num_envs, not use_cnn)  # (num_actors, obs_dim)

            # select action:
            if isinstance(cl, Packnet):
                masked_params = cl.apply_mask(actor_params["params"], combined_mask, env_idx)
                pi, _ = actor.apply(masked_params, obs_batch, env_idx=env_idx)
            else:
                pi, _ = actor.apply(actor_params, obs_batch, env_idx=env_idx)

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
            rewards += sum(reward[a] for a in agents)
            soups += sum(info["soups"][a] for a in agents)

            return (env_state2, obs2, rewards, soups, rng), None

        (env_state, obs, total_rewards, total_soups, rng), _ = jax.lax.scan(
            one_step,
            (env_state, obs, total_rewards, total_soups, rng),
            xs=None,
            length=num_steps
        )

        avg_rewards = total_rewards.mean()
        avg_soups = total_soups.mean()
        return avg_rewards, avg_soups

    if isinstance(cl, Packnet):
        return evaluate_model_packnet
    else:
        return evaluate_env


def evaluate_all_envs(rng, actor_params, num_envs, evaluate_env):
    env_indices = jnp.arange(num_envs, dtype=jnp.int32)
    rngs = jax.random.split(rng, num_envs)
    eval_vmapped = jax.vmap(evaluate_env, in_axes=(0, None, 0))
    avg_rewards, avg_soups = eval_vmapped(rngs, actor_params, env_indices)
    return avg_rewards, avg_soups
