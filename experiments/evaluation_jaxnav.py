import jax
import jax.numpy as jnp

from experiments.utils import batchify, unbatchify


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
        total_rewards = jnp.zeros((num_envs,), jnp.float32)
        total_goals_reached = jnp.zeros((num_envs,), jnp.float32)
        total_episodes = jnp.zeros((num_envs,), jnp.float32)

        def one_step(carry, _):
            env_state, obs, rewards, goals_reached, episodes, rng = carry

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

            # accumulate per-agent rewards
            rewards += sum(reward[a] for a in agents)

            # accumulate GoalR and NumN (episode stats)
            goals_reached += info["GoalR"]
            episodes += info["NumC"]

            return (env_state2, obs2, rewards, goals_reached, episodes, rng), None

        (env_state, obs, total_rewards, total_goals_reached, total_episodes, rng), _ = jax.lax.scan(
            one_step,
            (env_state, obs, total_rewards, total_goals_reached, total_episodes, rng),
            xs=None,
            length=num_steps
        )

        # Success per parallel env: GoalR / NumN, averaged over parallel envs
        success_per_env = jnp.where(total_episodes > 0, total_goals_reached / total_episodes, 0.0)
        avg_success = success_per_env.mean()
        avg_rewards = total_rewards.mean()
        return avg_rewards, avg_success

    return evaluate_env


def evaluate_all_envs(rng, params, num_envs, evaluate_env):
    env_indices = jnp.arange(num_envs, dtype=jnp.int32)
    rngs = jax.random.split(rng, num_envs)
    eval_vmapped = jax.vmap(evaluate_env, in_axes=(0, None, 0))
    avg_rewards, avg_success = eval_vmapped(rngs, params, env_indices)
    return avg_rewards, avg_success
