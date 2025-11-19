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

        def one_step(carry, _):
            env_state, obs, rewards, rng = carry

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
            rewards += sum(reward[a] for a in agents)

            return (env_state2, obs2, rewards, rng), None

        (env_state, obs, total_rewards, rng), _ = jax.lax.scan(
            one_step,
            (env_state, obs, total_rewards, rng),
            xs=None,
            length=num_steps
        )

        avg_rewards = total_rewards.mean()
        return avg_rewards

    return evaluate_env


def evaluate_all_envs(rng, params, num_envs, evaluate_env):
    env_indices = jnp.arange(num_envs, dtype=jnp.int32)
    rngs = jax.random.split(rng, num_envs)
    eval_vmapped = jax.vmap(evaluate_env, in_axes=(0, None, 0))
    avg_rewards = eval_vmapped(rngs, params, env_indices)
    return avg_rewards
