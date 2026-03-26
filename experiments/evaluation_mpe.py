"""
Evaluation helpers for MPE SimpleSpread.

Metrics per task:
  - avg_rewards          : mean total reward per episode across parallel envs
  - avg_coverage_fraction: mean coverage_fraction (num_covered/num_landmarks) per step [0,1]
  - avg_num_covered      : mean number of landmarks covered per step
"""

import jax
import jax.numpy as jnp

from experiments.utils import batchify, unbatchify


def make_eval_fn(
    reset_switch,
    step_switch,
    network,
    agents,
    num_envs: int,
    num_steps: int,
    use_cnn: bool,
):
    """Return a jitted ``evaluate_env(rng, params, env_idx)`` closure."""

    @jax.jit
    def evaluate_env(rng, params, env_idx):
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, num_envs)
        obs, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(
            reset_rng
        )

        total_rewards = jnp.zeros((num_envs,), jnp.float32)
        total_coverage_fraction = jnp.zeros((num_envs,), jnp.float32)
        total_num_covered = jnp.zeros((num_envs,), jnp.float32)

        def one_step(carry, _):
            env_state, obs, rewards, coverage_fraction, num_covered, rng = carry

            obs_batch = batchify(
                obs, agents, len(agents) * num_envs, not use_cnn
            )
            pi, _, _ = network.apply(params, obs_batch, env_idx=env_idx)
            action = pi.mode()

            env_act = unbatchify(action, agents, num_envs, len(agents))
            env_act = {k: v.flatten() for k, v in env_act.items()}

            rng, sub = jax.random.split(rng)
            step_rng = jax.random.split(sub, num_envs)
            obs2, env_state2, reward, done, info = jax.vmap(
                lambda k, s, a: step_switch(k, s, a, jnp.int32(env_idx))
            )(step_rng, env_state, env_act)

            rewards += sum(reward[a] for a in agents)
            coverage_fraction += info.get("coverage_fraction", jnp.zeros((num_envs,)))
            num_covered += info.get("num_covered", jnp.zeros((num_envs,)))

            return (env_state2, obs2, rewards, coverage_fraction, num_covered, rng), None

        (_, _, total_rewards, total_coverage_fraction, total_num_covered, _), _ = jax.lax.scan(
            one_step,
            (env_state, obs, total_rewards, total_coverage_fraction, total_num_covered, rng),
            xs=None,
            length=num_steps,
        )

        return (
            total_rewards.mean(),
            total_coverage_fraction.mean() / num_steps,  # per-step average [0,1]
            total_num_covered.mean() / num_steps,
        )

    return evaluate_env


def evaluate_all_envs(rng, params, num_envs, evaluate_env):
    env_indices = jnp.arange(num_envs, dtype=jnp.int32)
    rngs = jax.random.split(rng, num_envs)
    eval_vmapped = jax.vmap(evaluate_env, in_axes=(0, None, 0))
    avg_rewards, avg_coverage_fraction, avg_num_covered = eval_vmapped(rngs, params, env_indices)
    return avg_rewards, avg_coverage_fraction, avg_num_covered
