"""
Evaluation helpers for SMAX continual learning.

Metrics per task:
  - avg_returns  : mean total reward per episode
  - win_rate     : fraction of episodes where allies won (all enemies dead, ≥1 ally alive)
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

        total_returns = jnp.zeros((num_envs,), jnp.float32)
        total_wins = jnp.zeros((num_envs,), jnp.float32)
        num_episodes = jnp.zeros((num_envs,), jnp.float32)

        def one_step(carry, _):
            env_state, obs, returns, wins, episodes, rng = carry

            obs_batch = batchify(obs, agents, len(agents) * num_envs, not use_cnn)
            pi, _, _ = network.apply(params, obs_batch, env_idx=env_idx)
            action = pi.mode()

            env_act = unbatchify(action, agents, num_envs, len(agents))
            env_act = {k: v.flatten() for k, v in env_act.items()}

            rng, sub = jax.random.split(rng)
            step_rng = jax.random.split(sub, num_envs)
            obs2, env_state2, reward, done, info = jax.vmap(
                lambda k, s, a: step_switch(k, s, a, jnp.int32(env_idx))
            )(step_rng, env_state, env_act)

            episode_ended = jnp.array([done["__all__"]] if isinstance(done, dict) else done)
            if isinstance(done, dict):
                episode_ended = done.get("__all__", jnp.zeros((num_envs,), jnp.bool_))

            returns += sum(reward[a] for a in agents)
            won = info.get("won", jnp.zeros((num_envs,), jnp.bool_))
            wins += jnp.where(episode_ended, won.astype(jnp.float32), 0.0)
            episodes += episode_ended.astype(jnp.float32)

            return (env_state2, obs2, returns, wins, episodes, rng), None

        (_, _, total_returns, total_wins, num_episodes, _), _ = jax.lax.scan(
            one_step,
            (env_state, obs, total_returns, total_wins, num_episodes, rng),
            xs=None,
            length=num_steps,
        )

        # win_rate = wins / max(episodes, 1)
        win_rate = total_wins / jnp.maximum(num_episodes, 1.0)

        return total_returns.mean(), win_rate.mean()

    return evaluate_env


def evaluate_all_envs(rng, params, num_envs, evaluate_env):
    env_indices = jnp.arange(num_envs, dtype=jnp.int32)
    rngs = jax.random.split(rng, num_envs)
    eval_vmapped = jax.vmap(evaluate_env, in_axes=(0, None, 0))
    avg_returns, avg_win_rate = eval_vmapped(rngs, params, env_indices)
    return avg_returns, avg_win_rate
