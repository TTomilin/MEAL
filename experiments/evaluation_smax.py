"""
Evaluation helpers for SMAX continual learning.

Metrics per task:
  - avg_returns        : mean total reward per episode
  - avg_kill_fraction  : mean fraction of enemies killed per episode [0, 1]
                         (1.0 = all enemies dead = win; 0.4 = 2/5 enemies killed, etc.)
"""

import jax
import jax.numpy as jnp

from experiments.continual.packnet import Packnet
from experiments.utils import batchify, unbatchify


def make_eval_fn(
        cl,
        reset_switch,
        step_switch,
        network,
        agents,
        num_envs: int,
        num_steps: int,
        use_cnn: bool,
        eval_deterministic: bool = False,
        seed: int = 0,
):
    """Return a jitted ``evaluate_env(cl_state, rng, params, env_idx)`` closure."""

    @jax.jit
    def evaluate_env(cl_state, rng, params, env_idx):
        if eval_deterministic:
            rng = jax.random.PRNGKey(env_idx + seed)
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, num_envs)
        obs, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(
            reset_rng
        )

        total_returns = jnp.zeros((num_envs,), jnp.float32)
        total_kill_fracs = jnp.zeros((num_envs,), jnp.float32)
        num_episodes = jnp.zeros((num_envs,), jnp.float32)

        mask = None
        if isinstance(cl, Packnet):
            mask = cl.get_eval_mask(env_idx, cl_state)

        def one_step(carry, _):
            env_state, obs, returns, kill_fracs, episodes, rng = carry

            obs_batch = batchify(obs, agents, len(agents) * num_envs, not use_cnn)
            if isinstance(cl, Packnet):
                masked_params = cl.apply_mask(params, mask)
                pi, _, _ = network.apply(masked_params, obs_batch, env_idx=env_idx)
            else:
                pi, _, _ = network.apply(params, obs_batch, env_idx=env_idx)
            action = pi.mode()

            env_act = unbatchify(action, agents, num_envs, len(agents))
            env_act = {k: v.flatten() for k, v in env_act.items()}

            rng, sub = jax.random.split(rng)
            step_rng = jax.random.split(sub, num_envs)
            obs2, env_state2, reward, done, info = jax.vmap(
                lambda k, s, a: step_switch(k, s, a, jnp.int32(env_idx))
            )(step_rng, env_state, env_act)

            episode_ended = done.get("__all__", jnp.zeros((num_envs,), jnp.bool_))

            returns += sum(reward[a] for a in agents)
            kf = info.get("kill_fraction", jnp.zeros((num_envs,), jnp.float32))
            # Accumulate kill_fraction only at episode boundaries
            kill_fracs += jnp.where(episode_ended, kf, 0.0)
            episodes += episode_ended.astype(jnp.float32)

            return (env_state2, obs2, returns, kill_fracs, episodes, rng), None

        (_, _, total_returns, total_kill_fracs, num_episodes, _), _ = jax.lax.scan(
            one_step,
            (env_state, obs, total_returns, total_kill_fracs, num_episodes, rng),
            xs=None,
            length=num_steps,
        )

        avg_kill_fraction = total_kill_fracs / jnp.maximum(num_episodes, 1.0)
        return total_returns.mean(), avg_kill_fraction.mean()

    return evaluate_env


def evaluate_all_envs(cl_state, rng, params, num_envs, evaluate_env):
    env_indices = jnp.arange(num_envs, dtype=jnp.int32)
    rngs = jax.random.split(rng, num_envs)
    eval_vmapped = jax.vmap(evaluate_env, in_axes=(None, 0, None, 0))
    avg_returns, avg_kill_fraction = eval_vmapped(cl_state, rngs, params, env_indices)
    return avg_returns, avg_kill_fraction
