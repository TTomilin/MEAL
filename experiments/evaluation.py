import jax
import jax.numpy as jnp

from experiments.utils import batchify, unbatchify
from experiments.continual.packnet import Packnet


def make_eval_fn(cl, reset_switch, step_switch, network, agents, num_envs: int, num_steps: int, use_cnn: bool):
    """
    Returns a jitted evaluate_single_env(rng, params, env_idx) that *closes over*
    the static callables and constants so we don't pass non-arrays to jit.
    """

    @jax.jit
    def evaluate_env(cl_state, rng, params, env_idx):
        # Reset a batch of envs for shape parity with train
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, num_envs)
        obs, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

        # Accumulators across the horizon per parallel env
        total_rewards = jnp.zeros((num_envs,), jnp.float32)
        total_soups = jnp.zeros((num_envs,), jnp.float32)

        mask = None
        if isinstance(cl, Packnet):
            mask = cl.combine_masks(cl_state.masks, env_idx+1) # note that this collects all masks <= env_idx

        def one_step(carry, _):
            env_state, obs, rewards, soups, rng = carry

            # policy forward (greedy) on batched obs
            obs_batch = batchify(obs, agents, len(agents) * num_envs, not use_cnn)  # (num_actors, obs_dim)
            if isinstance(cl, Packnet):
                masked_params = cl.apply_mask(params["params"], mask, env_idx)
                pi, _, _ = network.apply(masked_params, obs_batch, env_idx=env_idx)
            else:
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

    return evaluate_env

def evaluate_all_envs(cl_state, rng, params, num_envs, evaluate_env):
    # Initialize accumulators
    total_rewards = jnp.zeros(num_envs, dtype=jnp.float32)
    total_soups = jnp.zeros(num_envs, dtype=jnp.float32)

    # Loop through all environments
    for i in range(num_envs):
        env_idx = i
        rng_env = jax.random.split(rng, num_envs)[i]
        avg_rewards, avg_soups = evaluate_env(cl_state, rng_env, params, env_idx)
        
        # Accumulate rewards and soups
        total_rewards = total_rewards.at[i].set(avg_rewards)
        total_soups = total_soups.at[i].set(avg_soups)

    return total_rewards, total_soups