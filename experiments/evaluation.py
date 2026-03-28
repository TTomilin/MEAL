import jax
import jax.numpy as jnp

from experiments.utils import batchify, unbatchify


def _jsd(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Jensen-Shannon divergence (log-base-2), bounded [0, 1].
    0 = identical action distributions (homogeneous), 1 = non-overlapping (fully heterogeneous).
    """
    m = 0.5 * (p + q)
    eps = 1e-10
    kl_pm = jnp.sum(jnp.where(p > eps, p * jnp.log2(p / (m + eps)), 0.0))
    kl_qm = jnp.sum(jnp.where(q > eps, q * jnp.log2(q / (m + eps)), 0.0))
    return jnp.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0)


def make_eval_fn(reset_switch, step_switch, network, agents, num_envs: int, num_steps: int, use_cnn: bool):
    """
    Returns a jitted evaluate_single_env(rng, params, env_idx) that *closes over*
    the static callables and constants so we don't pass non-arrays to jit.

    Returns (avg_rewards, avg_soups, avg_het) where avg_het is the mean
    Jensen-Shannon divergence between agent-0 and agent-1 action distributions
    across the episode (0 = homogeneous, 1 = fully heterogeneous).
    """
    num_actions = network.action_dim

    @jax.jit
    def evaluate_env(rng, params, env_idx):
        # Reset a batch of envs for shape parity with train
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, num_envs)
        obs, env_state = jax.vmap(lambda k: reset_switch(k, jnp.int32(env_idx)))(reset_rng)

        n_agents = len(agents)

        # Accumulators across the horizon per parallel env
        total_rewards = jnp.zeros((num_envs,), jnp.float32)
        total_soups = jnp.zeros((num_envs,), jnp.float32)
        # Per-agent, per-env action count accumulators: (n_agents, num_envs, num_actions)
        counts = jnp.zeros((n_agents, num_envs, num_actions), jnp.float32)

        def one_step(carry, _):
            env_state, obs, rewards, soups, counts, rng = carry

            # policy forward (greedy) on batched obs
            obs_batch = batchify(obs, agents, n_agents * num_envs, not use_cnn)  # (n_agents*num_envs, obs_dim)
            pi, _, _ = network.apply(params, obs_batch, env_idx=env_idx)
            action = pi.mode()  # (n_agents * num_envs,)

            # accumulate per-agent action counts
            # batchify lays out rows as: agent0_env0..envN, agent1_env0..envN, ...
            action_2d = action.reshape(n_agents, num_envs)       # (n_agents, num_envs)
            counts = counts + jax.nn.one_hot(action_2d, num_actions)  # (n_agents, num_envs, num_actions)

            # unbatch to dict of (num_envs,)
            env_act = unbatchify(action, agents, num_envs, n_agents)
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

            return (env_state2, obs2, rewards, soups, counts, rng), None

        (env_state, obs, total_rewards, total_soups, counts, rng), _ = jax.lax.scan(
            one_step,
            (env_state, obs, total_rewards, total_soups, counts, rng),
            xs=None,
            length=num_steps
        )

        avg_rewards = total_rewards.mean()
        avg_soups = total_soups.mean()

        # Normalise counts to per-env probability distributions
        # counts: (n_agents, num_envs, num_actions)
        probs = counts / (counts.sum(axis=-1, keepdims=True) + 1e-10)  # (n_agents, num_envs, num_actions)

        # Average pairwise JSD across all unique agent pairs and all envs
        # n_agents is static (Python int), so the loop is unrolled at trace time
        pair_jsds = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                jsd_ij = jax.vmap(_jsd)(probs[i], probs[j])  # (num_envs,)
                pair_jsds.append(jsd_ij.mean())

        if pair_jsds:
            avg_het = jnp.mean(jnp.stack(pair_jsds))
        else:
            avg_het = jnp.zeros((), jnp.float32)

        return avg_rewards, avg_soups, avg_het

    return evaluate_env


def evaluate_all_envs(rng, params, num_envs, evaluate_env):
    env_indices = jnp.arange(num_envs, dtype=jnp.int32)
    rngs = jax.random.split(rng, num_envs)
    eval_vmapped = jax.vmap(evaluate_env, in_axes=(0, None, 0))
    return eval_vmapped(rngs, params, env_indices)
