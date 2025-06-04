import jax
import jax.numpy as jnp
from flax import struct
from flax.core.frozen_dict import FrozenDict

from baselines.utils import make_task_onehot, copy_params


@struct.dataclass
class MASState:
    old_params: FrozenDict
    importance: FrozenDict
    reg_weights: FrozenDict  # a mask: ones for parameters to regularize, zeros otherwise


def init_cl_state(params: FrozenDict, regularize_critic: bool, regularize_heads: bool) -> MASState:
    """
    Initialize old_params with the current parameters, importance with zeros.
    """
    old_params = copy_params(params)
    reg_weights = build_reg_weights(params, regularize_critic, regularize_heads)
    importance = jax.tree_map(lambda x: jnp.zeros_like(x), old_params)
    return MASState(old_params=old_params, importance=importance, reg_weights=reg_weights)


def update_mas_state(mas_state: MASState,
                     new_params: FrozenDict,
                     new_importance: FrozenDict) -> MASState:
    """
    After finishing task i, store a snapshot of the new parameters and the new importance values.
    """
    return MASState(
        old_params=copy_params(new_params),
        importance=new_importance,
        reg_weights=mas_state.reg_weights
    )


def build_reg_weights(params: FrozenDict, regularize_critic: bool = False, regularize_heads: bool = True) -> FrozenDict:
    def _assign_reg_weight(path, x):
        # Join the keys in the path to a string.
        path_str = "/".join(str(key) for key in path)
        # Exclude head parameters: do not regularize if parameter is in actor_head or critic_head.
        if not regularize_heads:
            if "actor_head" in path_str or "critic_head" in path_str:
                return jnp.zeros_like(x)
        # If we're not regularizing the critic, then exclude any parameter from critic branches.
        if not regularize_critic and "critic" in path_str.lower():
            return jnp.zeros_like(x)
        # Otherwise, regularize (the trunk).
        return jnp.ones_like(x)

    return jax.tree_util.tree_map_with_path(_assign_reg_weight, params)


def compute_mas_importance(
        config,
        train_state,
        env,
        network,
        env_idx=0,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
        max_episodes=5,
        max_steps=500,
        normalize_importance=False
):
    """
    Perform rollouts and compute MAS importance by averaging the squared gradients of
    the output’s L2 norm. That is, for each state x, we compute
        L = 1/2 * ||f_\theta(x)||^2
    then accumulate (dL/dθ)^2 across steps/states.
    """

    # Initialize importance accumulation to zeros
    importance_init = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x),
        train_state.params
    )

    def l2_norm_output(params, obs_dict):
        # We'll sum over all agents the 1/2 * || f_\theta(obs) ||^2.
        # You can pick what "output" means: maybe just the policy logits,
        # or the value function, or both. Here, let's do policy logits + value.
        total_loss = 0.0
        for agent_id, obs_val in obs_dict.items():
            pi, v = network.apply(params, obs_val, env_idx=env_idx)
            # Example: L2 norm of the concatenation of logits and the value.
            # shape: (1, action_dim + 1)
            # sum of squares / 2
            logits_and_v = jnp.concatenate([pi.logits, jnp.expand_dims(v, -1)], axis=-1)
            l2 = 0.5 * jnp.sum(logits_and_v ** 2)
            total_loss += l2
        return total_loss

    def single_episode_importance(rng_ep, importance_accum):
        rng, rng_reset = jax.random.split(rng_ep)
        obs, state = env.reset(rng_reset)
        done = False
        step_count = 0

        while (not done) and (step_count < max_steps):
            # Prepare obs for all agents as a batch
            flat_obs = {}
            for agent_id, obs_v in obs.items():
                expected_shape = env.observation_space().shape
                if obs_v.ndim == len(expected_shape):
                    obs_v = jnp.expand_dims(obs_v, axis=0)  # (1, ...)
                v_b = jnp.reshape(obs_v, (obs_v.shape[0], -1))  # make it (1, obs_dim)
                # If we use task_id
                if config.use_task_id:
                    onehot = make_task_onehot(env_idx, config.seq_length)
                    onehot = jnp.expand_dims(onehot, axis=0)
                    v_b = jnp.concatenate([v_b, onehot], axis=1)
                flat_obs[agent_id] = v_b

            # Optional: step environment with some policy actions
            # (not necessary for importance computation, but you'd do it for exploring states)

            # Grad wrt L2 norm of the outputs
            def mas_loss_fn(p):
                return l2_norm_output(p, flat_obs)

            grads = jax.grad(mas_loss_fn)(train_state.params)
            grads_sqr = jax.tree_util.tree_map(lambda g: g ** 2, grads)
            # Accumulate
            importance_accum = jax.tree_util.tree_map(
                lambda acc, gs: acc + gs, importance_accum, grads_sqr
            )

            # Step environment or break if done
            rng, rng_step = jax.random.split(rng)
            actions = {}
            for i, agent_id in enumerate(env.agents):
                pi, _v = network.apply(train_state.params, flat_obs[agent_id], env_idx=env_idx)
                actions[agent_id] = jnp.squeeze(pi.sample(seed=rng_step), axis=0)

            next_obs, next_state, reward, done_info, _info = env.step(rng_step, state, actions)
            done = done_info["__all__"]
            obs, state = next_obs, next_state
            step_count += 1

        return importance_accum, step_count

    # Main loop
    importance_accum = importance_init
    total_steps = 0
    rngs = jax.random.split(key, max_episodes)

    for ep_i in range(max_episodes):
        importance_accum, ep_steps = single_episode_importance(rngs[ep_i], importance_accum)
        total_steps += ep_steps

    # Average over total steps
    if total_steps > 0:
        importance_accum = jax.tree_util.tree_map(
            lambda x: x / float(total_steps),
            importance_accum
        )

    # Optional normalization
    if normalize_importance and total_steps > 0:
        total_abs = jax.tree_util.tree_reduce(
            lambda acc, x: acc + jnp.sum(jnp.abs(x)),
            importance_accum,
            0.0
        )
        param_count = jax.tree_util.tree_reduce(
            lambda acc, x: acc + x.size,
            importance_accum,
            0
        )
        importance_mean = total_abs / (param_count + 1e-8)
        importance_accum = jax.tree_util.tree_map(
            lambda x: x / (importance_mean + 1e-8),
            importance_accum
        )

    return importance_accum


def compute_mas_loss(params: FrozenDict,
                     mas_state: MASState,
                     mas_coef: float
                     ) -> float:
    """
    0.5 * mas_coef * sum( importance * (params - old_params)^2 ).
    """

    def penalty(p, old_p, imp, w):
        return w * imp * (p - old_p) ** 2

    penalty_tree = jax.tree_util.tree_map(
        lambda p_, op_, im_, w: penalty(p_, op_, im_, w),
        params, mas_state.old_params, mas_state.importance, mas_state.reg_weights
    )

    penalty_sum = jax.tree_util.tree_reduce(lambda acc, x: acc + x.sum(), penalty_tree, 0.0)
    return 0.5 * mas_coef * penalty_sum
