import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from baselines.utils import build_reg_weights
from cl_methods.base import RegCLMethod, CLState


class MAS(RegCLMethod):
    """
    Memory-Aware Synapses (Aljundi 2018).
    """
    name = "mas"

    def init_state(self,
                   params: FrozenDict,
                   regularize_critic: bool,
                   regularize_heads: bool) -> CLState:
        return CLState(
            old_params=jax.tree_map(lambda x: x.copy(), params),
            importance=jax.tree_map(jnp.zeros_like, params),
            mask=build_reg_weights(params, regularize_critic, regularize_heads)
        )

    def update_state(self,
                     cl_state: CLState,
                     new_params: FrozenDict,
                     new_importance: FrozenDict,
                     **_) -> CLState:
        return CLState(old_params=new_params, importance=new_importance, mask=cl_state.mask)

    def penalty(self,
                params: FrozenDict,
                cl_state: CLState,
                coef: float) -> jnp.ndarray:
        def _term(p, o, ω, m):
            return m * ω * (p - o) ** 2

        tot = jax.tree_util.tree_map(_term, params, cl_state.old_params, cl_state.importance, cl_state.mask)
        tot = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(), tot, 0.)
        denom = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(), cl_state.mask, 0.) + 1e-8
        return 0.5 * coef * tot / denom

    def compute_importance(self,
                           params: FrozenDict,
                           env,
                           net,
                           env_idx: int,
                           key: jax.random.PRNGKey,
                           use_cnn: bool = True,
                           max_episodes: int = 5,
                           max_steps: int = 500,
                           norm_importance: bool = False):
        return compute_importance(params, env, net, env_idx, key, use_cnn, max_episodes, max_steps, norm_importance)


def compute_importance(params,
                       env,
                       net,
                       env_idx: int,
                       rng,
                       use_cnn: bool = True,
                       max_episodes=5,
                       max_steps=500,
                       norm_importance=False) -> FrozenDict:
    """
    Perform rollouts and compute MAS importance by averaging the squared gradients of
    the output’s L2 norm. That is, for each state x, we compute
        L = 1/2 * ||f_\theta(x)||^2
    then accumulate (dL/dθ)^2 across steps/states.
    """

    # Initialize importance accumulation to zeros
    importance_init = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x),
        params
    )

    def l2_norm_output(params, obs_dict):
        # We'll sum over all agents the 1/2 * || f_\theta(obs) ||^2.
        # You can pick what "output" means: maybe just the policy logits,
        # or the value function, or both. Here, let's do policy logits + value.
        total_loss = 0.0
        for agent_id, obs_val in obs_dict.items():
            pi, v = net.apply(params, obs_val, env_idx=env_idx)
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
                if not use_cnn:
                    obs_v = jnp.reshape(obs_v, (obs_v.shape[0], -1))  # make it (1, obs_dim)
                flat_obs[agent_id] = obs_v

            # Optional: step environment with some policy actions
            # (not necessary for importance computation, but you'd do it for exploring states)

            # Grad wrt L2 norm of the outputs
            def mas_loss_fn(p):
                return l2_norm_output(p, flat_obs)

            grads = jax.grad(mas_loss_fn)(params)
            grads_sqr = jax.tree_util.tree_map(lambda g: g ** 2, grads)
            # Accumulate
            importance_accum = jax.tree_util.tree_map(
                lambda acc, gs: acc + gs, importance_accum, grads_sqr
            )

            # Step environment or break if done
            rng, rng_step = jax.random.split(rng)
            actions = {}
            for i, agent_id in enumerate(env.agents):
                pi, _v = net.apply(params, flat_obs[agent_id], env_idx=env_idx)
                actions[agent_id] = jnp.squeeze(pi.sample(seed=rng_step), axis=0)

            next_obs, next_state, reward, done_info, _info = env.step(rng_step, state, actions)
            done = done_info["__all__"]
            obs, state = next_obs, next_state
            step_count += 1

        return importance_accum, step_count

    # Main loop
    importance_accum = importance_init
    total_steps = 0
    rngs = jax.random.split(rng, max_episodes)

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
    if norm_importance and total_steps > 0:
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
