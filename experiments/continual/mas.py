import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.continual.base import RegCLMethod, CLState
from experiments.utils import batchify, unbatchify


class MAS(RegCLMethod):
    """
    Memory-Aware Synapses (Aljundi 2018).
    """
    name = "mas"

    def update_state(self, cl_state: CLState, new_params: FrozenDict, new_importance: FrozenDict) -> CLState:
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

    def make_importance_fn(self, reset_switch, step_switch, network, agents, use_cnn: bool, max_episodes: int,
                           max_steps: int, norm_importance: bool, stride: int):
        """
        Returns a jitted function:
          mas_importance(params, env_idx, rng, max_episodes=5, max_steps=500,
                         norm_importance=False, stride=1)
        that computes MAS importance.
        """
        num_agents = len(agents)

        @jax.jit
        def mas_importance(params, env_idx: jnp.int32, rng: jax.random.PRNGKey):
            # zeros tree
            importance0 = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)

            def one_episode(carry, _):
                imp_accum, total_steps, rng = carry
                rng, r = jax.random.split(rng)
                obs, state = reset_switch(r, env_idx)
                done = jnp.array(0.0, dtype=jnp.float32)

                def one_step(carry, t):
                    imp_accum, state, obs, rng, done, steps = carry

                    # batchify observations once per step: (A, obs_dim or HWC)
                    obs_b = batchify(obs, agents, num_agents, not use_cnn)

                    def l2_norm_loss(p):
                        pi, v, _ = network.apply(p, obs_b, env_idx=env_idx)
                        v = v.reshape(num_agents, 1)
                        vec = jnp.concatenate([pi.logits, v], axis=-1)
                        return 0.5 * jnp.sum(vec * vec) / vec.shape[0]

                    # grads of l2 wrt params
                    grads = jax.grad(l2_norm_loss)(params)

                    # only accumulate if not done; optionally subsample by stride
                    alpha = (t % stride == 0).astype(jnp.float32)
                    factor = (1.0 - done) * alpha
                    grads2 = jax.tree_util.tree_map(lambda g: (g * g) * factor, grads)
                    imp_accum = jax.tree_util.tree_map(lambda a, g: a + g, imp_accum, grads2)

                    # step env with greedy/sample action (doesn't matter for MAS)
                    rng, s1, s2 = jax.random.split(rng, 3)
                    pi, _, _ = network.apply(params, obs_b, env_idx=env_idx)
                    action = pi.sample(seed=s1)  # (A,)
                    env_act = unbatchify(action, agents, 1, num_agents)  # dict -> (1,) then flatten
                    env_act = {k: v.flatten() for k, v in env_act.items()}

                    obs2, state2, _rew, done_dict, _info = step_switch(s2, state, env_act, env_idx)
                    done2 = done_dict["__all__"].astype(jnp.float32)

                    steps = steps + (1.0 - done)
                    done = jnp.maximum(done, done2)

                    return (imp_accum, state2, obs2, rng, done, steps), None

                (imp_accum, state, obs, rng, done, steps), _ = jax.lax.scan(
                    one_step,
                    (imp_accum, state, obs, rng, done, jnp.array(0.0, jnp.float32)),
                    xs=jnp.arange(max_steps)
                )

                return (imp_accum, total_steps + steps, rng), None

            (importance_accum, total_steps, rng), _ = jax.lax.scan(
                one_episode,
                (importance0, jnp.array(0.0, jnp.float32), rng),
                xs=jnp.arange(max_episodes)
            )

            # average over visited steps
            importance_accum = jax.tree_util.tree_map(
                lambda x: x / (total_steps + 1e-8), importance_accum
            )

            if norm_importance:
                total_abs = jax.tree_util.tree_reduce(
                    lambda acc, x: acc + jnp.sum(jnp.abs(x)), importance_accum, 0.0
                )
                n_params = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, importance_accum, 0)
                mean_abs = total_abs / (n_params + 1e-8)
                importance_accum = jax.tree_util.tree_map(
                    lambda x: x / (mean_abs + 1e-8), importance_accum
                )

            return importance_accum

        return mas_importance
