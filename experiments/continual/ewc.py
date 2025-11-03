import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.continual.base import RegCLMethod, CLState
from experiments.utils import batchify, unbatchify


class EWC(RegCLMethod):
    """
    Diagonal Elastic-Weight-Consolidation (Kirkpatrick 2017).

    Modes
    -----
    • "last"   – keep only the Fisher from the previous task
    • "online" – running exponential average with decay λ
    • "multi"  – accumulate *sum* of Fishers (standard EWC)
    """
    name = "ewc"

    def __init__(self, mode="last", decay: float = 0.9):
        assert mode in {"last", "online", "multi"}
        self.mode = mode
        self.decay = decay

    def update_state(self, cl_state: CLState, new_params: FrozenDict, new_fisher: FrozenDict) -> CLState:

        if self.mode == "last":
            fish = new_fisher

        elif self.mode == "multi":
            fish = jax.tree_util.tree_map(jnp.add, cl_state.importance, new_fisher)

        else:  # "online"
            fish = jax.tree_util.tree_map(
                lambda old_f, f_new: self.decay * old_f + (1. - self.decay) * f_new,
                cl_state.importance, new_fisher)

        return CLState(old_params=new_params, importance=fish, mask=cl_state.mask)

    def penalty(self, params: FrozenDict, cl_state: CLState, coef: float) -> jnp.ndarray:

        def _term(p, o, f, m):
            return m * f * (p - o) ** 2

        tot = jax.tree_util.tree_map(_term,
                                     params, cl_state.old_params,
                                     cl_state.importance, cl_state.mask)
        tot = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(), tot, 0.)
        denom = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(),
                                          cl_state.mask, 0.) + 1e-8
        return 0.5 * coef * tot / denom

    def make_importance_fn(self, reset_switch, step_switch, network, agents, use_cnn: bool, max_episodes: int,
                           max_steps: int, norm_importance: bool, stride: int):
        """
        Returns a jitted function:
            fisher(params, env_idx, rng) -> FrozenDict
        that computes diagonal Fisher using only reset/step switches and the network.
        """
        num_agents = len(agents)

        @jax.jit
        def fisher(params: FrozenDict, env_idx: jnp.int32, rng: jax.random.PRNGKey) -> FrozenDict:
            fisher0 = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)

            def one_episode(carry, _):
                rng, acc = carry
                rng, r = jax.random.split(rng)
                obs, state = reset_switch(r, env_idx)

                def one_step(carry, _):
                    obs, state, acc, rng = carry
                    rng, s1, s2 = jax.random.split(rng, 3)

                    # batchify to (A, obs_dim) or (A, H, W, C)
                    obs_b = batchify(obs, agents, num_agents, not use_cnn)

                    # forward once
                    pi, _, _ = network.apply(params, obs_b, env_idx=env_idx)
                    acts = jax.lax.stop_gradient(pi.sample(seed=s1))  # (A,)

                    # grad log π(a|obs) wrt params
                    def logp(p):
                        pi_p, _, _ = network.apply(p, obs_b, env_idx=env_idx)
                        return jnp.sum(pi_p.log_prob(acts))

                    g = jax.grad(logp)(params)
                    g2 = jax.tree_util.tree_map(lambda x: x * x, g)
                    acc = jax.tree_util.tree_map(jnp.add, acc, g2)

                    # env step with these actions
                    env_act = unbatchify(acts, agents, 1, num_agents)
                    env_act = {k: v.flatten() for k, v in env_act.items()}
                    obs2, state2, _r, _d, _info = step_switch(s2, state, env_act, env_idx)

                    return (obs2, state2, acc, rng), None

                (obs, state, acc, rng), _ = jax.lax.scan(
                    one_step, (obs, state, acc, rng), xs=None, length=max_steps
                )
                return (rng, acc), None

            (rng, fisher_acc), _ = jax.lax.scan(
                one_episode, (rng, fisher0), xs=None, length=max_episodes
            )

            # average over steps and episodes
            fisher_acc = jax.tree_util.tree_map(
                lambda x: x / (max_episodes * max_steps + 1e-8), fisher_acc
            )

            if norm_importance:
                total_abs = jax.tree_util.tree_reduce(lambda a, x: a + jnp.sum(jnp.abs(x)), fisher_acc, 0.0)
                n_params = jax.tree_util.tree_reduce(lambda a, x: a + x.size, fisher_acc, 0)
                mean_abs = total_abs / (n_params + 1e-8)
                fisher_acc = jax.tree_util.tree_map(lambda x: x / (mean_abs + 1e-8), fisher_acc)

            return fisher_acc

        return fisher
