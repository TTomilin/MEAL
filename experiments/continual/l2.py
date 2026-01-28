import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.continual.base import RegCLMethod, RegCLState


class L2(RegCLMethod):
    """
    Simple L2 regularisation towards the parameters learned on the
    *previous* task (Kirkpatrick 2017 “baseline”).
    """
    name = "l2"

    def update_state(self, cl_state: RegCLState, new_params: FrozenDict, new_importance: FrozenDict) -> RegCLState:
        # we only need to remember θᵗ
        return RegCLState(old_params=new_params, importance=cl_state.importance, mask=cl_state.mask)

    def penalty(self,
                params: FrozenDict,
                cl_state: RegCLState,
                coef: float) -> jnp.ndarray:
        diff2 = jax.tree_util.tree_map(
            lambda p, o, m: m * (p - o) ** 2,
            params, cl_state.old_params, cl_state.mask)

        tot = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(), diff2, 0.)
        denom = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(), cl_state.mask, 0.) + 1e-8
        return 0.5 * coef * tot / denom

    # ── importance function factory (to satisfy unified interface) ───────────
    def make_importance_fn(self, reset_switch, step_switch, network, agents, use_cnn: bool, max_episodes: int,
                           max_steps: int, norm_importance: bool, stride: int) -> callable:
        # Returns a jitted function with the same call signature but producing zeros.
        @jax.jit
        def importance_fn(params: FrozenDict, env_idx: jnp.int32, rng):
            return jax.tree.map(jnp.zeros_like, params)

        return importance_fn
