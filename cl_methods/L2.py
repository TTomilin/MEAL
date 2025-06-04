import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from baselines.utils import build_reg_weights
from cl_methods.base import RegCLMethod, CLState


class L2(RegCLMethod):
    """
    Simple L2 regularisation towards the parameters learned on the
    *previous* task (Kirkpatrick 2017 “baseline”).
    """
    name = "l2"

    def init_state(self,
                   params: FrozenDict,
                   regularize_critic: bool,
                   regularize_heads: bool) -> CLState:
        return CLState(
            old_params=jax.tree_map(lambda x: x.copy(), params),
            importance=None,
            mask=build_reg_weights(params, regularize_critic, regularize_heads)
        )

    def update_state(self, cl_state: CLState, new_params: FrozenDict, new_importance: FrozenDict) -> CLState:
        # we only need to remember θᵗ
        return CLState(old_params=new_params, importance=None, mask=cl_state.mask)

    def penalty(self,
                params: FrozenDict,
                cl_state: CLState,
                coef: float) -> jnp.ndarray:
        diff2 = jax.tree_util.tree_map(
            lambda p, o, m: m * (p - o) ** 2,
            params, cl_state.old_params, cl_state.mask)

        tot = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(), diff2, 0.)
        denom = jax.tree_util.tree_reduce(lambda a, b: a + b.sum(),
                                          cl_state.mask, 0.) + 1e-8
        return 0.5 * coef * tot / denom
