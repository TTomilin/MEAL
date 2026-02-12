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
