import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from typing import Tuple

from experiments.continual.base import RegCLMethod, RegCLState, CLState


class L2(RegCLMethod):
    """
    Simple L2 regularisation towards the parameters learned on the
    *previous* task (Kirkpatrick 2017 “baseline”).
    """
    name = "l2"

    def update_state(self, cl_state: Tuple[CLState, CLState], new_actor_params: FrozenDict, new_critic_params: FrozenDict, 
                     new_actor_importance, new_critic_importance) -> RegCLState:
        actor_cl_state, critic_cl_state = cl_state # unpack

        # we only need to remember θᵗ
        new_actor_state = RegCLState(old_params=new_actor_params, importance=actor_cl_state.importance, mask=actor_cl_state.mask)
        new_critic_state = RegCLState(old_params=new_critic_params, importance=critic_cl_state.importance, mask=critic_cl_state.mask)
        return new_actor_state, new_critic_state

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
