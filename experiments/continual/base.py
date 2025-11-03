from typing import Protocol

import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict


@struct.dataclass
class CLState:
    old_params: FrozenDict
    importance: FrozenDict
    mask: FrozenDict


class RegCLMethod(Protocol):
    """Minimal interface every regularization-based CL method must expose."""
    name: str

    # ---- state update -------------------------------------------------------
    def update_state(self, cl_state: CLState, new_params: FrozenDict, new_importance: FrozenDict) -> CLState: ...

    # ---- penalty ------------------------------------------------------------
    def penalty(self, params: FrozenDict, cl_state: CLState, coef: float) -> jnp.ndarray: ...

    # ---- importance function ------------------------------------------------
    def make_importance_fn(self, reset_switch, step_switch, network, agents, use_cnn, max_episodes: int, max_steps: int,
                           norm_importance: bool, stride: int) -> callable: ...
