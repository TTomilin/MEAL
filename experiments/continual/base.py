from typing import Protocol

import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict


@struct.dataclass
class CLState:
    mask: FrozenDict

@struct.dataclass
class RegCLState(CLState):
    old_params: FrozenDict
    importance: FrozenDict

class CLMethod(Protocol):
    """Minimal interface any CL method must expose."""
    name: str

    # ---- state update -------------------------------------------------------
    def update_state(self, cl_state: CLState, new_params: FrozenDict, new_importance: FrozenDict) -> CLState:
        return cl_state # do nothing by default
    # ---- importance function ------------------------------------------------
    def make_importance_fn(self, reset_switch, step_switch, network, agents, use_cnn, max_episodes: int, max_steps: int,
                           norm_importance: bool, stride: int) -> callable:
        return (lambda a, b, c: 0) # return zero function by default

class RegCLMethod(CLMethod):
    """Minimal interface every regularization-based CL method must expose."""
    # ---- penalty ------------------------------------------------------------
    def penalty(self, params: FrozenDict, cl_state: CLState, coef: float) -> jnp.ndarray: ...


