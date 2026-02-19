import jax

from typing import Protocol, Tuple

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
    def update_state(self, cl_state: Tuple[CLState, CLState], new_actor_params: FrozenDict, new_critic_params: FrozenDict,
                     new_actor_importance: FrozenDict, new_critic_importance: FrozenDict) -> CLState:
        return cl_state # return state unchanged by default

    # ---- importance function ------------------------------------------------
    def make_importance_fn(self, reset_switch, step_switch, actor, critic, agents, use_cnn, max_episodes: int, max_steps: int,
                           norm_importance: bool, stride: int) -> Tuple[callable, callable]:
        @jax.jit
        def importance_fn(params: FrozenDict, env_idx: jnp.int32, rng):
            return jax.tree.map(jnp.zeros_like, params)
        
        return importance_fn, importance_fn # return zero maps by default

class RegCLMethod(CLMethod):
    """Minimal interface every regularization-based CL method must expose."""

    # ---- penalty ------------------------------------------------------------
    def penalty(self, params: FrozenDict, cl_state: RegCLState, coef: float) -> jnp.ndarray:
        return 0.0 # default return zero (e.g. for agem and packnet)
