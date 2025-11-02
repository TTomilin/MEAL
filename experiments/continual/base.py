from typing import Protocol

import jax
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

    # ---- life-cycle ---------------------------------------------------------
    def init_state(self, params: FrozenDict, regularise_critic: bool, regularise_heads: bool) -> CLState: ...

    # ---- state update -------------------------------------------------------
    def update_state(self, cl_state: CLState, new_params: FrozenDict, new_importance: FrozenDict) -> CLState: ...

    # ---- penalty ------------------------------------------------------------
    def penalty(self, params: FrozenDict, cl_state: CLState, coef: float) -> jnp.ndarray: ...

    # ---- importance weights -------------------------------------------------
    def compute_importance(self,
                           params: FrozenDict,
                           reset_switch,
                           step_switch,
                           network,
                           env_idx: int,
                           rng: jax.random.PRNGKey,
                           agents,
                           use_cnn: bool,
                           max_episodes: int,
                           max_steps: int,
                           norm_importance: bool,
                           stride: int) -> FrozenDict: ...


