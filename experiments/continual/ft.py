import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.continual.base import RegCLMethod, RegCLState


class FT(RegCLMethod):
    """Plain fine-tuning: keep training, add **zero** regularization."""

    name = "ft"

    # ─── state update: nothing to store ─────────────────────────────────────
    def update_state(self, cl_state: RegCLState, new_params: FrozenDict, new_importance: FrozenDict) -> RegCLState:
        return cl_state  # no change

    # ─── penalty: always zero ───────────────────────────────────────────────
    def penalty(self, params: FrozenDict, cl_state: RegCLState, coef: float):
        return jnp.array(0.0, dtype=jnp.float32)
