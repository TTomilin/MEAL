import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from baselines.utils import build_reg_weights
from cl_methods.base import RegCLMethod, CLState


class FT(RegCLMethod):
    """Plain fine-tuning: keep training, add **zero** regularisation."""

    name = "ft"

    # ─── life-cycle ──────────────────────────────────────────────────────────
    def init_state(
            self,
            params: FrozenDict,
            regularize_critic: bool,
            regularize_heads: bool
    ) -> CLState:
        # dummy mask only to satisfy the dataclass; never used
        dummy_mask = build_reg_weights(params, regularize_critic, regularize_heads)
        return CLState(old_params=params, importance=None, mask=dummy_mask)

    # ─── state update: nothing to store ─────────────────────────────────────
    def update_state(
            self,
            cl_state: CLState,
            new_params: FrozenDict,
            new_importance: FrozenDict
    ) -> CLState:
        return cl_state  # no change

    # ─── penalty: always zero ───────────────────────────────────────────────
    def penalty(
            self,
            params: FrozenDict,
            cl_state: CLState,
            coef: float
    ):
        return jnp.array(0.0, dtype=jnp.float32)

    # ─── importance weights: not used ───────────────────────────────────────
    def compute_importance(
            self, *_, **__
    ):
        return None
