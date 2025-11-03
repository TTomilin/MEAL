import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.utils import build_reg_weights
from experiments.continual.base import RegCLMethod, CLState


class FT(RegCLMethod):
    """Plain fine-tuning: keep training, add **zero** regularization."""

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

    # ── importance function factory (to satisfy unified interface) ───────────
    def make_importance_fn(self, reset_switch, step_switch, network, agents, use_cnn: bool, max_episodes: int,
                           max_steps: int, norm_importance: bool, stride: int) -> callable:
        # Returns a jitted function with the same call signature but producing zeros.
        @jax.jit
        def importance_fn(params: FrozenDict, env_idx: jnp.int32, rng):
            return jax.tree.map(jnp.zeros_like, params)
        return importance_fn
