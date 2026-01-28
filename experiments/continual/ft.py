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

    # ── importance function factory (to satisfy unified interface) ───────────
    def make_importance_fn(self, reset_switch, step_switch, network, agents, use_cnn: bool, max_episodes: int,
                           max_steps: int, norm_importance: bool, stride: int) -> callable:
        # Returns a jitted function with the same call signature but producing zeros.
        @jax.jit
        def importance_fn(params: FrozenDict, env_idx: jnp.int32, rng):
            return jax.tree.map(jnp.zeros_like, params)

        return importance_fn
