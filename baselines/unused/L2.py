import os
import jax
import jax.numpy as jnp
from flax import struct
from flax.core.frozen_dict import FrozenDict
from baselines.utils import copy_params

@struct.dataclass
class CLState:
    old_params: FrozenDict
    reg_weights: FrozenDict


def init_cl_state(params: FrozenDict, regularize_critic: bool = False) -> CLState:
    """Initialize old_params = current params, reg_weights = all 1 (or 0 if skipping critic)."""
    old_params = copy_params(params)
    reg_weights = build_reg_weights(params, regularize_critic)
    return CLState(old_params=old_params, reg_weights=reg_weights)


def update_cl_state(cl_state: CLState, new_params: FrozenDict) -> CLState:
    """
    When starting a new task, overwrite old_params with new_params.
    Keep the same reg_weights. (Do not accumulate from earlier tasks.)
    """
    return CLState(
        old_params=copy_params(new_params),
        reg_weights=cl_state.reg_weights
    )


def build_reg_weights(params: FrozenDict, regularize_critic: bool = False) -> FrozenDict:
    def _assign_reg_weight(path, x):
        # Join the keys in the path to a string.
        path_str = "/".join(str(key) for key in path)
        # Exclude head parameters: do not regularize if parameter is in actor_head or critic_head.
        if "actor_head" in path_str or "critic_head" in path_str:
            return jnp.zeros_like(x)
        # If we're not regularizing the critic, then exclude any parameter from critic branches.
        if not regularize_critic and "critic" in path_str.lower():
            return jnp.zeros_like(x)
        # Otherwise, regularize (the trunk).
        return jnp.ones_like(x)

    return jax.tree_util.tree_map_with_path(_assign_reg_weight, params)


def compute_l2_reg_loss(params: FrozenDict,
                        cl_state: CLState,
                        seq_idx: int,
                        cl_reg_coef: float) -> jnp.ndarray:
    """L2 regularization vs. a single old_params, scaled by cl_reg_coef if seq_idx>0."""

    def _tree_loss(new_p, old_p, w):
        diff = new_p - old_p
        return jnp.sum(w * diff ** 2)

    # If seq_idx == 0, no penalty:
    coef = jnp.where(seq_idx > 0, cl_reg_coef, 0.0)

    loss_tree = jax.tree_util.tree_map(
        lambda p, op, w: _tree_loss(p, op, w),
        params, cl_state.old_params, cl_state.reg_weights
    )
    total_reg_loss = jax.tree_util.tree_reduce(lambda acc, x: acc + x, loss_tree, 0.0)

    # Count how many are actually being penalized (where reg_weights != 0).
    def _count_params(p, w):
        return jnp.sum(w)  # ignoring partial matches

    count_tree = jax.tree_util.tree_map(_count_params, params, cl_state.reg_weights)
    param_count = jax.tree_util.tree_reduce(lambda acc, x: acc + x, count_tree, 0.0)
    param_count = jnp.maximum(param_count, 1.0)

    return coef * (total_reg_loss / param_count)


