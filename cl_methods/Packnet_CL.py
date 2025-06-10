import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from cl_methods.base import RegCLMethod
from cl_methods.Packnet import Packnet, PacknetState


class PacknetCL(RegCLMethod):
    """Wrapper to use PackNet with the generic IPPO_CL script."""

    name = "packnet"

    def __init__(self, seq_length, prune_instructions=0.5, train_finetune_split=(1, 1)):
        self.packnet = Packnet(seq_length=seq_length,
                               prune_instructions=prune_instructions,
                               train_finetune_split=train_finetune_split)

    # ------------------------------------------------------------------
    # Interface required by RegCLMethod
    # ------------------------------------------------------------------
    def init_state(self, params: FrozenDict, regularize_critic: bool, regularize_heads: bool):
        mask_tree = self.packnet.init_mask_tree(params["params"])
        return PacknetState(masks=mask_tree, current_task=0, train_mode=True)

    def update_state(self, cl_state: PacknetState, new_params: FrozenDict, new_importance: FrozenDict):
        # PackNet updates are handled externally after each task
        return cl_state

    def penalty(self, params: FrozenDict, cl_state: PacknetState, coef: float):
        # PackNet does not use a regularisation penalty
        return jnp.array(0.0, dtype=jnp.float32)

    def compute_importance(self, *args, **kwargs):
        # No importance weights needed for PackNet
        return None

    # ------------------------------------------------------------------
    # Convenience wrappers for specific PackNet functionality
    # ------------------------------------------------------------------
    def on_backwards_end(self, state: PacknetState, train_state, params_copy):
        return self.packnet.on_backwards_end(state, train_state, params_copy)

    def on_train_end(self, params, state: PacknetState):
        return self.packnet.on_train_end(params, state)

    def on_finetune_end(self, state: PacknetState):
        return self.packnet.on_finetune_end(state)
