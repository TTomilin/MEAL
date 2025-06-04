import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from datetime import datetime
import copy
import pickle
import math
import flax
import flax.linen as nn
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from typing import Sequence, NamedTuple, Any, Optional, List, Tuple
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper

from jax_marl.registration import make
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import wandb
from functools import partial
from dataclasses import dataclass, field
import tyro
from tensorboardX import SummaryWriter
from pathlib import Path

class CBPDense(nn.Module):
    """Dense layer *plus* CBP bookkeeping (utility, age, counter)."""
    features: int
    name: str
    eta: float = 0.0   # utility decay, off by default
    kernel_init: callable = orthogonal(np.sqrt(2))
    bias_init: callable = constant(0.0)
    activation: str = "tanh"

    def setup(self):
        self.dense = nn.Dense(self.features,
                              kernel_init=self.kernel_init,
                              bias_init=self.bias_init,
                              name=f"{self.name}_d")
        if self.activation == "relu":
            self.act_fn = nn.relu
        elif self.activation == "tanh":
            self.act_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation function {self.activation}")

        self.variable("cbp", f"{self.name}_util", lambda: jnp.zeros((self.features,)))
        self.variable("cbp", f"{self.name}_age", lambda: jnp.zeros((self.features,), jnp.int32))
        self.variable("cbp", f"{self.name}_ctr", lambda: jnp.zeros(()))  # scalar float

    def __call__(self, x, next_kernel, train: bool):
        # Forward pass
        y = self.dense(x)
        h = self.act_fn(y)
        if train:
            # fetch existing variables
            util = self.get_variable("cbp", f"{self.name}_util")
            age  = self.get_variable("cbp", f"{self.name}_age")

            # ------------ CBP utility + age update ------------
            w_abs_sum = jnp.sum(jnp.abs(next_kernel), axis=1)  # (n_units,)  # Flax weights are saved as (in, out). PyTorch is (out, in).
            abs_neuron_output = jnp.abs(h)  
            contrib = jnp.mean(abs_neuron_output, axis=0) * w_abs_sum   # contribution =  |h| * Σ|w_out|  Mean over the batch (dim 0 of h)
            new_util = util * self.eta + (1 - self.eta) * contrib

            self.put_variable("cbp", f"{self.name}_util", new_util)
            self.put_variable("cbp", f"{self.name}_age",  age + 1)
        return h

def weight_reinit(key, shape):
    """Should be the same weight initializer used at model start (orthogonal(√2))."""
    return orthogonal(np.sqrt(2))(key, shape)


def cbp_step(
        params: FrozenDict,
        cbp_state: FrozenDict,
        *,
        rng: jax.random.PRNGKey,
        maturity: int,
        rho: float,   # replacement rate (0.0 - 1.0)
    ) -> Tuple[FrozenDict, FrozenDict, jax.random.PRNGKey]:
    """
    Pure JAX function: one CBP maintenance step.
    * increments replacement counter for every layer
    * performs (possibly zero) replacements based on counter
    """
    p, s = unfreeze(params), unfreeze(cbp_state)  # python dicts (outside jit)

    def _layer(layer_name, next_layer_name, rng):
        next_layer_no_d = next_layer_name.split("_d")[0]  # remove "_d" from name
        util = s[layer_name][f"{layer_name}_util"]
        age = s[layer_name][f"{layer_name}_age"]
        ctr = s[layer_name][f"{layer_name}_ctr"]
        mature_mask = age > maturity
        n_eligible = jnp.sum(mature_mask).astype(int)
        ctr += n_eligible * rho

        # number of whole neurons to replace
        n_rep = jnp.floor(ctr).astype(int)
        ctr -= jnp.float32(n_rep)
        s[layer_name][f"{layer_name}_ctr"] = ctr

        util_masked = jnp.where(mature_mask, util, jnp.inf)
        sorted_idx = jnp.argsort(util_masked)  # indices ascending by util
        # build inverse permutation to get each index’s rank
        ranks = jnp.zeros_like(sorted_idx).at[sorted_idx].set(jnp.arange(util.shape[0]))
        rep_mask = (ranks < n_rep) & mature_mask  # boolean mask of size n_units

        W_in = p[layer_name][f"{layer_name}_d"]["kernel"]
        b_in = p[layer_name][f"{layer_name}_d"]["bias"]
        if '_out' in next_layer_name:  # handle last layer (actor_out, critic_out) differently
            W_out = p[next_layer_name]["kernel"]
        else:
            W_out = p[next_layer_no_d][next_layer_name]["kernel"]
        rng, k_init = jax.random.split(rng)
        new_W = weight_reinit(k_init, W_in.shape)

        # apply mask
        W_in = jnp.where(rep_mask[None, :], new_W, W_in)
        b_in = jnp.where(rep_mask, 0.0, b_in)
        W_out = jnp.where(rep_mask[:, None], 0.0, W_out)

        # reset bookkeeping
        util = jnp.where(rep_mask, 0.0, util)
        age = jnp.where(rep_mask, 0, age)

        # write back
        p[layer_name][f"{layer_name}_d"]["kernel"] = W_in
        p[layer_name][f"{layer_name}_d"]["bias"] = b_in
        if '_out' in next_layer_name:
            p[next_layer_name]["kernel"] = W_out
        else:
            p[next_layer_no_d][next_layer_name]["kernel"] = W_out
        s[layer_name][f"{layer_name}_util"] = util
        s[layer_name][f"{layer_name}_age"] = age

        return rng

    rng = _layer("actor_fc1", "actor_fc2_d", rng)
    rng = _layer("actor_fc2", "actor_out", rng)
    rng = _layer("critic_fc1", "critic_fc2_d", rng)
    rng = _layer("critic_fc2", "critic_out", rng)

    return freeze(p), freeze(s), rng


class TrainStateCBP(TrainState):
    """TrainState with an *extra* collection that stores CBP variables."""
    cbp_state: FrozenDict
