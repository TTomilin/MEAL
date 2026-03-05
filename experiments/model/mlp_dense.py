import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
import distrax
from experiments.utils import get_layer_name

def choose_head(t: jnp.ndarray, n_heads: int, env_idx: int):
    b, tot = t.shape
    base = tot // n_heads
    return t.reshape(b, n_heads, base)[:, env_idx, :]

class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"
    num_tasks: int = 1
    use_multihead: bool = False
    shared_backbone: bool = False
    big_network: bool = False
    use_task_id: bool = False
    regularize_heads: bool = True
    use_layer_norm: bool = False
    track_dormant_ratio: bool = True
    dormant_threshold: float = 0.01

    @nn.compact
    def __call__(self, x, *, env_idx: int = 0):

        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        activations = [] if self.track_dormant_ratio else None
        actor_x = x
        critic_x = x

        # ------------------------------------------------------------------
        # Task ID one-hot (both actor & critic)
        # ------------------------------------------------------------------
        if self.use_task_id:
            ids = jnp.full((x.shape[0],), env_idx)
            task_onehot = jax.nn.one_hot(ids, self.num_tasks)
            actor_x = jnp.concatenate([actor_x, task_onehot], axis=-1)
            critic_x = jnp.concatenate([critic_x, task_onehot], axis=-1)

        # =========================
        # ======== ACTOR ==========
        # =========================
        a = nn.Dense(128,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0),
                     name=get_layer_name('actor', nn.Dense, 1))(actor_x)
        a = act_fn(a)
        if self.track_dormant_ratio:
            activations.append(a)
        if self.use_layer_norm:
            a = nn.LayerNorm(name=get_layer_name('actor', nn.LayerNorm, 1))(a)

        a = nn.Dense(128,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0),
                     name=get_layer_name('actor', nn.Dense, 2))(a)
        a = act_fn(a)
        if self.track_dormant_ratio:
            activations.append(a)
        if self.use_layer_norm:
            a = nn.LayerNorm(name=get_layer_name('actor', nn.LayerNorm, 2))(a)

        # -------- actor head --------------------------------------------------
        logits_dim = self.action_dim * (self.num_tasks if self.use_multihead else 1)
        all_logits = nn.Dense(
            logits_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name=get_layer_name('actor', nn.Dense, 3)
        )(a)

        logits = (
            choose_head(all_logits, self.num_tasks, env_idx)
            if self.use_multihead else all_logits
        )

        policy = distrax.Categorical(logits=logits)

        # =========================
        # ======== CRITIC =========
        # =========================
        c = nn.Dense(128,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0),
                     name=get_layer_name('critic', nn.Dense, 1))(critic_x)
        c = act_fn(c)
        if self.track_dormant_ratio:
            activations.append(c)
        if self.use_layer_norm:
            c = nn.LayerNorm(name=get_layer_name('critic', nn.LayerNorm, 1))(c)

        c = nn.Dense(128,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0),
                     name=get_layer_name('critic', nn.Dense, 2))(c)
        c = act_fn(c)
        if self.track_dormant_ratio:
            activations.append(c)
        if self.use_layer_norm:
            c = nn.LayerNorm(name=get_layer_name('critic', nn.LayerNorm, 2))(c)

        vdim = 1 * (self.num_tasks if self.use_multihead else 1)
        all_v = nn.Dense(
            vdim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name=get_layer_name('critic', nn.Dense, 3)
        )(c)

        v = (
            choose_head(all_v, self.num_tasks, env_idx)
            if self.use_multihead else all_v
        )

        value = jnp.squeeze(v, axis=-1)

        # ------------------------------------------------------------------
        # Dormant ratio across BOTH actor & critic hidden layers
        # ------------------------------------------------------------------
        dormant_ratio = 0.0
        if self.track_dormant_ratio and activations:
            flat = jnp.concatenate([layer.reshape(-1) for layer in activations])
            dormant = jnp.sum(jnp.abs(flat) < self.dormant_threshold)
            dormant_ratio = dormant / flat.size

        return policy, value, dormant_ratio