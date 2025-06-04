import numpy as np
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal


# ───────────────────────────────── helper ────────────────────────────────────

def choose_head(tensor: jnp.ndarray, num_heads: int, env_idx: int) -> jnp.ndarray:
    """Select the slice that corresponds to *env_idx* out of *num_heads*.

    The input is shaped (B, H·num_heads). We reshape to (B, num_heads, H)
    and grab the env‑specific slice.
    """
    B, tot = tensor.shape
    assert tot % num_heads == 0, "channels must be divisible by num_heads"
    base = tot // num_heads
    return tensor.reshape(B, num_heads, base)[:, env_idx, :]


# ─────────────────────────────── base conv block ─────────────────────────────

class CNN(nn.Module):
    """Tiny 3‑layer CNN ➜ 64‑unit projection with optional LayerNorm."""

    name_prefix: str  # "shared" | "actor" | "critic"
    activation: str = "relu"
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh

        def conv(name: str, x, kernel):
            x = nn.Conv(32, kernel, name=f"{self.name_prefix}_{name}",
                        kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = act(x)
            return x

        x = conv("conv1", x, (5, 5))
        x = conv("conv2", x, (3, 3))
        x = conv("conv3", x, (3, 3))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, name=f"{self.name_prefix}_proj",
                     kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        if self.use_layer_norm:
            x = nn.LayerNorm(name=f"{self.name_prefix}_proj_ln", epsilon=1e-5)(x)
        return x


# ─────────────────────────────── Actor‑Critic ────────────────────────────────

class ActorCritic(nn.Module):
    """CNN‑MLP Actor‑Critic with optional per‑task conditioning and LayerNorm."""

    action_dim: int
    activation: str = "relu"

    # Continual‑learning knobs
    num_tasks: int = 1
    use_multihead: bool = False
    shared_backbone: bool = True
    big_network: bool = False
    use_task_id: bool = False
    regularize_heads: bool = True
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, obs, *, env_idx: int = 0):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # ─── encoders ────────────────────────────────────────────────────
        cnn_kwargs = dict(activation=self.activation, use_layer_norm=self.use_layer_norm)
        if self.shared_backbone:
            trunk = CNN("shared", **cnn_kwargs)(obs)
            actor_emb = critic_emb = trunk
        else:
            actor_emb = CNN("actor", **cnn_kwargs)(obs)
            critic_emb = CNN("critic", **cnn_kwargs)(obs)

        # ─── task one‑hot concat ─────────────────────────────────────────
        if self.use_task_id:
            idxs = jnp.full((actor_emb.shape[0],), env_idx)
            task_onehot = jax.nn.one_hot(idxs, num_classes=self.num_tasks)
            actor_emb = jnp.concatenate([actor_emb, task_onehot], axis=-1)
            critic_emb = jnp.concatenate([critic_emb, task_onehot], axis=-1)

        # ─── actor branch ────────────────────────────────────────────────
        a = nn.Dense(128, name="actor_dense1",
                     kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_emb)
        a = act_fn(a)
        if self.use_layer_norm:
            a = nn.LayerNorm(name="actor_dense1_ln", epsilon=1e-5)(a)

        logits_dim = self.action_dim * (self.num_tasks if self.use_multihead else 1)
        logits_all = nn.Dense(logits_dim, name="actor_head",
                              kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        logits = choose_head(logits_all, self.num_tasks, env_idx) if self.use_multihead else logits_all
        pi = distrax.Categorical(logits=logits)

        # ─── critic branch ───────────────────────────────────────────────
        c = nn.Dense(128, name="critic_dense1",
                     kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic_emb)
        c = act_fn(c)
        if self.use_layer_norm:
            c = nn.LayerNorm(name="critic_dense1_ln", epsilon=1e-5)(c)

        vdim = 1 * (self.num_tasks if self.use_multihead else 1)
        v_all = nn.Dense(vdim, name="critic_head",
                         kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        v = choose_head(v_all, self.num_tasks, env_idx) if self.use_multihead else v_all
        v = jnp.squeeze(v, axis=-1)

        return pi, v
