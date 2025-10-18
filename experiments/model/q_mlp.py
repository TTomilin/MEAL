import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal


def choose_head(t: jnp.ndarray, n_heads: int, env_idx: int):
    b, tot = t.shape
    base = tot // n_heads
    return t.reshape(b, n_heads, base)[:, env_idx, :]


class MLPEncoder(nn.Module):
    hidden_size: int = 128
    activation: str = "relu"
    num_layers: int = 2
    big_network: bool = False
    use_layer_norm: bool = False

    def _act(self):
        return nn.relu if self.activation == "relu" else nn.tanh

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = x.reshape((x.shape[0], -1))
        act = self._act()
        hid = 256 if self.big_network else self.hidden_size

        for i in range(self.num_layers + self.big_network):
            x = nn.Dense(
                hid,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name=f"mlp_dense{i + 1}",
            )(x)
            x = act(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(epsilon=1e-5, name=f"mlp_ln{i + 1}")(x)
        return x


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 64
    # ––– ActorCritic-style switches –––
    encoder_type: str = "mlp"  # "mlp" | "cnn"
    activation: str = "relu"
    num_tasks: int = 1
    use_multihead: bool = False
    use_task_id: bool = False
    big_network: bool = False
    use_layer_norm: bool = False

    def _act(self):
        return nn.relu if self.activation == "relu" else nn.tanh

    def _encoder(self):
        return MLPEncoder(
            hidden_size=self.hidden_size,
            activation=self.activation,
            big_network=self.big_network,
            use_layer_norm=self.use_layer_norm,
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, env_idx: int = 0):
        x = self._encoder()(x)  # <-- swapped in!
        act = self._act()
        hid = 256 if self.big_network else self.hidden_size

        # optional additional layer
        x = nn.Dense(hid,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)

        # optionally append one-hot task id
        if self.use_task_id:
            ids = jnp.full((x.shape[0],), env_idx)
            x = jnp.concatenate([x, jax.nn.one_hot(ids, self.num_tasks)], axis=-1)

        out_dim = self.action_dim * (self.num_tasks if self.use_multihead else 1)
        all_q = nn.Dense(out_dim,
                         kernel_init=orthogonal(0.01),
                         bias_init=constant(0.0),
                         name="q_head")(x)
        q_values = choose_head(all_q, self.num_tasks, env_idx) if self.use_multihead else all_q
        return q_values
