import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal, constant


def choose_head(t: jnp.ndarray, n_heads: int, env_idx: int):
    b, tot = t.shape
    base = tot // n_heads
    return t.reshape(b, n_heads, base)[:, env_idx, :]


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"
    # continual-learning bells & whistles
    num_tasks: int = 1
    use_multihead: bool = False
    shared_backbone: bool = False
    big_network: bool = False
    use_task_id: bool = False
    regularize_heads: bool = True
    use_layer_norm: bool = False

    # ------------------------------------------------------------------ helpers
    def _act(self):  # pick activation fn once
        return nn.relu if self.activation == "relu" else nn.tanh

    def _dense(self, n, name, gain):
        return nn.Dense(n,
                        kernel_init=orthogonal(gain),
                        bias_init=constant(0.0),
                        name=name)

    # ------------------------------------------------------------------ forward
    @nn.compact
    def __call__(self, x, *, env_idx: int = 0):
        act = self._act()
        hid = 256 if self.big_network else 128

        # -------- shared trunk ------------------------------------------------
        if self.shared_backbone:
            for i in range(2 + self.big_network):  # 2 or 3 layers
                x = self._dense(hid, f"common_dense{i + 1}", np.sqrt(2))(x)
                x = act(x)
                if self.use_layer_norm:
                    x = nn.LayerNorm(name=f"common_ln{i + 1}", epsilon=1e-5)(x)
            trunk = x
            actor_in = critic_in = trunk
        else:
            # separate trunks – duplicate code for actor / critic
            def branch(prefix, inp):
                for i in range(2 + self.big_network):
                    inp = self._dense(hid, f"{prefix}_dense{i + 1}", np.sqrt(2))(inp)
                    inp = act(inp)
                    if self.use_layer_norm:
                        inp = nn.LayerNorm(name=f"{prefix}_ln{i + 1}", epsilon=1e-5)(inp)
                return inp

            actor_in = branch("actor", x)
            critic_in = branch("critic", x)

        # -------- append task one-hot ----------------------------------------
        if self.use_task_id:
            ids = jnp.full((x.shape[0],), env_idx)
            task_onehot = jax.nn.one_hot(ids, self.num_tasks)
            actor_in = jnp.concatenate([actor_in, task_onehot], axis=-1)
            critic_in = jnp.concatenate([critic_in, task_onehot], axis=-1)

        # -------- actor head --------------------------------------------------
        logits_dim = self.action_dim * (self.num_tasks if self.use_multihead else 1)
        all_logits = self._dense(logits_dim, "actor_head", 0.01)(actor_in)
        logits = choose_head(all_logits, self.num_tasks, env_idx) if self.use_multihead else all_logits
        pi = distrax.Categorical(logits=logits)

        # -------- critic head -------------------------------------------------
        vdim = 1 * (self.num_tasks if self.use_multihead else 1)
        all_v = self._dense(vdim, "critic_head", 1.0)(critic_in)
        v = choose_head(all_v, self.num_tasks, env_idx) if self.use_multihead else all_v
        v = jnp.squeeze(v, -1)

        return pi, v
