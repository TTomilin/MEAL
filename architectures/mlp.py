import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal, constant


def choose_head(t: jnp.ndarray, n_heads: int, env_idx):
    env_idx = jnp.asarray(env_idx, jnp.int32)          # dynamic
    b, tot = t.shape
    base = tot // n_heads
    t3 = t.reshape(b, n_heads, base)
    return jnp.take(t3, env_idx, axis=1)


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

    # ------------------------------------------------------------------ helpers
    def _act(self):  # pick activation fn once
        return nn.relu if self.activation == "relu" else nn.tanh

    def _dense(self, n, name, gain):
        return nn.Dense(n, kernel_init=orthogonal(gain), bias_init=constant(0.0), name=name)

    def _layer_dormant_ratio(self, h: jnp.ndarray, tag: str):
        """h: (batch, hidden). Implements Sokar et al.'s τ-dormant metric."""
        # mean |activation| per unit over the batch
        m = jnp.mean(jnp.abs(h), axis=0)             # shape: (H,)
        denom = jnp.mean(m) + 1e-12
        s = m / denom                                # normalized score
        ratio = jnp.mean(s <= self.dormant_threshold)
        # stash per-layer ratio for logging
        self.sow("metrics", "dormant_layer", {tag: ratio})
        return ratio

    # ------------------------------------------------------------------ forward
    @nn.compact
    def __call__(self, x, *, env_idx: int = 0):
        act = self._act()
        hid = 256 if self.big_network else 128

        per_layer_ratios = []  # collect to average later

        # -------- append task one-hot ----------------------------------------
        if self.use_task_id:
            ids = jnp.full((*x.shape[:-1],), env_idx)
            task_onehot = jax.nn.one_hot(ids, self.num_tasks)
            x = jnp.concatenate([x, task_onehot], axis=-1)

        # -------- shared trunk ------------------------------------------------
        if self.shared_backbone:
            for i in range(2 + self.big_network):  # 2 or 3 layers
                x = self._dense(hid, f"common_dense{i + 1}", np.sqrt(2))(x)
                x = act(x)
                if self.track_dormant_ratio:
                    per_layer_ratios.append(self._layer_dormant_ratio(x, f"shared_{i+1}"))
                if self.use_layer_norm:
                    x = nn.LayerNorm(name=f"common_ln{i + 1}", epsilon=1e-5)(x)
            trunk = x
            actor_in = critic_in = trunk
        else:
            # separate trunks – duplicate code for actor / critic
            def branch(prefix, inp):
                ratios = []
                for i in range(2 + self.big_network):
                    inp = self._dense(
                        hid, f"{prefix}_dense{i + 1}", np.sqrt(2))(inp)
                    inp = act(inp)
                    if self.track_dormant_ratio:
                        ratios.append(self._layer_dormant_ratio(inp, f"{prefix}_{i+1}"))
                    if self.use_layer_norm:
                        inp = nn.LayerNorm(name=f"{prefix}_ln{i + 1}", epsilon=1e-5)(inp)
                return inp, ratios

            actor_in, actor_ratios = branch("actor", x)
            critic_in, critic_ratios = branch("critic", x)
            if self.track_dormant_ratio:
                per_layer_ratios.extend(actor_ratios)
                per_layer_ratios.extend(critic_ratios)

        # -------- actor head --------------------------------------------------
        logits_dim = self.action_dim * \
            (self.num_tasks if self.use_multihead else 1)
        all_logits = self._dense(logits_dim, "actor_head", 0.01)(actor_in)

        logits = choose_head(all_logits, self.num_tasks,
                             env_idx) if self.use_multihead else all_logits
        pi = distrax.Categorical(logits=logits)

        # -------- critic head -------------------------------------------------
        vdim = 1 * (self.num_tasks if self.use_multihead else 1)
        all_v = self._dense(vdim, "critic_head", 1.0)(critic_in)
        v = choose_head(all_v, self.num_tasks,
                        env_idx) if self.use_multihead else all_v
        v = jnp.squeeze(v, -1)

        # -------- calculate dormant neuron ratio ------------------------------
        if self.track_dormant_ratio and per_layer_ratios:
            # average of per-layer dormant ratios
            dormant_ratio = jnp.mean(jnp.array(per_layer_ratios))
        else:
            dormant_ratio = jnp.array(0.0, dtype=jnp.float32)

        return pi, v, dormant_ratio
