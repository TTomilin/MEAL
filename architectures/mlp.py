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
        return nn.Dense(n,
                        kernel_init=orthogonal(gain),
                        bias_init=constant(0.0),
                        name=name)

    # ------------------------------------------------------------------ forward
    @nn.compact
    def __call__(self, x, *, env_idx: int = 0):
        act = self._act()
        hid = 256 if self.big_network else 128

        # Initialize list to collect activations for dormant ratio calculation
        activations = [] if self.track_dormant_ratio else None

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
                    activations.append(x)
                if self.use_layer_norm:
                    x = nn.LayerNorm(name=f"common_ln{i + 1}", epsilon=1e-5)(x)
            trunk = x
            actor_in = critic_in = trunk
        else:
            # separate trunks – duplicate code for actor / critic
            def branch(prefix, inp):
                branch_activations = []
                for i in range(2 + self.big_network):
                    inp = self._dense(
                        hid, f"{prefix}_dense{i + 1}", np.sqrt(2))(inp)
                    inp = act(inp)
                    if self.track_dormant_ratio:
                        branch_activations.append(inp)
                    if self.use_layer_norm:
                        inp = nn.LayerNorm(
                            name=f"{prefix}_ln{i + 1}", epsilon=1e-5)(inp)
                return inp, branch_activations

            actor_in, actor_activations = branch("actor", x)
            critic_in, critic_activations = branch("critic", x)

            if self.track_dormant_ratio:
                activations.extend(actor_activations)
                activations.extend(critic_activations)

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
        dormant_ratio = 0.0
        if self.track_dormant_ratio and activations:
            # Concatenate all activations and calculate dormant ratio
            all_activations = jnp.concatenate(
                [act_layer.flatten() for act_layer in activations])
            # Count neurons with activation below threshold
            dormant_count = jnp.sum(
                jnp.abs(all_activations) < self.dormant_threshold)
            total_count = all_activations.size
            dormant_ratio = dormant_count / total_count

        return pi, v, dormant_ratio
