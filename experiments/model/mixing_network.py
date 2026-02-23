import numpy as np

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class MixingNetwork(nn.Module):
    """
    QMIX monotonic mixing network.

    Takes individual agent Q-values and the global state, and combines them
    into a scalar Q_total via a two-layer hypernetwork with non-negative weights.
    Non-negativity enforces the IGM (Individual-Global-Max) property:
        argmax_a Q_total(s, a) = (argmax_{a1} Q_1, ..., argmax_{an} Q_n)

    Architecture (standard QMIX):
      Layer 1: W1 = |HyperW1(state)|, b1 = HyperB1(state)
               hidden = ELU(agent_qs @ W1 + b1)
      Layer 2: W2 = |HyperW2(state)|, b2 = MLP(state)
               Q_total = sum(hidden * W2) + b2
    """

    num_agents: int
    embed_dim: int = 32

    @nn.compact
    def __call__(self, agent_qs: jnp.ndarray, global_state: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            agent_qs:     (batch, num_agents)  individual Q-values for chosen actions
            global_state: (batch, state_dim)   concatenated (raw) agent observations

        Returns:
            q_total: (batch,)
        """
        batch = agent_qs.shape[0]

        # ── Layer 1 ───────────────────────────────────────────────────────────
        w1 = nn.Dense(self.num_agents * self.embed_dim, name="hw1",
                      kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(global_state)
        w1 = jnp.abs(w1).reshape(batch, self.num_agents, self.embed_dim)  # (B, A, E)

        b1 = nn.Dense(self.embed_dim, name="hb1",
                      kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(global_state)  # (B, E)

        hidden = jnp.einsum("ba,bae->be", agent_qs, w1) + b1  # (B, E)
        hidden = nn.elu(hidden)

        # ── Layer 2 ───────────────────────────────────────────────────────────
        w2 = nn.Dense(self.embed_dim, name="hw2",
                      kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(global_state)
        w2 = jnp.abs(w2)  # (B, E)  non-negative weights

        # Bias: small non-linear transform of state (no non-negativity required)
        b2 = nn.Dense(self.embed_dim, name="hb2_hidden",
                      kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(global_state)
        b2 = nn.relu(b2)
        b2 = nn.Dense(1, name="hb2_out",
                      kernel_init=orthogonal(0.01), bias_init=constant(0.0))(b2)  # (B, 1)

        q_total = jnp.sum(hidden * w2, axis=-1, keepdims=True) + b2  # (B, 1)
        return q_total.squeeze(-1)  # (B,)
