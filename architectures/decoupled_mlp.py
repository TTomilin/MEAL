import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
import distrax


def choose_head(t: jnp.ndarray, n_heads: int, env_idx: int):
    b, tot = t.shape
    base = tot // n_heads
    return t.reshape(b, n_heads, base)[:, env_idx, :]


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

        x = conv("conv1", x, (3, 3))
        x = conv("conv2", x, (3, 3))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, name=f"{self.name_prefix}_proj",
                     kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        if self.use_layer_norm:
            x = nn.LayerNorm(name=f"{self.name_prefix}_proj_ln", epsilon=1e-5)(x)
        return x


class Actor(nn.Module):
    """
    Actor network for MAPPO with multitask support.

    This network takes observations as input and outputs a 
    categorical distribution over actions.
    """
    action_dim: int
    activation: str = "tanh"
    # continual-learning bells & whistles
    num_tasks: int = 1
    use_multihead: bool = False
    use_task_id: bool = False
    use_cnn: bool = False
    use_layer_norm: bool = False
    # agent ID one-hot encoding
    use_agent_id: bool = False
    num_agents: int = 2
    num_envs: int = 16  # Number of parallel environments

    @nn.compact
    def __call__(self, x, *, env_idx: int = 0):
        # Choose the activation function based on input parameter.
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # CNN feature extraction if enabled
        if self.use_cnn:
            x = CNN("actor", activation=self.activation, use_layer_norm=self.use_layer_norm)(x)

        # -------- append agent ID one-hot ------------------------------------
        if self.use_agent_id:
            # Create agent IDs based on batch position
            # In MAPPO, observations are batched with all envs for agent 0, then all envs for agent 1, etc.
            # So agent_id = batch_position // num_envs
            batch_size = x.shape[0]
            agent_ids = jnp.arange(batch_size) // self.num_envs
            agent_onehot = jax.nn.one_hot(agent_ids, self.num_agents)
            x = jnp.concatenate([x, agent_onehot], axis=-1)

        # First hidden layer
        x = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = act_fn(x)
        if self.use_layer_norm:
            x = nn.LayerNorm(name="actor_dense1_ln", epsilon=1e-5)(x)

        # Second hidden layer
        x = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = act_fn(x)
        if self.use_layer_norm:
            x = nn.LayerNorm(name="actor_dense2_ln", epsilon=1e-5)(x)

        # -------- append task one-hot ----------------------------------------
        if self.use_task_id:
            ids = jnp.full((x.shape[0],), env_idx)
            task_onehot = jax.nn.one_hot(ids, self.num_tasks)
            x = jnp.concatenate([x, task_onehot], axis=-1)

        # -------- actor head --------------------------------------------------
        logits_dim = self.action_dim * (self.num_tasks if self.use_multihead else 1)
        all_logits = nn.Dense(
            logits_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(x)

        logits = choose_head(all_logits, self.num_tasks, env_idx) if self.use_multihead else all_logits

        # Create a categorical distribution using the logits
        pi = distrax.Categorical(logits=logits)
        return pi


class Critic(nn.Module):
    '''
    Critic network that estimates the value function with multitask support
    '''
    activation: str = "tanh"
    # continual-learning bells & whistles
    num_tasks: int = 1
    use_multihead: bool = False
    use_task_id: bool = False
    use_cnn: bool = False
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, *, env_idx: int = 0):
        # Choose activation function
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # CNN feature extraction if enabled
        if self.use_cnn:
            x = CNN("critic", activation=self.activation, use_layer_norm=self.use_layer_norm)(x)

        # First hidden layer
        critic = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        if self.use_layer_norm:
            critic = nn.LayerNorm(name="critic_dense1_ln", epsilon=1e-5)(critic)

        # Second hidden layer
        critic = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        if self.use_layer_norm:
            critic = nn.LayerNorm(name="critic_dense2_ln", epsilon=1e-5)(critic)

        # -------- append task one-hot ----------------------------------------
        if self.use_task_id:
            ids = jnp.full((x.shape[0],), env_idx)
            task_onehot = jax.nn.one_hot(ids, self.num_tasks)
            critic = jnp.concatenate([critic, task_onehot], axis=-1)

        # -------- critic head -------------------------------------------------
        vdim = 1 * (self.num_tasks if self.use_multihead else 1)
        all_v = nn.Dense(
            vdim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)

        v = choose_head(all_v, self.num_tasks, env_idx) if self.use_multihead else all_v
        value = jnp.squeeze(v, axis=-1)

        return value
