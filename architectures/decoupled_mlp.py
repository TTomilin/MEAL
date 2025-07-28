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

    @nn.compact
    def __call__(self, x, *, env_idx: int = 0):
        # Choose the activation function based on input parameter.
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # First hidden layer
        x = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = act_fn(x)

        # Second hidden layer
        x = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = act_fn(x)

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

    @nn.compact
    def __call__(self, x, *, env_idx: int = 0):
        # Choose activation function
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # First hidden layer
        critic = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        critic = activation(critic)

        # Second hidden layer
        critic = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)

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
