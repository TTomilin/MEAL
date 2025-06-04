import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


class Actor(nn.Module):
    """
    Actor network for MAPPO.
    
    This network takes observations as input and outputs a 
    categorical distribution over actions.
    """
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
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

        # Output layer to produce logits for the action distribution
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(x)

        # Create a categorical distribution using the logits
        pi = distrax.Categorical(logits=logits)
        return pi

class Critic(nn.Module):
    '''
    Critic network that estimates the value function
    '''
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
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
        
        # Output layer - produces a single value
        critic = nn.Dense(
            1, 
            kernel_init=orthogonal(1.0), 
            bias_init=constant(0.0)
        )(critic)
        
        # Remove the unnecessary dimension
        value = jnp.squeeze(critic, axis=-1)
        
        return value