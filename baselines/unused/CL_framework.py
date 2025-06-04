# Framework for continual learning experiments

import os
import sys
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax
import hydra
import distrax
import wandb
from omegaconf import OmegaConf
import gc
from jax import clear_caches
from typing import Sequence, NamedTuple, Any
from functools import partial
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze

import jax_marl
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.environments.overcooked_environment import overcooked_layouts
from baselines.utils import Transition, batchify, unbatchify, pad_observation_space, sample_discrete_action, get_rollout_for_visualization, visualize_environments
from jax_marl.wrappers.baselines import LogWrapper
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter

from baselines.ippo_algorithm import Config, ippo_train
from baselines.algorithms import ActorCritic

import tyro
from dotenv import load_dotenv




def initialize_networks(config, env, rng):
    '''
    Initialize the appropriate networks based on the algorithm to be used
    @param config: the configuration dictionary
    @param env: the environment
    @return: the initialized networks
    '''

    print("In initializing networks")

    # get the algorithm
    algorithm = config.alg_name
     
    if algorithm == "ippo":

        def linear_schedule(count):
            '''
            Linearly decays the learning rate depending on the number of minibatches and number of epochs
            returns the learning rate
            '''
            frac = 1.0 - ((count // (config.num_minibatches * config.update_epochs)) / config.num_updates)
            return config.lr * frac
        
        # create the actor-critic network
        network = ActorCritic(env.action_space().n, activation=config.activation)

        # initialize the network
        rng, network_rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape).flatten()
        network_params = network.init(network_rng, init_x)
        
        # Initialize the optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm), 
            optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
        )

        # Create the TrainState object
        train_state = TrainState.create(
        apply_fn=jax.jit(network.apply),   #CHECK IF THIS IS POSSIBLE 
            params=network_params,
            tx=tx,
        )

        return network, train_state, rng
    else:
        raise ValueError("Algorithm not recognized")
    

########################################################################################
########################################################################################
########################################################################################
########################################################################################

def make_train_fn(config: Config):
    """
    Create the training function for the continual learning experiment.
    @param config: the configuration dictionary
    """

    # @partial(jax.jit, static_argnums=(0,))
    def train_sequence(rng):
        '''
        Train on a sequence of tasks.
        @param rng: the random key for the experiment
        @return: the training output
        '''

        # Pad the environments to get a uniform shape
        padded_envs = pad_observation_space(config)
        envs = []
        for env_layout in padded_envs:
            # Create the environment object
            env = jax_marl.make(config.env_name, layout=env_layout)
            env = LogWrapper(env, replace_info=False)
            envs.append(env)
        
        print("Created environments")
        
        # add configuration items
        temp_env = envs[0]
        config.num_actors = temp_env.num_agents * config.num_envs
        config.num_updates = config.total_timesteps // config.num_steps // config.num_envs
        config.minibatch_size = (config.num_actors * config.num_steps) // config.num_minibatches


        # freeze(config)
        # print("Config is frozen")

        # REWARD SHAPING IN NEW VERSION
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.,
            end_value=0.,
            transition_steps=config.reward_shaping_horizon
        )

        # Initialize the network
        network, train_state, rng = initialize_networks(config, envs[0], rng)

        print("Initialized networks")

        # loop over environments
        rng, *env_rngs = jax.random.split(rng, len(envs)+1)
        
        def loop_over_envs(rng, network, train_state, envs, config):
            '''
            Loop over the environments in the sequence.
            @param rng: the random key for the experiment
            @param train_state: the current state of the training
            @param envs: the environments in the sequence
            @return: the updated training state
            '''
            print("In loop over envs")

            for env, env_rng in zip(envs, env_rngs):
                print(f"Training on environment {env}")

                if config.alg_name == "ippo":
                    runner_state = ippo_train(network, train_state, env, env_rng, config)
                    train_state = runner_state[0]

                    print(f"Finished training on environment")
                else:
                    raise ValueError("Algorithm not recognized")

            return runner_state
        
        # apply the loop_over_envs function to the environments
        runner_state = loop_over_envs(rng, network, train_state, envs, config)

        return runner_state

    return train_sequence



# @hydra.main(version_base=None, config_path="config", config_name=config_name)
def main():
    # set the device to GPU
    jax.config.update("jax_platform_name", "gpu")

    # config = OmegaConf.to_container(config)

    config = tyro.cli(Config)

    # generate a sequence of tasks
    seq_length = config.seq_length
    strategy = config.strategy
    layouts = config.layouts
    config.env_kwargs, config.layout_name = generate_sequence(seq_length, strategy, layouts=layouts)


    for layout_config in config.env_kwargs:
        # Extract the layout name
        layout_name = layout_config["layout"]

        # Set the layout in the config
        layout_config["layout"] = overcooked_layouts[layout_name]
    
    # Initialize WandB
    load_dotenv()
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project='Continual_IPPO', 
        config=config,
        sync_tensorboard=True,
        mode=config.wandb_mode,
        name=f'{config.alg_name}_{config.seq_length}_{config.strategy}'
    )
    
    # freeze(config)
    
    # intialize the summary writer
    writer = SummaryWriter(f"runs/{config.alg_name}_{config.seq_length}_{config.strategy}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    # Create the training loop
    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config.seed)
        rngs = jax.random.split(rng, config.num_seeds)
        train_jit = jax.jit(make_train_fn(config))
        output = jax.vmap(train_jit)(rngs)
    
    print("Finished training")

if __name__ == "__main__":
    main()