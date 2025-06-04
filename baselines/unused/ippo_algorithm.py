""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Optional
from flax.training.train_state import TrainState
# from flax.struct import dataclass
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jax_marl
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.environments.overcooked_environment import Overcooked
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from tensorboardX import SummaryWriter


import matplotlib.pyplot as plt
import wandb

from baselines.utils import Transition, batchify, unbatchify, pad_observation_space
from baselines.evaluation import evaluate_model
from functools import partial
from dataclasses import dataclass


############################
##### HELPER CLASSES   #####
############################


class ActorCritic(nn.Module):
    '''
    Class to define the actor-critic networks used in IPPO. Each agent has its own actor-critic network
    '''
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # ACTOR  
        actor_mean = nn.Dense(
            64, # number of neurons
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0) # sets the bias initialization to a constant value of 0
        )(x) # applies a dense layer to the input x

        actor_mean = activation(actor_mean) # applies the activation function to the output of the dense layer

        actor_mean = nn.Dense(
            64, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(actor_mean)

        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, 
            kernel_init=orthogonal(0.01), 
            bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean) # creates a categorical distribution over all actions (the logits are the output of the actor network)

        # CRITIC
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)

        critic = activation(critic)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)

        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        # returns the policy (actor) and state-value (critic) networks
        value = jnp.squeeze(critic, axis=-1)
        return pi, value #squeezed to remove any unnecessary dimensions
    

@dataclass(unsafe_hash=True)
class Config:
    lr: float = 1e-4
    num_envs: int = 16
    num_steps: int = 128
    total_timesteps: float = 5e6
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_shaping_horizon: float = 2.5e6
    activation: str = "tanh"
    env_name: str = "overcooked"
    alg_name: str = "ippo"
    
    seq_length: int = 5
    strategy: str = "random"
    layouts: Optional[Sequence[str]] = None
    env_kwargs: Optional[Sequence[dict]] = None
    layout_name: Optional[Sequence[str]] = None
    
    anneal_lr: bool = False
    seed: int = 30
    num_seeds: int = 1
    
    # Wandb settings
    wandb_mode: str = "online"
    entity: Optional[str] = ""
    project: str = "COOX"

    # to be computed during runtime
    num_actors: int = 0
    num_updates: int = 0
    minibatch_size: int = 0


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray


##########################################################
##########################################################
#######            TRAINING FUNCTION               #######
##########################################################
##########################################################




@partial(jax.jit, static_argnums=(0, 2, 3))
def ippo_train(network: ActorCritic, 
               train_state: TrainState, 
               env: Overcooked, 
               rng: int, 
               config: Config):
    '''
    Trains the network using IPPO
    @param rng: random number generator 
    returns the runner state and the metrics
    '''

    def linear_schedule(count):
        '''
        Linearly decays the learning rate depending on the number of minibatches and number of epochs
        returns the learning rate
        '''
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    # REWARD SHAPING IN NEW VERSION
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config.reward_shaping_horizon
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
    )
    train_state = train_state.replace(tx=tx)
    
    # Initialize and reset the environments
    rng, env_rng = jax.random.split(rng)
    reset_rng = jax.random.split(env_rng, config.num_envs) 
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng) 
    
    
    def _update_step(runner_state, unused):
        '''
        perform a single update step in the training loop
        @param runner_state: the carry state that contains all important training information
        returns the updated runner state and the metrics 
        '''

        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            '''
            selects an action based on the policy, calculates the log probability of the action, 
            and performs the selected action in the environment
            @param runner_state: the current state of the runner
            returns the updated runner state and the transition
            '''
            # Unpack the runner state
            train_state, env_state, last_obs, update_step, rng = runner_state

            # SELECT ACTION
            rng, sample_rng = jax.random.split(rng)

            # prepare the observations for the network
            obs_batch = batchify(last_obs, env.agents, config.num_actors)
            
            # apply the policy network to the observations to get the suggested actions and their values
            pi, value = network.apply(train_state.params, obs_batch)

            # sample the actions from the policy distribution 
            action = pi.sample(seed=sample_rng)
            log_prob = pi.log_prob(action)

            # format the actions to be compatible with the environment
            env_act = unbatchify(action, env.agents, config.num_envs, env.num_agents)
            env_act = {k:v.flatten() for k,v in env_act.items()}
            
            # STEP ENV
            # split the random number generator for stepping the environment
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config.num_envs)
            
            # simultaniously step all environments with the selected actions (parallelized over the number of environments with vmap)
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                rng_step, env_state, env_act
            )

            # REWARD SHAPING IN NEW VERSION
            info["reward"] = reward["agent_0"]
            current_timestep = update_step * config.num_steps * config.num_envs
            reward = jax.tree_util.tree_map(lambda x,y: x+y * rew_shaping_anneal(current_timestep), reward, info["shaped_reward"])

            # format the outputs of the environment to a 'transition' structure that can be used for analysis
            # info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info) # this is gone in the new version

            transition = Transition(
                batchify(done, env.agents, config.num_actors).squeeze(), 
                action,
                value,
                batchify(reward, env.agents, config.num_actors).squeeze(),
                log_prob,
                obs_batch
            )

            runner_state = (train_state, env_state, obsv, update_step, rng)
            return runner_state, (transition, info)
        
        # Apply the _env_step function a series of times, while keeping track of the runner state
        runner_state, (traj_batch, info) = jax.lax.scan(
            f=_env_step, 
            init=runner_state, 
            xs=None, 
            length=config.num_steps
        )  

        # unpack the runner state that is returned after the scan function
        train_state, env_state, last_obs, update_step, rng = runner_state

        # reshape the last observation to be compatible with the network
        last_obs_batch = batchify(last_obs, env.agents, config.num_actors)

        # apply the network to the batch of observations to get the value of the last state
        _, last_val = network.apply(train_state.params, last_obs_batch)
        # this returns the value network for the last observation batch

        def _calculate_gae(traj_batch, last_val):
            '''
            calculates the generalized advantage estimate (GAE) for the trajectory batch
            @param traj_batch: the trajectory batch
            @param last_val: the value of the last state
            returns the advantages and the targets
            '''
            def _get_advantages(gae_and_next_value, transition):
                '''
                calculates the advantage for a single transition
                @param gae_and_next_value: the GAE and value of the previous state
                @param transition: the transition to calculate the advantage for
                returns the updated GAE and the advantage
                '''
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config.gamma * next_value * (1 - done) - value # calculate the temporal difference
                gae = (
                    delta
                    + config.gamma * config.gae_lambda * (1 - done) * gae
                ) # calculate the GAE (used instead of the standard advantage estimate in PPO)
                
                return (gae, value), gae
            
            # iteratively apply the _get_advantages function to calculate the advantage for each step in the trajectory batch
            _, advantages = jax.lax.scan(
                f=_get_advantages,
                init=(jnp.zeros_like(last_val), last_val),
                xs=traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        # calculate the generalized advantage estimate (GAE) for the trajectory batch
        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            '''
            performs a single update epoch in the training loop
            @param update_state: the current state of the update
            returns the updated update_state and the total loss
            '''
            
            def _update_minbatch(train_state, batch_info):
                '''
                performs a single update minibatch in the training loop
                @param train_state: the current state of the training
                @param batch_info: the information of the batch
                returns the updated train_state and the total loss
                '''
                # unpack the batch information
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    '''
                    calculates the loss of the network
                    @param params: the parameters of the network
                    @param traj_batch: the trajectory batch
                    @param gae: the generalized advantage estimate
                    @param targets: the targets
                    @param network: the network
                    returns the total loss and the value loss, actor loss, and entropy
                    '''
                    # apply the network to the observations in the trajectory batch
                    pi, value = network.apply(params, traj_batch.obs) 
                    log_prob = pi.log_prob(traj_batch.action)

                    # calculate critic loss 
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config.clip_eps, config.clip_eps) 
                    value_losses = jnp.square(value - targets) 
                    value_losses_clipped = jnp.square(value_pred_clipped - targets) 
                    value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()) 

                    # Calculate actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob) 
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8) 
                    loss_actor_unclipped = ratio * gae 
                    loss_actor_clipped = (
                        jnp.clip(
                            ratio,
                            1.0 - config.clip_eps,
                            1.0 + config.clip_eps,
                        )
                        * gae
                    ) 

                    loss_actor = -jnp.minimum(loss_actor_unclipped, loss_actor_clipped) # calculate the actor loss as the minimum of the clipped and unclipped actor loss
                    loss_actor = loss_actor.mean() # calculate the mean of the actor loss
                    entropy = pi.entropy().mean() # calculate the entropy of the policy 

                    total_loss = (
                        loss_actor
                        + config.vf_coef * value_loss
                        - config.ent_coef * entropy
                    )

                    return total_loss, (value_loss, loss_actor, entropy)

                # returns a function with the same parameters as loss_fn that calculates the gradient of the loss function
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                # call the grad_fn function to get the total loss and the gradients
                total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                # apply the gradients to the network
                train_state = train_state.apply_gradients(grads=grads)

                return train_state, total_loss
            
            
            # unpack the update_state (because of the scan function)
            train_state, traj_batch, advantages, targets, rng = update_state
            
            # set the batch size and check if it is correct
            batch_size = config.minibatch_size * config.num_minibatches
            assert (
                batch_size == config.num_steps * config.num_actors
            ), "batch size must be equal to number of steps * number of actors"
            
            # create a batch of the trajectory, advantages, and targets
            batch = (traj_batch, advantages, targets)          

            # reshape the batch to be compatible with the network
            batch = jax.tree_util.tree_map(
                f=(lambda x: x.reshape((batch_size,) + x.shape[2:])), tree=batch
            )
            # split the random number generator for shuffling the batch
            rng, _rng = jax.random.split(rng)

            # creates random sequences of numbers from 0 to batch_size, one for each vmap 
            permutation = jax.random.permutation(_rng, batch_size)

            # shuffle the batch
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            ) # outputs a tuple of the batch, advantages, and targets shuffled 

            minibatches = jax.tree_util.tree_map(
                f=(lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:]))), tree=shuffled_batch,
            )

            # jax.debug.print(f"network: {network}")
            train_state, total_loss = jax.lax.scan(
                f=_update_minbatch, 
                init=train_state,
                xs=minibatches
            )
            
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, total_loss

        # create a tuple to be passed into the jax.lax.scan function
        update_state = (train_state, traj_batch, advantages, targets, rng)

        update_state, loss_info = jax.lax.scan( 
            f=_update_epoch, 
            init=update_state, 
            xs=None, 
            length=config.update_epochs
        )

        train_state = update_state[0]
        metric = info #returned from the env_step function

        # calculate the current timestep

        current_timestep = update_step * config.num_steps * config.num_envs

        # add the shaped rewards to the metric
        metric["shaped_reward"] = metric["shaped_reward"]["agent_0"]
        metric["shaped_reward_annealed"] = metric["shaped_reward"]*rew_shaping_anneal(current_timestep)

        # Update the step counter
        update_step = update_step + 1
        # update the metric with the current timestep
        metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
        metric["update_step"] = update_step
        metric["env_step"] = update_step * config.num_steps * config.num_envs

        # General section
        metric["General/update_step"] = update_step
        metric["General/env_step"] = update_step * config.num_steps * config.num_envs
        metric["General/lr"] = linear_schedule(update_step * config.update_epochs * config.num_minibatches)

        # Losses section
        total_loss, (value_loss, loss_actor, entropy) = loss_info
        metric["Losses/total_loss_mean"] = total_loss.mean()
        metric["Losses/total_loss_max"] = total_loss.max()
        metric["Losses/total_loss_min"] = total_loss.min()
        metric["Losses/total_loss_var"] = total_loss.var()

        metric["Losses/value_loss"] = value_loss.mean()
        metric["Losses/actor_loss"] = loss_actor.mean()
        metric["Losses/entropy"] = entropy.mean()

        # Rewards section
        metric["Rewards/shaped_reward"] = metric["shaped_reward"]
        metric["Rewards/shaped_reward_annealed"] = metric["shaped_reward_annealed"]
        metric["Rewards/episode_returns"] = metric["returned_episode_returns"]

        # Advantages and Targets section
        metric["Advantage_Targets/advantages"] = advantages.mean()
        metric["Advantage_Targets/targets"] = targets.mean()

        # Evaluation section
        for i in range(len(config.layout_name)):
            metric[f"Evaluation/{config.layout_name[i]}"] = jnp.nan
        

        def true_fun(metric):
            # freeze(config)

            # If update step is a multiple of 20, run the evaluation function
            rng, eval_rng = jax.random.split(rng)
            print("train_state: ", train_state)
            train_state_eval = jax.tree_util.tree_map(lambda x: x.copy(), train_state)

            evaluations = evaluate_model(train_state_eval, network, config, eval_rng)
            for i, evaluation in enumerate(evaluations):
                metric[f"Evaluation/{config.layout_name[i]}"] = evaluation
            return metric
        
        def false_fun(metric):
            return metric
                
        metric = jax.lax.cond((update_step % 200) == 0, true_fun, false_fun, metric)
        
        # Callback to log all sections at once
        def callback(metric):
            wandb.log(metric)

        # Use jax.debug.callback to log the metrics
        jax.debug.callback(callback, metric)
        
        rng = update_state[-1]
        runner_state = (train_state, env_state, last_obs, update_step, rng)

        return runner_state, _

    rng, train_rng = jax.random.split(rng)

    # initialize a carrier that keeps track of the states and observations of the agents
    runner_state = (train_state, env_state, obsv, 0, train_rng)
    
    # apply the _update_step function a series of times, while keeping track of the state 
    runner_state, _ = jax.lax.scan(
        f=_update_step, 
        init=runner_state, 
        xs=None, 
        length=config.num_updates
    )

    # Return the runner state after the training loop, and the metric arrays
    return {"runner_state": runner_state}

