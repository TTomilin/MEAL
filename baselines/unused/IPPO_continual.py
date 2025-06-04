import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper

from jax_marl.registration import make
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
from jax_marl.environments.overcooked_environment.layouts import counter_circuit_grid

from dotenv import load_dotenv

import hydra
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import wandb

from functools import partial

# Enable compile logging
jax.log_compiles(True)

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
    

class Transition(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray # whether the episode is done
    action: jnp.ndarray # the action taken
    value: jnp.ndarray # the value of the state
    reward: jnp.ndarray # the reward received
    log_prob: jnp.ndarray # the log probability of the action
    obs: jnp.ndarray # the observation
    # info: jnp.ndarray # additional information

############################
##### HELPER FUNCTIONS #####
############################



@partial(jax.jit, static_argnums=(1))
def evaluate_model(train_state, network, key):
    '''
    Evaluates the model by running 10 episodes on all environments and returns the average reward
    @param train_state: the current state of the training
    @param config: the configuration of the training
    returns the average reward
    '''

    def run_episode_while(env, key_r, network, network_params, max_steps=1000):
        """
        Run a single episode using jax.lax.while_loop 
        """
        class LoopState(NamedTuple):
            key: Any
            state: Any
            obs: Any
            done: bool
            total_reward: float
            step_count: int

        def loop_cond(state: LoopState):
            '''
            Checks if the episode is done or if the maximum number of steps has been reached
            @param state: the current state of the loop
            returns a boolean indicating whether the loop should continue
            '''
            return jnp.logical_and(~state.done, state.step_count < max_steps)

        def loop_body(state: LoopState):
            '''
            Performs a single step in the environment
            @param state: the current state of the loop
            returns the updated state
            '''
            key, state_env, obs, _, total_reward, step_count = state
            key, key_a0, key_a1, key_s = jax.random.split(key, 4)

            # Flatten observations
            flat_obs = {k: v.flatten() for k, v in obs.items()}

            def select_action(train_state, rng, obs):
                '''
                Selects an action based on the policy network
                @param params: the parameters of the network
                @param rng: random number generator
                @param obs: the observation
                returns the action
                '''
                network_apply = train_state.apply_fn
                params = train_state.params
                pi, value = network_apply(params, obs)
                return pi.sample(seed=rng), value


            # Get action distributions
            action_a1, _ = select_action(train_state, key_a0, flat_obs["agent_0"])
            action_a2, _ = select_action(train_state, key_a1, flat_obs["agent_1"])

            # Sample actions
            actions = {
                "agent_0": action_a1,
                "agent_1": action_a2
            }

            # Environment step
            next_obs, next_state, reward, done_step, info = env.step(key_s, state_env, actions)
            done = done_step["__all__"]
            reward = reward["agent_0"]  
            total_reward += reward
            step_count += 1

            return LoopState(key, next_state, next_obs, done, total_reward, step_count)

        # Initialize
        key, key_s = jax.random.split(key_r)
        obs, state = env.reset(key_s)
        init_state = LoopState(key, state, obs, False, 0.0, 0)

        # Run while loop
        final_state = jax.lax.while_loop(
            cond_fun=loop_cond,
            body_fun=loop_body,
            init_val=init_state
        )

        return final_state.total_reward

    # Loop through all environments
    all_avg_rewards = []

    envs = pad_observation_space(config)

    for env in envs:
        env = make(config["ENV_NAME"], layout=env)  # Create the environment

        # network.init(key_a, init_x)  # initializes the network with the observation space
        network_params = train_state.params

        # Run k episodes
        all_rewards = jax.vmap(lambda k: run_episode_while(env, k, network, network_params, 500))(
            jax.random.split(key, 5)
        )
        
        avg_reward = jnp.mean(all_rewards)
        all_avg_rewards.append(avg_reward)

    return all_avg_rewards

def batchify(x: dict, agent_list, num_actors):
    '''
    converts the observations of a batch of agents into an array of size (num_actors, -1) that can be used by the network
    @param x: dictionary of observations
    @param agent_list: list of agents
    @param num_actors: number of actors
    returns the batchified observations
    '''
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    '''
    converts the array of size (num_actors, -1) into a dictionary of observations for all agents
    @param x: array of observations
    @param agent_list: list of agents
    @param num_envs: number of environments
    @param num_actors: number of actors
    returns the unbatchified observations
    '''
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def pad_observation_space(config):
    '''
    Pads the observation space of the environment to be compatible with the network
    @param envs: the environment
    returns the padded observation space
    '''
    envs = []
    for env_args in config["ENV_KWARGS"]:
            # Create the environment
            env = make(config["ENV_NAME"], **env_args)
            envs.append(env)

    # find the environment with the largest observation space
    max_width, max_height = 0, 0
    for env in envs:
        max_width = max(max_width, env.layout["width"])
        max_height = max(max_height, env.layout["height"])
    
    # pad the observation space of all environments to be the same size by adding extra walls to the outside
    padded_envs = []
    for env in envs:
        # unfreeze the environment so that we can apply padding
        env = unfreeze(env.layout)  

        # calculate the padding needed
        width_diff = max_width - env["width"]
        height_diff = max_height - env["height"]

        # determine the padding needed on each side
        left = width_diff // 2
        right = width_diff - left
        top = height_diff // 2
        bottom = height_diff - top

        width = env["width"]

        # Adjust the indices of the observation space to match the padded observation space
        def adjust_indices(indices):
            '''
            adjusts the indices of the observation space
            @param indices: the indices to adjust
            returns the adjusted indices
            '''
            adjusted_indices = []

            for idx in indices:
                # Compute the row and column of the index
                row = idx // width
                col = idx % width
                
                # Shift the row and column by the padding
                new_row = row + top
                new_col = col + left
                
                # Compute the new index
                new_idx = new_row * (width + left + right) + new_col
                adjusted_indices.append(new_idx)
            
            return jnp.array(adjusted_indices)
        
        # adjust the indices of the observation space to account for the new walls
        env["wall_idx"] = adjust_indices(env["wall_idx"])
        env["agent_idx"] = adjust_indices(env["agent_idx"])
        env["goal_idx"] = adjust_indices(env["goal_idx"])
        env["plate_pile_idx"] = adjust_indices(env["plate_pile_idx"])
        env["onion_pile_idx"] = adjust_indices(env["onion_pile_idx"])
        env["pot_idx"] = adjust_indices(env["pot_idx"])

        # pad the observation space with walls
        padded_wall_idx = list(env["wall_idx"])  # Existing walls
        
        # Top and bottom padding
        for y in range(top):
            for x in range(max_width):
                padded_wall_idx.append(y * max_width + x)  # Top row walls

        for y in range(max_height - bottom, max_height):
            for x in range(max_width):
                padded_wall_idx.append(y * max_width + x)  # Bottom row walls

        # Left and right padding
        for y in range(top, max_height - bottom):
            for x in range(left):
                padded_wall_idx.append(y * max_width + x)  # Left column walls

            for x in range(max_width - right, max_width):
                padded_wall_idx.append(y * max_width + x)  # Right column walls

        env["wall_idx"] = jnp.array(padded_wall_idx)

        # set the height and width of the environment to the new padded height and width
        env["height"] = max_height
        env["width"] = max_width

        padded_envs.append(freeze(env)) # Freeze the environment to prevent further modifications

    return padded_envs

def sample_discrete_action(key, action_space):
    """Samples a discrete action based on the action space provided."""
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)

def get_rollout_for_visualization(config):
    '''
    Simulates the environment using the network
    @param train_state: the current state of the training
    @param config: the configuration of the training
    returns the state sequence
    '''

    # Add the padding
    envs = pad_observation_space(config)

    state_sequences = []
    for env_layout in envs:
        env = make(config["ENV_NAME"], layout=env_layout)

        key = jax.random.PRNGKey(0)
        key, key_r, key_a = jax.random.split(key, 3)

        done = False

        obs, state = env.reset(key_r)
        state_seq = [state]
        rewards = []
        shaped_rewards = []
        while not done:
            key, key_a0, key_a1, key_s = jax.random.split(key, 4)

            # Get the action space for each agent (assuming it's uniform and doesn't depend on the agent_id)
            action_space_0 = env.action_space()  # Assuming the method needs to be called
            action_space_1 = env.action_space()  # Same as above since action_space is uniform

            # Sample actions for each agent
            action_0 = sample_discrete_action(key_a0, action_space_0).item()  # Ensure it's a Python scalar
            action_1 = sample_discrete_action(key_a1, action_space_1).item()

            actions = {
                "agent_0": action_0,
                "agent_1": action_1
            }

            # STEP ENV
            obs, state, reward, done, info = env.step(key_s, state, actions)
            done = done["__all__"]
            rewards.append(reward["agent_0"])
            shaped_rewards.append(info["shaped_reward"]["agent_0"])

            state_seq.append(state)
        state_sequences.append(state_seq)

    return state_sequences
    
def visualize_environments(config):
    '''
    Visualizes the environments using the OvercookedVisualizer
    @param config: the configuration of the training
    returns None
    '''
    state_sequences = get_rollout_for_visualization(config)
    visualizer = OvercookedVisualizer()
    visualizer.animate(state_seq=state_sequences[0], agent_view_size=5, filename="initial_state_env1.gif")
    visualizer.animate(state_seq=state_sequences[1], agent_view_size=5, filename="initial_state_env2.gif")

    return None

##########################################################
##########################################################
#######            TRAINING FUNCTION               #######
##########################################################
##########################################################

def make_train(config):
    '''
    Creates a 'train' function that trains the network using PPO
    @param config: the configuration of the algorithm and environment
    returns the training function
    '''
    def train(rng):
        # step 1: make sure all envs are the same size and create the environments
        padded_envs = pad_observation_space(config)
        envs = []
        for env_layout in padded_envs:
            env = make(config["ENV_NAME"], layout=env_layout)
            env = LogWrapper(env, replace_info=False)
            envs.append(env)

      
        # set extra config parameters based on the environment
        temp_env = envs[0]
        config["NUM_ACTORS"] = temp_env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
        config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

        def linear_schedule(count):
            '''
            Linearly decays the learning rate depending on the number of minibatches and number of epochs
            returns the learning rate
            '''
            frac = 1.0 - ((count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"])
            return config["LR"] * frac
        

        # REWARD SHAPING IN NEW VERSION
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.,
            end_value=0.,
            transition_steps=config["REWARD_SHAPING_HORIZON"]
        )

        # step 2: initialize the network using the first environment
        network = ActorCritic(temp_env.action_space().n, activation=config["ACTIVATION"])

        # step 3: initialize the network parameters
        rng, network_rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape).flatten()
        network_params = network.init(network_rng, init_x)

        # step 4: initialize the optimizer
        if config["ANNEAL_LR"]: 
            # anneals the learning rate
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            # uses the default learning rate
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), 
                optax.adam(config["LR"], eps=1e-5)
            )

        # step 5: Initialize the training state      
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        @partial(jax.jit, static_argnums=(2,))
        def train_on_environment(rng, train_state, env):
            '''
            Trains the network using IPPO
            @param rng: random number generator 
            returns the runner state and the metrics
            '''

            print("Training on environment")

            # reset the learning rate and the optimizer
            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5)
                )
            train_state = train_state.replace(tx=tx)
            
            # Initialize and reset the environment 
            rng, env_rng = jax.random.split(rng) 
            reset_rng = jax.random.split(env_rng, config["NUM_ENVS"]) 
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng) 
            
            # TRAIN 
            # @profile
            def _update_step(runner_state, unused):
                '''
                perform a single update step in the training loop
                @param runner_state: the carry state that contains all important training information
                returns the updated runner state and the metrics 
                '''

                # COLLECT TRAJECTORIES
                # @profile
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
                    # split the random number generator for action selection
                    rng, _rng = jax.random.split(rng)

                    # prepare the observations for the network
                    obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                    # print("obs_shape", obs_batch.shape)
                    
                    # apply the policy network to the observations to get the suggested actions and their values
                    pi, value = network.apply(train_state.params, obs_batch)

                    # sample the actions from the policy distribution 
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)

                    # format the actions to be compatible with the environment
                    env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                    env_act = {k:v.flatten() for k,v in env_act.items()}
                    
                    # STEP ENV
                    # split the random number generator for stepping the environment
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                    
                    # simultaniously step all environments with the selected actions (parallelized over the number of environments with vmap)
                    obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                        rng_step, env_state, env_act
                    )

                    # REWARD SHAPING IN NEW VERSION
                    
                    # add the reward of one of the agents to the info dictionary
                    info["reward"] = reward["agent_0"]

                    current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

                    # add the shaped reward to the normal reward 
                    reward = jax.tree_util.tree_map(lambda x,y: x+y * rew_shaping_anneal(current_timestep), reward, info["shaped_reward"])

                    transition = Transition(
                        batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(), 
                        action,
                        value,
                        batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                        log_prob,
                        obs_batch
                    )

                    # runner_state = (train_state, env_state, obsv, update_step, rng)
                    return runner_state, (transition, info)
                
                # Apply the _env_step function a series of times, while keeping track of the runner state
                runner_state, (traj_batch, info) = jax.lax.scan(
                    f=_env_step, 
                    init=runner_state, 
                    xs=None, 
                    length=config["NUM_STEPS"]
                )  

                # unpack the runner state that is returned after the scan function
                train_state, env_state, last_obs, update_step, rng = runner_state

                # create a batch of the observations that is compatible with the network
                last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                # apply the network to the batch of observations to get the value of the last state
                _, last_val = network.apply(train_state.params, last_obs_batch)
                # this returns the value network for the last observation batch
                
                # @profile
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
                        @param gae_and_next_value: the GAE and value of the next state
                        @param transition: the transition to calculate the advantage for
                        returns the updated GAE and the advantage
                        '''
                        gae, next_value = gae_and_next_value
                        done, value, reward = (
                            transition.done,
                            transition.value,
                            transition.reward,
                        )
                        delta = reward + config["GAMMA"] * next_value * (1 - done) - value # calculate the temporal difference
                        gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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
                # @profile
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
                        
                        # @profile
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
                            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"]) 
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
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            ) 

                            loss_actor = -jnp.minimum(loss_actor_unclipped, loss_actor_clipped) # calculate the actor loss as the minimum of the clipped and unclipped actor loss
                            loss_actor = loss_actor.mean() # calculate the mean of the actor loss
                            entropy = pi.entropy().mean() # calculate the entropy of the policy 

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        # returns a function with the same parameters as loss_fn that calculates the gradient of the loss function
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                        # call the grad_fn function to get the total loss and the gradients
                        total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                        loss_information = total_loss, grads

                        # apply the gradients to the network
                        train_state = train_state.apply_gradients(grads=grads)

                        # Of course we also need to add the network to the carry here
                        return train_state, loss_information
                    
                    
                    # unpack the update_state (because of the scan function)
                    train_state, traj_batch, advantages, targets, rng = update_state
                    
                    # set the batch size and check if it is correct
                    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    assert (
                        batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
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
                        f=(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:]))), tree=shuffled_batch,
                    )

                    train_state, loss_information = jax.lax.scan(
                        f=_update_minbatch, 
                        init=train_state,
                        xs=minibatches
                    )
                    
                    total_loss, grads = loss_information 
                    avg_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)
                    update_state = (train_state, traj_batch, advantages, targets, rng)
                    return update_state, total_loss

                # create a tuple to be passed into the jax.lax.scan function
                update_state = (train_state, traj_batch, advantages, targets, rng)

                update_state, loss_info = jax.lax.scan( 
                    f=_update_epoch, 
                    init=update_state, 
                    xs=None, 
                    length=config["UPDATE_EPOCHS"]
                )

                # unpack update_state
                train_state, traj_batch, advantages, targets, rng = update_state
                metric = info
                current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]

                # Update the step counter
                update_step = update_step + 1
                # update the metric with the current timestep
                metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)

                # General section
                metric["General/update_step"] = update_step
                metric["General/env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                metric["General/learning_rate"] = linear_schedule(update_step * config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])

                # Losses section
                total_loss, (value_loss, loss_actor, entropy) = loss_info
                metric["Losses/total_loss"] = total_loss.mean()
                metric["Losses/total_loss_max"] = total_loss.max()
                metric["Losses/total_loss_min"] = total_loss.min()
                metric["Losses/total_loss_var"] = total_loss.var()
                metric["Losses/value_loss"] = value_loss.mean()
                metric["Losses/actor_loss"] = loss_actor.mean()
                metric["Losses/entropy"] = entropy.mean()

                # Rewards section
                metric["General/shaped_reward_agent0"] = metric["shaped_reward"]["agent_0"]
                metric["General/shaped_reward_agent1"] = metric["shaped_reward"]["agent_1"]
                metric["General/shaped_reward_annealed_agent0"] = metric["General/shaped_reward_agent0"] * rew_shaping_anneal(current_timestep)
                metric["General/shaped_reward_annealed_agent1"] = metric["General/shaped_reward_agent1"] * rew_shaping_anneal(current_timestep)

                # Advantages and Targets section
                metric["Advantage_Targets/advantages"] = advantages.mean()
                metric["Advantage_Targets/targets"] = targets.mean()

                # Evaluation section
                for i in range(len(config["LAYOUT_NAME"])):
                    metric[f"Evaluation/{config['LAYOUT_NAME'][i]}"] = jnp.nan
                

                # If update step is a multiple of 20, run the evaluation function
                rng, eval_rng = jax.random.split(rng)
                train_state_eval = jax.tree_util.tree_map(lambda x: x.copy(), train_state)

                def true_fun(metric):
                    evaluations = evaluate_model(train_state_eval, network, eval_rng)
                    for i, evaluation in enumerate(evaluations):
                        metric[f"Evaluation/{config['LAYOUT_NAME'][i]}"] = evaluation
                    return metric
                
                def false_fun(metric):
                    return metric
                
                metric = jax.lax.cond((update_step % 200) == 0, true_fun, false_fun, metric)

                def callback(metric):
                    wandb.log(
                        metric
                    )
                
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
                length=config["NUM_UPDATES"]
            )

            # Return the runner state after the training loop, and the metric arrays
            return runner_state

        # split the random number generator for training on the environments
        rng, *env_rngs = jax.random.split(rng, len(envs)+1)

        def loop_over_envs(rng, train_state, envs):
            '''
            Loops over the environments and trains the network
            @param rng: random number generator
            @param train_state: the current state of the training
            @param envs: the environments
            returns the runner state and the metrics
            '''
            # metrics = []
            for env_rng, env in zip(env_rngs, envs):
                runner_state = train_on_environment(env_rng, train_state, env)

                jax.debug.print("cache size of train_on_env: {cache_size}", cache_size=train_on_environment._cache_size())
                
                # metrics.append(metric)
                train_state = runner_state[0]
                # jax.clear_caches()
                print("done with env")
            return runner_state
        
        # apply the loop_over_envs function to the environments
        runner_state = loop_over_envs(rng, train_state, envs)

        return {"runner_state": runner_state}
    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_continual") 
def main(cfg):
    
    # check available devices
    print(jax.devices())

    # set the device to the first available GPU
    jax.config.update("jax_platform_name", "gpu")
    
    # set the config to global 
    global config

    # convert the config to a dictionary
    config = OmegaConf.to_container(cfg)

    # create the environments
    seq_length = config["SEQ_LENGTH"]
    strategy = config["STRATEGY"]
    config["ENV_KWARGS"], config["LAYOUT_NAME"] = generate_sequence(sequence_length=seq_length, strategy=strategy, layouts=None)

    # set the layout of the environment
    for layout_config in config["ENV_KWARGS"]:
        # Extract the layout name
        layout_name = layout_config["layout"]

        # Set the layout in the config
        layout_config["layout"] = overcooked_layouts[layout_name]

    # log in to wandb 
    load_dotenv()
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    # Initialize wandb
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"], 
        config=config, 
        mode = config["WANDB_MODE"],
        name = f'ippo_continual'
    )

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"]) 
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_jit = jax.jit(make_train(config))
        out = train_jit(rngs[1])
        # out = make_train(config)(rngs[1])
        # out = jax.vmap(train_jit)(rngs)
        # out = jax.vmap(make_train(config))(rngs)

    print("Done")
    

if __name__ == "__main__":
    print("Running main...")
    main()

