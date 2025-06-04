""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper
import jax_marl
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf

import matplotlib.pyplot as plt

# Set the global config variable
config = None

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # create the actor network   
        actor_mean = nn.Dense(
            64, # nr of layers
            kernel_init=orthogonal(np.sqrt(2)), # sets the weight initialization to orthogonal matrix with a scaling factor sqrt(2)
            bias_init=constant(0.0) # sets the bias initialization to a constant value of 0
        )(x) # applies a dense layer to the input x
        actor_mean = activation(actor_mean) # applies the activation function to the output of the dense layer
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean) # creates a categorical distribution over all actions (the logits are the output of the actor network)

        # create the critic network
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
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

############################
##### HELPER FUNCTIONS #####
############################

def get_rollout(train_state, config):
    '''
    Simulates the environment using the network
    @param train_state: the current state of the training
    @param config: the configuration of the training
    returns the state sequence
    '''
    env = jax_marl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env_params = env.default_params
    # env = LogWrapper(env) 

    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) # Sets up the network
    key = jax.random.PRNGKey(0) 
    key, key_r, key_a = jax.random.split(key, 3) 

    init_x = jnp.zeros(env.observation_space().shape) # initializes the observation space to zeros
    init_x = init_x.flatten() # flattens the observation space to a 1D array

    network.init(key_a, init_x) # initializes the network with the observation space
    network_params = train_state.params 

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        # breakpoint()
        obs = {k: v.flatten() for k, v in obs.items()}

        pi_0, _ = network.apply(network_params, obs["agent_0"])
        pi_1, _ = network.apply(network_params, obs["agent_1"])

        actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}
        # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
        # env_act = {k: v.flatten() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq

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


def linear_schedule(count):
    '''
    Linearly decays the learning rate depending on the number of minibatches and number of epochs
    returns the learning rate
    '''
    frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
    return config["LR"] * frac

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

def _calculate_gae(traj_batch, last_val):
    '''
    calculates the generalized advantage estimate (GAE) for the trajectory batch
    @param traj_batch: the trajectory batch
    @param last_val: the value of the last state
    returns the advantages and the targets
    '''
    # iteratively apply the _get_advantages function to calculate the advantage for each step in the trajectory batch
    _, advantages = jax.lax.scan(
        f=_get_advantages,
        init=(jnp.zeros_like(last_val), last_val),
        xs=traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value

def _loss_fn(params, traj_batch, gae, targets, network):
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


def _update_minbatch(train_state_and_network, batch_info):
    '''
    performs a single update minibatch in the training loop
    @param train_state: the current state of the training
    @param batch_info: the information of the batch
    returns the updated train_state and the total loss
    '''
    # unpack the batch information
    traj_batch, advantages, targets = batch_info

    # here was the definition of the loss function first

    # We need network in this function so we add it to the carry (first argument)
    train_state, network = train_state_and_network

    # jax.debug.print(f"train state: {type(train_state)}")

    # returns a function with the same parameters as loss_fn that calculates the gradient of the loss function
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

    # call the grad_fn function to get the total loss and the gradients
    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets, network)

    # apply the gradients to the network
    train_state = train_state.apply_gradients(grads=grads)

    # Of course we also need to add the network to the carry here
    return (train_state, network), total_loss

# UPDATE NETWORK
def _update_epoch(update_state, unused):
    '''
    performs a single update epoch in the training loop
    @param update_state: the current state of the update
    returns the updated update_state and the total loss
    '''

    # update minibatch function def
    
    # unpack the update_state (because of the scan function)
    train_state, traj_batch, advantages, targets, rng, network = update_state
    
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
    # jax.debug.print('permutation: {}', permutation)

    # shuffle the batch
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    ) # outputs a tuple of the batch, advantages, and targets shuffled 

    minibatches = jax.tree_util.tree_map(
        f=(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:]))), tree=shuffled_batch,
    )

    jax.debug.print(f"network: {network}")
    (train_state, _), total_loss = jax.lax.scan(
        f=_update_minbatch, 
        init=(train_state, network), 
        xs=minibatches
    )
    
    update_state = (train_state, traj_batch, advantages, targets, rng, network)
    return update_state, total_loss

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

    # Create config
    env = jax_marl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # set extra config parameters based on the environment
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])
    
    # log the environment
    env = LogWrapper(env)
    
    # linear schedule definition used to be here

    def train(rng):
        '''
        Trains the network using IPPO
        @param rng: random number generator 
        returns the runner state and the metrics
        '''
        # Initialize network
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        
        # Initialize the observation space 
        init_x = jnp.zeros(env.observation_space().shape) 
        init_x = init_x.flatten()
        # print(f"init_x: {init_x}")
        
        # Initialize the network parameters
        rng, network_rng = jax.random.split(rng)

        # Initialize the network parameters
        network_params = network.init(network_rng, init_x)           # VAN DEZE LINE SNAP IK NOG NIKS  
        # print(f"network_params: {network_params}")

        # Initialize the optimizer
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


        # Initialize the training state      
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        # Initialize environment 
        rng, env_rng = jax.random.split(rng) # env_rng shape is (1, 2)

        # create config["NUM_ENVS"] seeds for each environment 
        reset_rng = jax.random.split(env_rng, config["NUM_ENVS"]) # reset_rng shape is ('num_envs', 2)

        # create and reset the environment
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
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
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                # split the random number generator for action selection
                rng, _rng = jax.random.split(rng)

                # prepare the observations for the network
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                
                # apply the policy network to the observations to get the suggested actions and their values
                pi, value = network.apply(train_state.params, obs_batch)

                # sample the actions from the policy distribution
                # print(pi.sample(seed=_rng))
                # action = 0 
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

                # format the outputs of the environment to a 'transition' structure that can be used for analysis
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(), 
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info
                    
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition
            
            # Apply the _env_step function a series of times, while keeping track of the runner state
            runner_state, traj_batch = jax.lax.scan(
                f=_env_step, 
                init=runner_state, 
                xs=None, 
                length=config["NUM_STEPS"]
            ) 
            
            # unpack the runner state that is returned after the scan function
            train_state, env_state, last_obs, rng = runner_state

            # create a batch of the observations that is compatible with the network
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

            # apply the network to the batch of observations to get the value of the last state
            _, last_val = network.apply(train_state.params, last_obs_batch)

            # calculate the generalized advantage estimate (GAE) for the trajectory batch
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # _update_epoch definition used to be here

            # create a tuple to be passed into the jax.lax.scan function
            update_state = (train_state, traj_batch, advantages, targets, rng, network)

            update_state, unused_loss_info = jax.lax.scan( 
                f=_update_epoch, 
                init=update_state, 
                xs=None, 
                length=config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-2]
            
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, train_rng = jax.random.split(rng)

        # initialize a carrier that keeps track of the states and observations of the agents
        runner_state = (train_state, env_state, obsv, train_rng)
        
        # apply the _update_step function a series of times, while keeping track of the state 
        runner_state, metric = jax.lax.scan(
            f=_update_step, 
            init=runner_state, 
            xs=None, 
            length=config["NUM_UPDATES"]
        )

        # Return the runner state after the training loop, and the metric arrays
        return {"runner_state": runner_state, "metrics": metric}

    return train



@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_overcooked") 
def main(cfg):
    
    # set the config to global 
    global config

    # convert the config to a dictionary
    config = OmegaConf.to_container(cfg)

    # set the layout of the environment
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]["layout"]]

    # set the random number generator and number of seeds
    rng = jax.random.PRNGKey(30) # random number generator
    num_seeds = 1

    # run the training loop
    with jax.disable_jit(False):
        # vectorize the training function
        train_jit = jax.jit(jax.vmap(make_train(config))) 
        # split the rng into num_seed seeds
        rngs = jax.random.split(rng, num_seeds) 
        # run the train function num_seeds times
        out = train_jit(rngs)
    

    # Save results to a gif and a plot
    print('** Saving Results **')
    filename = f'{config["ENV_NAME"]}_cramped_room_new'
    rewards = out["metrics"]["returned_episode_returns"].mean(-1).reshape((num_seeds, -1)) 
    reward_mean = rewards.mean(0)  # mean 
    reward_std = rewards.std(0) / np.sqrt(num_seeds)  # standard error
    
    plt.plot(reward_mean)
    plt.fill_between(range(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    # compute standard error
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'{filename}.png')

    # animate first seed
    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    state_seq = get_rollout(train_state, config)
    viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

if __name__ == "__main__":
    print("Running main...")
    main()

