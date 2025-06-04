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
import wandb

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
    

class Actor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

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

        return pi
        

class Critic(): 
    
    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

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
        return value #squeezed to remove any unnecessary dimensions



class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

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
        transition.global_done,
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

def _actor_loss_fn(actor_params, traj_batch, gae, actor):
    # RERUN NETWORK
    pi = actor.apply(
        actor_params,
        (traj_batch.obs, traj_batch.avail_actions),
    )
    log_prob = pi.log_prob(traj_batch.action)

    # Calculate actor loss
    log_ratio = log_prob - traj_batch.log_prob
    ratio = jnp.exp(log_ratio) 
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

    # calculate the actor loss as the minimum of the clipped and unclipped actor loss
    loss_actor = -jnp.minimum(loss_actor_unclipped, loss_actor_clipped)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()

    # calculate the approximated KL divergence 
    approx_kl = ((ratio - 1) - log_ratio).mean()

    # calculate the fraction of the ratio that is clipped
    clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
    
    # calculate the total actor loss
    actor_loss = (
        loss_actor
        - config["ENT_COEF"] * entropy
    )
    return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

def _critic_loss_fn(critic_params, traj_batch, target, critic, targets):
    # RERUN NETWORK
    value = critic.apply(critic_params, traj_batch.world_state) 
    
    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )
    critic_loss = config["VF_COEF"] * value_loss

    return critic_loss, (value_loss)

def _update_minbatch(train_states, batch_info):
    '''
    performs a single update minibatch in the training loop
    @param train_state: the current state of the training
    @param batch_info: the information of the batch
    returns the updated train_state and the total loss
    '''
    # unpack the train_state
    train_state_actor, train_state_critic = train_states

    # unpack the batch information
    traj_batch, advantages, targets = batch_info

    # calculate the gradients of the loss functions
    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)

    # apply the loss functions to get the total loss and the gradients
    actor_loss, actor_grad = actor_grad_fn(train_state_actor.params, traj_batch, advantages)
    critic_loss, critic_grad = critic_grad_fn(train_state_critic.params, traj_batch, targets)
    print('actor_loss: ', actor_loss)
    print('critic_loss: ', critic_loss)

    # apply the gradients to the network
    actor_train_state = actor_train_state.apply_gradients(grads=actor_grad)
    critic_train_state = critic_train_state.apply_gradients(grads=critic_grad)

    # calculate the total loss
    total_loss = actor_loss[0] + critic_loss[0]
    loss_info = {
        "total_loss": total_loss,
        "actor_loss": actor_loss[0],
        "value_loss": critic_loss[0],
        "entropy": actor_loss[1][1],
        "ratio": actor_loss[1][2],
        "approx_kl": actor_loss[1][3],
        "clip_frac": actor_loss[1][4],
    }

    # return the updated train_states and the loss information
    return (actor_train_state, critic_train_state), loss_info

# UPDATE NETWORK
def _update_epoch(update_state, unused):
    '''
    performs a single update epoch in the training loop
    @param update_state: the current state of the update
    returns the updated update_state and the total loss
    '''

    # unpack the update_state (because of the scan function)
    train_states, traj_batch, advantages, targets, rng = update_state

    # create a batch
    batch = (traj_batch, advantages.squeeze(), targets.squeeze()) 

    # permutate the batch
    rng, _rng = jax.random.split(rng)
    permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

    # shuffle the batch
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    ) # outputs a tuple of the batch, advantages, and targets shuffled 

    # split the batch into minibatches
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(
            a=jnp.reshape(x,[x.shape[0], config["NUM_MINIBATCHES"], -1]+ list(x.shape[2:]),),
            axis1=1,
            axis2=0,
        ),
        shuffled_batch,
    )
    # update the minibatches and calculate the loss
    train_states, total_loss = jax.lax.scan(
        f=_update_minbatch, 
        init=train_states, 
        xs=minibatches
    )

    # return the updated update_state and the total loss
    update_state = (train_states, traj_batch, advantages, targets, rng)
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
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]
    
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
        actor = Actor(env.action_space().n, activation=config["ACTIVATION"])
        critic = Critic()

        rng, rng_actor, rng_critic = jax.random.split(rng, 3)

        # Initialize the actor network
        actor_init_x = (
            jnp.zeros((env.observation_space().shape)),
            jnp.zeros((env.action_space(env.agents[0]).n)),
        )
        actor_network_params = actor.init(rng_actor, actor_init_x)
        
        # Initialize the critic network
        critic_init_x = jnp.zeros((658,)) # NOTE hardcoded >:(                  #TODO: aanpassen naar overcooked
    
        critic_network_params = critic.init(rng_critic, critic_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic_network_params,
            tx=critic_tx,
        )
        
        # Initialize the environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        
        # TRAIN LOOP
        def _update_step(carry, unused):
            '''
            perform a single update step in the training loop
            @param runner_state: the carry state that contains all important training information
            returns the updated runner state and the metrics 
            '''
            runner_state, update_steps = carry

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                '''
                selects an action based on the policy, calculates the log probability of the action, 
                and performs the selected action in the environment
                @param runner_state: the current state of the runner
                returns the updated runner state and the transition
                '''
                train_states, env_state, last_obs, last_done, rng = runner_state

                # SELECT ACTION
                # split the random number generator for action selection
                rng, _rng = jax.random.split(rng)

                # get all available actions 
                avail_actions = jax.vmap(env.get_legal_actions)(env_state.env_state)

                # stop the computation of the gradient before feeding the avail_actions to the network
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )

                # create a batch of the last observations
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                actor_input = (
                    obs_batch[np.newaxis, :],
                    avail_actions[None, :],
                )
                
                # apply the actor network to get the policy distribution
                actor_train_state = train_states[0]
                pi = actor.apply(actor_train_state.params, actor_input)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # format the actions to be compatible with the environment
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = jax.tree_map(lambda x: x.squeeze(), env_act)

                # GET VALUE 
                # get the complete state 
                world_state = last_obs["world_state"].swapaxes(0,1)
                world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                # Apply the critic network to the whole state
                critic_train_state = train_states[1]
                value = critic.apply(critic_train_state.params, world_state)

                
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
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                    avail_actions,
                )

                runner_state = (train_states, env_state, obsv, done_batch, rng)
                return runner_state, transition
            
            # Apply the _env_step function a series of times, while keeping track of the runner state
            runner_state, traj_batch = jax.lax.scan(
                f=_env_step, 
                init=runner_state, 
                xs=None, 
                length=config["NUM_STEPS"]
            ) 
            
            # unpack the runner state that is returned after the scan function
            train_states, env_state, last_obs, last_done, rng = runner_state

            # use the last observation to calculate the value of the last state
            last_world_state = last_obs["world_state"].swapaxes(0,1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))

            # apply the critic network to the last state
            critic_train_state = train_states[1]
            last_val = critic.apply(critic_train_state.params, last_world_state)
            last_val = last_val.squeeze()

            # calculate the generalized advantage estimate (GAE) for last value
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # create a tuple to be passed into the jax.lax.scan function
            update_state = (train_states, traj_batch, advantages, targets, rng)

            update_state, loss_info = jax.lax.scan( 
                f=_update_epoch, 
                init=update_state, 
                xs=None, 
                length=config["UPDATE_EPOCHS"]
            )

            train_states = update_state[0]
            metric = traj_batch.info
            loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
            metric["loss"] = loss_info
            rng = update_state[-1]

            # CALLBACK
            def callback(metric):
                '''
                logs the metrics to the console and wandb
                @param metric: the metric to log
                '''
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                        **metric["loss"],
                    }
                )


            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)

        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            _rng,
        )

        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        # we don't need to return the metric because it is already logged in the callback

        return {"runner_state": runner_state}
    return train

@hydra.main(version_base=None, config_path="config", config_name="mappo") 
def main(cfg):
    
    # set the config to global 
    global config

    # convert the config to a dictionary
    config = OmegaConf.to_container(cfg)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MAPPO", "FF", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    # set the layout of the environment
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]["layout"]]

    # set the random number generator and number of seeds
    rng = jax.random.PRNGKey(30) # random number generator
    num_seeds = 1

    # run the training loop
    with jax.disable_jit(True):
        # vectorize the training function
        train_jit = jax.jit(jax.vmap(make_train(config))) 
        # split the rng into num_seed seeds
        rngs = jax.random.split(rng, num_seeds) 
        # run the train function num_seeds times
        out = train_jit(rngs)
    

    # # Save results to a gif and a plot
    # print('** Saving Results **')
    # # filename = f'{config["ENV_NAME"]}_cramped_room_new'
    # filename = 'mappo'
    # rewards = out["metrics"]["returned_episode_returns"].mean(-1).reshape((num_seeds, -1)) 
    # reward_mean = rewards.mean(0)  # mean 
    # reward_std = rewards.std(0) / np.sqrt(num_seeds)  # standard error
    
    # plt.plot(reward_mean)
    # plt.fill_between(range(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    # # compute standard errorƒ
    # plt.xlabel("Update Step")
    # plt.ylabel("Return")
    # plt.savefig(f'{filename}.png')

    # # animate first seed
    # train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    # state_seq = get_rollout(train_state, config)
    # viz = OvercookedVisualizer()
    # # agent_view_size is hardcoded as it determines the padding around the layout.
    # viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

if __name__ == "__main__":
    print("Running main...")
    main()
