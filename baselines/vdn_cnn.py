import os
import copy
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import tyro
from datetime import datetime
from tensorboardX import SummaryWriter

import flax
import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.core.frozen_dict import freeze, unfreeze
import flashbax as fbx
import wandb
from dotenv import load_dotenv

from jax_marl import make
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.wrappers.baselines import (
    LogWrapper,
    CTRolloutManager,
)
from jax_marl.environments.overcooked_environment import overcooked_layouts
from architectures.q_network import QNetwork
from baselines.utils_vdn import (
    Timestep,
    CustomTrainState,
    batchify,
    unbatchify,
    eps_greedy_exploration,
    get_greedy_actions
)
import uuid


@dataclass
class Config:
    total_timesteps: float = 5e6
    num_envs: int = 32
    num_steps: int = 1
    hidden_size: int = 64
    eps_start: float = 1.0
    eps_finish: float = 0.05
    eps_decay: float = 0.1
    max_grad_norm: int = 1
    num_epochs: int = 4
    lr: float = 0.00007
    lr_linear_decay: bool = True
    lambda_: float = 0.5 
    gamma: float = 0.99
    tau: float = 1
    buffer_size: int = 1e5
    buffer_batch_size: int = 128
    learning_starts: int = 1e3
    target_update_interval: int = 10

    rew_shaping_horizon: float = 2.5e6
    test_during_training: bool = True
    test_interval: float = 0.05 #fraction 
    test_num_steps: int = 400
    test_num_envs: int = 256 # number of episodes to average over
    eval_num_episodes: int = 10
    seed: int = 30

    # Sequence settings 
    seq_length: int = 2
    strategy: str = "random"
    layouts: Optional[Sequence[str]] = field(default_factory=lambda: [])
    env_kwargs: Optional[Sequence[dict]] = None
    layout_name: Optional[Sequence[str]] = None

    # Wandb settings
    wandb_mode: str = "online"
    entity: Optional[str] = ""
    project: str = "COOX"
    tags: List[str] = field(default_factory=list)
    wandb_log_all_seeds: bool = False
    env_name: str = "overcooked"
    alg_name: str = "vdn"
    network_name: str = "cnn"
    cl_method_name: str = "none"
    group: str = "none"

    # To be computed during runtime
    num_updates: int = 0


###################################################
############### MAIN FUNCTION #####################
###################################################

def main(): 
    # set the device 
    jax.config.update("jax_platform_name", "gpu")
    print("device: ", jax.devices())

    # load the config
    config = tyro.cli(Config)

    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    unique_id = uuid.uuid4()
    run_name = f'{config.alg_name}_seq{config.seq_length}_{config.strategy}_{timestamp}_{unique_id}'
    exp_dir = os.path.join("runs", run_name)

    # Initialize WandB
    load_dotenv()
    wandb_tags = config.tags if config.tags is not None else []
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project=config.project, 
        config=config,
        sync_tensorboard=True,
        mode=config.wandb_mode,
        name=run_name,
        tags=wandb_tags,
    )

    # Set up Tensorboard
    writer = SummaryWriter(exp_dir)

    # add the hyperparameters to the tensorboard
    rows = []
    for key, value in vars(config).items():
        value_str = str(value).replace("\n", "<br>")
        value_str = value_str.replace("|", "\\|")  # escape pipe chars if needed
        rows.append(f"|{key}|{value_str}|")

    table_body = "\n".join(rows)
    markdown = f"|param|value|\n|-|-|\n{table_body}"
    writer.add_text("hyperparameters", markdown)

    def pad_observation_space():
        '''
        Function that pads the observation space of all environments to be the same size by adding extra walls to the outside.
        This way, the observation space of all environments is the same, and compatible with the network
        returns the padded environments
        '''
        envs = []
        for env_args in config.env_kwargs:
                # Create the environment
                env = make(config.env_name, **env_args)
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

    @jax.jit
    def evaluate_model(rng, train_state):
        '''
        Evaluates the current model on all environments in the sequence
        @param rng: the random number generator
        @param train_state: the current training state
        returns the metrics: the average returns of the evaluated episodes
        '''
        def evaluate_on_environment(test_env, rng, train_state):
            '''
            Evaluates the current model on a single environment
            @param rng: the random number generator
            @param train_state: the current training state
            returns the metrics: the average returns of the evaluated episodes
            '''
            def evaluation_step(step_state, unused):
                last_obs, env_state, rng = step_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.params,
                    batchify(last_obs, env.agents),  # (num_agents, num_envs, num_actions)
                )  # (num_agents, num_envs, num_actions)
                actions = jnp.argmax(q_vals, axis=-1)
                actions = unbatchify(actions, env.agents)
                new_obs, new_env_state, rewards, dones, infos = test_env.batch_step(
                    rng_s, env_state, actions
                )
                step_state = (new_obs, new_env_state, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            rng, _rng = jax.random.split(rng)
            step_state, (rewards, dones, infos) = jax.lax.scan(
                evaluation_step,
                (init_obs, env_state, _rng),
                None,
                config.test_num_steps,
            )
            metrics = jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        infos["returned_episode_returns"],
                        jnp.nan,
                    )
                )
            return metrics

        evaluation_returns = []

        for env in test_envs:
            all_returns = jax.vmap(lambda k: evaluate_on_environment(env, k, train_state))(
                jax.random.split(rng, config.eval_num_episodes)
            )
            
            mean_returns = jnp.mean(all_returns)
            evaluation_returns.append(mean_returns)

        return evaluation_returns

    # generate the sequence of environments
    config.env_kwargs, config.layout_name = generate_sequence(
        sequence_length=config.seq_length, 
        strategy=config.strategy, 
        layout_names=config.layouts, 
        seed=config.seed
    )

    # Create the environments
    padded_envs = pad_observation_space()
    train_envs = []
    test_envs = []
    for env_layout in padded_envs:
        env = make(config.env_name, layout=env_layout)
        env = LogWrapper(env, replace_info=False)
        train_env = CTRolloutManager(
            env, batch_size=config.num_envs, preprocess_obs=False
        )
        test_env = CTRolloutManager(
            env, batch_size=config.test_num_envs, preprocess_obs=False
        )
        train_envs.append(train_env)
        test_envs.append(test_env)
    
        
    config.num_updates = (
        config.total_timesteps // config.num_steps // config.num_envs
    )

    eps_scheduler = optax.linear_schedule(
        config.eps_start,
        config.eps_finish,
        config.eps_decay * config.num_updates,
    )

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config.rew_shaping_horizon
    )

    # Initialize the network
    rng = jax.random.PRNGKey(config.seed)
    init_env = train_envs[0]
    network = QNetwork(
        action_dim=init_env.max_action_space,
        hidden_size=config.hidden_size,
    )

    rng, agent_rng = jax.random.split(rng)

    init_x = jnp.zeros((1, *init_env.observation_space().shape))
    init_network_params = network.init(agent_rng, init_x)

    lr_scheduler = optax.linear_schedule(
        config.lr,
        1e-10,
        (config.num_epochs) * config.num_updates,
    )

    lr = lr_scheduler if config.lr_linear_decay else config.lr

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.radam(learning_rate=lr),
    )

    train_state = CustomTrainState.create(
        apply_fn=jax.jit(network.apply),
        params=init_network_params,
        target_network_params=init_network_params,
        tx=tx,
    )

    # Create the replay buffer
    buffer = fbx.make_flat_buffer(
        max_length=int(config.buffer_size),
        min_length=int(config.buffer_batch_size),
        sample_batch_size=int(config.buffer_batch_size),
        add_sequences=False,
        add_batch_size=int(config.num_envs * config.num_steps),
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    rng, init_rng = jax.random.split(rng)

    init_obs, init_env_state = init_env.batch_reset(init_rng)
    init_actions = {
        agent: init_env.batch_sample(init_rng, agent) for agent in env.agents
    }
    init_obs, _, init_rewards, init_dones, init_infos = init_env.batch_step(
        init_rng, init_env_state, init_actions
    )
    init_avail_actions = init_env.get_valid_actions(init_env_state.env_state)
    init_timestep = Timestep(
        obs=init_obs,
        actions=init_actions,
        avail_actions=init_avail_actions,
        rewards=init_rewards,
        dones=init_dones,
    )
    init_timestep_unbatched = jax.tree.map(
        lambda x: x[0], init_timestep
    )  # remove the NUM_ENV dim
    buffer_state = buffer.init(init_timestep_unbatched)

    @partial(jax.jit, static_argnums=(2))
    def train_on_environment(rng, train_state, train_env, env_counter):
        '''
        Trains the agent on a single environment
        @param rng: the random number generator
        @param train_state: the current training state
        @param train_env: the environment to train on
        returns the updated training state
        '''

        print("Training on environment")

        # for each new environment, we want to start with a fresh buffer
        buffer_state = buffer.init(init_timestep_unbatched)

        # reset the learning rate
        lr = lr_scheduler if config.lr_linear_decay else config.lr
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.radam(learning_rate=lr),
        )
        new_optimizer = tx.init(train_state.params)
        train_state = train_state.replace(tx=tx, opt_state=new_optimizer, n_updates=0)

        # # reset the entire training state
        # train_state = train_state.replace(params=init_network_params)
        # train_state = train_state.replace(target_network_params=init_network_params)

        def _update_step(runner_state, unused):

            train_state, buffer_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                '''
                steps the environment for a single step
                @param carry: the current state of the environment
                returns the new state of the environment and the timestep
                '''
                last_obs, env_state, rng = carry

                rng, rng_action, rng_step = jax.random.split(rng, 3)

                # Compute Q-values for all agents
                q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.params,
                    batchify(last_obs, env.agents), 
                )  # (num_agents, num_envs, num_actions)

                # retrieve the valid actions
                avail_actions = train_env.get_valid_actions(env_state.env_state)

                # perform epsilon-greedy exploration
                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_action, env.num_agents)
                new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions, env.agents)
                )
                actions = unbatchify(new_action, env.agents)

                new_obs, new_env_state, rewards, dones, infos = train_env.batch_step(
                    rng_step, env_state, actions
                )

                # add shaped reward
                shaped_reward = infos.pop("shaped_reward")
                shaped_reward["__all__"] = batchify(shaped_reward, env.agents).sum(axis=0)
                rewards = jax.tree.map(
                    lambda x, y: x + y * rew_shaping_anneal(train_state.timesteps),
                    rewards,
                    shaped_reward,
                )

                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    avail_actions=avail_actions,
                    rewards=rewards,
                    dones=dones,
                )
                return (new_obs, new_env_state, rng), (timestep, infos)

            # step the env
            rng, _rng = jax.random.split(rng)
            carry, (timesteps, infos) = jax.lax.scan(
                f=_step_env,
                init=(*expl_state, _rng),
                xs=None,
                length=config.num_steps,
            )
            expl_state = carry[:2]

            # update the steps count of the train state
            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config.num_steps * config.num_envs
            )

            # prepare the timesteps for the buffer
            timesteps = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), timesteps
            )  # (num_envs*num_steps, ...)

            # add the timesteps to the buffer
            buffer_state = buffer.add(buffer_state, timesteps)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):

                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience # collects a minibatch of size buffer_batch_size

                q_next_target = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.target_network_params, batchify(minibatch.second.obs, env.agents)
                )  # (num_agents, batch_size, ...)
                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)

                vdn_target = minibatch.first.rewards["__all__"] + (
                    1 - minibatch.first.dones["__all__"]
                ) * config.gamma * jnp.sum(
                    q_next_target, axis=0
                )  # sum over agents

                def _loss_fn(params):
                    q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                        params, batchify(minibatch.first.obs, env.agents)
                    )  # (num_agents, batch_size, ...)

                    # get logits of the chosen actions
                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals,
                        batchify(minibatch.first.actions, env.agents)[..., jnp.newaxis],
                        axis=-1,
                    ).squeeze()  # (num_agents, batch_size, )

                    chosen_action_q_vals = jnp.sum(chosen_action_q_vals, axis=0)
                    loss = jnp.mean((chosen_action_q_vals - vdn_target) ** 2)

                    return loss, chosen_action_q_vals.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)

            # Check if learning should happen
            can_train = buffer.can_sample(buffer_state)
            has_enough_timesteps = train_state.timesteps > config.learning_starts
            is_learn_time = can_train & has_enough_timesteps

            # Define learning and no-op functions
            def perform_learning(train_state, rng):
                return jax.lax.scan(
                    f=_learn_phase, 
                    init=(train_state, rng), 
                    xs=None, 
                    length=config.num_epochs
                )

            def do_nothing(train_state, rng):
                return (train_state, rng), (jnp.zeros(config.num_epochs), jnp.zeros(config.num_epochs))

            # Conditionally execute learning
            (train_state, rng), (loss, qvals) = jax.lax.cond(
                is_learn_time,
                perform_learning,
                do_nothing,
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config.target_update_interval == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config.tau,
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,

            )
            
            def compute_action_gap(q_vals):
                '''
                Computes the action gap
                @param q_vals: the Q-values
                returns the action gap
                '''
                top_2_q_vals, _ = jax.lax.top_k(q_vals, 2)
                top_q = top_2_q_vals[0]
                second_q = top_2_q_vals[1]
                return top_q - second_q

            # UPDATE METRICS
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "General/env_step": train_state.timesteps,
                "General/update_steps": train_state.n_updates,
                "General/grad_steps": train_state.grad_steps,
                "General/learning_rate": lr_scheduler(train_state.n_updates),
                "Losses/loss": loss.mean(),
                "Values/qvals": qvals.mean(),
                "General/epsilon": eps_scheduler(train_state.n_updates),
                "General/action_gap": compute_action_gap(qvals),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            def evaluate_and_log(rng, update_steps, test_metrics):
                '''
                Evaluates the model and logs the metrics
                @param rng: the random number generator
                @param update_steps: the number of update steps
                returns the metrics
                '''
                rng, eval_rng = jax.random.split(rng)
                
                def log_metrics(metrics, update_steps, env_counter):
                    # evaluate the model
                    test_metrics = evaluate_model(eval_rng, train_state)
                    for i, eval_metric in enumerate(test_metrics):
                        metrics[f"Evaluation/evaluation_{config.layout_name[i]}"] = eval_metric

                    # log the metrics
                    def callback(args):
                        metrics, update_steps, env_counter = args
                        real_step = (int(env_counter) - 1) * config.num_updates + int(update_steps)
                        for k, v in metrics.items():
                            writer.add_scalar(k, v, real_step)
                    
                    jax.experimental.io_callback(callback, None, (metrics, update_steps, env_counter))
                    return None
                        
                def do_not_log(metrics, update_steps, env_counter):
                    return None

                # conditionally evaluate and log the metrics
                jax.lax.cond((train_state.n_updates % int(config.num_updates * config.test_interval)) == 0, 
                             log_metrics, 
                             do_not_log, 
                             metrics, update_steps, env_counter)

            # Evaluate the model and log metrics    
            evaluate_and_log(rng, train_state.n_updates, test_metrics)

            runner_state = (train_state, buffer_state, expl_state, test_metrics, rng)

            return runner_state, None
        
        
        rng, eval_rng = jax.random.split(rng)
        test_metrics = evaluate_model(eval_rng, train_state)

        rng, reset_rng = jax.random.split(rng)
        obs, env_state = train_env.batch_reset(reset_rng)
        expl_state = (obs, env_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, expl_state, test_metrics, _rng)

        runner_state, metrics = jax.lax.scan(
            f=_update_step, 
            init=runner_state, 
            xs=None, 
            length=config.num_updates
        )

        return runner_state, metrics
    
    def loop_over_envs(rng, train_state, train_envs, test_envs, buffer_state):
        '''
        Loops over the environments and trains the network
        @param rng: random number generator
        @param train_state: the current state of the training
        @param train_envs: the training environments
        @param test_envs: the test environments
        returns the runner state and the metrics
        '''
        # split the random number generator for training on the environments
        rng, *env_rngs = jax.random.split(rng, len(train_envs)+1)

        # counter for the environment 
        env_counter = 1

        for env_rng, train_env, test_env in zip(env_rngs, train_envs, test_envs):
            runner_state, metrics = train_on_environment(env_rng, train_state, train_env, env_counter)

            # update the train state and buffer state
            train_state = runner_state[0]

            # save the model
            path = f"checkpoints/overcooked/{run_name}/model_env_{env_counter}"
            save_params(path, train_state)

            # update the environment counter
            env_counter += 1

        return runner_state

    def save_params(path, train_state):
        '''
        Saves the parameters of the network
        @param path: the path to save the parameters
        @param train_state: the current state of the training
        returns None
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    {"params": train_state.params}
                )
            )
        print('model saved to', path)

    # train the network
    rng, train_rng = jax.random.split(rng)

    runner_state = loop_over_envs(train_rng, train_state, train_envs, test_envs, buffer_state)
        

if __name__ == "__main__":
    main()