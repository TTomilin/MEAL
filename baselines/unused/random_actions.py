import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jax_marl
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
from jax_marl.environments.overcooked_environment.layouts import counter_circuit_grid
import hydra
from omegaconf import OmegaConf
from flax.linen.initializers import constant, orthogonal

import matplotlib.pyplot as plt
import wandb
import tracemalloc
from jax import clear_caches
import gc
from memory_profiler import profile

from functools import partial

class Transition(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray  # whether the episode is done
    action: jnp.ndarray  # the action taken
    reward: jnp.ndarray  # the reward received
    obs: jnp.ndarray  # the observation



############################
##### HELPER FUNCTIONS #####
############################

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def pad_observation_space(config):
    envs = []
    for env_args in config["ENV_KWARGS"]:
        env = jax_marl.make(config["ENV_NAME"], **env_args)
        envs.append(env)

    max_width, max_height = 0, 0
    for env in envs:
        max_width = max(max_width, env.layout["width"])
        max_height = max(max_height, env.layout["height"])
    
    padded_envs = []
    for env in envs:
        env = unfreeze(env.layout)
        width_diff = max_width - env["width"]
        height_diff = max_height - env["height"]
        left = width_diff // 2
        right = width_diff - left
        top = height_diff // 2
        bottom = height_diff - top
        width = env["width"]

        def adjust_indices(indices):
            adjusted_indices = []
            for idx in indices:
                row = idx // width
                col = idx % width
                new_row = row + top
                new_col = col + left
                new_idx = new_row * (width + left + right) + new_col
                adjusted_indices.append(new_idx)
            return jnp.array(adjusted_indices)

        env["wall_idx"] = adjust_indices(env["wall_idx"])
        env["agent_idx"] = adjust_indices(env["agent_idx"])
        env["goal_idx"] = adjust_indices(env["goal_idx"])
        env["plate_pile_idx"] = adjust_indices(env["plate_pile_idx"])
        env["onion_pile_idx"] = adjust_indices(env["onion_pile_idx"])
        env["pot_idx"] = adjust_indices(env["pot_idx"])

        padded_wall_idx = list(env["wall_idx"])
        for y in range(top):
            for x in range(max_width):
                padded_wall_idx.append(y * max_width + x)
        for y in range(max_height - bottom, max_height):
            for x in range(max_width):
                padded_wall_idx.append(y * max_width + x)
        for y in range(top, max_height - bottom):
            for x in range(left):
                padded_wall_idx.append(y * max_width + x)
            for x in range(max_width - right, max_width):
                padded_wall_idx.append(y * max_width + x)

        env["wall_idx"] = jnp.array(padded_wall_idx)
        env["height"] = max_height
        env["width"] = max_width
        padded_envs.append(freeze(env))
    return padded_envs

def sample_discrete_action(key, action_space):
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)

def get_rollout_for_visualization(config):
    envs = pad_observation_space(config)
    state_sequences = []
    for env_layout in envs:
        env = jax_marl.make(config["ENV_NAME"], layout=env_layout)
        key = jax.random.PRNGKey(0)
        key, key_r, key_a = jax.random.split(key, 3)
        done = False
        obs, state = env.reset(key_r)
        state_seq = [state]
        rewards = []
        shaped_rewards = []
        while not done:
            key, key_a0, key_a1, key_s = jax.random.split(key, 4)
            action_space_0 = env.action_space()
            action_space_1 = env.action_space()
            action_0 = sample_discrete_action(key_a0, action_space_0).item()
            action_1 = sample_discrete_action(key_a1, action_space_1).item()
            actions = {"agent_0": action_0, "agent_1": action_1}
            obs, state, reward, done, info = env.step(key_s, state, actions)
            done = done["__all__"]
            rewards.append(reward["agent_0"])
            shaped_rewards.append(info["shaped_reward"]["agent_0"])
            state_seq.append(state)
        state_sequences.append(state_seq)
    return state_sequences

def visualize_environments(config):
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
    Creates a 'train' function that executes random actions in the environment instead of training.
    @param config: the configuration of the algorithm and environment
    returns the observational metrics
    '''
    @profile
    def train(rng):
        # step 1: make sure all envs are the same size and create the environments
        padded_envs = pad_observation_space(config)
        envs = []
        for env_layout in padded_envs:
            env = jax_marl.make(config["ENV_NAME"], layout=env_layout)
            env = LogWrapper(env, replace_info=False)
            envs.append(env)

        
        # check if all envs have the same layout and size
        # for env in envs:
        #     print(env.width, env.height, env.obs_shape, env.agent_view_size, env.layout, env.agents)

      
        # set extra config parameters based on the environment
        temp_env = envs[0]
        config["NUM_ACTORS"] = temp_env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
        config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

        # @partial(jax.jit, static_argnums=(1))
        def train_on_environment(rng, env):
            '''
            Executes random actions in the environment instead of training.
            @param rng: random number generator 
            returns observational metrics
            '''
            rng, env_rng = jax.random.split(rng)
            reset_rng = jax.random.split(env_rng, config["NUM_ENVS"]) 
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            update_step = 0

            def _update_step(runner_state, unused):
                '''
                Perform an environment step with random actions only.
                '''
                def _env_step(runner_state, unused):
                    '''
                    Selects random actions and performs a step in the environment.
                    '''
                    env_state, last_obs, update_step, rng = runner_state
                    rng, key_a0, key_a1, key_s = jax.random.split(rng, 4)
                    action_0 = jnp.broadcast_to(sample_discrete_action(key_a0, env.action_space()), (config["NUM_ENVS"],))
                    action_1 = jnp.broadcast_to(sample_discrete_action(key_a1, env.action_space()), (config["NUM_ENVS"],))
                    actions = {"agent_0": action_0, "agent_1": action_1}

                    obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                        jax.random.split(key_s, config["NUM_ENVS"]), env_state, actions
                    )
                    runner_state = (env_state, obsv, update_step, rng)
                    metrics = {"reward": reward["agent_0"], "done": done["__all__"]}
                    return runner_state, metrics

                runner_state, metrics = jax.lax.scan(
                    f=_env_step,
                    init=runner_state,
                    xs=None,
                    length=config["NUM_STEPS"]
                )

                return runner_state, metrics

            rng, train_rng = jax.random.split(rng)
            runner_state = (env_state, obsv, update_step, train_rng)
            
            runner_state, metrics = jax.lax.scan(
                f=_update_step,
                init=runner_state,
                xs=None,
                length=config["NUM_UPDATES"]
            )

            return runner_state, metrics

        def loop_over_envs(rng, envs):
            metrics = []

            # check if the train_on_environment function is in cache
            # runner_state, metric = train_on_environment(rng, envs[0])
            # jax.debug.print("cache size before looping: {}", train_on_environment._cache_size())


            for env_rng, env in zip(jax.random.split(rng, len(envs)+1)[1:], envs):

                runner_state, metric = train_on_environment(env_rng, env)

                # object = train_on_environment
                # runner_state, metric = object(env_rng, env)

                # jitted = jax.jit(train_on_environment, static_argnums=(1))
                # runner_state, metric = jitted(env_rng, env)
                # metrics.append(metric)


                # # check if the train_on_environment function is in cache
                # jax.debug.print("cache size: {}", jitted._cache_size())

                # del jitted
                # gc.collect()
            
            return runner_state, metrics

        runner_state, metrics = loop_over_envs(rng, envs)
        return {"runner_state": runner_state, "metrics": metrics}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_continual") 
def main(cfg):
    jax.config.update("jax_platform_name", "gpu")
    global config
    config = OmegaConf.to_container(cfg)

    config["ENV_KWARGS"], config["LAYOUT_NAME"] = generate_sequence(
        sequence_length=config["SEQ_LENGTH"], 
        strategy=config["STRATEGY"], 
        layouts=None
    )

    for layout_config in config["ENV_KWARGS"]:
        layout_name = layout_config["layout"]
        layout_config["layout"] = overcooked_layouts[layout_name]

    wandb.init(
        project="ippo-overcooked", 
        config=config, 
        mode=config["WANDB_MODE"],
        name="random_actions"
    )

    with jax.disable_jit(False):   
        rng = jax.random.PRNGKey(config["SEED"])  
        # rngs = jax.random.split(rng, config["NUM_SEEDS"])

        # # trace the make_train function
        # traced_train = jax.make_jaxpr(make_train(config))
        # print(traced_train(rngs[0]))

        jitted_train = jax.jit(make_train(config))
        out = jitted_train(rng)
        # out = jax.vmap(jitted_train)(rngs)  
        # out = jax.vmap(make_train(config))(rngs)  


    print("Done")


if __name__ == "__main__":
    print("Running main...")
    main()
