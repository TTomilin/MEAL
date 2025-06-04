import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["LINE_PROFILE"] = "1"

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Sequence, NamedTuple, Any
import hydra
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jax_marl
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
from omegaconf import OmegaConf
import wandb
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
import line_profiler 

from functools import partial

class Transition(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray  # whether the episode is done
    action: jnp.ndarray  # the action taken
    reward: jnp.ndarray  # the reward received
    obs: jnp.ndarray  # the observation


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


def make_train(config):
    def train(rng):
        padded_envs = pad_observation_space(config)
        envs = []
        for env_layout in padded_envs:
            env = jax_marl.make(config["ENV_NAME"], layout=env_layout)
            env = LogWrapper(env, replace_info=False)
            envs.append(env)

        # set extra config parameters based on the environment
        temp_env = envs[0]
        config["NUM_ACTORS"] = temp_env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
        config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

        def train_on_single_env(rng, env):
            rng, env_rng = jax.random.split(rng)
            reset_rng = jax.random.split(env_rng, config["NUM_ENVS"]) 
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            update_step = 0

            def _update_step(runner_state, _):
                metrics = []
                
                @line_profiler.profile
                def _env_step(runner_state, _):
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
            return runner_state


        def loop_over_envs(rng, envs):
            for env_rng, env in zip(jax.random.split(rng, len(envs)+1)[1:], envs):
                runner_state = train_on_single_env(env_rng, env)
                print("Done with env")

            
        
        loop_over_envs(rng, envs)
        

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
    #     # out = jax.vmap(jitted_train)(rngs)  
    #     # out = jax.vmap(make_train(config))(rngs)  

        jitted_train = jax.jit(make_train(config))
        out = jitted_train(rng)

    print("Done")


if __name__ == "__main__":
    print("Running main...")
    main()
