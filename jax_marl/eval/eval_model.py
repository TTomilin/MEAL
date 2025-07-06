#!/usr/bin/env python3
"""
evaluate_checkpoint.py

Load a Flax model checkpoint, run N episodes in the Overcooked environment,
record a GIF of the first episode, and report average reward and soups delivered.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from flax.serialization import from_bytes

from jax_marl.eval.overcooked_visualizer import OvercookedVisualizer
from jax_marl.registration import make
from architectures.mlp import ActorCritic as MLPActorCritic
from architectures.cnn import ActorCritic as CNNActorCritic


def load_checkpoint(ckpt_path: Path, train_state: TrainState) -> TrainState:
    """Load Flax TrainState parameters from a checkpoint file."""
    raw = ckpt_path.read_bytes()
    restored = from_bytes({"params": train_state.params}, raw)
    return train_state.replace(params=restored["params"])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Overcooked model checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the .ckpt file containing the model parameters.")
    parser.add_argument("--env_name", type=str, default="overcooked",
                        help="Name of the environment to load.")
    parser.add_argument("--env_kwargs", type=json.loads, required=True,
                        help="JSON dict of keyword args for env creation, e.g. '{\"layout\":\"simple_layout\"}'.")
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Number of episodes to run for evaluation.")
    parser.add_argument("--gif_path", type=Path, default=Path("./first_episode.gif"),
                        help="Output path for the first episode GIF.")
    parser.add_argument("--use_cnn", action="store_true",
                        help="Whether the model uses a CNN backbone (default: MLP).")
    args = parser.parse_args()

    # Create environment
    env = make(args.env_name, **args.env_kwargs)
    num_agents = env.num_agents

    # Initialize network & TrainState
    ActorCritic = CNNActorCritic if args.use_cnn else MLPActorCritic
    net = ActorCritic(env.action_space().n, activation="relu", seq_length=10,
                      use_multihead=False, shared_backbone=False,
                      big_network=False, use_task_id=False,
                      regularize_heads=False, use_layer_norm=False)

    obs_shape = env.observation_space().shape
    if not args.use_cnn:
        obs_dim = int(jnp.prod(jnp.array(obs_shape)))
        init_x = jnp.zeros((1, obs_dim))
    else:
        init_x = jnp.zeros((1, *obs_shape))
    rng = jax.random.PRNGKey(0)
    params = net.init(rng, init_x)

    # Wrap into TrainState for loading
    dummy_state = TrainState.create(apply_fn=net.apply, params=params, tx=None)
    state = load_checkpoint(args.checkpoint, dummy_state)

    # Run evaluation episodes
    total_rewards = []
    total_soups = []
    first_states = []
    viz = OvercookedVisualizer(num_agents=num_agents)

    rng = jax.random.PRNGKey(42)
    for ep in range(args.n_episodes):
        rng, subkey = jax.random.split(rng)
        obs, sim_state = env.reset(subkey)
        done = False
        ep_reward = 0.0
        ep_soup = 0.0
        frames = [sim_state]

        while not done:
            # Prepare batched obs for each agent
            batched = {}
            for ag, o in obs.items():
                o_b = o if o.ndim == len(obs_shape) else o[None]
                if not args.use_cnn:
                    o_b = o_b.reshape((1, -1))
                batched[ag] = o_b

            # Sample actions
            keys = jax.random.split(subkey, num_agents)
            actions = {}
            for i, ag in enumerate(env.agents):
                pi, _ = net.apply(state.params, batched[ag], env_idx=0)
                act = jnp.squeeze(pi.sample(seed=keys[i]), axis=0)
                actions[ag] = act

            rng, step_key = jax.random.split(rng)
            next_obs, next_state, reward, done_info, info = env.step(step_key, sim_state, actions)
            done = done_info["__all__"]
            ep_reward += float(reward["agent_0"])
            ep_soup += float(info["soups"]["agent_0"] + info["soups"]["agent_1"])

            if ep == 0:
                frames.append(next_state)

            obs, sim_state = next_obs, next_state

        total_rewards.append(ep_reward)
        total_soups.append(ep_soup)

    # Save GIF of first episode
    out_dir = args.gif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    viz.animate(frames, agent_view_size=5, task_idx=0, task_name="eval_first", exp_dir=str(out_dir))

    # Report metrics
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_soup = sum(total_soups) / len(total_soups)
    print(f"Ran {args.n_episodes} episodes")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average soups delivered: {avg_soup:.2f}")


if __name__ == "__main__":
    main()


