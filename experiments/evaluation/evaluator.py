#!/usr/bin/env python3
"""
evaluate_checkpoint.py

Load Flax model checkpoints for a sequence of tasks, run N episodes in the Overcooked environment,
record a GIF of each episode, render gameplay in real-time, and report average reward and soups delivered for each task.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax.serialization import from_bytes
from flax.training.train_state import TrainState

from experiments.model.cnn import ActorCritic as CNNActorCritic
from experiments.model.mlp import ActorCritic as MLPActorCritic
from meal import make_env, create_sequence
from meal.visualization.visualizer import OvercookedVisualizer


def load_checkpoint(ckpt_path: Path, train_state: TrainState) -> TrainState:
    """Load Flax TrainState parameters from a checkpoint file."""
    raw = ckpt_path.read_bytes()
    restored = from_bytes({"params": train_state.params}, raw)
    return train_state.replace(params=restored["params"])


def load_config(ckpt_path: Path) -> dict:
    """Load configuration from a checkpoint config file if it exists."""
    config_path = Path(f"{ckpt_path}_config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a sequence of saved Overcooked model checkpoints."
    )
    parser.add_argument("--checkpoint_dir", type=Path, required=True,
                        help="Path to the directory containing checkpoint files.")
    parser.add_argument("--env_name", type=str, default="overcooked",
                        help="Name of the environment to load.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed for environment generation, should match training seed.")
    parser.add_argument("--sequence_length", type=int, default=4,
                        help="Number of tasks in the sequence.")
    parser.add_argument("--strategy", type=str, default="generate",
                        help="Strategy for environment generation (random, ordered, generate).")
    parser.add_argument("--height_min", type=int, default=6,
                        help="Minimum height for generated environments.")
    parser.add_argument("--height_max", type=int, default=7,
                        help="Maximum height for generated environments.")
    parser.add_argument("--width_min", type=int, default=6,
                        help="Minimum width for generated environments.")
    parser.add_argument("--width_max", type=int, default=7,
                        help="Maximum width for generated environments.")
    parser.add_argument("--wall_density", type=float, default=0.15,
                        help="Density of walls in generated environments.")
    parser.add_argument("--n_episodes", type=int, default=1,
                        help="Number of episodes to run for evaluation per task.")
    parser.add_argument("--gif_dir", type=Path, default=Path("./eval_gifs"),
                        help="Output directory for GIFs.")
    parser.add_argument("--use_cnn", action="store_true",
                        help="Whether the model uses a CNN backbone (default: MLP).")
    parser.add_argument("--render_realtime", action="store_true",
                        help="Whether to render the environment in real-time.")
    args = parser.parse_args()

    # Check if the first checkpoint has a config file
    first_checkpoint_path = args.checkpoint_dir / "model_env_1"
    first_config = load_config(first_checkpoint_path) if first_checkpoint_path.exists() else None

    # Use config from checkpoint if available, otherwise use command line args
    if first_config:
        print("Found configuration file. Using stored configuration.")
        # Extract environment generation parameters from config
        sequence_length = first_config.get("num_tasks", args.sequence_length)
        strategy = first_config.get("strategy", args.strategy)
        seed = first_config.get("seed", args.seed)
        height_min = first_config.get("height_min", args.height_min)
        height_max = first_config.get("height_max", args.height_max)
        width_min = first_config.get("width_min", args.width_min)
        width_max = first_config.get("width_max", args.width_max)
        wall_density = first_config.get("wall_density", args.wall_density)

        # Update args with loaded config for later use
        args.sequence_length = sequence_length
        args.use_cnn = first_config.get("use_cnn", args.use_cnn)
    else:
        print("No configuration file found. Using command line arguments.")
        sequence_length = args.sequence_length
        strategy = args.strategy
        seed = args.seed
        height_min = args.height_min
        height_max = args.height_max
        width_min = args.width_min
        width_max = args.width_max
        wall_density = args.wall_density

    # Generate the sequence of environments
    env_kwargs_list, layout_names = create_sequence(
        sequence_length=sequence_length,
        strategy=strategy,
        seed=seed,
        height_rng=(height_min, height_max),
        width_rng=(width_min, width_max),
        wall_density=wall_density
    )

    # Make sure the output directory exists
    args.gif_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    num_agents = 2  # Default for Overcooked
    viz = OvercookedVisualizer(num_agents=num_agents)

    # Evaluate each task in the sequence
    for task_idx in range(args.sequence_length):
        # Find the checkpoint for this task
        checkpoint_path = args.checkpoint_dir / f"model_env_{task_idx + 1}"
        if not checkpoint_path.exists():
            print(f"No checkpoint found for task {task_idx} at {checkpoint_path}. Skipping.")
            continue
        print(f"Using checkpoint: {checkpoint_path}")

        # Load config for this checkpoint if available
        checkpoint_config = load_config(checkpoint_path)

        if checkpoint_config and checkpoint_config.get("env_kwargs"):
            # Use stored environment kwargs and layout name
            env_kwargs = checkpoint_config["env_kwargs"]
            layout_name = checkpoint_config.get("layout_name", f"task_{task_idx}")
            print(f"Using stored environment configuration for task {task_idx}: {layout_name}")
        else:
            # Fall back to generated sequence if available
            if task_idx < len(env_kwargs_list):
                env_kwargs = env_kwargs_list[task_idx]
                layout_name = layout_names[task_idx]
                print(f"Using generated environment for task {task_idx}: {layout_name}")
            else:
                print(f"No environment configuration available for task {task_idx}. Skipping.")
                continue

        print(f"\n--- Evaluating Task {task_idx}: {layout_name} ---")

        # Create environment for this task
        env = make_env(args.env_name, **env_kwargs)
        num_agents = env.num_agents

        # Initialize network & TrainState
        # Use stored network configuration if available
        if checkpoint_config:
            use_cnn = checkpoint_config.get("use_cnn", args.use_cnn)
            activation = checkpoint_config.get("activation", "relu")
            num_tasks = checkpoint_config.get("num_tasks", args.sequence_length)
            use_multihead = checkpoint_config.get("use_multihead", True)
            shared_backbone = checkpoint_config.get("shared_backbone", False)
            big_network = checkpoint_config.get("big_network", False)
            use_task_id = checkpoint_config.get("use_task_id", True)
            regularize_heads = checkpoint_config.get("regularize_heads", False)
            use_layer_norm = checkpoint_config.get("use_layer_norm", True)
        else:
            use_cnn = args.use_cnn
            activation = "relu"
            num_tasks = args.sequence_length
            use_multihead = True
            shared_backbone = False
            big_network = False
            use_task_id = True
            regularize_heads = False
            use_layer_norm = True

        ActorCritic = CNNActorCritic if use_cnn else MLPActorCritic
        net = ActorCritic(env.action_space().n, activation=activation,
                          num_tasks=num_tasks,
                          use_multihead=use_multihead,
                          shared_backbone=shared_backbone,
                          big_network=big_network,
                          use_task_id=use_task_id,
                          regularize_heads=regularize_heads,
                          use_layer_norm=use_layer_norm)

        obs_shape = env.observation_space().shape
        if not use_cnn:
            obs_dim = int(jnp.prod(jnp.array(obs_shape)))
            init_x = jnp.zeros((1, obs_dim))
        else:
            init_x = jnp.zeros((1, *obs_shape))
        rng = jax.random.PRNGKey(0)
        params = net.init(rng, init_x)

        # Wrap into TrainState for loading
        # Use optax.identity() as a dummy optimizer that doesn't modify parameters
        dummy_optimizer = optax.identity()
        dummy_state = TrainState.create(apply_fn=net.apply, params=params, tx=dummy_optimizer)
        state = load_checkpoint(checkpoint_path, dummy_state)

        # Run evaluation episodes for this task
        total_rewards = []
        total_soups = []
        all_frames = []

        rng = jax.random.PRNGKey(42 + task_idx)  # Different seed for each task
        for ep in range(args.n_episodes):
            print(f"Running episode {ep + 1}/{args.n_episodes}")
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
                    if not use_cnn:
                        o_b = o_b.reshape((1, -1))
                    batched[ag] = o_b

                # Sample actions
                keys = jax.random.split(subkey, num_agents)
                actions = {}
                for i, ag in enumerate(env.agents):
                    pi, _, _ = net.apply(state.params, batched[ag], env_idx=0)
                    act = jnp.squeeze(pi.sample(seed=keys[i]), axis=0)
                    actions[ag] = act

                rng, step_key = jax.random.split(rng)
                next_obs, next_state, reward, done_info, info = env.step(step_key, sim_state, actions)
                done = done_info["__all__"]
                ep_reward += float(reward["agent_0"])
                ep_soup += float(info["soups"]["agent_0"] + info["soups"]["agent_1"])

                # Save frame for GIF
                frames.append(next_state)

                # Real-time rendering if requested
                if args.render_realtime:
                    viz.render(next_state, show=True)
                    time.sleep(0.1)  # Small delay to make rendering visible

                obs, sim_state = next_obs, next_state

            total_rewards.append(ep_reward)
            total_soups.append(ep_soup)

            # Save frames from this episode
            if ep == 0:  # Only save GIF for the first episode
                all_frames = frames

        # Save GIF for this task
        gif_path = args.gif_dir / f"task_{task_idx + 1}_{layout_name}.gif"
        viz.animate(all_frames, gif_path)

        # Report metrics for this task
        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_soup = sum(total_soups) / len(total_soups)
        print(f"Task {task_idx} ({layout_name}) Results:")
        print(f"  Ran {args.n_episodes} episodes")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average soups delivered: {avg_soup:.2f}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
