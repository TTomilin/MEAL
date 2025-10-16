"""
Four Agent Level 3 Environment with GIF Rendering

This example demonstrates how to:
1. Create a 4-agent environment at level 3 difficulty (hard)
2. Run an episode with random actions sampled from the action space
3. Render the episode as a GIF

The script creates a challenging environment with 4 agents working together
in a procedurally generated hard-difficulty kitchen layout.
"""


import meal
import jax
import jax.numpy as jnp
import os
from meal.visualization.visualizer import OvercookedVisualizer

def main():
    print("Four Agent Level 3 Environment with GIF Rendering")
    print("=" * 60)

    num_agents = 4
    difficulty = 'hard'

    # Set random seed for reproducibility
    seed = 42
    key = jax.random.PRNGKey(seed)

    # Create a 4-agent environment with hard difficulty (level 3)
    print(f"Creating a {num_agents}-agent environment with level 3 (hard) difficulty...")
    env = meal.make_env('overcooked', difficulty=difficulty, num_agents=num_agents)

    print(f"Environment created: {type(env).__name__}")
    print(f"Number of agents: {env.num_agents}")
    print(f"Action space: {env.action_space().n}")
    print(f"Observation space: {env.observation_space().shape}")
    print()

    # Reset environment and collect states for GIF
    print("Running episode with random actions...")
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    # Store states for GIF creation
    state_sequence = [state]

    # Run episode for a reasonable number of steps
    max_steps = 100
    episode_reward = 0

    for step in range(max_steps):
        # Sample random actions from action space
        key, action_key = jax.random.split(key)
        subkeys = jax.random.split(action_key, num_agents)
        actions = {agent: env.action_space(agent).sample(sk).item() for agent, sk in zip(env.agents, subkeys)}

        # Take step in environment
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(step_key, state, actions)

        # Store state for GIF
        state_sequence.append(state)

        # Sum per-agent reward
        episode_reward += float(jnp.sum(jnp.array([rewards[a] for a in env.agents])))

        # Print progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{max_steps}: Total reward = {episode_reward:.2f}")

    print(f"Episode finished! Total reward: {episode_reward:.2f}")
    print(f"Collected {len(state_sequence)} states for GIF")
    print()

    # Initialize the visualizer
    print("Initializing visualizer...")
    visualizer = OvercookedVisualizer(num_agents=num_agents)

    # Create output directory
    output_dir = "gifs"
    os.makedirs(output_dir, exist_ok=True)

    # Create GIF
    print("Creating GIF...")
    gif_path = os.path.join(output_dir, f"{num_agents}_agents_{difficulty}.gif")
    try:
        visualizer.animate(state_seq=state_sequence, out_path=gif_path)
        print(f"✓ GIF of {len(state_sequence)} frames created successfully: {gif_path}")

    except Exception as e:
        print(f"✗ Error creating GIF: {e}")
        print("This might be due to missing dependencies (PIL, pygame) or display issues.")
        print("The episode ran successfully, but GIF creation failed.")

    print()
    print("Example completed!")

if __name__ == "__main__":
    main()
