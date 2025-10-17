"""
Continual Learning Sequence Example

This example demonstrates how to generate and use sequences of environments for continual learning.
"""
import os

import meal
import jax
import jax.numpy as jnp

from meal.visualization.visualizer import OvercookedVisualizer


def evaluate_on_environment(env, key, num_episodes=1):
    """Simple evaluation function to test an environment."""
    total_rewards = []
    states = []
    
    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        states.append(state)
        
        episode_reward = 0
        max_steps = 25

        for _ in range(max_steps):
            key, action_key = jax.random.split(key)
            subkeys = jax.random.split(action_key, env.num_agents)
            actions = {
                agent: env.action_space(agent).sample(sk).item()
                for agent, sk in zip(env.agents, subkeys)
            }
            
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(step_key, state, actions)
            states.append(state)

            # Sum per-agent reward
            episode_reward += float(jnp.sum(jnp.array([rewards[a] for a in env.agents])))

        total_rewards.append(float(episode_reward))
    
    return jnp.mean(jnp.array(total_rewards)), states

def main():
    print("MEAL Continual Learning Sequence Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)

    # Define continual learning strategy
    strategy = 'curriculum'
    
    # Generate a curriculum-based continual learning sequence
    print("Generating curriculum-based CL sequence...")
    envs = meal.make_sequence(
        sequence_length=6,
        strategy=strategy,
        seed=42
    )
    
    print(f"Generated {len(envs)} environments")
    print()

    # Store states for GIF creation
    state_sequence = []

    # Initialize the visualizer
    print("Initializing visualizer...")
    visualizer = OvercookedVisualizer(num_agents=envs[0].num_agents)

    # Evaluate each environment in the sequence
    print("Evaluating each environment in the sequence:")
    print("-" * 40)
    
    for i, env in enumerate(envs):
        print(f"Environment {i + 1}: {env.task_name}")
        print(f"  Task ID: {env.task_id}")
        print(f"  Difficulty: {env.difficulty}")

        # Evaluate the environment
        key, eval_key = jax.random.split(key)
        avg_reward, states = evaluate_on_environment(env, eval_key)
        state_sequence.extend(states)
        print(f"  Average reward: {avg_reward:.2f}")
        print()
    
    print("=" * 50)

    print(f"Collected {len(state_sequence)} states for GIF")
    print()

    # Create output directory
    output_dir = "gifs"
    os.makedirs(output_dir, exist_ok=True)

    # Create GIF
    print("Creating GIF...")
    gif_path = os.path.join(output_dir, f"{envs[0].num_agents}_agents_{strategy}.gif")
    try:
        visualizer.animate(state_seq=state_sequence, out_path=gif_path, pad_to_max=True)
        print(f"✓ GIF of {len(state_sequence)} frames created successfully: {gif_path}")

    except Exception as e:
        print(f"✗ Error creating GIF: {e}")
        print("This might be due to missing dependencies (PIL, pygame) or display issues.")
        print("The episode ran successfully, but GIF creation failed.")
    
    print("Continual learning sequence example completed!")

if __name__ == "__main__":
    main()