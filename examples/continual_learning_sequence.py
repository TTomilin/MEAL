"""
Continual Learning Sequence Example

This example demonstrates how to generate and use sequences of environments for continual learning.
"""

import meal
import jax
import jax.numpy as jnp

def evaluate_on_environment(env, key, num_episodes=3):
    """Simple evaluation function to test an environment."""
    total_rewards = []
    
    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        
        episode_reward = 0
        done = False
        step_count = 0
        max_steps = 100  # Prevent infinite episodes
        
        while not done and step_count < max_steps:
            key, action_key = jax.random.split(key)
            actions = env.action_space().sample(action_key)
            
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(step_key, state, actions)
            
            episode_reward += jnp.sum(rewards)
            done = jnp.any(dones)
            step_count += 1
        
        total_rewards.append(float(episode_reward))
    
    return jnp.mean(jnp.array(total_rewards))

def main():
    print("MEAL Continual Learning Sequence Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Generate a curriculum-based continual learning sequence
    print("Generating curriculum-based CL sequence...")
    cl_envs = meal.make_cl_sequence(
        sequence_length=6,
        strategy='curriculum',
        seed=42
    )
    
    print(f"Generated {len(cl_envs)} environments")
    print()
    
    # Evaluate each environment in the sequence
    print("Evaluating each environment in the sequence:")
    print("-" * 40)
    
    for i, env in enumerate(cl_envs):
        print(f"Environment {i + 1}: {env._cl_task_name}")
        print(f"  Strategy: {env._cl_strategy}")
        print(f"  Task ID: {env._cl_task_id}")
        
        # Evaluate the environment
        key, eval_key = jax.random.split(key)
        avg_reward = evaluate_on_environment(env, eval_key)
        print(f"  Average reward: {avg_reward:.2f}")
        print()
    
    print("=" * 50)
    
    # Generate a random sequence
    print("Generating random-based CL sequence...")
    random_envs = meal.make_cl_sequence(
        sequence_length=4,
        strategy='random',
        seed=123
    )
    
    print(f"Generated {len(random_envs)} random environments")
    for i, env in enumerate(random_envs):
        print(f"  {i + 1}. {env._cl_task_name}")
    print()
    
    # Generate a procedurally generated sequence
    print("Generating procedurally generated CL sequence...")
    generated_envs = meal.make_cl_sequence(
        sequence_length=3,
        strategy='generate',
        seed=456,
        height_rng=(6, 8),
        width_rng=(6, 8),
        wall_density=0.2
    )
    
    print(f"Generated {len(generated_envs)} procedurally generated environments")
    for i, env in enumerate(generated_envs):
        print(f"  {i + 1}. {env._cl_task_name}")
    print()
    
    print("Continual learning sequence example completed!")

if __name__ == "__main__":
    main()