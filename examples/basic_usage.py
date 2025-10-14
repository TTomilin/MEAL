"""
Basic MEAL Environment Usage Example

This example demonstrates how to use the gym-style API to create and interact with MEAL environments.
"""

import meal
import jax
import jax.numpy as jnp

def main():
    print("MEAL Basic Usage Example")
    print("=" * 40)

    # List available environments
    print("Available environments:")
    for env_id in meal.list_envs():
        print(f"  - {env_id}")
    print()

    # Create a basic environment using gym-style API
    print("Creating environment using meal.make_env()...")
    env = meal.make_env('overcooked')
    print(f"Environment created: {type(env).__name__}")
    print()

    # Alternative: use the original make function
    print("Creating environment using meal.make()...")
    env2 = meal.make('overcooked')
    print(f"Environment created: {type(env2).__name__}")
    print()

    # Initialize environment
    key = jax.random.PRNGKey(42)
    key, reset_key = jax.random.split(key)

    print("Resetting environment...")
    obs, state = env.reset(reset_key)
    print(f"Observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"Observation keys: {list(obs.keys())}")
        for obs_key, value in obs.items():
            print(f"  {obs_key} shape: {value.shape}")
    else:
        print(f"Observation shape: {obs.shape}")
    print(f"State type: {type(state)}")
    print()

    # Take a few random actions
    print("Taking random actions...")
    for step in range(3):
        key, action_key = jax.random.split(key)
        actions = env.action_space().sample(action_key)

        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(step_key, state, actions)

        print(f"Step {step + 1}:")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {rewards}")
        print(f"  Done: {dones}")
        print()

    print("Basic usage example completed!")

if __name__ == "__main__":
    main()
