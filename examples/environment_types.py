"""
Environment Types Example

This example demonstrates the different types of environments available in MEAL
and how to configure them with various parameters.
"""

import meal
import jax
import jax.numpy as jnp

def print_env_info(env, env_name):
    """Print basic information about an environment."""
    print(f"Environment: {env_name}")
    print(f"  Type: {type(env).__name__}")
    print(f"  Action space: {env.action_space()}")
    print(f"  Observation space: {env.observation_space()}")
    
    # Reset to get observation shape
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    print(f"  Observation shape: {obs.shape}")
    print(f"  Number of agents: {env.num_agents}")
    print()

def main():
    print("MEAL Environment Types Example")
    print("=" * 40)
    
    # 1. Standard Overcooked environment
    print("1. Standard Overcooked Environment")
    print("-" * 30)
    env1 = meal.make_env('overcooked')
    print_env_info(env1, 'overcooked')
    
    # 2. Partially Observable Overcooked
    print("2. Partially Observable Overcooked")
    print("-" * 30)
    env2 = meal.make_env('overcooked_po')
    print_env_info(env2, 'overcooked_po')
    
    # 3. N-Agent Overcooked (3 agents)
    print("3. N-Agent Overcooked (3 agents)")
    print("-" * 30)
    env3 = meal.make_env('overcooked_n_agent', num_agents=3)
    print_env_info(env3, 'overcooked_n_agent (3 agents)')
    
    # 4. N-Agent Overcooked (4 agents)
    print("4. N-Agent Overcooked (4 agents)")
    print("-" * 30)
    env4 = meal.make_env('overcooked_n_agent', num_agents=4)
    print_env_info(env4, 'overcooked_n_agent (4 agents)')
    
    # 5. Environment with specific layout
    print("5. Environment with Specific Layout")
    print("-" * 30)
    # Note: This would require a specific layout, but we'll show the concept
    try:
        env5 = meal.make_env('overcooked', layout='simple')
        print_env_info(env5, 'overcooked (simple layout)')
    except Exception as e:
        print(f"Layout 'simple' not available: {e}")
        print("Using default layout instead...")
        env5 = meal.make_env('overcooked')
        print_env_info(env5, 'overcooked (default layout)')
    
    # 6. Demonstrate environment interaction
    print("6. Environment Interaction Demo")
    print("-" * 30)
    
    env = meal.make_env('overcooked')
    key = jax.random.PRNGKey(42)
    
    # Reset environment
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial state type: {type(state)}")
    
    # Take a few steps
    for step in range(3):
        key, action_key = jax.random.split(key)
        actions = env.action_space().sample(action_key)
        
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(step_key, state, actions)
        
        print(f"Step {step + 1}:")
        print(f"  Actions taken: {actions}")
        print(f"  Rewards received: {rewards}")
        print(f"  Episode done: {dones}")
        print(f"  Info keys: {list(info.keys()) if hasattr(info, 'keys') else 'N/A'}")
    
    print()
    
    # 7. Show all available environments
    print("7. All Available Environments")
    print("-" * 30)
    all_envs = meal.list_envs()
    for i, env_id in enumerate(all_envs, 1):
        print(f"  {i}. {env_id}")
    
    print()
    print("Environment types example completed!")

if __name__ == "__main__":
    main()