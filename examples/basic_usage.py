"""
Basic MEAL Environment Usage Example

This example demonstrates how to use the gym-style API to create and interact with MEAL environments.
"""

import jax

import meal


def main():
    env = meal.make_env('overcooked')

    key = jax.random.PRNGKey(42)
    key, reset_key = jax.random.split(key)

    obs, state = env.reset(reset_key)

    for step in range(3):
        key, action_key = jax.random.split(key)
        subkeys = jax.random.split(action_key, env.num_agents)
        actions = {agent: env.action_space(agent).sample(sk).item() for agent, sk in zip(env.agents, subkeys)}

        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(step_key, state, actions)

        print(f"Step {step + 1}:")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {rewards}")
        print(f"  Done: {dones}")


if __name__ == "__main__":
    main()
