"""
Partial Observability Example

This example demonstrates the difference between the fully observable `Overcooked`
environment and the partially observable `OvercookedPO` variant:

1. Create both a full-observability env and a partial-observability env on the
   same layout, and compare their observation shapes.
2. Run a random-action episode on the partial-observability env.
3. Render a GIF of the full kitchen with each agent's visible window overlaid
   as a translucent, agent-coloured patch (uses `env.get_agent_view_masks`).

Higher difficulty levels shrink the agent's field of view (fewer tiles ahead/
behind/to the sides) - see `meal/env/overcooked/difficulty_config.py`.
"""

import os

import jax
import jax.numpy as jnp

import meal
from meal.visualization.visualizer_po import OvercookedVisualizerPO


def main():
    print("Partial Observability Example")
    print("=" * 60)

    difficulty = "hard"  # controls both the generated layout size and view_ahead/sides/behind
    num_agents = 2
    seed = 42

    full_env = meal.make_env("overcooked", difficulty=difficulty, num_agents=num_agents, seed=seed)
    po_env = meal.make_env(
        "overcooked_po",
        layout=full_env.layout,
        layout_name=full_env.layout_name,
        num_agents=num_agents,
        difficulty=difficulty,
    )

    print(f"Layout: {full_env.layout_name} ({full_env.width}x{full_env.height})  |  Difficulty (PO view size): {difficulty}")
    print(f"Full observation shape: {full_env.observation_space().shape}")
    print(f"PO observation shape:   {po_env.observation_space().shape}")
    print(f"  view_ahead={po_env.view_ahead}  view_sides={po_env.view_sides}  view_behind={po_env.view_behind}")
    print()

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = po_env.reset(reset_key)
    state_sequence = [state]

    max_steps = 100
    episode_reward = 0.0
    action_dim = po_env.action_space().n

    print("Running episode with random actions...")
    for step in range(max_steps):
        key, action_key, step_key = jax.random.split(key, 3)
        act_keys = jax.random.split(action_key, num_agents)
        actions = {
            agent: jax.random.randint(act_keys[i], shape=(), minval=0, maxval=action_dim)
            for i, agent in enumerate(po_env.agents)
        }
        obs, state, rewards, dones, info = po_env.step(step_key, state, actions)
        state_sequence.append(state)
        episode_reward += float(jnp.sum(jnp.array([rewards[a] for a in po_env.agents])))
        if dones.get("__all__", False):
            break

    print(f"Episode finished! Total reward: {episode_reward:.2f}")
    print(f"Collected {len(state_sequence)} states for GIF")
    print()

    output_dir = "gifs"
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"partial_observability_{difficulty}.gif")

    print("Creating GIF with per-agent view overlay...")
    try:
        # OvercookedVisualizerPO tints each agent's currently visible tiles using
        # env.get_agent_view_masks(state), so you can see exactly what each agent
        # can and can't observe as it moves through the kitchen.
        visualizer = OvercookedVisualizerPO(num_agents=num_agents)
        visualizer.animate(state_seq=state_sequence, out_path=gif_path, env=po_env)
        print(f"✓ GIF of {len(state_sequence)} frames created successfully: {gif_path}")
    except Exception as e:
        print(f"✗ Error creating GIF: {e}")

    print()
    print("Example completed!")


if __name__ == "__main__":
    main()
