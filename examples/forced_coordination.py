"""
Forced Coordination Example

`forced_coord` is a classic Overcooked layout where a wall splits the kitchen
in two: one agent's side has the onion piles and pot, the other agent's side
has the plate piles and the delivery goal. Neither agent can complete a soup
alone - onions and dishes must be handed across the counter that separates
the two halves. This is a good stress test for emergent hand-off behaviour.

This example also shows the `agent_restrictions` mechanism, which enforces
role specialisation explicitly (rather than just via layout geometry): agent 0
is forbidden from picking up plates, agent 1 is forbidden from picking up
onions, so they *must* divide the labor even on layouts where nothing
physically stops one agent from doing everything.
"""

import os

import jax
import jax.numpy as jnp

import meal
from meal.visualization.visualizer import OvercookedVisualizer


def rollout(env, key, max_steps=200):
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    state_sequence = [state]

    episode_reward = 0.0
    action_dim = env.action_space().n
    for _ in range(max_steps):
        key, action_key, step_key = jax.random.split(key, 3)
        act_keys = jax.random.split(action_key, env.num_agents)
        actions = {
            agent: jax.random.randint(act_keys[i], shape=(), minval=0, maxval=action_dim)
            for i, agent in enumerate(env.agents)
        }
        obs, state, rewards, dones, info = env.step(step_key, state, actions)
        state_sequence.append(state)
        episode_reward += float(jnp.sum(jnp.array([rewards[a] for a in env.agents])))
        if dones.get("__all__", False):
            break

    return state_sequence, episode_reward


def main():
    print("Forced Coordination Example")
    print("=" * 60)

    seed = 42
    key = jax.random.PRNGKey(seed)
    output_dir = "gifs"
    os.makedirs(output_dir, exist_ok=True)
    visualizer = OvercookedVisualizer(num_agents=2)

    # 1) The `forced_coord` layout: a wall splits ingredients from plates/goal.
    print("Layout 'forced_coord': ingredients and pot on one side, plates and")
    print("delivery goal on the other - agents must pass items across the divide.")
    env = meal.make_env("overcooked", layout_name="forced_coord", num_agents=2)
    key, sub_key = jax.random.split(key)
    state_sequence, reward = rollout(env, sub_key)
    print(f"  Episode reward: {reward:.2f}  ({len(state_sequence)} states)")
    gif_path = os.path.join(output_dir, "forced_coordination_layout.gif")
    visualizer.animate(state_seq=state_sequence, out_path=gif_path)
    print(f"  ✓ {gif_path}")
    print()

    # 2) `cramped_room` + agent_restrictions: no physical wall, but agent 0 may
    #    only ever touch onions/pots and agent 1 may only ever touch plates.
    print("Layout 'cramped_room' + agent_restrictions: agent_0 cannot pick")
    print("plates, agent_1 cannot pick onions - coordination is enforced by")
    print("role, not geometry.")
    restricted_env = meal.make_env(
        "overcooked",
        layout_name="cramped_room",
        num_agents=2,
        agent_restrictions={
            "agent_0_cannot_pick_plates": True,
            "agent_1_cannot_pick_onions": True,
        },
    )
    key, sub_key = jax.random.split(key)
    state_sequence, reward = rollout(restricted_env, sub_key)
    print(f"  Episode reward: {reward:.2f}  ({len(state_sequence)} states)")
    gif_path = os.path.join(output_dir, "forced_coordination_restrictions.gif")
    visualizer.animate(state_seq=state_sequence, out_path=gif_path)
    print(f"  ✓ {gif_path}")

    print()
    print("Example completed!")


if __name__ == "__main__":
    main()
