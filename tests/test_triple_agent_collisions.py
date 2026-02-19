#!/usr/bin/env python
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from meal.env.layouts.presets import cramped_room
from meal.env.overcooked import Overcooked
from meal.visualization.visualizer import OvercookedVisualizer


def test_triple_agent_collisions():
    # ---------------------------------------------------------------------
    # 1.  Create 3-agent env (deterministic spawn -> we know the layout)
    # ---------------------------------------------------------------------
    env = Overcooked(layout=FrozenDict(cramped_room),
                     num_agents=3,
                     random_reset=False,
                     max_steps=50,
                     start_idx=(6, 8, 12))  # fixed spawn for all 3 agents
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    init_pos = np.asarray(state.agent_pos)  # (3,2)  uint32
    print("initial positions:", init_pos)  # sanity-check

    # ---------------------------------------------------------------------
    # 2.  Collect states for GIF
    # ---------------------------------------------------------------------
    viz = OvercookedVisualizer(num_agents=3)
    states = [state]

    # ---------------------------------------------------------------------
    # 3.  Steps: scripted collisions
    # ---------------------------------------------------------------------
    A = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5}

    step_actions = [
        dict(agent_0=jnp.uint32(A['R']),
             agent_1=jnp.uint32(A['L']),
             agent_2=jnp.uint32(A['U'])),
        dict(agent_0=jnp.uint32(A['R']),
             agent_1=jnp.uint32(A['D']),
             agent_2=jnp.uint32(A['L'])),
        dict(agent_0=jnp.uint32(A['D']),
             agent_1=jnp.uint32(A['L']),
             agent_2=jnp.uint32(A['R'])),
        dict(agent_0=jnp.uint32(A['R']),
             agent_1=jnp.uint32(A['L']),
             agent_2=jnp.uint32(A['U'])),
    ]

    for step, act in enumerate(step_actions, 1):
        rng, key = jax.random.split(rng)
        obs, state, _, done, _ = env.step_env(key, state, act)
        states.append(state)

    # ---------------------------------------------------------------------
    # 4.  Save GIF
    # ---------------------------------------------------------------------
    gif_path = Path("gifs/three_agent_collision.gif")
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    viz.animate(states, out_path=str(gif_path))
    print("GIF saved to", gif_path)

    # ---------------------------------------------------------------------
    # 5.  Assertions – positions after 4 steps form a cycle
    # ---------------------------------------------------------------------
    final_pos = np.asarray(state.agent_pos)

    assert np.array_equal(final_pos[0], init_pos[1]), "chef-1 didn't move to chef-2's position!"
    assert np.array_equal(final_pos[1], init_pos[2]), "chef-2 didn't move to chef-3's position!"
    assert np.array_equal(final_pos[2], init_pos[0]), "chef-3 didn't move to chef-1's position!"

    print("Collision test passed ✅")


if __name__ == "__main__":
    test_triple_agent_collisions()
