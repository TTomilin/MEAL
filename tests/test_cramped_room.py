#!/usr/bin/env python
import os
import sys
from os import makedirs

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from meal.env.layouts.presets import cramped_room
from meal.env.overcooked import Overcooked, DELIVERY_REWARD
from meal.visualization.visualizer import OvercookedVisualizer


def test_cramped_room_double_agent():
    # ---------------------------------------------------------------------
    # 1. Set up env (deterministic reset -> we know the spawn)
    # ---------------------------------------------------------------------
    env = Overcooked(layout=FrozenDict(cramped_room), num_agents=2, random_reset=False,
                     max_steps=400, cook_time=5)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    viz = OvercookedVisualizer(num_agents=2)
    states = [state]

    # ---------------------------------------------------------------------
    # 2. Pre-baked action list (see analysis for reasoning)
    # ---------------------------------------------------------------------
    A = {  # human-readable aliases
        'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5,
    }

    # "pick onion -> pot" pattern
    onion_cycle = [A['L'], A['I'], A['R'], A['U'], A['I']]
    actions = onion_cycle * 3  # 3 onions
    actions += [A['S']] * 5  # wait for cooking (5->0 with cook_time=5)
    actions += [
        A['D'],  # step down
        A['L'],  # step left
        A['D'],  # step down, now facing plate-pile
        A['I'],  # take plate
        A['U'],  # back up
        A['R'],  # step right
        A['U'],  # turn up to pot, face pot
        A['I'],  # scoop soup (now holding dish)
        A['D'],  # step down toward goal line
        A['R'],  # step right
        A['D'],  # turn down, facing the serving window
        A['I'],  # deliver!
    ]

    actions_agent_1 = [A['S']] * 50

    # ---------------------------------------------------------------------
    # 3. Roll out, asserting everything we care about
    # ---------------------------------------------------------------------
    total_reward = 0.0
    total_shaped_0 = 0.0
    total_shaped_1 = 0.0

    for t, act in enumerate(actions):
        rng, step_key = jax.random.split(rng)
        obs, state, rew, done, info = env.step_env(
            step_key, state, {"agent_0": jnp.uint32(act), "agent_1": jnp.uint32(actions_agent_1[t])}
        )

        total_reward += float(rew["agent_0"])
        total_shaped_0 += float(info["shaped_reward"]["agent_0"])
        total_shaped_1 += float(info["shaped_reward"]["agent_1"])
        states.append(state)

    # ---------------------------------------------------------------------
    # 4. Write GIF
    # ---------------------------------------------------------------------
    gif_path = "gifs/double_agent_cramped_room.gif"
    makedirs("gifs", exist_ok=True)
    viz.animate(states, out_path=gif_path)
    print(f"GIF saved to {gif_path}")

    # ---------------------------------------------------------------------
    # 5. Assertions
    # ---------------------------------------------------------------------
    expected_shaped = 3 * 3 + 3 + 5  # 3 onions + plate + soup = 17
    assert np.isclose(total_shaped_0, expected_shaped), f"shaped reward {total_shaped_0} != {expected_shaped}"
    assert total_reward >= float(DELIVERY_REWARD), "didn't get delivery reward!"
    assert done["__all__"] is False, "episode ended prematurely"

    print(f"Success! total_reward = {total_reward}")
    print(f"Success! total_shaped_0 = {total_shaped_0}")
    print(f"Success! total_shaped_1 = {total_shaped_1}")


if __name__ == "__main__":
    test_cramped_room_double_agent()
