#!/usr/bin/env python
import os
import sys
from os import makedirs

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from meal.env.layouts.presets import cramped_room
from meal.env.overcooked import DELIVERY_REWARD
from meal.env.overcooked_po import OvercookedPO
from meal.visualization.visualizer_po import OvercookedVisualizerPO


def test_cramped_room_po():
    # ---------------------------------------------------------------------
    # 1. Set up PO env (deterministic reset -> we know the spawn)
    # ---------------------------------------------------------------------
    env = OvercookedPO(
        layout=FrozenDict(cramped_room),
        num_agents=2,
        random_reset=False,
        max_steps=400,
        cook_time=5,
        view_ahead=3,
        view_behind=1,
        view_sides=1,
    )
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    viz = OvercookedVisualizerPO(num_agents=2)
    states = [state]

    # ---------------------------------------------------------------------
    # 2. Pre-baked action list (same as original test)
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
    # 4. Write GIF with PO visualization
    # ---------------------------------------------------------------------
    gif_path = "gifs/double_agent_cramped_room_po.gif"
    makedirs("gifs", exist_ok=True)
    viz.animate(states, out_path=gif_path, env=env)
    print(f"PO GIF saved to {gif_path}")
    print(f"View areas are highlighted with:")
    print("- Light red for agent 0's view area")
    print("- Light blue for agent 1's view area")
    print("- Purple where both agents can see")

    # ---------------------------------------------------------------------
    # 5. Assertions (adapted for PO environment)
    # ---------------------------------------------------------------------
    expected_shaped_min = 10  # Minimum reasonable shaped reward for completing the task
    assert total_shaped_0 >= expected_shaped_min, f"shaped reward {total_shaped_0} too low (< {expected_shaped_min})"
    assert total_reward >= float(DELIVERY_REWARD), "didn't get delivery reward!"
    assert done["__all__"] is False, "episode ended prematurely"

    print(f"Success! total_reward = {total_reward}")
    print(f"Success! total_shaped_0 = {total_shaped_0}")
    print(f"Success! total_shaped_1 = {total_shaped_1}")

    print(f"\nPartial Observability Configuration:")
    print(f"- View ahead: {env.view_ahead} tiles")
    print(f"- View behind: {env.view_behind} tiles")
    print(f"- View sides: {env.view_sides} tiles")


if __name__ == "__main__":
    test_cramped_room_po()
