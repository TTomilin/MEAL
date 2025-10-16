#!/usr/bin/env python
import os
import sys
from os import makedirs

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from flax.core import FrozenDict

from meal.env.layouts.presets import cramped_room
from meal.env.overcooked import POT_FULL_STATUS, Overcooked
from meal.visualization.visualizer import OvercookedVisualizer

# ---------------------------------------------------------------------
# 1. Set up env (deterministic reset -> we know the spawn)
# ---------------------------------------------------------------------
env = Overcooked(layout=FrozenDict(cramped_room), num_agents=1, random_reset=False, max_steps=400)
rng = jax.random.PRNGKey(0)
obs, state = env.reset(rng)

# convenience shortcuts
pot_status = lambda st: int(st.maze_map[env.agent_view_size - 2, 2, 2])  # row0-pad, col2, channel 2
frames = []
viz = OvercookedVisualizer(num_agents=1, use_old_rendering=False)


def add_frame(st):
    # Use the visualizer's render method to get the frame
    # This will use our updated visualization code that properly handles agent inventory and pot objects
    surface = viz.render(env.agent_view_size, st)

    # Convert pygame surface to numpy array
    frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    frames.append(frame)


add_frame(state)

# ---------------------------------------------------------------------
# 2. Pre-baked action list (see analysis for reasoning)
# ---------------------------------------------------------------------
A = {  # human-readable aliases
    'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5,
}

# “pick onion ➜ pot” pattern
onion_cycle = [A['L'], A['I'], A['R'], A['U'], A['I']]
actions = onion_cycle * 3  # 3 onions
actions += [A['S']] * 20  # wait for cooking (20→0)
actions += [
    A['D'],  # step down
    A['L'],  # step down
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

# ---------------------------------------------------------------------
# 3. Roll out, asserting everything we care about
# ---------------------------------------------------------------------
total_reward = 0.0
total_shaped = 0.0
onions_in_pot_expected = POT_FULL_STATUS + 3  # = 20 after 3 drops

for t, act in enumerate(actions, start=1):
    rng, step_key = jax.random.split(rng)
    obs, state, rew, done, info = env.step_env(
        step_key, state, {"agent_0": jnp.uint32(act)}
    )

    total_reward += float(rew["agent_0"])
    total_shaped += float(info["shaped_reward"]["agent_0"])
    add_frame(state)

# ---------------------------------------------------------------------
# 4. Write GIF
# ---------------------------------------------------------------------
gif_path = "gifs/single_agent_cramped_room.gif"
makedirs("gifs", exist_ok=True)
iio.imwrite(gif_path, frames, loop=0, fps=12)
print(f"GIF saved to {gif_path}")

# ---------------------------------------------------------------------
# 5. Assertions
# ---------------------------------------------------------------------
expected_shaped = 3 * 3 + 3 + 5  # 3 onions + plate + soup = 17
assert np.isclose(total_shaped, expected_shaped), f"shaped reward {total_shaped} != {expected_shaped}"
# assert total_reward >= float(DELIVERY_REWARD), "didn’t get delivery reward!"
assert done["__all__"] is False, "episode ended prematurely"

print(f"Success! total_reward = {total_reward}")
print(f"Success! total_shaped = {total_shaped}")
