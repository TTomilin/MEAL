#!/usr/bin/env python
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from flax.core import FrozenDict

from meal.env.common import OBJECT_TO_INDEX
from meal.env.layouts.presets import cramped_room
from meal.env.overcooked import Overcooked
from meal.visualization.visualizer import OvercookedVisualizer

# ---------------------------------------------------------------------
# 1.  Create 3-agent env (deterministic spawn → we know the layout)
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
# 2.  Helper to add a frame for the GIF
# ---------------------------------------------------------------------
viz = OvercookedVisualizer(num_agents=3, use_old_rendering=False)
frames = []


def add_frame(st):
    pad = env.agent_view_size - 2
    grid = np.asarray(st.maze_map[pad:-pad, pad:-pad, :])

    # Use the new visualization logic
    from collections import namedtuple
    from meal.visualization.rendering.actions import Direction

    # Convert grid to format expected by StateVisualizer
    grid_str = viz._convert_grid_to_str(grid)

    # Create a mapping from environment direction indices to visualization direction tuples
    ENV_DIR_IDX_TO_VIZ_DIR = {
        0: Direction.NORTH,  # (0, -1)
        1: Direction.SOUTH,  # (0, 1)
        2: Direction.EAST,  # (1, 0)
        3: Direction.WEST  # (-1, 0)
    }

    # Create mock players based on agent positions and directions
    MockPlayer = namedtuple('MockPlayer', ['position', 'orientation', 'held_object'])
    players = []

    # Find agent positions in the grid
    agent_positions = []
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x, 0] == OBJECT_TO_INDEX['agent']:
                agent_positions.append((x, y))

    # Create players for each agent position
    for i, pos in enumerate(agent_positions):
        # Convert environment direction index to visualization direction tuple
        # Convert JAX array to int before using as dictionary key
        dir_idx = int(st.agent_dir_idx[i])
        orientation = ENV_DIR_IDX_TO_VIZ_DIR[dir_idx]

        # Create a player with no held object
        players.append(MockPlayer(position=pos, orientation=orientation, held_object=None))

    # Create a mock state
    MockState = namedtuple('MockState', ['players', 'objects'])
    mock_state = MockState(players=players, objects={})

    # Render using StateVisualizer
    surface = viz.state_visualizer.render_state(mock_state, grid_str)
    frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    frames.append(frame)


add_frame(state)

# ---------------------------------------------------------------------
# 3.  One step: chef-0 tries to move right, chef-1 left  → collision
#               chef-2 stays put
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
    add_frame(state)

# ---------------------------------------------------------------------
# 4.  Save GIF so you can watch it
# ---------------------------------------------------------------------
gif_path = Path("gifs/three_agent_collision.gif")
gif_path.parent.mkdir(parents=True, exist_ok=True)
iio.imwrite(gif_path, frames, loop=0, fps=4)
print("GIF saved to", gif_path)

# ---------------------------------------------------------------------
# 5.  Assertions – they must still stand where they started
# ---------------------------------------------------------------------
final_pos = np.asarray(state.agent_pos)

assert np.array_equal(final_pos[0], init_pos[1]), "chef-1 didn't move to chef-2's position!"
assert np.array_equal(final_pos[1], init_pos[2]), "chef-2 didn't move to chef-3's position!"
assert np.array_equal(final_pos[2], init_pos[0]), "chef-3 didn't move to chef-1's position!"

print("Collision test passed ✅")
