#!/usr/bin/env python
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from jax_marl.environments.overcooked_environment.layouts import cramped_room
from jax_marl.environments.overcooked_environment.overcooked_n_agent import Overcooked
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer

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
viz = OvercookedVisualizer(num_agents=3)
frames = []


def add_frame(st):
    pad = env.agent_view_size - 2
    grid = np.asarray(st.maze_map[pad:-pad, pad:-pad, :])
    frame = viz._render_grid(grid,
                             agent_dir_idx=np.atleast_1d(st.agent_dir_idx),
                             agent_inv=np.atleast_1d(st.agent_inv))
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
