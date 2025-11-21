import os
import sys
from os import makedirs

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from meal.env.overcooked import Overcooked
from meal.wrappers.slippery_tiles import SlipperyTiles
from meal.visualization.visualizer import OvercookedVisualizer


def make_empty_rect_layout(H: int = 6, W: int = 14) -> FrozenDict:
    """
    Create a simple empty layout:
    - Size x Size grid
    - Outer border is walls (counters)
    - Interior is walkable floor
    - Two fixed agent spawn points: left and right on the middle row
    - One goal, one onion pile, one plate pile, one pot on the bottom inner row
    """
    # -------------------------------------------------------------------------
    # Walls: outer border
    # -------------------------------------------------------------------------
    wall_idx = []
    for y in range(H):
        for x in range(W):
            if x == 0 or x == W - 1 or y == 0 or y == H - 1:
                wall_idx.append(y * W + x)
    wall_idx = jnp.array(wall_idx, dtype=jnp.uint32)

    # -------------------------------------------------------------------------
    # Agents: middle row, left and right
    # -------------------------------------------------------------------------
    mid_y = H // 2
    left_x = 1
    right_x = W - 2
    agent_idx = jnp.array(
        [
            mid_y * W + left_x,   # agent_0 on the left
            mid_y * W + right_x,  # agent_1 on the right
        ],
        dtype=jnp.uint32,
    )

    # -------------------------------------------------------------------------
    # Objects on bottom inner row
    # (row just above the bottom wall: y = H - 2, x in [1, W-2])
    # -------------------------------------------------------------------------
    bottom_y = H - 2

    # Keep them reasonably spaced; assume size >= 8 is fine for your use case
    goal_pos      = (2, bottom_y)
    onion_pile_pos = (4, bottom_y)
    plate_pile_pos = (6, bottom_y)
    pot_pos       = (W - 3, bottom_y)

    goal_idx = jnp.array(
        [goal_pos[1] * W + goal_pos[0]],
        dtype=jnp.uint32,
    )
    onion_pile_idx = jnp.array(
        [onion_pile_pos[1] * W + onion_pile_pos[0]],
        dtype=jnp.uint32,
    )
    plate_pile_idx = jnp.array(
        [plate_pile_pos[1] * W + plate_pile_pos[0]],
        dtype=jnp.uint32,
    )
    pot_idx = jnp.array(
        [pot_pos[1] * W + pot_pos[0]],
        dtype=jnp.uint32,
    )

    layout = FrozenDict(
        {
            "height": H,
            "width": W,
            "wall_idx": wall_idx,
            "goal_idx": goal_idx,
            "onion_pile_idx": onion_pile_idx,
            "plate_pile_idx": plate_pile_idx,
            "pot_idx": pot_idx,
            "agent_idx": agent_idx,
        }
    )
    return layout


def test_slippery_tiles_two_agents_meet():
    """
    Visual test of slipperiness:
    - Large empty grid with border walls
    - Agent 0 always *tries* to move right
    - Agent 1 always *tries* to move left
    - SlipperyTiles with slip_prob=1.0 so they constantly slip
    - Records a GIF of the resulting chaos
    """
    print("=== SLIPPERY TILES VISUAL TEST: two agents walking towards each other ===")

    layout = make_empty_rect_layout()

    base_env = Overcooked(
        layout=layout,
        layout_name="slippery_empty_grid",
        num_agents=2,
        random_reset=False,
        max_steps=200,
    )
    env = SlipperyTiles(base_env, slip_prob=1.0)

    # Action indices (must match Overcooked.Actions)
    # 0: up, 1: down, 2: right, 3: left, 4: stay, 5: interact
    A_RIGHT = 2
    A_LEFT = 3

    horizon = 60
    actions_agent_0 = [A_RIGHT] * horizon
    actions_agent_1 = [A_LEFT] * horizon

    # Reset env
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    # Collect underlying Overcooked states (for the visualizer)
    states = [state.env_state]

    for t in range(horizon):
        rng, step_key = jax.random.split(rng)
        action_dict = {
            "agent_0": jnp.uint32(actions_agent_0[t]),
            "agent_1": jnp.uint32(actions_agent_1[t]),
        }
        obs, state, rew, done, info = env.step(step_key, state, action_dict)
        states.append(state.env_state)

        if done["__all__"]:
            break

    # Save GIF
    makedirs("gifs", exist_ok=True)
    gif_path = "gifs/test_slippery_tiles_two_agents_meet.gif"
    viz = OvercookedVisualizer()
    viz.animate(states, gif_path)
    print(f"\nGIF saved to {gif_path}")
    print("🎢 Slippery tiles test finished.")


if __name__ == "__main__":
    test_slippery_tiles_two_agents_meet()
