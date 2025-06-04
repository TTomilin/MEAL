#!/usr/bin/env python3
from pathlib import Path

import jax
import numpy as np
from PIL import Image

from jax_marl.environments import Overcooked
from jax_marl.environments.overcooked_environment.layouts import (
    hard_layouts,
    medium_layouts,
    easy_layouts,
)
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer, TILE_PIXELS


def crop_to_minimal(state, agent_view_size: int):
    """
    Remove the padding that `make_overcooked_map` adds for agent view.
    Leaves exactly the original grid (outer wall + walkable area).
    """
    pad = agent_view_size - 1  # 5 → 3 with default settings
    if pad == 0:  # in case the view size is changed
        return state.maze_map
    return state.maze_map[pad:-pad, pad:-pad, :]


def save_start_states(grouped_layouts, base_dir: str = "../../assets/screenshots"):
    base_dir = Path(base_dir)
    key = jax.random.PRNGKey(0)
    vis = OvercookedVisualizer()

    for diff, layouts in grouped_layouts.items():
        out_dir = base_dir / diff
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, layout in layouts.items():
            key, subkey = jax.random.split(key)

            env = Overcooked(layout=layout)
            _, state = env.reset(subkey)

            grid = np.asarray(crop_to_minimal(state, env.agent_view_size))

            img = vis._render_grid(
                grid,
                tile_size=TILE_PIXELS,
                highlight_mask=None,
                agent_dir_idx=state.agent_dir_idx,
                agent_inv=state.agent_inv
            )

            img_path = out_dir / f"{name}.png"
            Image.fromarray(img).save(img_path)
            print("Saved", img_path)


if __name__ == "__main__":
    save_start_states(
        {
            "easy": easy_layouts,
            "medium": medium_layouts,
            "hard": hard_layouts,
        }
    )
