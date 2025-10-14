#!/usr/bin/env python3
from pathlib import Path

import jax
import numpy as np
from PIL import Image

from meal.env import Overcooked
from meal.env.overcooked.presets import (
    hard_layouts_legacy,
    medium_layouts_legacy,
    easy_layouts_legacy,
)
from meal.visualization.visualizer import OvercookedVisualizer, TILE_PIXELS


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

            img = vis.render_grid(
                grid,
                tile_size=TILE_PIXELS,
                agent_dir_idx=state.agent_dir_idx,
                agent_inv=state.agent_inv
            )

            img_path = out_dir / f"{name}.png"
            Image.fromarray(img).save(img_path)
            print("Saved", img_path)


if __name__ == "__main__":
    save_start_states(
        {
            "easy": easy_layouts_legacy,
            "medium": medium_layouts_legacy,
            "hard": hard_layouts_legacy,
        }
    )
