#!/usr/bin/env python3
from pathlib import Path

import jax
from PIL import Image

from meal.env import Overcooked
from meal.env.layouts.presets import hard_layouts, medium_layouts, easy_layouts
from meal.visualization.visualizer import OvercookedVisualizer


def save_start_states(grouped_layouts, base_dir: str = "../../assets/screenshots"):
    screenshots_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "assets" / "screenshots"
    key = jax.random.PRNGKey(0)
    vis = OvercookedVisualizer()

    for diff, layouts in grouped_layouts.items():
        out_dir = screenshots_dir / diff
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, layout in layouts.items():
            key, subkey = jax.random.split(key)

            env = Overcooked(layout=layout)
            _, state = env.reset(subkey)

            img = vis.render(state)

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
