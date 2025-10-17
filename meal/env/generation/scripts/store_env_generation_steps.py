#!/usr/bin/env python3
"""
Store the key stages of a procedurally generated Overcooked kitchen.

Usage examples
--------------
# easy layout with fixed seed
python env_gen_steps.py --difficulty easy --seed 7

# custom ranges
python env_gen_steps.py --h-min 9 --h-max 10 --density 0.22
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from meal.env.common import (
    FLOOR, WALL, GOAL, ONION_PILE, PLATE_PILE, POT, AGENT,
    OBJECT_TO_INDEX, COLOR_TO_INDEX,
)
from meal.env.generation.layout_generator import place_tiles, remove_unreachable_items
from meal.env.generation.layout_validator import evaluate_grid
from meal.visualization.visualizer import OvercookedVisualizer

# ──────────────────────────────────────────────────────────────────────────
# Configurable output folder
OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "assets" / "screenshots" / "env_generation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# Char→tensor helpers (red chef, blue chef)
_AGENT_COLORS = [COLOR_TO_INDEX["red"], COLOR_TO_INDEX["blue"]]
_STATIC = {
    FLOOR: (OBJECT_TO_INDEX["empty"], COLOR_TO_INDEX["grey"], 0),
    WALL: (OBJECT_TO_INDEX["wall"], COLOR_TO_INDEX["grey"], 0),
    GOAL: (OBJECT_TO_INDEX["goal"], COLOR_TO_INDEX["green"], 0),
    POT: (OBJECT_TO_INDEX["pot"], COLOR_TO_INDEX["black"], 0),
    ONION_PILE: (OBJECT_TO_INDEX["onion_pile"], COLOR_TO_INDEX["yellow"], 0),
    PLATE_PILE: (OBJECT_TO_INDEX["plate_pile"], COLOR_TO_INDEX["white"], 0),
}


def grid_to_tensor(char_grid: list[list[str]]) -> np.ndarray:
    h, w = len(char_grid), len(char_grid[0])
    out, agent_id = np.zeros((h, w, 3), np.int8), 0
    for i, row in enumerate(char_grid):
        for j, ch in enumerate(row):
            if ch == AGENT:
                out[i, j] = (OBJECT_TO_INDEX["agent"], _AGENT_COLORS[agent_id], 0)
                agent_id += 1
            else:
                out[i, j] = _STATIC.get(ch, _STATIC[FLOOR])
    return out


def save_png(vis: OvercookedVisualizer, char_grid, fname):
    img = vis.render_grid(char_grid)
    Image.fromarray(img).save(OUT_DIR / fname)


# ──────────────────────────────────────────────────────────────────────────
def try_generate(rng: random.Random, h_rng, w_rng, density):
    """Return list[grid_stage] if valid, else None."""
    h, w = rng.randint(*h_rng), rng.randint(*w_rng)

    # 0. blank + border
    g = [[FLOOR for _ in range(w)] for _ in range(h)]
    for i in range(h): g[i][0] = g[i][-1] = WALL
    for j in range(w): g[0][j] = g[-1][j] = WALL
    stages = [[row[:] for row in g]]  # deep-copy

    # 1. interactive tiles
    for sym in (GOAL, ONION_PILE, PLATE_PILE):
        place_tiles(g, sym, rng.randint(1, 2), rng)
    # Always place exactly 2 pots
    place_tiles(g, POT, 2, rng)
    stages.append([row[:] for row in g])

    # 2. walls to reach density
    int_cells = (h - 2) * (w - 2)
    target = round(density * int_cells)
    cur = sum(ch != FLOOR for row in g[1:-1] for ch in row[1:-1])
    place_tiles(g, WALL, max(0, target - cur), rng)
    stages.append([row[:] for row in g])

    # 3. agents + cleanup
    place_tiles(g, AGENT, 2, rng)
    remove_unreachable_items(g)

    # Ensure exactly 2 pots remain after cleanup
    pot_count = sum(1 for row in g for cell in row if cell == POT)
    if pot_count < 2:
        additional_pots_needed = 2 - pot_count
        if not place_tiles(g, POT, additional_pots_needed, rng):
            return None  # Retry if we can't place additional pots

    stages.append([row[:] for row in g])

    # validate
    ok, reason = evaluate_grid("\n".join("".join(r) for r in g))
    return stages if ok else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-attempts", type=int, default=100)

    # difficulty presets
    diff = p.add_mutually_exclusive_group()
    diff.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")

    # manual overrides
    p.add_argument("--h-min", type=int)
    p.add_argument("--h-max", type=int)
    p.add_argument("--w-min", type=int)
    p.add_argument("--w-max", type=int)
    p.add_argument("--density", type=float)

    args = p.parse_args()
    # difficulty → defaults
    if args.difficulty == "easy":
        h_rng = w_rng = (6, 7)
        density = 0.15
    elif args.difficulty == "medium":
        h_rng = w_rng = (8, 9)
        density = 0.25
    elif args.difficulty == "hard":
        h_rng = w_rng = (10, 11)
        density = 0.35
    else:  # custom
        h_rng = (args.h_min or 6, args.h_max or 11)
        w_rng = (args.w_min or 6, args.w_max or 11)
        density = args.density if args.density is not None else 0.20

    seed = args.seed if args.seed is not None else random.randint(0, 2 ** 31 - 1)
    rng = random.Random(seed)
    vis = OvercookedVisualizer()

    for attempt in range(1, args.max_attempts + 1):
        stages = try_generate(rng, h_rng, w_rng, density)
        if stages is not None:
            for k, stage in enumerate(stages):
                save_png(vis, stage, f"step_{k + 1}.png")
            print(f"Valid layout after {attempt} attempt(s) → PNGs saved in {OUT_DIR}")
            break
    else:
        raise RuntimeError(f"No valid layout in {args.max_attempts} attempts.")


if __name__ == "__main__":
    main()
