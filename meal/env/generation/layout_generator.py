from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict

from meal.env.common import FLOOR, WALL, GOAL, ONION_PILE, PLATE_PILE, POT, AGENT
from meal.env.generation.layout_validator import evaluate_grid, UNPASSABLE_TILES, INTERACTIVE_TILES
from meal.env.layouts.presets import layout_grid_to_dict
from meal.env.utils.difficulty_config import get_difficulty_params


###############################################################################
# ─── Generator ───────────────────────────────────────────────────────────────
###############################################################################

def _random_empty_cell(grid: List[List[str]], rng: random.Random) -> Optional[Tuple[int, int]]:
    """Return a random (row, col) index of a FLOOR cell or ``None`` if none exist."""
    empties = [
        (i, j)
        for i in range(1, len(grid) - 1)
        for j in range(1, len(grid[0]) - 1)
        if grid[i][j] == FLOOR
    ]
    if not empties:
        return None
    return rng.choice(empties)


def place_tiles(
        grid: List[List[str]],
        tile_symbol: str,
        count: int,
        rng: random.Random,
) -> bool:
    """Place *count* copies of *tile_symbol* on random FLOOR cells.

    Returns ``True`` on success, ``False`` if not enough empty space was available.
    """
    for _ in range(count):
        cell = _random_empty_cell(grid, rng)
        if cell is None:
            return False
        i, j = cell
        grid[i][j] = tile_symbol
    return True


def remove_unreachable_items(grid: List[List[str]]) -> bool:
    """Remove interactive tiles that are not reachable by any agent and replace unreachable floor tiles with walls.

    Returns True if any items were removed or floor tiles were replaced, False otherwise.
    """
    # Find agent positions
    agents = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == AGENT]
    if len(agents) < 1:
        return False  # No agents to check reachability

    # Get reachable positions for each agent
    height, width = len(grid), len(grid[0])

    # Create a temporary function to get reachable positions
    def get_reachable_positions(start_i: int, start_j: int):
        # Initialize tracking structures
        visited = [[False for _ in range(width)] for _ in range(height)]
        reachable_positions = set()

        def dfs(i: int, j: int):
            # Check bounds and if already visited
            if (i < 0 or i >= height or j < 0 or j >= width or visited[i][j]):
                return

            # Check if we can occupy this cell
            if grid[i][j] not in (FLOOR, AGENT):
                # Mark as visited to prevent revisiting
                visited[i][j] = True
                return

            visited[i][j] = True
            reachable_positions.add((i, j))

            # Continue DFS in all directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for di, dj in directions:
                dfs(i + di, j + dj)

        # Start DFS from agent position
        dfs(start_i, start_j)
        return reachable_positions

    # Get reachable positions for each agent
    all_reachable = set()
    for agent_i, agent_j in agents:
        agent_reachable = get_reachable_positions(agent_i, agent_j)
        all_reachable.update(agent_reachable)

    # Find interactive tiles
    interactive_tiles = []
    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch in INTERACTIVE_TILES:
                interactive_tiles.append((i, j, ch))

    # Check if each interactive tile is reachable by at least one agent
    items_removed = False
    for i, j, tile_type in interactive_tiles:
        # Check if any adjacent position is reachable
        is_reachable = False
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < height and 0 <= nj < width and
                    (ni, nj) in all_reachable):
                is_reachable = True
                break

        # If not reachable, replace with FLOOR
        if not is_reachable:
            grid[i][j] = FLOOR
            items_removed = True

    # Replace unreachable floor tiles with walls
    for i in range(height):
        for j in range(width):
            # Skip border cells and non-floor cells
            if (i == 0 or i == height - 1 or j == 0 or j == width - 1 or grid[i][j] != FLOOR):
                continue

            # If this floor tile is not reachable, replace it with a wall
            if (i, j) not in all_reachable:
                grid[i][j] = WALL
                items_removed = True

    return items_removed


def generate_random_layout(
        num_agents: int = 2,
        difficulty: str | None = None,
        height_rng: Tuple[int, int] = (5, 10),
        width_rng: Tuple[int, int] = (5, 10),
        wall_density: float = 0.15,
        seed: Optional[int] = None,
        max_attempts: int = 2000,
        allow_invalid: bool = False,
):
    """Generate and return a random solvable Overcooked layout.

    The procedure is:
    1. Draw random width/height.
    2. Add an outer border of walls.
    3. Place interactive tiles (goal, pot, onion pile, plate pile).
    4. Compute how many *additional* walls are needed so that the total
       fraction of unpassable internal cells equals ``wall_density``.
    5. Place those walls.
    6. Finally place the agents.
    7. Remove any interactive tiles that are not reachable by any agent.

    The process is retried up to ``max_attempts`` times if any stage runs
    out of empty cells or the resulting grid fails the solvability check.
    """
    rng = random.Random(seed)

    if difficulty:
        params = get_difficulty_params(difficulty)
        height_rng = params["height_rng"]
        width_rng = params["width_rng"]
        wall_density = params["wall_density"]

    for attempt in range(1, max_attempts + 1):
        height = rng.randint(*height_rng)
        width = rng.randint(*width_rng)

        # Initialise grid with FLOOR
        grid = [[FLOOR for _ in range(width)] for _ in range(height)]

        # Outer walls
        for i in range(height):
            grid[i][0] = grid[i][-1] = WALL
        for j in range(width):
            grid[0][j] = grid[-1][j] = WALL

        # 1. Interactive tiles -------------------------------------------------
        # Up to two of each interactive type
        for symbol in (GOAL, POT, ONION_PILE, PLATE_PILE):
            # Generate more pots than other items to increase throughput potential
            copies = rng.randint(2, 3) if symbol == POT else rng.randint(1, 2)
            if not place_tiles(grid, symbol, copies, rng):
                print(f"[Attempt {attempt}] Not enough space for {symbol}. Retrying…")
                break  # go to next attempt
        else:  # executed if the loop *didn't* break: all good so far
            # 2. Additional walls so that density matches ----------------------
            internal_cells = (height - 2) * (width - 2)
            current_unpassable = sum(
                1
                for i in range(1, height - 1)
                for j in range(1, width - 1)
                if grid[i][j] in UNPASSABLE_TILES
            )
            target_unpassable = int(round(wall_density * internal_cells))
            additional_walls_needed = max(0, target_unpassable - current_unpassable)

            if not place_tiles(grid, WALL, additional_walls_needed, rng):
                print(f"[Attempt {attempt}] Could not reach desired wall density. Retrying…")
                continue  # next attempt

            # 3. Agents --------------------------------------------------------
            if not place_tiles(grid, AGENT, num_agents, rng):
                print(f"[Attempt {attempt}] Not enough space for agents. Retrying…")
                continue

            # 4. Remove unreachable items --------------------------------------
            items_removed = remove_unreachable_items(grid)
            if items_removed:
                print(
                    f"[Attempt {attempt}] Removed unreachable interactive tiles and replaced unreachable floor tiles with walls.")

            # 5. Ensure exactly 2 pots are in the layout -----------------------------
            pot_count = sum(1 for row in grid for cell in row if cell == POT)
            if pot_count < 2:
                print(
                    f"[Attempt {attempt}] Only {pot_count} pots remain after removing unreachable items. Adding more pots.")
                additional_pots_needed = 2 - pot_count
                if not place_tiles(grid, POT, additional_pots_needed, rng):
                    print(f"[Attempt {attempt}] Could not add more pots. Retrying...")
                    continue  # next attempt

            # Convert to string and validate -----------------------------------
            grid_str = "\n".join("".join(row) for row in grid)
            is_valid, reason = evaluate_grid(grid_str, num_agents=num_agents)
            if is_valid or allow_invalid:
                return grid_str, layout_grid_to_dict(grid_str)

            print(f"[Attempt {attempt}] Generated layout not solvable: {reason}. Retrying…")

    raise RuntimeError(
        f"Failed to generate a solvable layout in {max_attempts} attempts."
    )


###############################################################################
# ─── Matplotlib preview ─────────────────────────────────────────────────────
###############################################################################

_TILE_COLOUR = {
    WALL: (0.5, 0.5, 0.5),
    GOAL: (0.1, 0.1, 0.1),
    ONION_PILE: (1, 0.9, 0.2),
    PLATE_PILE: (0.2, 0.2, 0.8),
    POT: (0.9, 0.2, 0.2),
    AGENT: (0.1, 0.7, 0.3),
    FLOOR: (1, 1, 1),
}


def mpl_show(grid_str: str, title: str | None = None):
    rows = grid_str.strip().split("\n")
    height, width = len(rows), len(rows[0])
    img = np.zeros((height, width, 3))
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            img[y, x] = _TILE_COLOUR[ch]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width / 2, height / 2))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="black", lw=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


###############################################################################
# ─── Overcooked viewer ──────────────────────────────────────────────────────
###############################################################################

def oc_show(layout: FrozenDict, num_agents: int = 2):
    from meal.env import Overcooked
    from meal.visualization.visualizer import OvercookedVisualizer

    env = Overcooked(layout=layout, layout_name="random_gen", random_reset=False, num_agents=num_agents)
    _, state = env.reset(jax.random.PRNGKey(0))
    vis = OvercookedVisualizer(num_agents)
    vis.render(state, show=True)


###############################################################################
# ─── CLI ────────────────────────────────────────────────────────────────────
###############################################################################

def main(argv=None):
    parser = argparse.ArgumentParser("Random Overcooked layout generator")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--num_agents", type=int, default=2, help="number of agents in layout")
    parser.add_argument("--height-min", type=int, default=10, help="minimum layout height")
    parser.add_argument("--height-max", type=int, default=11, help="maximum layout height")
    parser.add_argument("--width-min", type=int, default=10, help="minimum layout width")
    parser.add_argument("--width-max", type=int, default=11, help="maximum layout width")
    parser.add_argument("--wall-density", type=float, default=0.35, help="fraction of unpassable internal cells")
    parser.add_argument("--difficulty", type=str, choices=["easy", "med", "medium", "hard"],
                        help="difficulty level (overrides height, width, and wall density)")
    parser.add_argument("--num-envs", type=int, default=1, help="number of environments to generate")
    parser.add_argument("--show", action="store_true", help="preview with matplotlib")
    parser.add_argument("--oc", action="store_true", help="open JAX‑MARL Overcooked viewer")
    parser.add_argument("--save", action="store_true", help="save PNG to assets/screenshots/generated/")
    args = parser.parse_args(argv)

    from meal.env import Overcooked
    from meal.visualization.visualizer import OvercookedVisualizer

    # Override parameters based on difficulty
    if args.difficulty:
        if args.difficulty == "easy":
            args.height_min = args.width_min = 6
            args.height_max = args.width_max = 7
            args.wall_density = 0.15
        elif args.difficulty == "med" or args.difficulty == "medium":
            args.height_min = args.width_min = 8
            args.height_max = args.width_max = 9
            args.wall_density = 0.25
        elif args.difficulty == "hard":
            args.height_min = args.width_min = 10
            args.height_max = args.width_max = 11
            args.wall_density = 0.35

    # Generate environments
    layouts = []
    for i in range(args.num_envs):
        # Use a different seed for each environment if seed is provided
        env_seed = None if args.seed is None else args.seed + i

        grid_str, layout = generate_random_layout(
            num_agents=args.num_agents,
            height_rng=(args.height_min, args.height_max),
            width_rng=(args.width_min, args.width_max),
            wall_density=args.wall_density,
            seed=env_seed,
        )
        layouts.append((grid_str, layout, env_seed))
        print(f"Environment {i + 1}/{args.num_envs}:")
        print(grid_str)

    if args.show and layouts:
        mpl_show(layouts[0][0], "Random kitchen")

    if args.oc and layouts:
        oc_show(layouts[0][1], args.num_agents)

    if args.save and layouts:
        # Determine the base output directory
        base_dir = Path(__file__).parent.parent.parent.parent / "assets" / "screenshots"

        # Create difficulty-specific directory if difficulty is specified
        if args.difficulty:
            out_dir = base_dir / args.difficulty
        else:
            out_dir = base_dir / "generated"

        out_dir.mkdir(parents=True, exist_ok=True)

        # Find the highest existing index for gen_X files
        existing_files = list(out_dir.glob("gen_*.png"))
        highest_index = 0

        for file in existing_files:
            filename = file.name
            if filename.startswith("gen_") and filename.endswith(".png"):
                try:
                    index = int(filename[4:-4])  # Extract number between "gen_" and ".png"
                    highest_index = max(highest_index, index)
                except ValueError:
                    # If the filename doesn't follow the pattern, ignore it
                    pass

        from PIL import Image

        # Save each generated environment
        for i, (_, layout, env_seed) in enumerate(layouts):
            env = Overcooked(layout=layout, layout_name="generated", random_reset=False, num_agents=args.num_agents)
            _, state = env.reset(jax.random.PRNGKey(env_seed or 0))
            vis = OvercookedVisualizer(args.num_agents)
            img = vis.render(state)

            # Create filename with auto-incrementing index
            file_index = highest_index + i + 1
            file_name = f"{args.num_agents}_agents_gen_{file_index}.png"

            Image.fromarray(img).save(out_dir / file_name)
            print(f"Saved generated layout {i + 1}/{len(layouts)} to {out_dir / file_name}")


if __name__ == "__main__":
    main()
