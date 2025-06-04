#!/usr/bin/env python
"""Test script for the improved Overcooked layout validator.

This script generates a list of layouts, validates each layout using the new constraints,
plots the layouts for visual validation, and prints the reason why invalid layouts are invalid.
"""
import random

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from jax_marl.environments import Overcooked
from jax_marl.environments.overcooked_environment.env_generator import (
    generate_random_layout, layout_grid_to_dict
)
from jax_marl.environments.overcooked_environment.env_validator import (
    evaluate_grid, WALL, FLOOR, AGENT, GOAL, ONION_PILE, POT
)
from jax_marl.gridworld.grid_viz import TILE_PIXELS
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer


def create_invalid_layout(issue_type, seed=None):
    """Create an invalid layout with a specific issue."""
    rng = random.Random(seed)

    # Start with a valid layout
    grid_str, _ = generate_random_layout(
        num_agents=2,
        height_rng=(6, 8),
        width_rng=(6, 8),
        wall_density=0.25,
        seed=seed,
    )

    rows = grid_str.strip().split("\n")
    grid = [list(r) for r in rows]
    height, width = len(grid), len(grid[0])

    if issue_type == "unreachable_onions":
        # Find all onions and surround them with walls
        for i in range(height):
            for j in range(width):
                if grid[i][j] == ONION_PILE:
                    # Surround with walls if possible
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width and grid[ni][nj] == FLOOR:
                            grid[ni][nj] = WALL

    elif issue_type == "no_onion_to_pot_path":
        # Find all pots and surround them with walls
        for i in range(height):
            for j in range(width):
                if grid[i][j] == POT:
                    # Surround with walls if possible
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width and grid[ni][nj] == FLOOR:
                            grid[ni][nj] = WALL

    elif issue_type == "no_pot_to_delivery_path":
        # Find all delivery zones and surround them with walls
        for i in range(height):
            for j in range(width):
                if grid[i][j] == GOAL:
                    # Surround with walls if possible
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width and grid[ni][nj] == FLOOR:
                            grid[ni][nj] = WALL

    elif issue_type == "useless_agent":
        # Find one agent and isolate it
        agent_found = False
        for i in range(height):
            for j in range(width):
                if grid[i][j] == AGENT and not agent_found:
                    agent_found = True
                    # Surround with walls if possible
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width and grid[ni][nj] == FLOOR:
                            grid[ni][nj] = WALL

    # Convert back to string
    return "\n".join("".join(row) for row in grid)


def visualize_layout(name, grid_str, is_valid, reason):
    """Visualize a layout using the OvercookedVisualizer."""
    # Convert grid string to FrozenDict layout
    layout_dict = layout_grid_to_dict(grid_str)

    # Check if the layout has all required tile types
    # If not, add dummy ones to avoid errors in the Overcooked environment's reset method
    required_keys = ["plate_pile_idx", "onion_pile_idx", "pot_idx", "goal_idx"]
    missing_keys = [key for key in required_keys if len(layout_dict[key]) == 0]

    if missing_keys:
        # Find wall positions that aren't already used for something else
        wall_idx = layout_dict["wall_idx"]
        used_idx = set()
        for key in ["agent_idx", "goal_idx", "onion_pile_idx", "pot_idx", "plate_pile_idx"]:
            if key in layout_dict:
                # Convert JAX array to list of integers before adding to set
                used_idx.update([int(idx) for idx in layout_dict[key]])

        # Find available wall positions
        available_idx = [int(idx) for idx in wall_idx if int(idx) not in used_idx]

        # If we have enough available positions, add dummy tiles
        if len(available_idx) >= len(missing_keys):
            layout_dict = dict(layout_dict)
            for i, key in enumerate(missing_keys):
                layout_dict[key] = jnp.array([available_idx[i]])
            layout_dict = FrozenDict(layout_dict)

    # Create Overcooked environment with this layout
    env = Overcooked(layout=layout_dict, layout_name=name, random_reset=False)

    # Reset the environment to get the initial state
    _, state = env.reset(jax.random.PRNGKey(0))

    # Crop the grid to remove padding
    padding = env.agent_view_size - 2
    grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])

    # Create visualizer and render the grid
    vis = OvercookedVisualizer()

    # Add a title with validation status
    status = "VALID" if is_valid else "INVALID"
    title = f"{name} - {status}"
    if not is_valid:
        title += f": {reason}"

    print(f"\nDisplaying: {title}")
    print("Close the window to continue to the next layout...")

    # Render the grid with the agent directions
    vis.render_grid(grid, tile_size=TILE_PIXELS, agent_dir_idx=state.agent_dir_idx, title=title)


def create_specific_test_case(test_case):
    """Create specific test cases for testing the validator."""
    if test_case == "multi_pot_one_usable":
        # Create a layout with 2 pots, one usable and one not
        grid_str = """WWWWWWWW
WAAOW  W
W  W   W
W  WPPPW
W      W
W  X   W
WWWWWWWW"""
        return grid_str
    elif test_case == "agent_can_pass_plates":
        # Create a layout where an agent is surrounded by walls but can pass plates to the other agent
        grid_str = """WWWWWWWW
W  O   W
W      W
WWAWWPPW
W  A   W
W  X   W
WWWWWWWW"""
        return grid_str
    elif test_case == "complex_handoff":
        # Create a layout where agents need to cooperate through multiple handoffs
        grid_str = """WWWWWWWWWW
W  O     W
W        W
WWAWWWWWWW
W        W
W        W
W   A    W
W        W
W   X    W
WWWWWWWWWW"""
        return grid_str
    elif test_case == "agent_next_to_interactive":
        # Create a layout where an agent is next to an interactive tile but surrounded by walls
        grid_str = """WWWWWWWW
W  O   W
W      W
WWPAWWWW
W  A   W
W  X   W
WWWWWWWW"""
        return grid_str
    elif test_case == "multiple_pots_complex":
        # Create a layout with multiple pots, some usable and some not
        grid_str = """WWWWWWWWWW
W  O     W
W        W
WWPWWWWWWW
W  A     W
W        W
W   A    W
W   P    W
W   X    W
WWWWWWWWWW"""
        return grid_str
    else:
        raise ValueError(f"Unknown test case: {test_case}")


def test_layouts():
    """Test a variety of layouts with the improved validator."""
    # Create a list of layouts to test
    layouts = []

    # Add some valid layouts
    for seed in range(5):
        try:
            grid_str, _ = generate_random_layout(
                num_agents=2,
                height_rng=(6, 8),
                width_rng=(6, 8),
                wall_density=0.25,
                seed=seed,
            )
            layouts.append(("valid_" + str(seed), grid_str))
        except RuntimeError:
            print(f"Could not generate valid layout with seed {seed}")

    # Add some invalid layouts
    for issue_type in ["unreachable_onions", "no_onion_to_pot_path", "no_pot_to_delivery_path", "useless_agent"]:
        for seed in range(3):
            try:
                grid_str = create_invalid_layout(issue_type, seed=seed + 100)
                layouts.append((issue_type + "_" + str(seed), grid_str))
            except Exception as e:
                print(f"Error creating invalid layout {issue_type} with seed {seed}: {e}")

    # Add specific test cases
    layouts.append(("multi_pot_one_usable", create_specific_test_case("multi_pot_one_usable")))
    layouts.append(("agent_can_pass_plates", create_specific_test_case("agent_can_pass_plates")))
    layouts.append(("complex_handoff", create_specific_test_case("complex_handoff")))
    layouts.append(("agent_next_to_interactive", create_specific_test_case("agent_next_to_interactive")))
    layouts.append(("multiple_pots_complex", create_specific_test_case("multiple_pots_complex")))

    # Validate each layout
    results = []
    for name, grid_str in layouts:
        is_valid, reason = evaluate_grid(grid_str)
        results.append((name, grid_str, is_valid, reason))

    # Visualize each layout using the OvercookedVisualizer
    for name, grid_str, is_valid, reason in results:
        visualize_layout(name, grid_str, is_valid, reason)

    # Print the results
    print("\nValidation Results:")
    print("=" * 80)
    for name, _, is_valid, reason in results:
        status = "VALID" if is_valid else "INVALID"
        print(f"{name}: {status}")
        if not is_valid:
            print(f"  Reason: {reason}")
    print("=" * 80)


if __name__ == "__main__":
    test_layouts()
