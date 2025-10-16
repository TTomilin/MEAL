from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

FLOOR: str = " "
WALL: str = "W"
GOAL: str = "X"
ONION_PILE: str = "O"
PLATE_PILE: str = "B"
POT: str = "P"
AGENT: str = "A"

OBJECT_TO_INDEX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "onion": 3,
    "onion_pile": 4,
    "plate": 5,
    "plate_pile": 6,
    "goal": 7,
    "pot": 8,
    "dish": 9,
    "agent": 10,
}

COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100]),
    'white': np.array([255, 255, 255]),
    'black': np.array([25, 25, 25]),
    'orange': np.array([230, 180, 0]),
}

COLOR_TO_INDEX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'purple': 3,
    'yellow': 4,
    'grey': 5,
    'white': 6,
    'black': 7,
    'orange': 8,
}

OBJECT_INDEX_TO_VEC = jnp.array([
    jnp.array([OBJECT_TO_INDEX['unseen'], 0, 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['wall'], COLOR_TO_INDEX['grey'], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['onion'], COLOR_TO_INDEX["yellow"], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['onion_pile'], COLOR_TO_INDEX["yellow"], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['plate'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['plate_pile'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['goal'], COLOR_TO_INDEX['green'], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['pot'], COLOR_TO_INDEX['black'], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['dish'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], 0], dtype=jnp.uint8),  # Default color and direction
])

# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array([
    # Pointing right (positive X)
    (0, -1),  # NORTH
    (0, 1),  # SOUTH
    (1, 0),  # EAST
    (-1, 0),  # WEST
], dtype=jnp.int8)


def make_overcooked_map(
        wall_map,
        goal_pos,
        agent_pos,
        agent_dir_idx,
        plate_pile_pos,
        onion_pile_pos,
        pot_pos,
        pot_status,
        onion_pos,
        plate_pos,
        dish_pos,
        num_agents=2):
    agent_pos = agent_pos[:num_agents]
    agent_dir_idx = agent_dir_idx[:num_agents]

    # Expand maze map to H x W x C
    empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
    wall = jnp.array([OBJECT_TO_INDEX['wall'], COLOR_TO_INDEX['grey'], 0], dtype=jnp.uint8)
    maze_map = jnp.array(jnp.expand_dims(wall_map, -1), dtype=jnp.uint8)
    maze_map = jnp.where(maze_map > 0, wall, empty)

    # Add agents
    def _get_agent_updates(agent_dir_idx, agent_pos, agent_idx):
        agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'] + agent_idx * 2, agent_dir_idx],
                          dtype=jnp.uint8)
        agent_x, agent_y = agent_pos
        return agent_x, agent_y, agent

    agent_x_vec, agent_y_vec, agent_vec = jax.vmap(_get_agent_updates, in_axes=(0, 0, 0))(agent_dir_idx, agent_pos,
                                                                                          jnp.arange(num_agents))
    maze_map = maze_map.at[agent_y_vec, agent_x_vec, :].set(agent_vec)

    # Add goals
    goal = jnp.array([OBJECT_TO_INDEX['goal'], COLOR_TO_INDEX['green'], 0], dtype=jnp.uint8)
    goal_x = goal_pos[:, 0]
    goal_y = goal_pos[:, 1]
    maze_map = maze_map.at[goal_y, goal_x, :].set(goal)

    # Add onions
    onion_x = onion_pile_pos[:, 0]
    onion_y = onion_pile_pos[:, 1]
    onion_pile = jnp.array([OBJECT_TO_INDEX['onion_pile'], COLOR_TO_INDEX["yellow"], 0], dtype=jnp.uint8)
    maze_map = maze_map.at[onion_y, onion_x, :].set(onion_pile)

    # Add plates
    plate_x = plate_pile_pos[:, 0]
    plate_y = plate_pile_pos[:, 1]
    plate_pile = jnp.array([OBJECT_TO_INDEX['plate_pile'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8)
    maze_map = maze_map.at[plate_y, plate_x, :].set(plate_pile)

    # Add pots
    pot_x = pot_pos[:, 0]
    pot_y = pot_pos[:, 1]
    pots = jnp.stack(
        [jnp.array([OBJECT_TO_INDEX['pot'], COLOR_TO_INDEX["black"], status]) for status in pot_status],
        dtype=jnp.uint8
    )
    maze_map = maze_map.at[pot_y, pot_x, :].set(pots)

    if len(onion_pos) > 0:
        onion_x = onion_pos[:, 0]
        onion_y = onion_pos[:, 1]
        onion = jnp.array([OBJECT_TO_INDEX['onion'], COLOR_TO_INDEX["yellow"], 0], dtype=jnp.uint8)
        maze_map = maze_map.at[onion_y, onion_x, :].set(onion)
    if len(plate_pos) > 0:
        plate_x = plate_pos[:, 0]
        plate_y = plate_pos[:, 1]
        plate = jnp.array([OBJECT_TO_INDEX['plate'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8)
        maze_map = maze_map.at[plate_y, plate_x, :].set(plate)
    if len(dish_pos) > 0:
        dish_x = dish_pos[:, 0]
        dish_y = dish_pos[:, 1]
        dish = jnp.array([OBJECT_TO_INDEX['dish'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8)
        maze_map = maze_map.at[dish_y, dish_x, :].set(dish)

    return maze_map
