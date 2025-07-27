import math
from collections import namedtuple

import numpy as np
import pygame
import wandb

import jax_marl.eval.grid_rendering as rendering
from jax_marl.environments.overcooked.common import OBJECT_TO_INDEX, COLOR_TO_INDEX, COLORS
from jax_marl.eval.visualization.state_visualizer import StateVisualizer
from jax_marl.eval.window import Window

# Define a namedtuple for mock objects to be used in visualization
MockObject = namedtuple('MockObject', ['name', 'ingredients'])

# INDEX_TO_COLOR = [k for k,v in COLOR_TO_INDEX.items()]
INDEX_TO_COLOR = [k for k, _ in sorted(COLOR_TO_INDEX.items(), key=lambda p: p[1])]

TILE_PIXELS = 64  # Increased from 32 to 64 for higher resolution


def _colour_to_agent_index(col_idx: int) -> int:
    """
    Overcooked colours go 0,2,4,…  → map even index → consecutive agent id.
    e.g. 0→0 (red), 2→1 (blue), 4→2 (green) …
    """
    return col_idx // 2


class OvercookedVisualizer:
    """
    A renderer / GIF-maker that works with any number of agents.

    Parameters
    ----------
    num_agents : int
        How many inventories / headings you’ll pass in `agent_inv`
        and `agent_dir_idx`.
    use_old_rendering : bool
        If True, use the old rendering logic. If False (default), use the new rendering logic
        from the overcooked_ai repository.
    pot_full_status : int
        The pot status value when the pot is full and starts cooking (default: 20).
        Should match the environment's pot_full_status.
    pot_empty_status : int
        The pot status value when the pot is empty (default: 23).
        Should match the environment's pot_empty_status.
    """
    tile_cache: dict[tuple, np.ndarray] = {}

    def __init__(self, num_agents: int = 2, use_old_rendering: bool = False, 
                 pot_full_status: int = 20, pot_empty_status: int = 23):
        self.window: Window | None = None
        self._num_agents = num_agents
        self.use_old_rendering = use_old_rendering

        # Store configurable pot status values for rendering
        self.pot_full_status = pot_full_status
        self.pot_empty_status = pot_empty_status

        # Initialize the new state visualizer if using new rendering
        if not self.use_old_rendering:
            self.state_visualizer = StateVisualizer(
                player_colors=["red", "blue", "green", "purple"][:num_agents],
                tile_size=TILE_PIXELS
            )

    def _create_held_object_from_inventory(self, inventory_value):
        """
        Create a MockObject based on the agent's inventory value.

        Parameters
        ----------
        inventory_value : int
            The inventory value from the environment state

        Returns
        -------
        MockObject or None
            A MockObject with appropriate name and ingredients, or None if inventory is empty
        """
        if inventory_value == OBJECT_TO_INDEX['empty']:
            return None
        elif inventory_value == OBJECT_TO_INDEX['onion']:
            return MockObject(name='onion', ingredients=None)
        elif inventory_value == OBJECT_TO_INDEX['plate']:
            return MockObject(name='dish', ingredients=None)  # 'dish' is the visualization name for an empty plate
        elif inventory_value == OBJECT_TO_INDEX['dish']:
            # Dish with soup (onion soup)
            return MockObject(name='soup', ingredients=['onion', 'onion', 'onion'])
        else:
            # Default case, should not happen
            return None

    def _create_mock_objects(self, grid, env_state):
        """
        Create mock objects for visualization based on the grid and environment state.

        Parameters
        ----------
        grid : numpy.ndarray
            The grid representation of the environment
        env_state : State
            The environment state

        Returns
        -------
        dict
            A dictionary of mock objects for visualization
        """
        objects = {}

        # Check if we have pot positions and status information
        if not hasattr(env_state, 'pot_pos') or not hasattr(env_state, 'maze_map'):
            return objects

        # Find pots in the grid and create mock soup objects for them
        height, width = grid.shape[:2]
        padding = (env_state.maze_map.shape[0] - height) // 2 if hasattr(env_state.maze_map, 'shape') else 0

        for i, pot_pos in enumerate(env_state.pot_pos):
            # Get pot status from maze_map
            pot_x, pot_y = pot_pos
            pot_status = env_state.maze_map[padding + pot_y, padding + pot_x, 2]

            # Only create objects for pots that have ingredients or are cooking/ready
            if pot_status < self.pot_empty_status:  # Use configurable empty pot status
                obj_id = f"soup_{pot_x}_{pot_y}"

                # Determine ingredients and status
                # Convert JAX array to Python integer
                pot_status_int = int(pot_status)
                num_onions = min(3, self.pot_empty_status - pot_status_int) if pot_status_int >= self.pot_full_status else 3
                ingredients = ['onion'] * num_onions

                # Create a position tuple for the object
                position = (int(pot_x), int(pot_y))

                # Create a mock soup object
                is_ready = (pot_status == 0)  # Pot is ready when status is 0
                is_cooking = (pot_status < self.pot_full_status and pot_status > 0)  # Use configurable full pot status

                # Create a soup object with appropriate attributes
                soup_obj = type('MockSoup', (), {
                    'name': 'soup',
                    'position': position,
                    'ingredients': ingredients,
                    'is_ready': is_ready,
                    '_cooking_tick': pot_status if is_cooking else -1,
                    'cook_time': self.pot_full_status  # Use configurable cook time
                })

                objects[obj_id] = soup_obj

        # Scan the grid for individual items placed on counters (onions, plates, dishes)
        for y in range(height):
            for x in range(width):
                obj = grid[y, x, :]
                obj_type = obj[0]

                # Create objects for individual items placed on counters
                if obj_type == OBJECT_TO_INDEX['onion']:
                    obj_id = f"onion_{x}_{y}"
                    onion_obj = type('MockOnion', (), {
                        'name': 'onion',
                        'position': (int(x), int(y)),
                        'ingredients': None
                    })
                    objects[obj_id] = onion_obj
                elif obj_type == OBJECT_TO_INDEX['plate']:
                    obj_id = f"plate_{x}_{y}"
                    plate_obj = type('MockPlate', (), {
                        'name': 'dish',  # 'dish' is the visualization name for an empty plate
                        'position': (int(x), int(y)),
                        'ingredients': None
                    })
                    objects[obj_id] = plate_obj
                elif obj_type == OBJECT_TO_INDEX['dish']:
                    obj_id = f"dish_{x}_{y}"
                    dish_obj = type('MockDish', (), {
                        'name': 'soup',  # Dish with soup
                        'position': (int(x), int(y)),
                        'ingredients': ['onion', 'onion', 'onion'],
                        'is_ready': True
                    })
                    objects[obj_id] = dish_obj

        return objects

    # --------------------------------------------------------------------- #
    # Window helpers
    # --------------------------------------------------------------------- #

    def _lazy_init_window(self):
        if self.window is None:
            self.window = Window('Overcooked-viz')

    def _convert_grid_to_str(self, grid):
        """
        Convert the grid from the format used in the old rendering logic to the format expected by the StateVisualizer.

        StateVisualizer expects a grid of strings where each cell is one of:
        " " (empty), "X" (counter), "O" (onion dispenser), "T" (tomato dispenser), 
        "P" (pot), "D" (dish dispenser), or "S" (serving location).
        """
        height, width = grid.shape[:2]
        grid_str = []

        for y in range(height):
            row = []
            for x in range(width):
                obj = grid[y, x, :]
                obj_type = obj[0]

                if obj_type == OBJECT_TO_INDEX['empty']:
                    row.append(" ")  # Empty
                elif obj_type == OBJECT_TO_INDEX['wall']:
                    row.append("X")  # Counter
                elif obj_type == OBJECT_TO_INDEX['onion_pile']:
                    row.append("O")  # Onion dispenser
                elif obj_type == OBJECT_TO_INDEX['plate_pile']:
                    row.append("D")  # Dish dispenser
                elif obj_type == OBJECT_TO_INDEX['pot']:
                    row.append("P")  # Pot
                elif obj_type == OBJECT_TO_INDEX['goal']:
                    row.append("S")  # Serving location
                elif obj_type == OBJECT_TO_INDEX['agent']:
                    row.append(" ")  # Empty tile under agent
                else:
                    row.append("X")  # Default to counter for other objects

            grid_str.append(row)

        return grid_str

    def show(self, block=False):
        self._lazy_init_window()
        self.window.show(block=block)

    def render(self, agent_view_size, state, highlight=True, tile_size=TILE_PIXELS):
        """Method for rendering the state in a window. Esp. useful for interactive mode."""
        if self.use_old_rendering:
            return self._render_state(agent_view_size, state, highlight, tile_size)
        else:
            self._lazy_init_window()
            # Check if state is a LogEnvState (has env_state attribute)
            if hasattr(state, 'env_state'):
                env_state = state.env_state
            else:
                env_state = state

            # Extract grid from state
            padding = agent_view_size - 1  # 5→4 because map has +1 outer wall
            grid = np.asarray(env_state.maze_map[padding:-padding, padding:-padding, :])

            # Convert grid to format expected by StateVisualizer
            grid_str = self._convert_grid_to_str(grid)

            # Create a minimal state object for rendering
            from collections import namedtuple
            from jax_marl.eval.visualization.actions import Direction

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

            # Use agent positions directly from state instead of scanning grid
            # This ensures agents maintain their correct IDs and don't switch positions
            for i in range(self._num_agents):
                if i < len(env_state.agent_pos):
                    # Get agent position directly from state (x, y format)
                    pos = (int(env_state.agent_pos[i, 0]), int(env_state.agent_pos[i, 1]))

                    # Convert environment direction index to visualization direction tuple
                    # Convert JAX array to int before using as dictionary key
                    dir_idx = int(env_state.agent_dir_idx[i])
                    orientation = ENV_DIR_IDX_TO_VIZ_DIR[dir_idx]

                    # Create a player with appropriate held object based on inventory
                    held_object = None
                    if hasattr(env_state, 'agent_inv') and i < len(env_state.agent_inv):
                        held_object = self._create_held_object_from_inventory(int(env_state.agent_inv[i]))
                    players.append(MockPlayer(position=pos, orientation=orientation, held_object=held_object))

            # Create mock objects for pots
            objects = self._create_mock_objects(grid, env_state)

            # Create a mock state
            MockState = namedtuple('MockState', ['players', 'objects'])
            mock_state = MockState(players=players, objects=objects)

            # Render using StateVisualizer
            surface = self.state_visualizer.render_state(mock_state, grid_str)
            self.window.show_img(pygame.surfarray.array3d(surface).transpose(1, 0, 2))
            return surface

    def animate(self, state_seq, agent_view_size, task_idx, task_name, exp_dir, tile_size: int = TILE_PIXELS):
        """
        Make a GIF and log it to WandB.

        `state_seq` is the list you built in `record_gif(...)`; every item must
        expose a `.maze_map`, `.agent_dir_idx`, `.agent_inv`, etc.
        """
        import imageio.v3 as iio
        import os

        padding = agent_view_size - 1  # 5→4 because map has +1 outer wall

        if self.use_old_rendering:
            def get_frame(state):
                # Check if state is a LogEnvState (has env_state attribute)
                if hasattr(state, 'env_state'):
                    state = state.env_state
                grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
                # Render the state
                frame = self._render_grid(
                    grid,
                    tile_size=tile_size,
                    highlight_mask=None,
                    agent_dir_idx=np.atleast_1d(state.agent_dir_idx),
                    agent_inv=np.atleast_1d(state.agent_inv),
                    pot_full_status=self.pot_full_status,
                    pot_empty_status=self.pot_empty_status,
                )
                return frame

            frames = [get_frame(state) for state in state_seq]
        else:
            # Use the new rendering logic
            frames = []
            from collections import namedtuple
            from jax_marl.eval.visualization.actions import Direction

            # Create a mapping from environment direction indices to visualization direction tuples
            ENV_DIR_IDX_TO_VIZ_DIR = {
                0: Direction.NORTH,  # (0, -1)
                1: Direction.SOUTH,  # (0, 1)
                2: Direction.EAST,  # (1, 0)
                3: Direction.WEST  # (-1, 0)
            }

            for state in state_seq:
                # Check if state is a LogEnvState (has env_state attribute)
                if hasattr(state, 'env_state'):
                    env_state = state.env_state
                else:
                    env_state = state

                grid = np.asarray(env_state.maze_map[padding:-padding, padding:-padding, :])
                grid_str = self._convert_grid_to_str(grid)

                # Create mock players based on agent positions and directions
                MockPlayer = namedtuple('MockPlayer', ['position', 'orientation', 'held_object'])
                players = []

                # Use agent positions directly from state instead of scanning grid
                # This ensures agents maintain their correct IDs and don't switch positions
                for i in range(self._num_agents):
                    if i < len(env_state.agent_pos):
                        # Get agent position directly from state (x, y format)
                        pos = (int(env_state.agent_pos[i, 0]), int(env_state.agent_pos[i, 1]))

                        # Convert environment direction index to visualization direction tuple
                        # Convert JAX array to int before using as dictionary key
                        dir_idx = int(env_state.agent_dir_idx[i])
                        orientation = ENV_DIR_IDX_TO_VIZ_DIR[dir_idx]

                        # Create a player with appropriate held object based on inventory
                        held_object = None
                        if hasattr(env_state, 'agent_inv') and i < len(env_state.agent_inv):
                            held_object = self._create_held_object_from_inventory(int(env_state.agent_inv[i]))
                        players.append(MockPlayer(position=pos, orientation=orientation, held_object=held_object))

                # Create mock objects for pots
                objects = self._create_mock_objects(grid, env_state)

                # Create a mock state
                MockState = namedtuple('MockState', ['players', 'objects'])
                mock_state = MockState(players=players, objects=objects)

                # Render using StateVisualizer
                surface = self.state_visualizer.render_state(mock_state, grid_str)

                # Convert pygame surface to numpy array
                frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
                frames.append(frame)

        # Create directory if it doesn't exist
        os.makedirs(exp_dir, exist_ok=True)

        file_name = f"task_{task_idx}_{task_name}"
        file_path = f"{exp_dir}/{file_name}.gif"

        iio.imwrite(file_path, frames, loop=0, fps=10)
        if wandb.run is not None:
            wandb.log({file_name: wandb.Video(file_path, format="gif")})

    def render_grid(self, grid, tile_size=TILE_PIXELS, k_rot90=0, agent_dir_idx=None, agent_inv=None, title=None):
        self._lazy_init_window()

        if self.use_old_rendering:
            img = self._render_grid(
                grid,
                tile_size,
                highlight_mask=None,
                agent_dir_idx=agent_dir_idx,
                agent_inv=agent_inv,
                pot_full_status=self.pot_full_status,
                pot_empty_status=self.pot_empty_status,
            )
            # img = np.transpose(img, axes=(1,0,2))
            if k_rot90 > 0:
                img = np.rot90(img, k=k_rot90)
        else:
            # Convert grid to format expected by StateVisualizer
            grid_str = self._convert_grid_to_str(grid)

            # Create a minimal state object for rendering
            from collections import namedtuple
            from jax_marl.eval.visualization.actions import Direction

            # Create a mapping from environment direction indices to visualization direction tuples
            ENV_DIR_IDX_TO_VIZ_DIR = {
                0: Direction.NORTH,  # (0, -1)
                1: Direction.SOUTH,  # (0, 1)
                2: Direction.EAST,  # (1, 0)
                3: Direction.WEST  # (-1, 0)
            }

            # Create mock players based on agent_dir_idx and agent_inv
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
                if agent_dir_idx is not None and i < len(agent_dir_idx):
                    # Convert environment direction index to visualization direction tuple
                    # Convert JAX array to int before using as dictionary key
                    dir_idx = int(agent_dir_idx[i])
                    orientation = ENV_DIR_IDX_TO_VIZ_DIR[dir_idx]
                else:
                    orientation = Direction.SOUTH  # Default orientation (facing downwards)

                # Create a player with appropriate held object based on inventory
                held_object = None
                if agent_inv is not None and i < len(agent_inv):
                    held_object = self._create_held_object_from_inventory(int(agent_inv[i]))
                players.append(MockPlayer(position=pos, orientation=orientation, held_object=held_object))

            # Create mock objects for pots
            # Since we don't have a full state object here, we'll create a minimal one for the _create_mock_objects method
            minimal_state = type('MinimalState', (), {
                'pot_pos': [],  # We don't have pot positions here
                'maze_map': grid  # Use the grid as a substitute
            })
            objects = self._create_mock_objects(grid, minimal_state)

            # Create a mock state
            MockState = namedtuple('MockState', ['players', 'objects'])
            mock_state = MockState(players=players, objects=objects)

            # Render using StateVisualizer
            surface = self.state_visualizer.render_state(mock_state, grid_str)

            # Convert pygame surface to numpy array
            img = pygame.surfarray.array3d(surface).transpose(1, 0, 2)

            if k_rot90 > 0:
                img = np.rot90(img, k=k_rot90)

        if title is not None and hasattr(self, "window"):
            self.window.set_caption(title)  # one-liner caption

        self.window.show_img(img)
        return img

    def _render_state(self, agent_view_size, state, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the state
        """
        self._lazy_init_window()

        # Check if state is a LogEnvState (has env_state attribute)
        if hasattr(state, 'env_state'):
            env_state = state.env_state
        else:
            env_state = state

        padding = agent_view_size - 1  # 5→4 because map has +1 outer wall
        grid = np.asarray(env_state.maze_map[padding:-padding, padding:-padding, :])
        grid_offset = np.array([1, 1])
        h, w = grid.shape[:2]
        # === Compute highlight mask
        highlight_mask = np.zeros(shape=(h, w), dtype=bool)

        if highlight:
            f_vec = env_state.agent_dir
            r_vec = np.array([-f_vec[1], f_vec[0]])

            fwd_bound1 = env_state.agent_pos
            fwd_bound2 = env_state.agent_pos + env_state.agent_dir * (agent_view_size - 1)
            side_bound1 = env_state.agent_pos - r_vec * (agent_view_size // 2)
            side_bound2 = env_state.agent_pos + r_vec * (agent_view_size // 2)

            min_bound = np.min(np.stack([
                fwd_bound1,
                fwd_bound2,
                side_bound1,
                side_bound2]) + grid_offset, 0)

            min_y = min(max(min_bound[1], 0), highlight_mask.shape[0] - 1)
            min_x = min(max(min_bound[0], 0), highlight_mask.shape[1] - 1)

            max_y = min(max(min_bound[1] + agent_view_size, 0), highlight_mask.shape[0] - 1)
            max_x = min(max(min_bound[0] + agent_view_size, 0), highlight_mask.shape[1] - 1)

            highlight_mask[min_y:max_y, min_x:max_x] = True

        # Render the whole grid
        img = self._render_grid(
            grid,
            tile_size,
            highlight_mask=highlight_mask if highlight else None,
            agent_dir_idx=np.atleast_1d(env_state.agent_dir_idx),
            agent_inv=np.atleast_1d(env_state.agent_inv),
            pot_full_status=self.pot_full_status,
            pot_empty_status=self.pot_empty_status,
        )
        self.window.show_img(img)
        return img

    @classmethod
    def _render_obj(
            cls,
            obj,
            img,
            pot_full_status=20,
            pot_empty_status=23):
        # Render each kind of object
        obj_type = obj[0]
        color = INDEX_TO_COLOR[obj[1]]

        if obj_type == OBJECT_TO_INDEX['wall']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['goal']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['agent']:
            agent_dir_idx = obj[2]
            tri_fn = rendering.point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )
            # tri_fn = rendering.rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir_idx)

            # A bit hacky, but needed so that actions order matches the one of Overcooked-AI
            direction_reording = [3, 1, 0, 2]
            # Convert JAX array to int before using as list index
            dir_idx = int(agent_dir_idx)
            direction = direction_reording[dir_idx]
            tri_fn = rendering.rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction)

            rendering.fill_coords(img, tri_fn, COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['empty']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['onion_pile']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            onion_fns = [rendering.point_in_circle(*coord, 0.15) for coord in [(0.5, 0.15), (0.3, 0.4), (0.8, 0.35),
                                                                               (0.4, 0.8), (0.75, 0.75)]]
            [rendering.fill_coords(img, onion_fn, COLORS[color]) for onion_fn in onion_fns]
        elif obj_type == OBJECT_TO_INDEX['onion']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            onion_fn = rendering.point_in_circle(0.5, 0.5, 0.15)
            rendering.fill_coords(img, onion_fn, COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['plate_pile']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            plate_fns = [rendering.point_in_circle(*coord, 0.2) for coord in [(0.3, 0.3), (0.75, 0.42),
                                                                              (0.4, 0.75)]]
            [rendering.fill_coords(img, plate_fn, COLORS[color]) for plate_fn in plate_fns]
        elif obj_type == OBJECT_TO_INDEX['plate']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            plate_fn = rendering.point_in_circle(0.5, 0.5, 0.2)
            rendering.fill_coords(img, plate_fn, COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['dish']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            plate_fn = rendering.point_in_circle(0.5, 0.5, 0.2)
            rendering.fill_coords(img, plate_fn, COLORS[color])
            onion_fn = rendering.point_in_circle(0.5, 0.5, 0.13)
            rendering.fill_coords(img, onion_fn, COLORS["orange"])
        elif obj_type == OBJECT_TO_INDEX['pot']:
            cls._render_pot(obj, img, pot_full_status, pot_empty_status)
        # rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
        # pot_fns = [rendering.point_in_rect(0.1, 0.9, 0.3, 0.9),
        # 		   rendering.point_in_rect(0.1, 0.9, 0.20, 0.23),
        # 		   rendering.point_in_rect(0.4, 0.6, 0.15, 0.20),]
        # [rendering.fill_coords(img, pot_fn, COLORS[color]) for pot_fn in pot_fns]
        else:
            raise ValueError(f'Rendering object at index {obj_type} is currently unsupported.')

    @classmethod
    def _render_pot(
            cls,
            obj,
            img,
            pot_full_status=20,
            pot_empty_status=23):
        pot_status = obj[-1]
        num_onions = np.max([pot_empty_status - pot_status, 0])
        is_cooking = np.array((pot_status < pot_full_status) * (pot_status > 0))
        is_done = np.array(pot_status == 0)

        pot_fn = rendering.point_in_rect(0.1, 0.9, 0.33, 0.9)
        lid_fn = rendering.point_in_rect(0.1, 0.9, 0.21, 0.25)
        handle_fn = rendering.point_in_rect(0.4, 0.6, 0.16, 0.21)

        rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])

        # Render onions in pot
        if num_onions > 0 and not is_done:
            onion_fns = [rendering.point_in_circle(*coord, 0.13) for coord in
                         [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]]
            onion_fns = onion_fns[:num_onions]
            [rendering.fill_coords(img, onion_fn, COLORS["yellow"]) for onion_fn in onion_fns]
            if not is_cooking:
                lid_fn = rendering.rotate_fn(lid_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
                handle_fn = rendering.rotate_fn(handle_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)

        # Render done soup
        if is_done:
            soup_fn = rendering.point_in_rect(0.12, 0.88, 0.23, 0.35)
            rendering.fill_coords(img, soup_fn, COLORS["orange"])

        # Render the pot itself
        pot_fns = [pot_fn, lid_fn, handle_fn]
        [rendering.fill_coords(img, pot_fn, COLORS["black"]) for pot_fn in pot_fns]

        # Render progress bar
        if is_cooking:
            progress_fn = rendering.point_in_rect(0.1, 0.9 - (0.9 - 0.1) / pot_full_status * pot_status, 0.83, 0.88)
            rendering.fill_coords(img, progress_fn, COLORS["green"])

    @classmethod
    def _render_inv(
            cls,
            obj,
            img):
        # Render each kind of object
        obj_type = obj[0]
        if obj_type == OBJECT_TO_INDEX['empty']:
            pass
        elif obj_type == OBJECT_TO_INDEX['onion']:
            onion_fn = rendering.point_in_circle(0.75, 0.75, 0.15)
            rendering.fill_coords(img, onion_fn, COLORS["yellow"])
        elif obj_type == OBJECT_TO_INDEX['plate']:
            plate_fn = rendering.point_in_circle(0.75, 0.75, 0.2)
            rendering.fill_coords(img, plate_fn, COLORS["white"])
        elif obj_type == OBJECT_TO_INDEX['dish']:
            plate_fn = rendering.point_in_circle(0.75, 0.75, 0.2)
            rendering.fill_coords(img, plate_fn, COLORS["white"])
            onion_fn = rendering.point_in_circle(0.75, 0.75, 0.13)
            rendering.fill_coords(img, onion_fn, COLORS["orange"])
        else:
            raise ValueError(f'Rendering object at index {obj_type} is currently unsupported.')

    @classmethod
    def _render_tile(
            cls,
            obj,
            highlight=False,
            agent_dir_idx=None,
            agent_inv=None,
            tile_size=TILE_PIXELS,
            subdivs=3,
            pot_full_status=20,
            pot_empty_status=23
    ):
        """
        Render a tile and cache the result
        """
        if obj is not None and obj[0] == OBJECT_TO_INDEX["agent"]:
            if agent_inv is not None:
                colour_idx = int(obj[1])
                agent_idx = _colour_to_agent_index(colour_idx)
                if agent_idx < len(agent_inv):
                    sub_inv = np.array([agent_inv[agent_idx], -1, 0], np.int8)
                else:  # safeguard – blank if missing
                    sub_inv = np.array([OBJECT_TO_INDEX["empty"], -1, 0], np.int8)
            else:
                sub_inv = None

            if agent_dir_idx is not None:
                colour_idx = int(obj[1])
                agent_idx = _colour_to_agent_index(colour_idx)
                obj = np.array(obj)
                if agent_idx < len(agent_dir_idx):
                    obj[-1] = agent_dir_idx[agent_idx]

        else:
            sub_inv = None

        no_object = obj is None or (
                obj[0] in [OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['unseen']] \
                and obj[2] == 0
        )

        if not no_object:
            if sub_inv is not None and obj[0] == OBJECT_TO_INDEX['agent']:
                key = (*obj, *sub_inv, highlight, tile_size)
            else:
                key = (*obj, highlight, tile_size)
        else:
            key = (obj, highlight, tile_size)

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        rendering.fill_coords(img, rendering.point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if not no_object:
            OvercookedVisualizer._render_obj(obj, img, pot_full_status, pot_empty_status)
            # render inventory
            if sub_inv is not None and obj[0] == OBJECT_TO_INDEX['agent']:
                OvercookedVisualizer._render_inv(sub_inv, img)

        if highlight:
            rendering.highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = rendering.downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    @classmethod
    def _render_grid(
            cls,
            grid,
            tile_size=TILE_PIXELS,
            highlight_mask=None,
            agent_dir_idx=None,
            agent_inv=None,
            pot_full_status=20,
            pot_empty_status=23):
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=grid.shape[:2], dtype=bool)

        # Compute the total grid size in pixels
        width_px = grid.shape[1] * tile_size
        height_px = grid.shape[0] * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                obj = grid[y, x, :]
                if obj[0] in [OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['unseen']] \
                        and obj[2] == 0:
                    obj = None

                tile_img = OvercookedVisualizer._render_tile(
                    obj,
                    highlight=highlight_mask[y, x],
                    tile_size=tile_size,
                    agent_dir_idx=agent_dir_idx,
                    agent_inv=agent_inv,
                    pot_full_status=pot_full_status,
                    pot_empty_status=pot_empty_status,
                )

                ymin = y * tile_size
                ymax = (y + 1) * tile_size
                xmin = x * tile_size
                xmax = (x + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def close(self):
        self.window.close()
