import numpy as np
from jax_marl.eval.visualizer import OvercookedVisualizer
from jax_marl.environments.overcooked.common import COLORS, OBJECT_TO_INDEX

class OvercookedVisualizerPO(OvercookedVisualizer):
    """Visualizer for Partially Observable Overcooked with view area highlighting"""

    def __init__(self, num_agents: int = 2, use_old_rendering: bool = False, 
                 pot_full_status: int = 20, pot_empty_status: int = 23):
        super().__init__(num_agents, use_old_rendering, pot_full_status, pot_empty_status)

        # Colors for highlighting view areas
        self.view_colors = [
            np.array([255, 200, 200, 100]),  # Light red for agent 0
            np.array([200, 200, 255, 100]),  # Light blue for agent 1
        ]

    def render(self, agent_view_size, state, env=None, highlight_views=True, tile_size=32):
        """Render the environment with optional view area highlighting

        Args:
            agent_view_size: Not used in PO version (kept for compatibility)
            state: Environment state
            env: OvercookedPO environment instance (needed for view masks)
            highlight_views: Whether to highlight agent view areas
            tile_size: Size of each tile in pixels
        """
        if highlight_views and env is not None:
            # Get view masks for all agents
            view_masks = env.get_agent_view_masks(state)

            # The view masks have the same dimensions as the observation space
            # Get the observation space dimensions from the environment
            obs_height, obs_width = env.obs_shape[1], env.obs_shape[0]

            # Create highlight mask with the same dimensions as the view masks
            highlight_mask = np.zeros((obs_height, obs_width, 4), dtype=np.uint8)

            for agent_idx in range(min(len(view_masks), len(self.view_colors))):
                agent_key = f"agent_{agent_idx}"
                if agent_key in view_masks:
                    mask = np.array(view_masks[agent_key])

                    # Apply agent-specific color to view area
                    color = self.view_colors[agent_idx]
                    mask_height, mask_width = mask.shape

                    for i in range(mask_height):
                        for j in range(mask_width):
                            if mask[i, j]:
                                # Blend colors if multiple agents can see the same area
                                if highlight_mask[i, j, 3] > 0:  # Already has color
                                    highlight_mask[i, j] = (highlight_mask[i, j] + color) // 2
                                else:
                                    highlight_mask[i, j] = color

            return self._render_state(agent_view_size, state, highlight=True, 
                                    tile_size=tile_size, highlight_mask=highlight_mask)
        else:
            return super().render(agent_view_size, state, highlight=True, tile_size=tile_size)

    def _render_state(self, agent_view_size, state, highlight=True, tile_size=32, highlight_mask=None):
        """Render state with optional view area highlighting"""

        if self.use_old_rendering:
            # Use old rendering method (triangles)
            img = super()._render_state(agent_view_size, state, highlight=False, tile_size=tile_size)
        else:
            # Use new rendering method (chef images) - similar to base class render method
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

            # Create mock players and objects for new rendering
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
            for i in range(self._num_agents):
                if i < len(env_state.agent_pos):
                    # Get agent position directly from state (x, y format)
                    pos = (int(env_state.agent_pos[i, 0]), int(env_state.agent_pos[i, 1]))

                    # Convert environment direction index to visualization direction tuple
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
            import pygame
            img = pygame.surfarray.array3d(surface).transpose(1, 0, 2)

        # Check if rendering was successful
        if img is None:
            print("Warning: Base rendering failed, returning None")
            return None

        # Apply view area highlighting if provided
        if highlight_mask is not None:
            # Resize highlight mask to match image size
            img_height, img_width = img.shape[:2]
            mask_height, mask_width = highlight_mask.shape[:2]

            # Calculate scaling factors
            scale_y = img_height / mask_height
            scale_x = img_width / mask_width

            # Create overlay
            overlay = np.zeros((img_height, img_width, 4), dtype=np.uint8)

            for i in range(mask_height):
                for j in range(mask_width):
                    if highlight_mask[i, j, 3] > 0:  # Has alpha
                        # Map to image coordinates
                        img_y_start = int(i * scale_y)
                        img_y_end = int((i + 1) * scale_y)
                        img_x_start = int(j * scale_x)
                        img_x_end = int((j + 1) * scale_x)

                        overlay[img_y_start:img_y_end, img_x_start:img_x_end] = highlight_mask[i, j]

            # Blend overlay with image
            alpha = overlay[:, :, 3:4] / 255.0
            img_rgba = np.concatenate([img, np.ones((img_height, img_width, 1), dtype=np.uint8) * 255], axis=2)
            img_rgba = img_rgba.astype(np.float32)
            overlay = overlay.astype(np.float32)

            blended = img_rgba * (1 - alpha) + overlay * alpha
            img = blended[:, :, :3].astype(np.uint8)

        return img

    def _create_mock_objects(self, grid, env_state):
        """
        Override parent method to adjust coordinates for padding removed from grid.

        Parameters
        ----------
        grid : numpy.ndarray
            The cropped grid representation of the environment
        env_state : State
            The environment state

        Returns
        -------
        dict
            A dictionary of mock objects for visualization with adjusted coordinates
        """
        objects = {}

        # Check if we have pot positions and status information
        if not hasattr(env_state, 'pot_pos') or not hasattr(env_state, 'maze_map'):
            return objects

        # Calculate padding that was removed from the grid
        height, width = grid.shape[:2]
        # Use the same padding calculation as in _render_state
        # The grid was extracted with: grid = env_state.maze_map[padding:-padding, padding:-padding, :]
        # So padding = agent_view_size - 1, but we need to get agent_view_size
        # We can calculate it from the maze_map and grid sizes
        if hasattr(env_state.maze_map, 'shape'):
            maze_height = env_state.maze_map.shape[0]
            padding = (maze_height - height) // 2
        else:
            padding = 0

        # Find pots in the grid and create mock soup objects for them
        for i, pot_pos in enumerate(env_state.pot_pos):
            # Get pot status from maze_map (using original coordinates)
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

                # Create a position tuple for the object - ADJUST FOR PADDING
                adjusted_x = int(pot_x) - padding
                adjusted_y = int(pot_y) - padding

                # Ensure the adjusted position is within grid bounds
                if (0 <= adjusted_x < width and 0 <= adjusted_y < height):
                    position = (adjusted_x, adjusted_y)

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
        # These coordinates are already correct since they're from the cropped grid
        for y in range(height):
            for x in range(width):
                obj = grid[y, x, :]
                obj_type = obj[0]

                # Create objects for individual items placed on counters
                if obj_type == OBJECT_TO_INDEX['onion']:
                    obj_id = f"onion_{x}_{y}"
                    onion_obj = type('MockOnion', (), {
                        'name': 'onion',
                        'position': (int(x), int(y)),  # Use correct x,y order
                        'ingredients': None
                    })
                    objects[obj_id] = onion_obj
                elif obj_type == OBJECT_TO_INDEX['plate']:
                    obj_id = f"plate_{x}_{y}"
                    plate_obj = type('MockPlate', (), {
                        'name': 'dish',  # 'dish' is the visualization name for an empty plate
                        'position': (int(x), int(y)),  # Use correct x,y order
                        'ingredients': None
                    })
                    objects[obj_id] = plate_obj
                elif obj_type == OBJECT_TO_INDEX['dish']:
                    obj_id = f"dish_{x}_{y}"
                    dish_obj = type('MockDish', (), {
                        'name': 'soup',  # Dish with soup
                        'position': (int(x), int(y)),  # Use correct x,y order
                        'ingredients': ['onion', 'onion', 'onion'],
                        'is_ready': True,
                        '_cooking_tick': -1,  # Not cooking (already ready)
                        'cook_time': self.pot_full_status  # Use configurable cook time
                    })
                    objects[obj_id] = dish_obj

        return objects

    def animate(self, state_seq, agent_view_size, task_idx, task_name, exp_dir, env=None, tile_size: int = 32):
        """
        Make a GIF with view area highlighting for PO environments.

        Args:
            state_seq: List of states to animate
            agent_view_size: Agent view size (kept for compatibility)
            task_idx: Task index
            task_name: Task name
            exp_dir: Export directory
            env: OvercookedPO environment instance (needed for view masks)
            tile_size: Size of each tile in pixels
        """
        import imageio.v3 as iio
        import os

        # Generate frames using the PO render method with view highlighting
        frames = []
        for state in state_seq:
            # Use the PO render method which supports view highlighting
            frame = self.render(
                agent_view_size=agent_view_size,
                state=state,
                env=env,
                highlight_views=True,
                tile_size=tile_size
            )
            if frame is not None:
                frames.append(frame)

        if not frames:
            print("Warning: No frames generated for PO visualization")
            return

        # Create directory if it doesn't exist
        os.makedirs(exp_dir, exist_ok=True)

        file_name = f"task_{task_idx}_{task_name}"
        file_path = f"{exp_dir}/{file_name}.gif"

        iio.imwrite(file_path, frames, loop=0, fps=10)

        # Log to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({file_name: wandb.Video(file_path, format="gif")})
        except ImportError:
            pass
