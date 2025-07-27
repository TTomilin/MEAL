import numpy as np
from jax_marl.eval.visualizer import OvercookedVisualizer
from jax_marl.environments.overcooked.common import COLORS

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
