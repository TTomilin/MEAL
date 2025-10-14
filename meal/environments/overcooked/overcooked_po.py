from typing import Dict

import chex
import jax.numpy as jnp

from meal.environments import spaces
from meal.environments.overcooked.common import (
    OBJECT_TO_INDEX)
from meal.environments.overcooked.overcooked import Overcooked, State

# Constants for partial observability
DEFAULT_VIEW_AHEAD = 3
DEFAULT_VIEW_BEHIND = 1
DEFAULT_VIEW_SIDES = 1


class OvercookedPO(Overcooked):
    """Partially Observable Overcooked Environment

    Agents can only observe a limited area around them:
    - view_ahead: number of tiles visible ahead of the agent
    - view_behind: number of tiles visible behind the agent  
    - view_sides: number of tiles visible to the sides of the agent
    """

    def __init__(
            self,
            layout=None,
            layout_name="cramped_room",
            random_reset: bool = True,
            max_steps: int = 400,
            task_id: int = 0,
            num_agents: int = 2,
            agent_restrictions: dict = None,
            soup_cook_time: int = 20,
            view_ahead: int = DEFAULT_VIEW_AHEAD,
            view_behind: int = DEFAULT_VIEW_BEHIND,
            view_sides: int = DEFAULT_VIEW_SIDES,
    ):
        """Initialize partially observable Overcooked environment

        Args:
            view_ahead: Number of tiles visible ahead of agent (default: 3)
            view_behind: Number of tiles visible behind agent (default: 1)
            view_sides: Number of tiles visible to sides of agent (default: 1)
            Other args same as base Overcooked environment
        """
        super().__init__(
            layout=layout,
            layout_name=layout_name,
            random_reset=random_reset,
            max_steps=max_steps,
            task_id=task_id,
            num_agents=num_agents,
            agent_restrictions=agent_restrictions,
            soup_cook_time=soup_cook_time
        )

        # Store partial observability parameters
        self.view_ahead = view_ahead
        self.view_behind = view_behind
        self.view_sides = view_sides

        # Calculate maximum view distance for any direction
        self.max_view_distance = max(view_ahead, view_behind, view_sides)

        # Calculate the partially observable dimensions for observation_space()
        # The observable area forms a grid: (view_sides*2 + 1) x (view_ahead + view_behind + 1)
        self.po_width = self.view_sides * 2 + 1  # 2 sides + agent position
        self.po_height = self.view_ahead + self.view_behind + 1  # ahead + behind + agent position

    def _get_agent_view_mask(self, agent_pos: chex.Array, agent_dir: int, height: int, width: int) -> chex.Array:
        """Create a boolean mask for what an agent can observe as a 5x5 grid

        Args:
            agent_pos: Agent position [x, y]
            agent_dir: Agent direction (0=North, 1=South, 2=East, 3=West)
            height: Grid height
            width: Grid width

        Returns:
            Boolean mask of shape (height, width) where True = observable
            Forms a 5x5 grid: 3 front, 1 back, 2 sides
        """
        mask = jnp.zeros((height, width), dtype=jnp.bool_)

        # Agent position
        agent_x, agent_y = agent_pos[0], agent_pos[1]

        # Direction vectors: [North, South, East, West]
        dir_vectors = jnp.array([
            [0, -1],  # North: up
            [0, 1],  # South: down
            [1, 0],  # East: right
            [-1, 0]  # West: left
        ])

        # Get forward, backward, left, right directions relative to agent orientation
        forward_vec = dir_vectors[agent_dir]
        backward_vec = -forward_vec

        # Perpendicular vectors for left/right using JAX-compatible operations
        # For North/South (0,1): left=West=[-1,0], right=East=[1,0]
        # For East/West (2,3): left=North=[0,-1], right=South=[0,1]
        north_south_facing = jnp.logical_or(agent_dir == 0, agent_dir == 1)
        left_vec = jnp.where(north_south_facing,
                             jnp.array([-1, 0]),  # West for North/South
                             jnp.array([0, -1]))  # North for East/West
        right_vec = jnp.where(north_south_facing,
                              jnp.array([1, 0]),  # East for North/South
                              jnp.array([0, 1]))  # South for East/West

        # Create a 5x5 grid centered on the agent
        # Grid extends: 3 forward, 1 backward, 2 to each side

        # Generate all positions in the 5x5 grid
        for forward_offset in range(-self.view_behind, self.view_ahead + 1):
            for side_offset in range(-self.view_sides, self.view_sides + 1):
                # Calculate the actual position
                pos = (jnp.array([agent_x, agent_y]) +
                       forward_offset * forward_vec +
                       side_offset * left_vec)

                x, y = pos[0], pos[1]

                # Check if position is within bounds
                valid = jnp.logical_and(
                    jnp.logical_and(x >= 0, x < width),
                    jnp.logical_and(y >= 0, y < height)
                )

                # Set the mask if valid
                mask = jnp.where(valid, mask.at[y, x].set(True), mask)

        return mask

    def _extract_agent_view(self, full_obs: chex.Array, agent_pos: chex.Array, agent_dir: int) -> chex.Array:
        """Extract the observable area around an agent from the full observation

        Args:
            full_obs: Full observation of shape (width, height, 26)
            agent_pos: Agent position [x, y]
            agent_dir: Agent direction (0=North, 1=South, 2=East, 3=West)

        Returns:
            Partial observation of shape (po_width, po_height, 26)
        """
        full_height, full_width = full_obs.shape[1], full_obs.shape[0]

        # Direction vectors: [North, South, East, West]
        dir_vectors = jnp.array([
            [0, -1],  # North: up
            [0, 1],   # South: down
            [1, 0],   # East: right
            [-1, 0]   # West: left
        ])

        # Get forward, backward, left, right directions relative to agent orientation
        forward_vec = dir_vectors[agent_dir]

        # Perpendicular vectors for left/right
        north_south_facing = jnp.logical_or(agent_dir == 0, agent_dir == 1)
        left_vec = jnp.where(north_south_facing,
                             jnp.array([-1, 0]),  # West for North/South
                             jnp.array([0, -1]))  # North for East/West

        # Create the partial observation
        partial_obs = jnp.zeros((self.po_width, self.po_height, 26), dtype=jnp.uint8)

        # Fill with 'unseen' by default
        partial_obs = partial_obs.at[:, :, :].set(OBJECT_TO_INDEX['unseen'])

        # Create coordinate grids for vectorized operations
        po_y_coords, po_x_coords = jnp.meshgrid(
            jnp.arange(self.po_height), 
            jnp.arange(self.po_width), 
            indexing='ij'
        )

        # Calculate offsets for all positions at once
        forward_offsets = po_y_coords - self.view_behind  # -view_behind to +view_ahead
        side_offsets = po_x_coords - self.view_sides      # -view_sides to +view_sides

        # Calculate full observation positions for all partial observation positions
        # Shape: (po_height, po_width, 2)
        full_positions = (agent_pos[None, None, :] + 
                         forward_offsets[:, :, None] * forward_vec[None, None, :] + 
                         side_offsets[:, :, None] * left_vec[None, None, :])

        # Extract x and y coordinates
        full_x_coords = full_positions[:, :, 0]  # Shape: (po_height, po_width)
        full_y_coords = full_positions[:, :, 1]  # Shape: (po_height, po_width)

        # Check bounds for all positions at once
        valid_mask = jnp.logical_and(
            jnp.logical_and(full_x_coords >= 0, full_x_coords < full_width),
            jnp.logical_and(full_y_coords >= 0, full_y_coords < full_height)
        )

        # Convert to integer indices (safe because we're using them in array indexing, not int())
        full_x_indices = jnp.clip(full_x_coords.astype(jnp.int32), 0, full_width - 1)
        full_y_indices = jnp.clip(full_y_coords.astype(jnp.int32), 0, full_height - 1)

        # Extract observations for all valid positions
        # Use advanced indexing to get all observations at once
        extracted_obs = full_obs[full_x_indices, full_y_indices, :]  # Shape: (po_height, po_width, 26)

        # Apply the valid mask - keep extracted observations where valid, 'unseen' elsewhere
        unseen_obs = jnp.full_like(extracted_obs, OBJECT_TO_INDEX['unseen'])
        partial_obs = jnp.where(valid_mask[:, :, None], extracted_obs, unseen_obs)

        # Transpose to match expected output shape (po_width, po_height, 26)
        partial_obs = jnp.transpose(partial_obs, (1, 0, 2))

        return partial_obs

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return partially observable observations for each agent

        Each agent only sees a limited area around them based on view parameters.
        The returned observations have shape (po_width, po_height, 26) which matches
        the declared observation_space().
        """
        # Get full observations first (same as parent class)
        full_obs = super().get_obs(state)

        # Create partial observations for each agent
        partial_obs = {}

        for agent_idx in range(self.num_agents):
            agent_key = f"agent_{agent_idx}"

            # Get agent position and direction
            agent_pos = state.agent_pos[agent_idx]
            agent_dir = state.agent_dir_idx[agent_idx]

            # Extract the observable area around the agent
            agent_obs = self._extract_agent_view(full_obs[agent_key], agent_pos, agent_dir)
            partial_obs[agent_key] = agent_obs

        return partial_obs

    def get_agent_view_masks(self, state: State) -> Dict[str, chex.Array]:
        """Get view masks for all agents (useful for visualization)

        Returns:
            Dictionary mapping agent names to their view masks
        """
        width = self.obs_shape[0]
        height = self.obs_shape[1]

        # Handle LogEnvState wrapper - extract the actual environment state
        if hasattr(state, 'env_state'):
            env_state = state.env_state
        else:
            env_state = state

        masks = {}
        for agent_idx in range(self.num_agents):
            agent_key = f"agent_{agent_idx}"
            agent_pos = env_state.agent_pos[agent_idx]
            agent_dir = env_state.agent_dir_idx[agent_idx]
            masks[agent_key] = self._get_agent_view_mask(agent_pos, agent_dir, height, width)

        return masks

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment.

        For partially observable environments, the observation space reflects
        the actual observable area around each agent, not the full environment.
        """
        return spaces.Box(0, 255, (self.po_width, self.po_height, 26))

    @property
    def name(self) -> str:
        """Environment name"""
        return "OvercookedPO"
