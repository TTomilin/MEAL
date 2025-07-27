from enum import IntEnum
from typing import Tuple, Dict

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core.frozen_dict import FrozenDict
from jax import lax

from jax_marl.environments import MultiAgentEnv
from jax_marl.environments import spaces
from jax_marl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC,
    make_overcooked_map)
from jax_marl.environments.overcooked.layouts import overcooked_layouts as layouts, layout_grid_to_dict
from jax_marl.environments.overcooked.overcooked import Overcooked, Actions, State

# Constants for partial observability
DEFAULT_VIEW_AHEAD = 3
DEFAULT_VIEW_BEHIND = 1
DEFAULT_VIEW_SIDES = 2

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
        random_reset: bool = False,
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
            [0, 1],   # South: down  
            [1, 0],   # East: right
            [-1, 0]   # West: left
        ])

        # Get forward, backward, left, right directions relative to agent orientation
        forward_vec = dir_vectors[agent_dir]
        backward_vec = -forward_vec

        # Perpendicular vectors for left/right using JAX-compatible operations
        # For North/South (0,1): left=West=[-1,0], right=East=[1,0]
        # For East/West (2,3): left=North=[0,-1], right=South=[0,1]
        north_south_facing = jnp.logical_or(agent_dir == 0, agent_dir == 1)
        left_vec = jnp.where(north_south_facing, 
                            jnp.array([-1, 0]),   # West for North/South
                            jnp.array([0, -1]))   # North for East/West
        right_vec = jnp.where(north_south_facing,
                             jnp.array([1, 0]),    # East for North/South  
                             jnp.array([0, 1]))    # South for East/West

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

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return partially observable observations for each agent

        Each agent only sees a limited area around them based on view parameters.
        Unseen areas are filled with 'unseen' object type.
        """
        width = self.obs_shape[0]
        height = self.obs_shape[1]
        n_channels = self.obs_shape[2]
        padding = (state.maze_map.shape[0] - height) // 2

        # Get full observations first (same as parent class)
        full_obs = super().get_obs(state)

        # Create partial observations for each agent
        partial_obs = {}

        for agent_idx in range(self.num_agents):
            agent_key = f"agent_{agent_idx}"

            # Get agent position and direction
            agent_pos = state.agent_pos[agent_idx]
            agent_dir = state.agent_dir_idx[agent_idx]

            # Create view mask for this agent
            view_mask = self._get_agent_view_mask(agent_pos, agent_dir, height, width)

            # Apply mask to full observation
            agent_obs = full_obs[agent_key].copy()

            # Create unseen mask (inverse of view mask)
            unseen_mask = ~view_mask

            # Set unseen areas to 'unseen' object type
            unseen_layer = jnp.zeros((height, width), dtype=jnp.uint8)
            unseen_layer = jnp.where(unseen_mask, OBJECT_TO_INDEX['unseen'], 0)

            # Apply unseen mask to all channels
            for channel in range(n_channels):
                agent_obs = agent_obs.at[:, :, channel].set(
                    jnp.where(unseen_mask, unseen_layer, agent_obs[:, :, channel])
                )

            partial_obs[agent_key] = agent_obs

        return partial_obs

    def get_agent_view_masks(self, state: State) -> Dict[str, chex.Array]:
        """Get view masks for all agents (useful for visualization)

        Returns:
            Dictionary mapping agent names to their view masks
        """
        width = self.obs_shape[0] 
        height = self.obs_shape[1]

        masks = {}
        for agent_idx in range(self.num_agents):
            agent_key = f"agent_{agent_idx}"
            agent_pos = state.agent_pos[agent_idx]
            agent_dir = state.agent_dir_idx[agent_idx]
            masks[agent_key] = self._get_agent_view_mask(agent_pos, agent_dir, height, width)

        return masks

    @property
    def name(self) -> str:
        """Environment name"""
        return "OvercookedPO"
