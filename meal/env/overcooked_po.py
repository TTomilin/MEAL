from typing import Dict

import chex
import jax.numpy as jnp

from meal.env.overcooked import Overcooked, State
from meal.env.utils import spaces
from meal.env.utils.difficulty_config import get_difficulty_params

# Constants for partial observability (default: easy difficulty)
DEFAULT_VIEW_AHEAD = 1
DEFAULT_VIEW_SIDES = 1
DEFAULT_VIEW_BEHIND = 0


class OvercookedPO(Overcooked):
    """Partially Observable Overcooked Environment

    Agents can only observe a limited area around them:
    - view_ahead: number of tiles visible ahead of the agent
    - view_behind: number of tiles visible behind the agent  
    - view_sides: number of tiles visible to the sides of the agent

    This yields a local window aligned to the where the agent is facing:
      width  = 2*view_sides + 1
      height = view_behind + 1 (self) + view_ahead
    """

    def __init__(
            self,
            difficulty: str | None = None,
            view_ahead: int = DEFAULT_VIEW_AHEAD,
            view_sides: int = DEFAULT_VIEW_SIDES,
            view_behind: int = DEFAULT_VIEW_BEHIND,
            **kwargs,
    ):
        """Initialize partially observable Overcooked environment

        Args:
            view_ahead: Number of tiles visible ahead of agent (default: 1)
            view_sides: Number of tiles visible to sides of agent (default: 1)
            view_behind: Number of tiles visible behind agent (default: 0)
            Other args same as base Overcooked environment
        """
        super().__init__(difficulty=difficulty, **kwargs)

        # Store partial observability parameters
        if difficulty:
            params = get_difficulty_params(difficulty)
            view_ahead = params["view_ahead"]
            view_behind = params["view_behind"]
            view_sides = params["view_sides"]
        self.view_ahead = view_ahead
        self.view_behind = view_behind
        self.view_sides = view_sides

        # Calculate the partially observable dimensions for observation_space()
        # The observable area forms a grid: (view_sides*2 + 1) x (view_ahead + view_behind + 1)
        self.po_width = self.view_sides * 2 + 1  # 2 sides + agent position
        self.po_height = self.view_ahead + self.view_behind + 1  # ahead + behind + agent position

        # Channel count stays identical to the full obs
        self.obs_shape = (self.width, self.height, self.obs_channels)

    # --------------------------- helpers ---------------------------

    @staticmethod
    def _dir_vecs():
        # (N,S,E,W) as dx,dy; we’ll use (y,x) indexing later carefully
        return jnp.array([
            [0, -1],  # North: up
            [0, 1],  # South: down
            [1, 0],  # East: right
            [-1, 0],  # West: left
        ], dtype=jnp.int32)

    def _extract_agent_view(self, full_obs_hwC: chex.Array, agent_pos_xy: chex.Array, agent_dir_idx: int) -> chex.Array:
        """
        Crop a (po_height x po_width) window aligned to agent orientation from a full (H,W,C) tensor.
        • full_obs_hwC shape: (H, W, C)
        • agent_pos_xy      : (2,) with (x, y)
        • returns           : (po_height, po_width, C)
        """
        H, W, C = full_obs_hwC.shape
        # orientation-aligned axes: forward/back along dir, left/right perpendicular
        dir_vecs = self._dir_vecs()  # (4,2) in (dx,dy)
        agent_dir_idx = jnp.asarray(agent_dir_idx, jnp.int32)
        fwd_dx, fwd_dy = dir_vecs[agent_dir_idx, 0], dir_vecs[agent_dir_idx, 1]
        # left is a 90° CCW rotation of forward: (-dy, dx)
        left_dx, left_dy = -fwd_dy, fwd_dx

        # grids in (row=y, col=x) order for the local window
        # rows: back .. self .. ahead (size = po_height)
        # cols: left .. self .. right (size = po_width)
        row_offsets = jnp.arange(-self.view_behind, self.view_ahead + 1, dtype=jnp.int32)  # length po_height
        col_offsets = jnp.arange(-self.view_sides, self.view_sides + 1, dtype=jnp.int32)  # length po_width
        # Meshgrid in (row, col) = (po_height, po_width)
        off_rows, off_cols = jnp.meshgrid(row_offsets, col_offsets, indexing="ij")  # (Hpo,Wpo)

        # Map local (row/col) offsets to global (x,y) using forward/left basis
        ax, ay = agent_pos_xy[0].astype(jnp.int32), agent_pos_xy[1].astype(jnp.int32)
        # Δx = off_rows * fwd_dx + off_cols * left_dx
        # Δy = off_rows * fwd_dy + off_cols * left_dy
        dx = off_rows * fwd_dx + off_cols * left_dx
        dy = off_rows * fwd_dy + off_cols * left_dy

        gx = ax + dx  # (Hpo,Wpo)
        gy = ay + dy  # (Hpo,Wpo)

        # Valid in-bounds mask
        valid = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)

        # Safe indices (clip) for gather; we’ll zero invalid later
        ix = jnp.clip(gx, 0, W - 1)
        iy = jnp.clip(gy, 0, H - 1)

        # Gather: note full_obs is (H,W,C) so we index [iy, ix]
        patch = full_obs_hwC[iy, ix, :]  # (Hpo, Wpo, C)

        # Zero out anything outside bounds (unseen = zeros per channel)
        patch = jnp.where(valid[..., None], patch, jnp.zeros_like(patch))

        return patch

    # ------------------------ main overrides -----------------------

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """
        Returns per-agent partial observations with shape (po_height, po_width, C),
        where C == self.obs_channels (same channels as full obs).
        """
        # First get the full per-agent obs from the base env: dict of (H,W,C)
        full = super().get_obs(state)

        # Vectorized crop per agent
        po = {}
        for i in range(self.num_agents):
            key = f"agent_{i}"
            agent_pos_xy = state.agent_pos[i]  # (x,y)
            agent_dir_idx = state.agent_dir_idx[i]  # int
            po[key] = self._extract_agent_view(full[key], agent_pos_xy, agent_dir_idx)

        return po

    def get_agent_view_masks(self, state: State) -> Dict[str, chex.Array]:
        # Handle LogWrapper state transparently
        if hasattr(state, "env_state"):
            state = state.env_state

        H = self.height
        W = self.width

        dir_vecs = self._dir_vecs()
        masks: Dict[str, chex.Array] = {}

        for i in range(self.num_agents):
            ax = state.agent_pos[i, 0].astype(jnp.int32)
            ay = state.agent_pos[i, 1].astype(jnp.int32)

            fdx, fdy = dir_vecs[state.agent_dir_idx[i]]
            ldx, ldy = -fdy, fdx

            # Build forward/behind rows
            rows_forward = jnp.arange(0, self.view_ahead + 1, dtype=jnp.int32)  # 1..ahead
            rows_backward = jnp.arange(-self.view_behind, 0, dtype=jnp.int32)  # -behind..-1
            row_offsets = jnp.concatenate([rows_backward, rows_forward], axis=0)

            # Build lateral columns including center only if the “spine” is needed
            col_offsets = jnp.arange(-self.view_sides, self.view_sides + 1, dtype=jnp.int32)  # -sides..0..+sides

            off_rows, off_cols = jnp.meshgrid(row_offsets, col_offsets, indexing="ij")
            gx = ax + (off_rows * fdx + off_cols * ldx)
            gy = ay + (off_rows * fdy + off_cols * ldy)

            valid = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)

            # clip only for the indexing op; write 'valid' directly so OOB never gets set
            ix = jnp.clip(gx, 0, W - 1)
            iy = jnp.clip(gy, 0, H - 1)

            mask = jnp.zeros((H, W), dtype=jnp.bool_)
            mask = mask.at[iy, ix].set(valid)  # <— write the validity mask directly
            masks[f"agent_{i}"] = mask

        return masks

    def observation_space(self) -> spaces.Box:
        # Match get_obs output: (Hpo, Wpo, C)
        return spaces.Box(0, 255, (self.po_height, self.po_width, self.obs_channels), dtype=jnp.uint8)

    @property
    def name(self) -> str:
        return "OvercookedPO"
