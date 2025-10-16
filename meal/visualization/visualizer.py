from __future__ import annotations

import os
from typing import Sequence, Optional

import numpy as np
import pygame
from PIL import Image

from meal.visualization.adapters import to_drawable_state
from meal.visualization.bridge_stateviz import render_drawable_with_stateviz
from meal.visualization.cache import TileCache
from meal.visualization.painter import render_frame
from meal.visualization.rendering.state_visualizer import StateVisualizer
from meal.visualization.types import DrawableState
from meal.visualization.window import Window


class OvercookedVisualizer:
    """
    Numpy-only renderer that uses DrawableState + TileCache + adapter.
    Produces RGB arrays; can display via Window or write GIFs.
    """

    def __init__(self, num_agents: int = 2, pot_full: int = 20, pot_empty: int = 23, tile_px: int = 64):
        self.num_agents = num_agents
        self.pot_full = pot_full
        self.pot_empty = pot_empty
        self.tile_px = tile_px
        self.cache = TileCache(max_items=4096)
        self.window: Optional[Window] = None
        self.state_visualizer = StateVisualizer(tile_size=tile_px)

    def _lazy_window(self):
        if self.window is None:
            self.window = Window("Kitchen")

    # ---------- single-frame ----------
    def render(self, env_state_raw, show: bool = False) -> np.ndarray:
        """
        env_state_raw: raw state or log wrapper with .env_state
        Returns RGB ndarray (H*tile_px, W*tile_px, 3).
        """
        dstate: DrawableState = to_drawable_state(
            env_state_raw, pot_full=self.pot_full, pot_empty=self.pot_empty, num_agents=self.num_agents,
        )
        frame = render_frame(dstate, tile_px=self.tile_px, cache=self.cache)
        if show:
            self._lazy_window()
            self.window.show_img(frame)
        return frame

    # ---------- sequence / GIF ----------
    def animate(self, state_seq: Sequence[object], out_path: str, fps: int = 10, pad_to_max: bool = True) -> str:
        """
        Render a sequence of env states to an animated GIF (palette-safe).
        """

        # 1) convert
        dseq = [
            to_drawable_state(s, pot_full=self.pot_full, pot_empty=self.pot_empty, num_agents=self.num_agents)
            for s in state_seq
        ]

        # 2) optional pad all to common WxH (so GIF doesn't jump)
        if pad_to_max:
            maxH = max(ds.H for ds in dseq)
        maxW = max(ds.W for ds in dseq)
        dseq = [ds.pad_to(maxH, maxW) for ds in dseq]

        # 3) paint frames
        frames = []
        for ds in dseq:
            surface = render_drawable_with_stateviz(ds, self.state_visualizer)
            frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)  # (H, W, 3)
            frames.append(frame.copy())  # copy to detach from pygame surface buffer

        # 4) write the GIF
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        pil_frames = [Image.fromarray(f, mode="RGB") for f in frames]
        duration_ms = int(1000 / fps)

        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=False,
            duration=duration_ms,
            loop=0,
            disposal=2,
        )

        return out_path
