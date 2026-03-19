from __future__ import annotations

import dataclasses
import os
from typing import Sequence, Optional, List

import numpy as np
import pygame
import wandb
import imageio.v3 as iio

from meal.visualization.adapters import to_drawable_state, char_grid_to_drawable_state
from meal.visualization.bridge_stateviz import render_drawable_with_stateviz
from meal.visualization.cache import TileCache
from meal.visualization.renderer_config import RENDERER_VERSION
from meal.visualization.types import DrawableState, Tile, Obj
from meal.visualization.window import Window

FLASH_DURATION = 5  # frames to show the delivery flash


def _make_state_visualizer(renderer_version: str, tile_px: int):
    """Instantiate the correct renderer backend."""
    if renderer_version == "v2":
        from meal.visualization.rendering.state_visualizer_v2 import StateVisualizerV2
        return StateVisualizerV2(tile_size=tile_px)
    else:
        from meal.visualization.rendering.state_visualizer import StateVisualizer
        return StateVisualizer(tile_size=tile_px)


def _detect_deliveries(prev_ds: DrawableState, curr_ds: DrawableState) -> tuple:
    """Return serve-tile coords where a delivery occurred between prev and curr states."""
    serve_tiles = set()
    for y, row in enumerate(prev_ds.grid):
        for x, tile in enumerate(row):
            if tile == Tile.SERVE:
                serve_tiles.add((x, y))

    if not serve_tiles:
        return ()

    prev_players = {p.id: p for p in prev_ds.players}
    curr_players = {p.id: p for p in curr_ds.players}

    delivered = set()
    for pid, prev_p in prev_players.items():
        if prev_p.held == Obj.DISH:
            curr_p = curr_players.get(pid)
            if curr_p is None or curr_p.held != Obj.DISH:
                px, py = prev_p.pos
                for sx, sy in serve_tiles:
                    if abs(px - sx) + abs(py - sy) <= 1:
                        delivered.add((sx, sy))
                        break

    return tuple(delivered)


class OvercookedVisualizer:
    """
    Numpy-only renderer that uses DrawableState + TileCache + adapter.
    Produces RGB arrays; can display via Window or write GIFs.
    """

    def __init__(
        self,
        num_agents: int = 2,
        pot_full: int = 20,
        pot_empty: int = 23,
        tile_px: int = 64,
        renderer_version: Optional[str] = None,
    ):
        self.num_agents = num_agents
        self.pot_full = pot_full
        self.pot_empty = pot_empty
        self.tile_px = tile_px
        self.cache = TileCache(max_items=4096)
        self.window: Optional[Window] = None
        # renderer_version from caller overrides the global default in renderer_config.py
        rv = renderer_version if renderer_version is not None else RENDERER_VERSION
        self.state_visualizer = _make_state_visualizer(rv, tile_px)

    def _lazy_window(self):
        if self.window is None:
            self.window = Window("Kitchen")

    def _drawable_state_to_frame(self, drawable_state: DrawableState) -> np.ndarray:
        surface = render_drawable_with_stateviz(drawable_state, self.state_visualizer)
        frame = pygame.surfarray.pixels3d(surface)
        frame = np.ascontiguousarray(frame.transpose(1, 0, 2))  # (H, W, 3)
        return frame

    def _render_drawable_state(self, drawable_state: DrawableState, show: bool = False) -> np.ndarray:
        frame = self._drawable_state_to_frame(drawable_state)
        if show:
            self._lazy_window()
            self.window.show_img(frame)
        return frame

    # ---------- single-frame ----------
    def render(self, env_state, show: bool = False) -> np.ndarray:
        """
        env_state: raw state or log wrapper with .env_state
        Returns RGB ndarray (H*tile_px, W*tile_px, 3).
        """
        dstate: DrawableState = to_drawable_state(
            env_state, pot_full=self.pot_full, pot_empty=self.pot_empty, num_agents=self.num_agents,
        )
        return self._render_drawable_state(dstate, show)

    def render_grid(self, char_grid: List[List[str]], show: bool = False) -> np.ndarray:
        """
        Render a character grid directly without needing to create a full environment state.

        Args:
            char_grid: 2D list of character strings representing the grid layout
            show: whether to display the rendered image in a window

        Returns:
            RGB ndarray (H*tile_px, W*tile_px, 3)
        """
        # Convert character grid to DrawableState
        drawable_state = char_grid_to_drawable_state(char_grid)
        return self._render_drawable_state(drawable_state, show)

    # ---------- sequence ----------
    def animate(self, state_seq: Sequence[object], out_path: str, task_idx: int = 0, fps: int = 10, pad_to_max: bool = False, env = None) -> str:
        """
        Render a sequence of env states to GIF or MP4.
        """

        # 1) convert
        state_seq = [
            to_drawable_state(s, pot_full=self.pot_full, pot_empty=self.pot_empty, num_agents=self.num_agents)
            for s in state_seq
        ]

        # 2) optional pad all layouts to the largest WxH
        if pad_to_max:
            maxH = max(drawable_state.H for drawable_state in state_seq)
            maxW = max(ds.W for ds in state_seq)
            state_seq = [ds.pad_to(maxH, maxW) for ds in state_seq]

        # 2b) detect delivery events and mark flash frames
        delivery_map: dict[int, set] = {}
        for i in range(len(state_seq) - 1):
            deliveries = _detect_deliveries(state_seq[i], state_seq[i + 1])
            if deliveries:
                for j in range(i + 1, min(i + 1 + FLASH_DURATION, len(state_seq))):
                    delivery_map.setdefault(j, set()).update(deliveries)
        if delivery_map:
            state_seq = [
                dataclasses.replace(ds, delivery_positions=tuple(delivery_map.get(i, ())))
                for i, ds in enumerate(state_seq)
            ]

        # 3) paint the frames
        frames = []
        for drawable_state in state_seq:
            frames.append(self._drawable_state_to_frame(drawable_state))
        frames = np.stack(frames, axis=0)  # shape (T, H, W, 3)

        # 4) write to file (format-aware, artifact-free)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self._save_frames(frames, out_path, fps)

        # 5) log to wandb
        if wandb.run is not None:
            wandb.log({f"task_{task_idx}": wandb.Video(out_path, format="mp4")})

        return out_path

    def _save_frames(self, frames: np.ndarray, out_path: str, fps: int) -> None:
        """
        Write *frames* (T, H, W, 3) uint8 to *out_path* without lossy artifacts.

        GIF — uses PIL with a single global colour palette computed from a
              representative sample of all frames, then applies that palette to
              every frame with NO dithering.  This guarantees that identical
              pixels in different frames map to the same palette index and
              therefore produce identical bytes → no flickering.

        Video — uses lossless H.264 (libx264rgb, CRF 0, RGB pixel format).
                No YUV colour-space conversion, no blocking artefacts.
        """
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".gif":
            self._save_gif(frames, out_path, fps)
        else:
            self._save_video(frames, out_path, fps)

    @staticmethod
    def _save_gif(frames: np.ndarray, out_path: str, fps: int) -> None:
        from PIL import Image

        pil_frames = [Image.fromarray(f, mode="RGB") for f in frames]
        T, H, W = frames.shape[:3]

        # Build one global palette from a uniform sample of frames (up to 10).
        # Sampling avoids excessive memory use on long episodes.
        sample_count = min(10, T)
        sample_idx = np.linspace(0, T - 1, sample_count, dtype=int)
        combined = Image.new("RGB", (W, H * sample_count))
        for row, idx in enumerate(sample_idx):
            combined.paste(pil_frames[idx], (0, row * H))
        palette_ref = combined.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

        # Quantize every frame to that fixed palette with NO dithering.
        # Without dithering, identical RGB pixels always map to the same index
        # regardless of surrounding pixels → static areas are truly static.
        quantized = [f.quantize(palette=palette_ref, dither=Image.Dither.NONE) for f in pil_frames]

        duration_ms = int(1000 / fps)
        quantized[0].save(
            out_path,
            save_all=True,
            append_images=quantized[1:],
            optimize=False,
            duration=duration_ms,
            loop=0,
            disposal=2,
        )

    @staticmethod
    def _save_video(frames: np.ndarray, out_path: str, fps: int) -> None:
        # libx264rgb: lossless H.264 in native RGB (no YUV conversion, no chroma
        # subsampling).  CRF 0 = mathematically lossless.  Files are moderate in
        # size for short sprite-art clips.
        iio.imwrite(
            out_path,
            frames,
            fps=fps,
            plugin="FFMPEG",
            codec="libx264rgb",
            pixelformat="rgb24",
            output_params=["-crf", "0", "-preset", "fast"],
        )
