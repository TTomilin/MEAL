from __future__ import annotations

from typing import Sequence, Optional, Dict

import numpy as np

from meal.visualization.visualizer import OvercookedVisualizer


def _alpha_over(dst_rgb: np.ndarray, src_rgba: np.ndarray) -> np.ndarray:
    """
    Standard 'source-over' alpha composite.
    dst_rgb: (H,W,3) uint8
    src_rgba: (H,W,4) uint8
    returns: (H,W,3) uint8
    """
    if src_rgba is None:
        return dst_rgb
    a = (src_rgba[..., 3:4].astype(np.float32)) / 255.0
    out = dst_rgb.astype(np.float32) * (1.0 - a) + src_rgba[..., :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


class OvercookedVisualizerPO(OvercookedVisualizer):
    """
    Same renderer as OvercookedVisualizer, plus optional overlay that highlights
    each agent's partially observable window using env.get_agent_view_masks(state).

    Usage:
        viz = OvercookedVisualizerPO(num_agents=env.num_agents, tile_px=64)
        frame = viz.render(state, env=env, show=True)  # highlights if env is provided
        viz.animate(state_seq, out_path="x.gif", env=env, task_idx=i)
    """

    def __init__(
        self,
        num_agents: int = 2,
        pot_full: int = 20,
        pot_empty: int = 23,
        tile_px: int = 64,
        renderer_version=None,
    ):
        super().__init__(
            num_agents=num_agents,
            pot_full=pot_full,
            pot_empty=pot_empty,
            tile_px=tile_px,
            renderer_version=renderer_version,
        )
        self._init_view_colors()  # semi-transparent

    def _init_view_colors(self, alpha: int = 90):
        """
        Build per-agent RGBA overlay colors using the same player_colors ordering
        as StateVisualizer (chef hat colors).
        """
        # Map the names used by StateVisualizer to nice overlay RGBs
        name_to_rgb = {
            "red": (255, 90, 90),
            "green": (120, 220, 120),
            "orange": (255, 180, 80),
            "blue": (100, 150, 255),
            "purple": (180, 120, 255),
            "yellow": (255, 235, 120),
            "teal": (120, 235, 235),
            "pink": (255, 165, 210),
        }

        # Pull the configured order directly from the StateVisualizer.
        # With the v2 renderer player_colors defaults to [], so we fall back to
        # auto-generated colours (same golden-ratio hues used for the hats).
        from meal.visualization.rendering.sprite_loader import agent_color as _agent_color
        palette = []
        for name in self.state_visualizer.player_colors:
            rgb = name_to_rgb.get(name, (200, 200, 200))  # sane fallback for unknown names
            palette.append(np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.uint8))

        if not palette:
            # v2 renderer (empty player_colors): generate per-agent colours automatically
            for i in range(self.num_agents):
                rgb = _agent_color(i)
                palette.append(np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.uint8))
        elif len(palette) < self.num_agents:
            # cycle through the supplied palette if there are more agents than colours
            extra = [palette[i % len(palette)] for i in range(self.num_agents)]
            palette = extra

        self.view_colors = palette

    # --- internal: build an RGBA overlay (H*tile_px, W*tile_px, 4) from boolean masks ---
    def _build_view_overlay(self, env, state) -> Optional[np.ndarray]:
        """
        Query env.get_agent_view_masks(state) -> dict[str]:(H,W) bool,
        upsample each mask to pixels (tile_px x tile_px per cell), colorize, alpha-compose layers.
        """
        # get masks (expects OvercookedPO implementation)
        try:
            masks: Dict[str, np.ndarray] = env.get_agent_view_masks(state)  # dict of (H,W) booleans
        except Exception:
            return None
        if not masks:
            return None

        # infer H,W from one mask; compute pixel dims from state renderer tile size
        any_key = next(iter(masks))
        H, W = masks[any_key].shape
        Hp, Wp = H * self.tile_px, W * self.tile_px

        overlay = np.zeros((Hp, Wp, 4), dtype=np.uint8)

        # upsample helper via Kronecker product (fast)
        ones = np.ones((self.tile_px, self.tile_px), dtype=np.uint8)

        for agent_idx in range(self.num_agents):
            key = f"agent_{agent_idx}"
            if key not in masks:
                continue
            mask = np.asarray(masks[key], dtype=np.uint8)  # (H,W) {0,1}
            big = np.kron(mask, ones).astype(bool)  # (Hp, Wp)

            color = self.view_colors[agent_idx % len(self.view_colors)]
            # Direct paint: each visible tile gets this agent's color.
            # Using assignment (not alpha-over stacking) means the alpha stays
            # constant at the configured value regardless of how many agents
            # share the same tile.  Agents with higher indices paint over lower
            # ones in overlapping areas, which is fine for visualization.
            overlay[big] = color

        return overlay

    # --- public API: mirrors base, but adds `env` and overlays when provided ---

    def render(self, env_state, show: bool = False, env=None) -> np.ndarray:
        """
        If `env` is provided and supports get_agent_view_masks(state), highlight PO windows.
        """
        base = super().render(env_state, show=False)  # (Hp,Wp,3)
        overlay = self._build_view_overlay(env, env_state) if env is not None else None
        out = _alpha_over(base, overlay) if overlay is not None else base
        if show:
            self._lazy_window()
            self.window.show_img(out)
        return out

    def animate(
            self,
            state_seq: Sequence[object],
            out_path: str,
            task_idx: int = 0,
            fps: int = 10,
            pad_to_max: bool = False,
            env=None,
    ) -> str:
        """
        Same as base animate, but if `env` is given we overlay PO windows on every frame.
        """
        import os
        import wandb

        frames = []
        for s in state_seq:
            frame = self.render(s, show=False, env=env)
            frames.append(frame)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self._save_frames(np.stack(frames, axis=0), out_path, fps)

        if wandb.run is not None:
            ext = os.path.splitext(out_path)[1].lower()
            fmt = "gif" if ext == ".gif" else "mp4"
            wandb.log({f"task_{task_idx}": wandb.Video(out_path, format=fmt)})

        return out_path
