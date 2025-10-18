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

    def __init__(self, num_agents: int = 2, pot_full: int = 20, pot_empty: int = 23, tile_px: int = 64):
        super().__init__(num_agents=num_agents, pot_full=pot_full, pot_empty=pot_empty, tile_px=tile_px)
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

        # Pull the configured order directly from the StateVisualizer
        # (defaults to ["red","green","orange","blue","purple"] unless you changed it)
        palette = []
        for name in self.state_visualizer.player_colors:
            rgb = name_to_rgb.get(name, (200, 200, 200))  # sane fallback for unknown names
            palette.append(np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.uint8))
        # keep at least num_agents colors by cycling
        if len(palette) < self.num_agents:
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
            big = np.kron(mask, ones)  # (Hp, Wp)

            color = self.view_colors[agent_idx % len(self.view_colors)]
            layer = np.zeros_like(overlay)
            # paint only where mask is 1
            layer[..., 0] = big * color[0]
            layer[..., 1] = big * color[1]
            layer[..., 2] = big * color[2]
            layer[..., 3] = big * color[3]

            # composite this agent layer over the running overlay (source-over)
            overlay = self._alpha_over_rgba(overlay, layer)

        return overlay

    @staticmethod
    def _alpha_over_rgba(dst_rgba: np.ndarray, src_rgba: np.ndarray) -> np.ndarray:
        """
        Source-over for two RGBA uint8 images; returns RGBA uint8.
        """
        if src_rgba is None:
            return dst_rgba
        dst = dst_rgba.astype(np.float32) / 255.0
        src = src_rgba.astype(np.float32) / 255.0
        sa = src[..., 3:4]
        da = dst[..., 3:4]

        out_a = sa + da * (1.0 - sa)
        # avoid div-by-zero when out_a == 0
        out_rgb = np.where(
            out_a > 0,
            (src[..., :3] * sa + dst[..., :3] * da * (1.0 - sa)) / out_a,
            0.0,
        )
        out = np.concatenate([out_rgb, out_a], axis=-1)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

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
        # 1) drawable conversion + (optional) pad — reuse parent logic
        #    but we’ll just render frames directly since parent already batches conversion internally.
        frames = []
        for s in state_seq:
            frame = self.render(s, show=False, env=env)
            frames.append(frame)

        # 2) write GIF (reuse parent’s writer for consistency)
        from PIL import Image
        import os, wandb

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

        if wandb.run is not None:
            wandb.log({f"task_{task_idx}": wandb.Video(out_path, format="gif")})

        return out_path
