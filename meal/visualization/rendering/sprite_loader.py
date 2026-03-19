"""
Sprite loader for the v2 renderer.

Instead of slicing frames from large sprite-sheet PNGs, the v2 renderer reads
each sprite from its own individual PNG file (produced by
scripts/extract_sprites.py).  This module provides:

  DirectorySpriteLoader  – drop-in replacement for MultiFramePygameImage
  colorize_surface       – tint a surface to any RGB colour (used for hats)
  agent_color            – generate a unique colour for any agent index
"""

import colorsys
import os
from typing import Tuple

import numpy as np
import pygame


# ---------------------------------------------------------------------------
# Per-file sprite loader
# ---------------------------------------------------------------------------

class DirectorySpriteLoader:
    """
    Load sprites from individual PNG files in a flat directory.

    Interface is compatible with MultiFramePygameImage.blit_on_surface() so
    the two can be swapped transparently.

    Images are loaded lazily and cached in memory.
    """

    def __init__(self, directory: str):
        self.directory = directory
        self._cache: dict = {}

    def _load(self, frame_name: str) -> pygame.Surface:
        if frame_name not in self._cache:
            path = os.path.join(self.directory, frame_name + ".png")
            self._cache[frame_name] = pygame.image.load(path)
        return self._cache[frame_name]

    def blit_on_surface(
        self,
        surface: pygame.Surface,
        top_left_pixel_position,
        frame_name: str,
        **kwargs,
    ):
        """Blit the named sprite onto *surface* at *top_left_pixel_position*."""
        sprite = self._load(frame_name)
        surface.blit(sprite, top_left_pixel_position, **kwargs)

    def get_surface(self, frame_name: str) -> pygame.Surface:
        """Return the sprite surface (not a copy – do not mutate)."""
        return self._load(frame_name)


# ---------------------------------------------------------------------------
# Dynamic hat colorisation
# ---------------------------------------------------------------------------

def colorize_surface(
    surface: pygame.Surface,
    target_rgb: Tuple[int, int, int],
) -> pygame.Surface:
    """
    Return a *new* surface tinted to *target_rgb*, preserving luminance.

    The function converts every pixel to a grey-scale luminance value using
    the standard ITU-R BT.601 weights, then multiplies by *target_rgb*.
    This produces natural-looking shading / highlights in any colour.

    target_rgb is normalised so its brightest channel maps to 255, ensuring
    hats are as vivid as the original template regardless of the chosen hue.

    The alpha channel is preserved unchanged.
    """
    result = surface.copy()

    # pixels3d gives a (W, H, 3) view; mutations are applied directly.
    pixels = pygame.surfarray.pixels3d(result)

    # Use the HSV "value" (max channel) as the luminance.
    # This is important because the hat template is a monochromatic red sprite
    # (R≈255, G≈0, B≈0). The ITU-weighted luma formula would assign it only
    # 0.299 brightness, producing dark results. max(R,G,B) correctly treats
    # a saturated primary as fully bright and preserves shading ratios.
    lum = np.max(pixels.astype(np.float32), axis=2) / 255.0  # shape (W, H)

    # Normalise target so max channel = 255, preserving hue but maximising brightness
    tr, tg, tb = target_rgb
    max_comp = max(tr, tg, tb) or 1
    scale = 255.0 / max_comp
    tr = min(255, int(tr * scale))
    tg = min(255, int(tg * scale))
    tb = min(255, int(tb * scale))

    pixels[:, :, 0] = np.clip(lum * tr, 0, 255).astype(np.uint8)
    pixels[:, :, 1] = np.clip(lum * tg, 0, 255).astype(np.uint8)
    pixels[:, :, 2] = np.clip(lum * tb, 0, 255).astype(np.uint8)

    del pixels   # release the surface lock
    return result


# ---------------------------------------------------------------------------
# Automatic colour generation
# ---------------------------------------------------------------------------

# Named colours supported for backward compatibility with v1 player_colors
_NAMED_COLORS: dict[str, Tuple[int, int, int]] = {
    "red":    (220,  20,  20),
    "green":  ( 20, 160,  20),
    "blue":   ( 30,  30, 210),
    "orange": (230, 140,   0),
    "purple": (140,  50, 200),
}


# Ordered palette matching the v1 player_colors list: red, green, orange, blue, purple
_V1_AGENT_COLORS = [
    _NAMED_COLORS["red"],
    _NAMED_COLORS["green"],
    _NAMED_COLORS["orange"],
    _NAMED_COLORS["blue"],
    _NAMED_COLORS["purple"],
]


def agent_color(agent_idx: int) -> Tuple[int, int, int]:
    """
    Return a perceptually distinct RGB colour for *agent_idx*.

    Agents 0-4 use the same colours as the v1 renderer in the same order
    (red, green, orange, blue, purple).  Higher indices use golden-ratio
    hue spacing in HSV space.
    """
    if agent_idx < len(_V1_AGENT_COLORS):
        return _V1_AGENT_COLORS[agent_idx]
    golden_ratio = 0.618033988749895
    hue = (agent_idx * golden_ratio + 0.1) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return (int(r * 255), int(g * 255), int(b * 255))


def resolve_color(color_spec, agent_idx: int) -> Tuple[int, int, int]:
    """
    Convert a colour specification to an (R, G, B) tuple.

    Accepts:
      - (R, G, B) tuple  → returned as-is
      - str name         → looked up in _NAMED_COLORS
      - None             → auto-generated from agent_idx
    """
    if color_spec is None:
        return agent_color(agent_idx)
    if isinstance(color_spec, tuple):
        return color_spec
    if isinstance(color_spec, str):
        return _NAMED_COLORS.get(color_spec, agent_color(agent_idx))
    return agent_color(agent_idx)
