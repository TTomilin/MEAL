"""
Rendering backend selector.

Change RENDERER_VERSION to switch between:

  "v1"  –  Original spritesheet-based renderer (StateVisualizer).
            Agent colours are limited to the 5 pre-drawn hat variants:
            red, green, orange, blue, purple.

  "v2"  –  File-per-sprite renderer with dynamic agent colourisation
            (StateVisualizerV2).  Any number of agents get unique colours
            generated automatically; no extra sprite assets required.
            Sprites live in meal/visualization/data/sprites/.
"""

RENDERER_VERSION: str = "v2"
