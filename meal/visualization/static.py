import os
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
DATA_DIR = os.path.join(_current_dir, "data")
GRAPHICS_DIR = os.path.join(DATA_DIR, "graphics")
SPRITES_DIR = os.path.join(DATA_DIR, "sprites")   # individual-file sprites (v2)
FONTS_DIR = os.path.join(DATA_DIR, "fonts")
