import os
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
DATA_DIR = os.path.join(_current_dir, "data")
GRAPHICS_DIR = os.path.join(DATA_DIR, "graphics")
FONTS_DIR = os.path.join(DATA_DIR, "fonts")
