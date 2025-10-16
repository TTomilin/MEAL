from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np

from meal.visualization.cache import TileCache, TileKey
from meal.visualization.types import DrawableState, Tile, Obj, Dir

RGB = np.ndarray


# ---------- tiny drawing primitives ----------
def rect(img: RGB, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
    img[y0:y1, x0:x1] = color


def circle(img: RGB, cx: int, cy: int, r: int, color: Tuple[int, int, int]) -> None:
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    img[mask] = color


def triangle(img: RGB, pts: List[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
    # simple barycentric raster (good enough at tile scale)
    (x0, y0), (x1, y1), (x2, y2) = pts
    xmin, xmax = max(min(x0, x1, x2), 0), min(max(x0, x1, x2), img.shape[1] - 1)
    ymin, ymax = max(min(y0, y1, y2), 0), min(max(y0, y1, y2), img.shape[0] - 1)
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if denom == 0:
        return
    X, Y = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    w1 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / denom
    w2 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / denom
    w3 = 1 - w1 - w2
    mask = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
    sub = img[ymin:ymax + 1, xmin:xmax + 1]
    sub[mask] = color


# ---------- palette ----------
COLORS = {
    "floor": (210, 180, 140),
    "counter": (100, 100, 100),
    "serve": (0, 120, 0),
    "pot_body": (30, 30, 30),
    "pot_liquid": (240, 140, 0),
    "onion": (240, 225, 140),
    "plate": (240, 240, 240),
    "chef": [
        (220, 40, 40),
        (120, 60, 200),
        (40, 180, 80),
        (240, 140, 0),
        (60, 120, 220),
    ],
    "progress": (40, 180, 60),
    "border": (60, 60, 60),
}


def _chef_color(pid: int) -> Tuple[int, int, int]:
    return COLORS["chef"][pid % len(COLORS["chef"])]


# ---------- per-tile painters (cached) ----------
def _tile_base(tile_px: int) -> RGB:
    return np.zeros((tile_px, tile_px, 3), dtype=np.uint8)


def _paint_floor(tile_px: int) -> RGB:
    img = _tile_base(tile_px)
    rect(img, 0, 0, tile_px, tile_px, COLORS["floor"])
    return img


def _paint_counter(tile_px: int) -> RGB:
    img = _tile_base(tile_px)
    rect(img, 0, 0, tile_px, tile_px, COLORS["counter"])
    return img


def _paint_serve(tile_px: int) -> RGB:
    img = _tile_base(tile_px)
    rect(img, 0, 0, tile_px, tile_px, COLORS["floor"])
    overlay = img.copy()
    rect(overlay, 0, 0, tile_px, tile_px, COLORS["serve"])
    # alpha blend 50%
    img = ((img.astype(np.int16) + overlay.astype(np.int16)) // 2).astype(np.uint8)
    rect(img, 0, 0, tile_px, tile_px, img[0, 0])  # already blended area
    return img


def _paint_dispenser(tile_px: int) -> RGB:
    # shared look for onion/plate dispensers (distinct item painted later if needed)
    return _paint_counter(tile_px)


def _paint_pot(tile_px: int, bucket: int) -> RGB:
    img = _tile_base(tile_px)
    rect(img, 0, 0, tile_px, tile_px, COLORS["floor"])
    # pot body
    pad = tile_px // 10
    rect(img, pad, pad * 3, tile_px - pad, tile_px - pad, COLORS["pot_body"])
    # progress bar at top
    bar_h = max(2, tile_px // 12)
    if bucket > 0:
        filled = int((tile_px - 2 * pad) * (1 - bucket / 20.0))
        rect(img, pad, pad, pad + filled, pad + bar_h, COLORS["progress"])
    return img


def _paint_agent(tile_px: int, dir_idx: int, pid: int, held: Optional[str]) -> RGB:
    img = _tile_base(tile_px)
    rect(img, 0, 0, tile_px, tile_px, COLORS["floor"])
    c = _chef_color(pid)
    # triangle arrow
    m = tile_px // 6
    if dir_idx == Dir.N.value:
        pts = [(m, tile_px - m), (tile_px - m, tile_px - m), (tile_px // 2, m)]
    elif dir_idx == Dir.S.value:
        pts = [(m, m), (tile_px - m, m), (tile_px // 2, tile_px - m)]
    elif dir_idx == Dir.E.value:
        pts = [(m, m), (m, tile_px - m), (tile_px - m, tile_px // 2)]
    else:
        pts = [(tile_px - m, m), (tile_px - m, tile_px - m), (m, tile_px // 2)]
    triangle(img, pts, c)
    # held item dot
    if held:
        cx, cy = tile_px - m * 2, tile_px - m * 2
        if held == Obj.ONION.value:
            circle(img, cx, cy, m // 2, COLORS["onion"])
        elif held in (Obj.PLATE.value, Obj.DISH.value, Obj.SOUP.value):
            circle(img, cx, cy, m // 2, COLORS["plate"])
    return img


# ---------- public API ----------
def render_frame(
        state: DrawableState,
        *,
        tile_px: int = 48,
        cache: Optional[TileCache] = None,
        draw_borders: bool = True,
) -> RGB:
    H, W = state.H, state.W
    frame = np.zeros((H * tile_px, W * tile_px, 3), dtype=np.uint8)

    def blit(tile_img: RGB, gx: int, gy: int) -> None:
        y0, x0 = gy * tile_px, gx * tile_px
        frame[y0:y0 + tile_px, x0:x0 + tile_px] = tile_img

    cache = cache or TileCache(max_items=2048)

    # 1) draw base tiles
    for y in range(H):
        for x in range(W):
            t = state.grid[y][x]
            if t is Tile.EMPTY:
                key = TileKey("FLOOR", None, None, None, None, tile_px)
                tile = cache.get_or_render(key, lambda: _paint_floor(tile_px))
            elif t is Tile.COUNTER:
                key = TileKey("COUNTER", None, None, None, None, tile_px)
                tile = cache.get_or_render(key, lambda: _paint_counter(tile_px))
            elif t is Tile.SERVE:
                key = TileKey("SERVE", None, None, None, None, tile_px)
                tile = cache.get_or_render(key, lambda: _paint_serve(tile_px))
            elif t in (Tile.ONION_DISPENSER, Tile.DISH_DISPENSER, Tile.TOMATO_DISPENSER):
                key = TileKey("DISP", None, None, None, None, tile_px)
                tile = cache.get_or_render(key, lambda: _paint_dispenser(tile_px))
            elif t is Tile.POT:
                # base pot; progress added by pot overlay later if you prefer. Here we bake progress into tile.
                # find pot status bucket if any pot on this cell
                pots_here = [p for p in state.pots if p.pos == (x, y)]
                bucket = pots_here[0].bucketed_status(20) if pots_here else 20
                key = TileKey("POT", None, None, None, bucket, tile_px)
                tile = cache.get_or_render(key, lambda b=bucket: _paint_pot(tile_px, b))
            else:
                key = TileKey("FALLBACK", None, None, None, None, tile_px)
                tile = cache.get_or_render(key, lambda: _paint_counter(tile_px))
            blit(tile, x, y)

            if draw_borders and t in (Tile.COUNTER, Tile.ONION_DISPENSER, Tile.DISH_DISPENSER, Tile.POT, Tile.SERVE):
                # 1px border
                y0, x0 = y * tile_px, x * tile_px
                frame[y0:y0 + tile_px, x0:x0 + 1] = COLORS["border"]
                frame[y0:y0 + tile_px, x0 + tile_px - 1:x0 + tile_px] = COLORS["border"]
                frame[y0:y0 + 1, x0:x0 + tile_px] = COLORS["border"]
                frame[y0 + tile_px - 1:y0 + tile_px, x0:x0 + tile_px] = COLORS["border"]

    # 2) draw loose items on top
    for kind, (x, y) in state.items:
        cx, cy = x * tile_px + tile_px // 2, y * tile_px + tile_px // 2
        if kind is Obj.ONION:
            circle(frame, cx, cy, max(3, tile_px // 8), COLORS["onion"])
        elif kind in (Obj.PLATE, Obj.DISH, Obj.SOUP):
            circle(frame, cx, cy, max(3, tile_px // 8), COLORS["plate"])

    # 3) draw agents last
    for p in state.players:
        held = p.held.value if p.held is not None else None
        key = TileKey("AGENT", p.dir.value, p.id, held, None, tile_px)
        tile = cache.get_or_render(
            key,
            lambda d=p.dir.value, pid=p.id, h=held: _paint_agent(tile_px, d, pid, h),
        )
        blit(tile, p.pos[0], p.pos[1])

    return frame
