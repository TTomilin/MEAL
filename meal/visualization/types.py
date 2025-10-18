from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Tuple, Sequence, Optional, List, Iterable, Union


class Tile(Enum):
    EMPTY = " "
    COUNTER = "X"  # walls/counters
    ONION_DISPENSER = "O"
    TOMATO_DISPENSER = "T"
    POT = "P"
    DISH_DISPENSER = "D"
    SERVE = "S"  # serving window

    @classmethod
    def from_token(cls, t: str | "Tile") -> "Tile":
        if isinstance(t, Tile):
            return t
        if not (isinstance(t, str) and len(t) == 1):
            raise ValueError(f"Tile token must be 1 char or Tile, got {t!r}")
        # map the single-char token to the enum
        for member in cls:
            if member.value == t:
                return member
        raise ValueError(f"Unknown tile token: {t!r}")

    def token(self) -> str:
        """Single-char token the renderer/adapter expects."""
        return self.value


Coord = Tuple[int, int]
TileLike = Union[Tile, str]


class Dir(Enum):
    N = 0
    S = 1
    E = 2
    W = 3

    @property
    def vec(self) -> Tuple[int, int]:
        return {
            Dir.N: (0, -1),
            Dir.S: (0, 1),
            Dir.E: (1, 0),
            Dir.W: (-1, 0),
        }[self]

    @property
    def name_short(self) -> str:
        return {Dir.N: "N", Dir.S: "S", Dir.E: "E", Dir.W: "W"}[self]

    def rotate_left(self) -> "Dir":
        return {Dir.N: Dir.W, Dir.W: Dir.S, Dir.S: Dir.E, Dir.E: Dir.N}[self]

    def rotate_right(self) -> "Dir":
        return {Dir.N: Dir.E, Dir.E: Dir.S, Dir.S: Dir.W, Dir.W: Dir.N}[self]


class Obj(Enum):
    ONION = "onion";
    PLATE = "plate";
    DISH = "dish";
    SOUP = "soup"


@dataclass(frozen=True, slots=True)
class Player:
    """
    Minimal, render-friendly player model.

    - `held` is a simple object class; `held_ingredients` (optional) lists
      ingredients when `held == Obj.SOUP` or `Obj.DISH`.
    """
    id: int
    pos: Coord
    dir: Dir
    held: Optional[Obj] = None
    held_ingredients: Optional[Sequence[Obj]] = None  # e.g., (Obj.ONION, Obj.ONION, Obj.ONION)

    # ---------- predicates ----------
    @property
    def has_item(self) -> bool:
        return self.held is not None

    @property
    def facing_vec(self) -> Coord:
        return self.dir.vec

    @property
    def forward_pos(self) -> Coord:
        x, y = self.pos
        dx, dy = self.dir.vec
        return (x + dx, y + dy)

    # ---------- builders (immutability-friendly) ----------
    def with_pose(self, pos: Coord | None = None, dir: Dir | None = None) -> "Player":
        return Player(
            id=self.id,
            pos=self.pos if pos is None else pos,
            dir=self.dir if dir is None else dir,
            held=self.held,
            held_ingredients=self.held_ingredients,
        )

    def with_held(
            self,
            held: Optional[Obj],
            ingredients: Optional[Sequence[Obj]] = None,
    ) -> "Player":
        return Player(
            id=self.id,
            pos=self.pos,
            dir=self.dir,
            held=held,
            held_ingredients=tuple(ingredients) if ingredients is not None else None,
        )

    def step_forward(self) -> "Player":
        return self.with_pose(self.forward_pos, self.dir)

    def turn_left(self) -> "Player":
        return self.with_pose(self.pos, self.dir.rotate_left())

    def turn_right(self) -> "Player":
        return self.with_pose(self.pos, self.dir.rotate_right())

    # ---------- export ----------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pos": self.pos,
            "dir": self.dir.name,
            "held": None if self.held is None else self.held.value,
            "held_ingredients": None
            if self.held_ingredients is None
            else [o.value for o in self.held_ingredients],
        }

    # ---------- validation ----------
    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError(f"player id must be >= 0 (got {self.id})")
        x, y = self.pos
        if not (isinstance(x, int) and isinstance(y, int)):
            raise TypeError("pos must be integer grid coordinates")
        if self.held is None and self.held_ingredients is not None:
            raise ValueError("held_ingredients provided but held is None")
        if self.held in (Obj.SOUP, Obj.DISH):
            if not self.held_ingredients:
                raise ValueError(f"{self.held.value} must carry ingredients")
            if len(self.held_ingredients) > 3:
                raise ValueError("max 3 ingredients supported")
        else:
            # Non-soup items should not carry ingredients
            if self.held not in (None,) and self.held_ingredients is not None:
                raise ValueError("ingredients only valid for SOUP/DISH")


@dataclass(frozen=True, slots=True)
class DrawableState:
    """
    Pure, renderer-friendly snapshot:

    grid   : 2D tokens (H rows × W cols) — each token is a Tile or 1-char string.
    players: sequence of Player (immutable).
    pots   : sequence of PotState (immutable).
    items  : list of loose items: (Obj, (x, y)).

    Invariants enforced on construction:
      - Rectangular grid (all rows same width), H>0, W>0.
      - All player/item/pot coordinates are inside the grid bounds.
      - No duplicate player ids.
    """

    grid: List[List[Tile]]
    players: Sequence["Player"]
    pots: Sequence["PotState"]
    items: Sequence[Tuple["Obj", Coord]]

    # ---------- constructors ----------
    @staticmethod
    def _normalize_tile(t: TileLike) -> Tile:
        return Tile.from_token(t)

    @classmethod
    def from_raw(
            cls,
            grid_tokens: Iterable[Iterable[TileLike]],
            players: Sequence["Player"],
            pots: Sequence["PotState"],
            items: Sequence[Tuple["Obj", Coord]],
    ) -> "DrawableState":
        grid_norm: List[List[Tile]] = [
            [cls._normalize_tile(t) for t in row] for row in grid_tokens
        ]
        inst = cls(grid=grid_norm, players=tuple(players), pots=tuple(pots), items=tuple(items))
        inst._validate()
        return inst

    # ---------- basic geometry ----------
    @property
    def H(self) -> int:
        return len(self.grid)

    @property
    def W(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def in_bounds(self, pos: Coord) -> bool:
        x, y = pos
        return 0 <= x < self.W and 0 <= y < self.H

    def tile(self, pos: Coord) -> Tile:
        x, y = pos
        return self.grid[y][x]

    # ---------- queries ----------
    def player_by_id(self, pid: int) -> Optional["Player"]:
        for p in self.players:
            if p.id == pid:
                return p
        return None

    def items_of(self, kind: "Obj") -> List[Tuple["Obj", Coord]]:
        return [it for it in self.items if it[0] == kind]

    # ---------- transforms (return new instances) ----------
    def with_grid(self, grid_tokens: Iterable[Iterable[TileLike]]) -> "DrawableState":
        return type(self).from_raw(grid_tokens, self.players, self.pots, self.items)

    def pad_to(self, target_h: int, target_w: int, fill: TileLike = Tile.EMPTY) -> "DrawableState":
        if target_h < self.H or target_w < self.W:
            raise ValueError(f"pad_to expects target >= current (got {target_h}x{target_w} vs {self.H}x{self.W})")
        fill_norm = self._normalize_tile(fill)

        pad_top = (target_h - self.H) // 2
        pad_left = (target_w - self.W) // 2

        # build padded grid
        new_grid: List[List[Tile]] = []
        # top rows
        for _ in range(pad_top):
            new_grid.append([fill_norm] * target_w)
        # middle rows
        for row in self.grid:
            new_grid.append([fill_norm] * pad_left + list(row) + [fill_norm] * (target_w - pad_left - self.W))
        # bottom rows
        while len(new_grid) < target_h:
            new_grid.append([fill_norm] * target_w)

        # shift coords of entities
        def shift(pos: Coord) -> Coord:
            return (pos[0] + pad_left, pos[1] + pad_top)

        players = tuple(p.with_pose(shift(p.pos), p.dir) for p in self.players)
        pots = tuple(replace(pot, pos=shift(pot.pos)) for pot in self.pots)
        items = tuple((kind, shift(pos)) for kind, pos in self.items)

        out = type(self)(grid=new_grid, players=players, pots=pots, items=items)
        out._validate()
        return out

    # ---------- integrity checks ----------
    def _validate(self) -> None:
        # grid rectangular & non-empty
        if not self.grid or not self.grid[0]:
            raise ValueError("grid must be non-empty")
        w0 = len(self.grid[0])
        for i, row in enumerate(self.grid):
            if len(row) != w0:
                raise ValueError(f"grid must be rectangular; row 0 width={w0}, row {i} width={len(row)}")

        # unique player ids
        ids = [p.id for p in self.players]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate player ids: {ids}")

        # bounds checks
        for p in self.players:
            if not self.in_bounds(p.pos):
                raise ValueError(f"player {p.id} out of bounds at {p.pos}")
        for pot in self.pots:
            if not self.in_bounds(pot.pos):
                raise ValueError(f"pot out of bounds at {pot.pos}")
        for kind, pos in self.items:
            if not self.in_bounds(pos):
                raise ValueError(f"item {kind} out of bounds at {pos}")

    # ---------- export ----------
    def to_dict(self) -> dict:
        return {
            "H": self.H,
            "W": self.W,
            "grid": [[t.value for t in self.grid[y]] for y in range(self.H)],
            "players": [p.to_dict() for p in self.players],
            "pots": [getattr(p, "to_dict", lambda: {
                "pos": p.pos, "status": p.status, "full": p.full, "empty": p.empty
            })() for p in self.pots],
            "items": [(kind.value, pos) for kind, pos in self.items],
        }


@dataclass(frozen=True, slots=True)
class PotState:
    """
    Minimal, portable representation of a cooking pot.

    Assumptions:
      - status ∈ [0, empty]
      - status == empty  → empty pot (0 onions)
      - full > status > 0 → cooking
      - status == 0      → ready (cooked)
      - full ≤ status < empty → filling (has ingredients but not cooking yet)
      - Max soup uses 3 onions.

    Attributes
    ----------
    pos : (x, y) grid coordinate.
    status : int
        Remaining ticks to cooked (0 when ready). Grows toward `empty` as the pot empties.
    full : int
        Threshold at/under which the pot is “full enough to start cooking”.
    empty : int
        Value that represents an empty pot with 0 onions.
    """
    pos: Tuple[int, int]
    status: int
    full: int = 20
    empty: int = 23

    # --------- state predicates ---------
    @property
    def is_ready(self) -> bool:
        """True when soup is fully cooked and waiting to be picked up."""
        return self.status == 0

    @property
    def is_cooking(self) -> bool:
        """True while actively cooking (strictly between 0 and full)."""
        return 0 < self.status < self.full

    @property
    def is_filling(self) -> bool:
        """True when ingredients are in the pot but cooking hasn't started yet."""
        return self.full <= self.status < self.empty

    @property
    def is_empty(self) -> bool:
        """True when the pot has no ingredients."""
        return self.status >= self.empty

    # --------- derived quantities ---------
    @property
    def num_onions(self) -> int:
        """
        How many onions appear in the pot (0..3).
        - While filling: increases as status approaches `full`.
        - While cooking or ready: visually show 3.
        """
        if self.is_empty:
            return 0
        if self.is_cooking or self.is_ready:
            return 3
        # filling region: map [full..empty) -> (3..0], but cap to [0..3]
        return max(0, min(3, self.empty - self.status))

    @property
    def cook_progress(self) -> float:
        """
        Normalized progress in [0, 1]:
          0.0 at start of cooking (status == full),
          1.0 when ready (status == 0).
        Returns 0.0 if not in cooking region.
        """
        if not self.is_cooking and not self.is_ready:
            return 0.0
        # Clamp to [0, full] just in case.
        s = max(0, min(self.status, self.full))
        return 1.0 - (s / float(self.full)) if self.full > 0 else 1.0

    def bucketed_status(self, buckets: int = 20) -> int:
        """
        Quantize status for sprite/caching (e.g., progress bar frames).
        Returns an integer in [0, buckets], where 0 == ready.
        """
        if self.is_ready:
            return 0
        if self.full <= 0:
            return buckets
        # Only quantize cooking range; filling/empty map to max bucket.
        if self.is_cooking:
            frac = self.status / float(self.full)
            return max(1, min(buckets, int(round(frac * buckets))))
        return buckets

    # --------- utils ---------
    def with_status(self, status: int) -> "PotState":
        """Return a copy with a different status (keeps validation semantics)."""
        return PotState(self.pos, status, self.full, self.empty)

    def to_dict(self) -> dict:
        """Handy for HUD/export layers."""
        return {
            "pos": self.pos,
            "status": self.status,
            "full": self.full,
            "empty": self.empty,
            "is_ready": self.is_ready,
            "is_cooking": self.is_cooking,
            "is_filling": self.is_filling,
            "is_empty": self.is_empty,
            "num_onions": self.num_onions,
            "cook_progress": self.cook_progress,
        }

    # --------- validation ---------
    def __post_init__(self) -> None:
        if self.full <= 0:
            raise ValueError(f"`full` must be > 0 (got {self.full})")
        if self.empty <= self.full:
            raise ValueError(f"`empty` must be > `full` (got empty={self.empty}, full={self.full})")
        if self.status < 0:
            raise ValueError(f"`status` must be >= 0 (got {self.status})")


@dataclass(frozen=True, slots=True)
class Ids:
    """Indices used in env.maze_map[..., 0] and inventory channels."""
    unseen: int = 0
    empty: int = 1
    wall: int = 2
    onion: int = 3
    onion_pile: int = 4
    plate: int = 5
    plate_pile: int = 6
    goal: int = 7
    pot: int = 8
    dish: int = 9
    agent: int = 10


@dataclass(frozen=True)
class TileKey:
    kind: str  # e.g., "FLOOR", "COUNTER", "AGENT", "POT"
    dir_idx: Optional[int]  # 0..3 for agents, else None
    agent_id: Optional[int]  # color/hat variant; None for non-agents
    held: Optional[str]  # "onion", "plate", "dish", "soup", or None
    pot_bucket: Optional[int]  # 0..K for progress, or None
    tile_px: int  # tile size in pixels
