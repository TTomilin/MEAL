from typing import Tuple, Optional, List, Dict

import numpy as np

from meal.visualization.types import Ids, Tile, Obj, Player, Dir, PotState, DrawableState


def _np(a) -> np.ndarray:
    """JAX/torch/tensor → numpy."""
    return np.asarray(a)


def _colour_to_agent_index(color_idx: int) -> int:
    """Overcooked palette: even color indices map to consecutive agent IDs: 0→0, 2→1, 4→2, ..."""
    return color_idx // 2


def _grid_tokens(maze_crop: np.ndarray, ids: Ids) -> List[List["Tile"]]:
    """Channel-0 object ids → Tile tokens."""
    H, W = maze_crop.shape[:2]
    out: List[List["Tile"]] = []
    for y in range(H):
        row: List["Tile"] = []
        for x in range(W):
            t = int(maze_crop[y, x, 0])
            if t == ids.empty:
                row.append(Tile.EMPTY)
            elif t == ids.wall:
                row.append(Tile.COUNTER)
            elif t == ids.onion_pile:
                row.append(Tile.ONION_DISPENSER)
            elif t == ids.plate_pile:
                row.append(Tile.DISH_DISPENSER)
            elif t == ids.pot:
                row.append(Tile.POT)
            elif t == ids.goal:
                row.append(Tile.SERVE)
            elif t == ids.agent:
                # Under an agent there is floor visually
                row.append(Tile.EMPTY)
            else:
                # default to counter for unknown cells to avoid transparent holes
                row.append(Tile.COUNTER)
        out.append(row)
    return out


def _held_from_inventory(inv_val: int, ids: Ids) -> Tuple[Optional["Obj"], Optional[Tuple["Obj", ...]]]:
    """Map env inventory integer → (held obj, ingredients)."""
    if inv_val == ids.empty:
        return None, None
    if inv_val == ids.onion:
        return Obj.ONION, None
    if inv_val == ids.plate:
        return Obj.PLATE, None
    if inv_val == ids.dish:
        # Plate with onion soup
        return Obj.DISH, (Obj.ONION, Obj.ONION, Obj.ONION)
    # Fallback: unknown items treated as none
    return None, None


def _players_from_state(env_state, num_agents: int, ids: Ids) -> List["Player"]:
    """
    Build Player list from env_state.{agent_pos, agent_dir_idx, agent_inv}.
    Falls back to scanning the grid if positions are missing.
    """
    players: List[Player] = []

    # Source A: explicit arrays
    has_pos = hasattr(env_state, "agent_pos")
    has_dir = hasattr(env_state, "agent_dir_idx")
    has_inv = hasattr(env_state, "agent_inv")

    if has_pos and has_dir:
        pos_arr = _np(env_state.agent_pos)  # shape (N,2), x,y
        dir_arr = _np(env_state.agent_dir_idx)  # shape (N,)
        inv_arr = _np(env_state.agent_inv) if has_inv else None
        N = min(num_agents, len(pos_arr))
        for pid in range(N):
            x, y = int(pos_arr[pid, 0]), int(pos_arr[pid, 1])
            d = int(dir_arr[pid])
            dir_enum = {0: Dir.N, 1: Dir.S, 2: Dir.E, 3: Dir.W}.get(d, Dir.S)
            if inv_arr is not None and pid < len(inv_arr):
                held, ingred = _held_from_inventory(int(inv_arr[pid]), ids)
            else:
                held, ingred = (None, None)
            players.append(Player(id=pid, pos=(x, y), dir=dir_enum, held=held, held_ingredients=ingred))
        return players

    # Source B: scan the grid for agents; derive IDs from color channel
    grid = _np(env_state.maze_map)
    H, W = grid.shape[:2]
    seen: Dict[int, Tuple[int, int]] = {}
    for y in range(H):
        for x in range(W):
            if int(grid[y, x, 0]) == ids.agent:
                color_idx = int(grid[y, x, 1])
                pid = _colour_to_agent_index(color_idx)
                # assume default facing south if dir array not available
                seen[pid] = (x, y)

    max_pid = max(seen.keys(), default=-1)
    N = max(num_agents, max_pid + 1)
    for pid in range(N):
        pos = seen.get(pid, (0, 0))
        dir_enum = Dir.S
        held, ingred = (None, None)
        players.append(Player(id=pid, pos=pos, dir=dir_enum, held=held, held_ingredients=ingred))
    return players


def _pots_from_state(maze_crop: np.ndarray, env_state, pot_full: int, pot_empty: int) -> List["PotState"]:
    """
    Use env_state.pot_pos (if present) + channel-2 status, else infer by scanning for POT tiles.
    Note: maze_crop is already cropped to visible region; adjust coordinates accordingly if needed.
    """
    pots: List[PotState] = []
    H, W = maze_crop.shape[:2]

    # Prefer explicit positions
    if hasattr(env_state, "pot_pos"):
        for p in _np(env_state.pot_pos):
            x, y = int(p[0]), int(p[1])
            if 0 <= y < H and 0 <= x < W:
                status = int(maze_crop[y, x, 2])
                pots.append(PotState(pos=(x, y), status=status, full=pot_full, empty=pot_empty))
        return pots

    # Fallback: scan for POT by channel 0; status from channel 2
    ids_guess = None  # we can’t access Ids here; caller controls cropping so just read channel 0 == pot
    for y in range(H):
        for x in range(W):
            # channel 0 pot assumed to be == Ids.pot, but if caller cropped + provided only POT cells, we skip that check.
            status = int(maze_crop[y, x, 2])
            # Heuristic: consider it a pot cell if status < pot_empty
            if 0 <= status < pot_empty:
                pots.append(PotState(pos=(x, y), status=status, full=pot_full, empty=pot_empty))
    return pots


def _loose_items_from_grid(maze_crop: np.ndarray, ids: Ids) -> List[Tuple["Obj", Tuple[int, int]]]:
    """Find standalone onions/plates/dishes on counters."""
    out: List[Tuple["Obj", Tuple[int, int]]] = []
    H, W = maze_crop.shape[:2]
    for y in range(H):
        for x in range(W):
            t = int(maze_crop[y, x, 0])
            if t == ids.onion:
                out.append((Obj.ONION, (x, y)))
            elif t == ids.plate:
                out.append((Obj.PLATE, (x, y)))
            elif t == ids.dish:
                out.append((Obj.DISH, (x, y)))
    return out


def char_grid_to_drawable_state(char_grid: List[List[str]]) -> "DrawableState":
    """Convert character grid to DrawableState for basic grid rendering."""
    from meal.env.common import FLOOR, WALL, GOAL, ONION_PILE, PLATE_PILE, POT, AGENT

    # Character to Tile mapping for direct grid rendering
    char_to_tile = {
        FLOOR: Tile.EMPTY,
        WALL: Tile.COUNTER,  # walls are rendered as counters
        GOAL: Tile.SERVE,    # goals are serving locations
        ONION_PILE: Tile.ONION_DISPENSER,
        PLATE_PILE: Tile.DISH_DISPENSER,
        POT: Tile.POT,
        # AGENT is handled separately as players
    }

    # Convert characters to tiles, handling agents separately
    grid_tiles = []
    players = []
    agent_id = 0

    for y, row in enumerate(char_grid):
        tile_row = []
        for x, ch in enumerate(row):
            if ch == AGENT:
                # Add agent as player, use floor tile for grid
                tile_row.append(Tile.EMPTY)
                players.append(Player(
                    id=agent_id,
                    pos=(x, y),
                    dir=Dir.N,  # default direction
                    held=None,
                    held_ingredients=None
                ))
                agent_id += 1
            else:
                # Convert character to tile
                tile = char_to_tile.get(ch, Tile.EMPTY)
                tile_row.append(tile)
        grid_tiles.append(tile_row)

    # Create DrawableState with empty pots and items for basic rendering
    return DrawableState.from_raw(
        grid_tokens=grid_tiles,
        players=players,
        pots=[],  # no pot states for basic rendering
        items=[]  # no loose items for basic rendering
    )


def to_drawable_state(env_state_raw, pot_full: int = 20, pot_empty: int = 23, num_agents: int = 2,
                      ids: Ids = Ids()) -> "DrawableState":
    """
    Convert an Overcooked env_state (or Log wrapper with .env_state) into a DrawableState.

    Parameters
    ----------
    env_state_raw : object
        Must expose at least .maze_map (H,W,C). Optional: .env_state, .agent_pos, .agent_dir_idx, .agent_inv, .pot_pos.
    pot_full / pot_empty : int
        Thresholds for pot semantics (must match environment).
    num_agents : int
        Expected number of agents (used when scanning).
    ids : Ids
        Object ID mapping for channel-0 and inventory values.

    Returns
    -------
    DrawableState
    """
    # unwrap LogEnvState if present
    env_state = getattr(env_state_raw, "env_state", env_state_raw)
    maze = _np(env_state.maze_map)

    # grid tiles
    grid_tokens = _grid_tokens(maze, ids)  # List[List[Tile]]

    # players
    players = _players_from_state(env_state, num_agents=num_agents, ids=ids)

    # pots
    pots = _pots_from_state(maze, env_state, pot_full=pot_full, pot_empty=pot_empty)

    # loose items
    items = _loose_items_from_grid(maze, ids)

    # done
    drawable = DrawableState.from_raw(
        grid_tokens=grid_tokens,
        players=players,
        pots=pots,
        items=items,
    )
    return drawable
