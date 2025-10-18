from __future__ import annotations

from collections import namedtuple

from meal.visualization.rendering.actions import Direction
from meal.visualization.rendering.state_visualizer import StateVisualizer, EMPTY, COUNTER, \
    ONION_DISPENSER, TOMATO_DISPENSER, POT, DISH_DISPENSER, SERVING_LOC
from meal.visualization.types import (
    DrawableState, Tile, Obj, Player as DPlayer, PotState, Dir
)

# --- helpers ---------------------------------------------------------------

_TILE_TO_CHAR = {
    Tile.EMPTY: EMPTY,  # " "
    Tile.COUNTER: COUNTER,  # "X"
    Tile.ONION_DISPENSER: ONION_DISPENSER,  # "O"
    Tile.TOMATO_DISPENSER: TOMATO_DISPENSER,  # "T"
    Tile.POT: POT,  # "P"
    Tile.DISH_DISPENSER: DISH_DISPENSER,  # "D"
    Tile.SERVE: SERVING_LOC,  # "S"
}

_DIR_TO_VIZ = {
    Dir.N: Direction.NORTH,
    Dir.S: Direction.SOUTH,
    Dir.E: Direction.EAST,
    Dir.W: Direction.WEST,
}


# --- pot objects for StateVisualizer.objects ------------------------------

def _soup_obj_from_pot(p: PotState):
    """
    Build a soup object with the exact fields the sprites use.
    NOTE: do NOT create soup for empty pots (no frames for 0-ingredient soups).
    """
    # 1) skip empty pots entirely
    if p.is_empty:
        return None

    # 2) pick ingredient count
    ingred = ("onion",) * 3 if p.is_cooking or p.is_ready else tuple("onion" for _ in range(p.num_onions))

    tick = p.status if p.is_cooking else -1  # -1 hides the timer
    cook_time = p.full

    return type(
        "SoupInPot",
        (),
        {
            "name": "soup",
            "position": p.pos,
            "ingredients": ingred,
            "is_ready": p.is_ready,
            "_cooking_tick": tick,
            "cook_time": cook_time,
        },
    )()


def _grid_chars_from_drawable(ds: DrawableState) -> list[list[str]]:
    """Convert DrawableState.grid (Tile enums) -> grid of exact single-char tokens."""
    return [[_TILE_TO_CHAR[t] for t in row] for row in ds.grid]


def _players_from_drawable(ds: DrawableState):
    MockPlayer = namedtuple('MockPlayer', ['position', 'orientation', 'held_object'])

    def held_from_player(p: DPlayer):
        if p.held is None:
            return None
        # StateVisualizer wants obj with fields: name, ingredients (for soups)
        if p.held == Obj.ONION:
            return type('MO', (), {'name': 'onion', 'ingredients': None})()
        if p.held == Obj.PLATE:
            return type('MO', (), {'name': 'dish', 'ingredients': None})()
        if p.held in (Obj.DISH, Obj.SOUP):
            ings = tuple(o.value for o in (p.held_ingredients or ()))
            return type('MO', (), {'name': 'soup', 'ingredients': ings})()
        return None

    players = []
    for p in ds.players:
        players.append(
            MockPlayer(
                position=p.pos,
                orientation=_DIR_TO_VIZ[p.dir],
                held_object=held_from_player(p),
            )
        )
    return players


def _objects_from_drawable(ds: DrawableState):
    objects = {}
    for pot in ds.pots:
        num_onions = pot.num_onions
        if num_onions > 0:
            ings = tuple('onion' for _ in range(num_onions))
            o = type('SoupObj', (), {
                'name': 'soup',
                'position': pot.pos,
                'ingredients': ings,
                'is_ready': pot.is_ready,
                '_cooking_tick': (pot.status if pot.is_cooking else -1),
                'cook_time': pot.full,
            })()
            objects[f"soup_{pot.pos[0]}_{pot.pos[1]}"] = o

    # Loose items
    for kind, pos in ds.items:
        if kind == Obj.ONION:
            o = type('OnionObj', (), {'name': 'onion', 'position': pos, 'ingredients': None})()
            objects[f"onion_{pos[0]}_{pos[1]}"] = o
        elif kind == Obj.PLATE:
            o = type('PlateObj', (), {'name': 'dish', 'position': pos, 'ingredients': None})()
            objects[f"plate_{pos[0]}_{pos[1]}"] = o
        elif kind in (Obj.DISH, Obj.SOUP):
            ings = ('onion', 'onion', 'onion')  # current env only makes onion soup
            o = type('DishObj', (), {
                'name': 'soup', 'position': pos, 'ingredients': ings,
                'is_ready': True, '_cooking_tick': -1, 'cook_time': 20
            })()
            objects[f"soupdish_{pos[0]}_{pos[1]}"] = o

    return objects


def render_drawable_with_stateviz(ds: DrawableState, sv: StateVisualizer, hud_data: dict | None = None,
                                  action_probs=None):
    """Return a pygame.Surface rendered with the new spritesheets."""
    grid_chars = _grid_chars_from_drawable(ds)
    MockState = namedtuple('MockState', ['players', 'objects'])
    mock_state = MockState(players=_players_from_drawable(ds), objects=_objects_from_drawable(ds))
    return sv.render_state(mock_state, grid_chars, hud_data=hud_data, action_probs=action_probs)
