"""
StateVisualizerV2 – drop-in replacement for StateVisualizer.

Differences from v1:
  • Sprites are loaded from individual PNG files in data/sprites/ instead of
    sliced from large sprite-sheet PNGs.
  • Agent hats are colourised at runtime via pixel arithmetic, so any number
    of agents can have unique, distinct colours without extra sprite assets.
  • player_colors accepts RGB tuples, colour-name strings, or can be left
    empty to trigger automatic golden-ratio colour generation per agent.

Everything else (rendering logic, HUD, cooking timers, action arrows, …) is
identical to StateVisualizer so the two can be swapped at a single call site.
"""

import copy
import math
import os

import pygame

from meal.visualization.rendering.actions import Direction, Action
from meal.visualization.rendering.sprite_loader import (
    DirectorySpriteLoader,
    colorize_surface,
    resolve_color,
)
from meal.visualization.rendering.spritesheet import (
    run_static_resizeable_window,
    scale_surface_by_factor,
    blit_on_new_surface_of_size,
    vstack_surfaces,
)
from meal.visualization.static import SPRITES_DIR, FONTS_DIR
from meal.visualization.utils.io import generate_temporary_file_path

# IPython helpers are only needed for notebook display; import lazily so the
# renderer works in non-Jupyter environments even if IPython is not installed.
def _ipython_show_slider(img_paths, label):
    from meal.visualization.utils.ipy_image_widgets import show_ipython_images_slider
    return show_ipython_images_slider(img_paths, label)

def _ipython_show_image(img_path):
    from meal.visualization.utils.ipy_image_widgets import show_image_in_ipython
    return show_image_in_ipython(img_path)

roboto_path = os.path.join(FONTS_DIR, "Roboto-Regular.ttf")

EMPTY = " "
COUNTER = "X"
ONION_DISPENSER = "O"
TOMATO_DISPENSER = "T"
POT = "P"
DISH_DISPENSER = "D"
SERVING_LOC = "S"


class StateVisualizerV2:
    """
    Sprite-per-file renderer with unlimited dynamic agent colours.

    Compatible API with StateVisualizer; swap by changing RENDERER_VERSION
    in meal/visualization/renderer_config.py.
    """

    UNSCALED_TILE_SIZE = 240

    DEFAULT_VALUES = {
        "height": None,
        "width": None,
        "tile_size": 150,
        "window_fps": 30,
        # Empty list → colours are auto-generated per agent (any number supported).
        # Accepts: [] | [(R,G,B), …] | ["red", "blue", …] | mix of both.
        "player_colors": [],
        "is_rendering_hud": True,
        "hud_font_size": 20,
        "hud_font_path": roboto_path,
        "hud_system_font_name": None,
        "hud_font_color": (255, 255, 255),
        "hud_data_default_key_order": [
            "all_orders",
            "bonus_orders",
            "time_left",
            "score",
            "potential",
        ],
        "hud_interline_size": 20,
        "hud_margin_bottom": 20,
        "hud_margin_top": 20,
        "hud_margin_left": 20,
        "hud_distance_between_orders": 10,
        "hud_order_size": 30,
        "is_rendering_cooking_timer": True,
        "show_timer_when_cooked": True,
        "cooking_timer_font_size": 40,
        "cooking_timer_font_path": roboto_path,
        "cooking_timer_system_font_name": None,
        "cooking_timer_font_color": (255, 0, 0),
        "grid": None,
        "background_color": (155, 101, 0),
        "is_rendering_action_probs": True,
        "is_rendering_borders": True,
    }

    TILE_TO_FRAME_NAME = {
        EMPTY: "floor",
        COUNTER: "counter",
        ONION_DISPENSER: "onions",
        TOMATO_DISPENSER: "tomatoes",
        POT: "pot",
        DISH_DISPENSER: "dishes",
        SERVING_LOC: "serve",
    }

    def __init__(self, **kwargs):
        params = copy.deepcopy(self.DEFAULT_VALUES)
        params.update(kwargs)
        self.configure(**params)
        self.reload_fonts()

        # Sprite loaders (one per sub-directory)
        self._terrain = DirectorySpriteLoader(os.path.join(SPRITES_DIR, "terrain"))
        self._chef_body = DirectorySpriteLoader(os.path.join(SPRITES_DIR, "chef", "body"))
        self._chef_hat = DirectorySpriteLoader(os.path.join(SPRITES_DIR, "chef", "hat"))
        self._objects = DirectorySpriteLoader(os.path.join(SPRITES_DIR, "objects"))
        self._soups = DirectorySpriteLoader(os.path.join(SPRITES_DIR, "soups"))

        # Cache for colourised hat surfaces  {(direction, agent_idx): Surface}
        self._hat_cache: dict = {}

        # Arrow / interact / stay overlays (same files as v1)
        from meal.visualization.static import GRAPHICS_DIR
        self._arrow_img = pygame.image.load(os.path.join(GRAPHICS_DIR, "arrow.png"))
        self._interact_img = pygame.image.load(os.path.join(GRAPHICS_DIR, "interact.png"))
        self._stay_img = pygame.image.load(os.path.join(GRAPHICS_DIR, "stay.png"))

    # ------------------------------------------------------------------
    # Configuration helpers  (identical to v1)
    # ------------------------------------------------------------------

    def reload_fonts(self):
        pygame.font.init()
        if not hasattr(self, "_fonts"):
            self._fonts = {}
        if self.is_rendering_hud:
            self.hud_font = self._init_font(
                self.hud_font_size, self.hud_font_path, self.hud_system_font_name
            )
        else:
            self.hud_font = None
        if self.is_rendering_cooking_timer:
            self.cooking_timer_font = self._init_font(
                self.cooking_timer_font_size,
                self.cooking_timer_font_path,
                self.cooking_timer_system_font_name,
            )
        else:
            self.cooking_timer_font = None

    @classmethod
    def configure_defaults(cls, **kwargs):
        cls._check_config_validity(kwargs)
        cls.DEFAULT_VALUES.update(copy.deepcopy(kwargs))

    def configure(self, **kwargs):
        StateVisualizerV2._check_config_validity(kwargs)
        for name, value in copy.deepcopy(kwargs).items():
            setattr(self, name, value)

    @staticmethod
    def default_hud_data(state, **kwargs):
        result = {
            "timestep": state.timestep,
            "all_orders": [r.to_dict() for r in state.all_orders],
            "bonus_orders": [r.to_dict() for r in state.bonus_orders],
        }
        result.update(copy.deepcopy(kwargs))
        return result

    @staticmethod
    def default_hud_data_from_trajectories(trajectories, trajectory_idx=0):
        rews = trajectories["ep_rewards"][trajectory_idx]
        scores = [sum(rews[:t]) for t in range(len(rews))]
        return [
            StateVisualizerV2.default_hud_data(state, score=scores[i])
            for i, state in enumerate(trajectories["ep_states"][trajectory_idx])
        ]

    # ------------------------------------------------------------------
    # Public rendering entry points (identical signatures to v1)
    # ------------------------------------------------------------------

    def display_rendered_trajectory(
        self,
        trajectories,
        trajectory_idx=0,
        hud_data=None,
        action_probs=None,
        img_directory_path=None,
        img_extension=".png",
        img_prefix="",
        ipython_display=True,
    ):
        states = trajectories["ep_states"][trajectory_idx]
        grid = trajectories["mdp_params"][trajectory_idx]["terrain"]
        if hud_data is None:
            if self.is_rendering_hud:
                hud_data = StateVisualizerV2.default_hud_data_from_trajectories(
                    trajectories, trajectory_idx
                )
            else:
                hud_data = [None] * len(states)
        if action_probs is None:
            action_probs = [None] * len(states)
        if not img_directory_path:
            img_directory_path = generate_temporary_file_path(
                prefix="overcooked_visualized_trajectory", extension=""
            )
        os.makedirs(img_directory_path, exist_ok=True)
        img_paths = []
        for i, state in enumerate(states):
            img_name = img_prefix + str(i) + img_extension
            img_path = os.path.join(img_directory_path, img_name)
            img_paths.append(
                self.display_rendered_state(
                    state=state,
                    hud_data=hud_data[i],
                    action_probs=action_probs[i],
                    grid=grid,
                    img_path=img_path,
                    ipython_display=False,
                    window_display=False,
                )
            )
        if ipython_display:
            return _ipython_show_slider(img_paths, "timestep")
        return img_directory_path

    def display_rendered_state(
        self,
        state,
        hud_data=None,
        action_probs=None,
        grid=None,
        img_path=None,
        ipython_display=False,
        window_display=False,
    ):
        assert window_display or img_path or ipython_display
        surface = self.render_state(state, grid, hud_data, action_probs=action_probs)
        if img_path is None and ipython_display:
            img_path = generate_temporary_file_path(
                prefix="overcooked_visualized_state_", extension=".png"
            )
        if img_path is not None:
            pygame.image.save(surface, img_path)
        if ipython_display:
            _ipython_show_image(img_path)
        if window_display:
            run_static_resizeable_window(surface, self.window_fps)
        return img_path

    def render_state(self, state, grid, hud_data=None, action_probs=None, delivery_positions=()):
        """Return a pygame.Surface for the given state (same API as v1)."""
        pygame.init()
        grid = grid or self.grid
        assert grid

        grid_surface = pygame.surface.Surface(self._unscaled_grid_pixel_size(grid))
        self._render_grid(grid_surface, grid, delivery_positions=delivery_positions)
        self._render_players(grid_surface, state.players)
        self._render_objects(grid_surface, state.objects, grid)

        if self.scale_by_factor != 1:
            grid_surface = scale_surface_by_factor(grid_surface, self.scale_by_factor)

        if self.is_rendering_cooking_timer:
            self._render_cooking_timers(grid_surface, state.objects, grid)

        if delivery_positions:
            self._render_delivery_flashes(grid_surface, delivery_positions)

        if self.is_rendering_action_probs and action_probs is not None:
            self._render_actions_probs(grid_surface, state.players, action_probs)

        if self.is_rendering_hud and hud_data:
            hud_width = self.width or grid_surface.get_width()
            hud_surface = pygame.surface.Surface(
                (hud_width, self._calculate_hud_height(hud_data))
            )
            hud_surface.fill(self.background_color)
            self._render_hud_data(hud_surface, hud_data)
            rendered_surface = vstack_surfaces([hud_surface, grid_surface], self.background_color)
        else:
            rendered_surface = grid_surface

        result_size = (
            self.width or rendered_surface.get_width(),
            self.height or rendered_surface.get_height(),
        )
        if result_size != rendered_surface.get_size():
            result_surface = blit_on_new_surface_of_size(
                rendered_surface, result_size, background_color=self.background_color
            )
        else:
            result_surface = rendered_surface

        return result_surface

    # ------------------------------------------------------------------
    # Internal properties / helpers
    # ------------------------------------------------------------------

    @property
    def scale_by_factor(self):
        return self.tile_size / self.UNSCALED_TILE_SIZE

    @property
    def hud_line_height(self):
        return self.hud_interline_size + self.hud_font_size

    @staticmethod
    def _check_config_validity(config):
        assert set(config.keys()).issubset(set(StateVisualizerV2.DEFAULT_VALUES.keys()))

    def _init_font(self, font_size, font_path=None, system_font_name=None):
        if system_font_name:
            key = "%i-sys:%s" % (font_size, system_font_name)
            font = self._fonts.get(key) or pygame.font.SysFont(system_font_name, font_size)
        else:
            key = "%i-path:%s" % (font_size, font_path)
            font = self._fonts.get(key) or pygame.font.Font(font_path, font_size)
        self._fonts[key] = font
        return font

    def _unscaled_grid_pixel_size(self, grid):
        return (
            len(grid[0]) * self.UNSCALED_TILE_SIZE,
            len(grid) * self.UNSCALED_TILE_SIZE,
        )

    def _position_in_unscaled_pixels(self, position):
        x, y = position
        return (self.UNSCALED_TILE_SIZE * x, self.UNSCALED_TILE_SIZE * y)

    def _position_in_scaled_pixels(self, position):
        x, y = position
        return (self.tile_size * x, self.tile_size * y)

    # ------------------------------------------------------------------
    # Colour resolution for agents
    # ------------------------------------------------------------------

    def _player_color_rgb(self, player_num: int):
        """Return the (R, G, B) colour for *player_num*."""
        if self.player_colors and player_num < len(self.player_colors):
            spec = self.player_colors[player_num]
        else:
            spec = None   # triggers auto-generation
        return resolve_color(spec, player_num)

    def _get_colorized_hat(
        self, direction_name: str, player_num: int, rgb
    ) -> pygame.Surface:
        """Return a colourised hat surface, using a per-direction cache."""
        key = (direction_name, player_num)
        if key not in self._hat_cache:
            template = self._chef_hat.get_surface(direction_name)
            self._hat_cache[key] = colorize_surface(template, rgb)
        return self._hat_cache[key]

    # ------------------------------------------------------------------
    # Render grid
    # ------------------------------------------------------------------

    def _render_grid(self, surface, grid, delivery_positions=()):
        delivery_set = set(delivery_positions)
        for y_tile, row in enumerate(grid):
            for x_tile, tile in enumerate(row):
                self._terrain.blit_on_surface(
                    surface,
                    self._position_in_unscaled_pixels((x_tile, y_tile)),
                    self.TILE_TO_FRAME_NAME[tile],
                )

                if tile in (COUNTER, ONION_DISPENSER, DISH_DISPENSER, POT, SERVING_LOC) \
                        and self.is_rendering_borders:
                    pos = self._position_in_unscaled_pixels((x_tile, y_tile))
                    pygame.draw.rect(
                        surface,
                        (60, 60, 60),
                        (pos[0], pos[1], self.UNSCALED_TILE_SIZE, self.UNSCALED_TILE_SIZE),
                        1,
                    )

                if tile == SERVING_LOC:
                    pos = self._position_in_unscaled_pixels((x_tile, y_tile))
                    overlay = pygame.Surface(
                        (self.UNSCALED_TILE_SIZE, self.UNSCALED_TILE_SIZE), pygame.SRCALPHA
                    )
                    if (x_tile, y_tile) in delivery_set:
                        overlay.fill((255, 215, 0, 200))
                    else:
                        overlay.fill((0, 100, 0, 128))
                    surface.blit(overlay, pos)

    # ------------------------------------------------------------------
    # Render players
    # ------------------------------------------------------------------

    def _render_players(self, surface, players):
        for player_num, player in enumerate(players):
            rgb = self._player_color_rgb(player_num)
            direction_name = Direction.DIRECTION_TO_NAME[player.orientation]

            held_obj = player.held_object
            if held_obj is None:
                held_name = ""
            elif held_obj.name == "soup":
                held_name = "soup-onion" if "onion" in held_obj.ingredients else "soup-tomato"
            else:
                held_name = held_obj.name

            body_frame = direction_name if not held_name else f"{direction_name}-{held_name}"
            pos = self._position_in_unscaled_pixels(player.position)

            self._chef_body.blit_on_surface(surface, pos, body_frame)
            hat = self._get_colorized_hat(direction_name, player_num, rgb)
            surface.blit(hat, pos)

    # ------------------------------------------------------------------
    # Render objects / soups
    # ------------------------------------------------------------------

    @staticmethod
    def _soup_frame_name(ingredients_names, status):
        num_onions = ingredients_names.count("onion")
        num_tomatoes = ingredients_names.count("tomato")
        return "soup_%s_tomato_%i_onion_%i" % (status, num_tomatoes, num_onions)

    def _render_objects(self, surface, objects, grid):
        def render_soup(obj):
            x_pos, y_pos = obj.position
            if (
                0 <= y_pos < len(grid)
                and 0 <= x_pos < len(grid[0])
                and grid[y_pos][x_pos] == POT
            ):
                status = "cooked" if obj.is_ready else "idle"
            else:
                status = "done"
            frame = self._soup_frame_name(obj.ingredients, status)
            self._soups.blit_on_surface(
                surface, self._position_in_unscaled_pixels(obj.position), frame
            )

        for obj in objects.values():
            if obj.name == "soup":
                render_soup(obj)
            else:
                self._objects.blit_on_surface(
                    surface,
                    self._position_in_unscaled_pixels(obj.position),
                    obj.name,
                )

    # ------------------------------------------------------------------
    # Cooking timers
    # ------------------------------------------------------------------

    def _render_cooking_timers(self, surface, objects, grid):
        for obj in objects.values():
            x_pos, y_pos = obj.position
            if (
                obj.name == "soup"
                and 0 <= y_pos < len(grid)
                and 0 <= x_pos < len(grid[0])
                and grid[y_pos][x_pos] == POT
            ):
                if obj._cooking_tick != -1 and (
                    obj._cooking_tick <= obj.cook_time or self.show_timer_when_cooked
                ):
                    text_surface = self.cooking_timer_font.render(
                        str(obj._cooking_tick), True, self.cooking_timer_font_color
                    )
                    tile_px_x, tile_px_y = self._position_in_scaled_pixels(obj.position)
                    font_pos = (
                        tile_px_x + int((self.tile_size - text_surface.get_width()) * 0.5),
                        tile_px_y + int((self.tile_size - text_surface.get_height()) * 0.9),
                    )
                    surface.blit(text_surface, font_pos)

    # ------------------------------------------------------------------
    # Delivery flash
    # ------------------------------------------------------------------

    def _render_delivery_flashes(self, surface, delivery_positions):
        for x_tile, y_tile in delivery_positions:
            cx = self.tile_size * x_tile + self.tile_size // 2
            cy = self.tile_size * y_tile + self.tile_size // 2
            outer_r = self.tile_size * 0.30
            inner_r = self.tile_size * 0.12
            pts = self._star_polygon(cx, cy, outer_r, inner_r)
            pygame.draw.polygon(surface, (255, 255, 255), pts)

    @staticmethod
    def _star_polygon(cx, cy, outer_r, inner_r, points=5):
        coords = []
        for i in range(points * 2):
            angle = math.pi / points * i - math.pi / 2
            r = outer_r if i % 2 == 0 else inner_r
            coords.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        return coords

    # ------------------------------------------------------------------
    # HUD  (identical to v1)
    # ------------------------------------------------------------------

    def _sorted_hud_items(self, hud_data):
        def _key(item):
            try:
                i = self.hud_data_default_key_order.index(item[0])
            except ValueError:
                i = 99999
            return (i, item[0])
        return sorted(hud_data.items(), key=_key)

    def _key_to_hud_text(self, key):
        return key.replace("_", " ").title() + ": "

    def _calculate_hud_height(self, hud_data):
        return (
            self.hud_margin_top
            + len(hud_data) * self.hud_line_height
            + self.hud_margin_bottom
        )

    def _render_hud_data(self, surface, hud_data):
        def hud_text_pos(line_num):
            return (
                self.hud_margin_left,
                self.hud_margin_top + self.hud_line_height * line_num,
            )

        def recipes_pos(text_surf, text_pos):
            return (text_pos[0] + text_surf.get_width(), text_pos[1])

        def get_recipes_surface(orders_dicts):
            order_w = order_h = self.hud_order_size
            unscaled = (self.UNSCALED_TILE_SIZE, self.UNSCALED_TILE_SIZE)
            total_w = (
                len(orders_dicts) * order_w
                + (len(orders_dicts) - 1) * self.hud_distance_between_orders
            )
            rec_surf = pygame.surface.Surface((total_w, order_h))
            rec_surf.fill(self.background_color)
            next_x = 0
            for od in orders_dicts:
                frame = self._soup_frame_name(od["ingredients"], "done")
                tile_surf = pygame.surface.Surface(unscaled)
                tile_surf.fill(self.background_color)
                self._soups.blit_on_surface(tile_surf, (0, 0), frame)
                scaled = pygame.transform.scale(tile_surf, (order_w, order_h))
                rec_surf.blit(scaled, (next_x, 0))
                next_x += order_w + self.hud_distance_between_orders
            return rec_surf

        order_keys = {"all_orders", "bonus_orders", "start_all_orders", "start_bonus_orders"}
        for line_num, (key, value) in enumerate(self._sorted_hud_items(hud_data)):
            hud_text = self._key_to_hud_text(key)
            if key not in order_keys:
                hud_text += str(value)
            text_surf = self.hud_font.render(hud_text, True, self.hud_font_color)
            text_pos = hud_text_pos(line_num)
            surface.blit(text_surf, text_pos)
            if key in order_keys and value:
                rec_surf = get_recipes_surface(value)
                assert rec_surf.get_width() + text_surf.get_width() <= surface.get_width()
                surface.blit(rec_surf, recipes_pos(text_surf, text_pos))

    # ------------------------------------------------------------------
    # Action probability arrows  (identical to v1)
    # ------------------------------------------------------------------

    def _render_on_tile_position(
        self, scaled_grid_surface, source_surface, tile_position,
        horizontal_align="left", vertical_align="top",
    ):
        left_x, top_y = self._position_in_scaled_pixels(tile_position)
        if horizontal_align == "left":
            x = left_x
        elif horizontal_align == "center":
            x = left_x + (self.tile_size - source_surface.get_width()) / 2
        elif horizontal_align == "right":
            x = left_x + self.tile_size - source_surface.get_width()
        else:
            raise ValueError("horizontal_align must be left/center/right")

        if vertical_align == "top":
            y = top_y
        elif vertical_align == "center":
            y = top_y + (self.tile_size - source_surface.get_height()) / 2
        elif vertical_align == "bottom":
            y = top_y + self.tile_size - source_surface.get_height()
        else:
            raise ValueError("vertical_align must be top/center/bottom")

        scaled_grid_surface.blit(source_surface, (x, y))

    def _render_actions_probs(self, surface, players, action_probs):
        direction_to_rotation = {
            Direction.NORTH: 0,
            Direction.WEST: 90,
            Direction.SOUTH: 180,
            Direction.EAST: 270,
        }
        direction_to_aligns = {
            Direction.NORTH: {"horizontal_align": "center", "vertical_align": "bottom"},
            Direction.WEST:  {"horizontal_align": "right",  "vertical_align": "center"},
            Direction.SOUTH: {"horizontal_align": "center", "vertical_align": "top"},
            Direction.EAST:  {"horizontal_align": "left",   "vertical_align": "center"},
        }

        rescaled_arrow    = pygame.transform.scale(self._arrow_img, (self.tile_size, self.tile_size))
        rescaled_interact = pygame.transform.scale(
            self._interact_img, (int(self.tile_size / math.sqrt(2)), self.tile_size)
        )
        rescaled_stay = pygame.transform.scale(
            self._stay_img, (int(self.tile_size / math.sqrt(2)), self.tile_size)
        )

        for player, probs in zip(players, action_probs):
            if probs is None:
                continue
            for action in Action.ALL_ACTIONS:
                size = math.sqrt(probs[Action.ACTION_TO_INDEX[action]])
                if action == "interact":
                    img = pygame.transform.rotozoom(rescaled_interact, 0, size)
                    self._render_on_tile_position(
                        surface, img, player.position,
                        horizontal_align="left", vertical_align="center",
                    )
                elif action == Action.STAY:
                    img = pygame.transform.rotozoom(rescaled_stay, 0, size)
                    self._render_on_tile_position(
                        surface, img, player.position,
                        horizontal_align="right", vertical_align="center",
                    )
                else:
                    position = Action.move_in_direction(player.position, action)
                    img = pygame.transform.rotozoom(
                        rescaled_arrow, direction_to_rotation[action], size
                    )
                    self._render_on_tile_position(
                        surface, img, position, **direction_to_aligns[action]
                    )
