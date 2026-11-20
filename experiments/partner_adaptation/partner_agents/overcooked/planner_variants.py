"""Configurable planner partners P01-P12 for continual partner adaptation.

One planner, `PlannerAgent`, parameterised by `PlannerParams`. The parameters are *runtime data* (they travel as the
`params` argument of `OvercookedPlannerPolicyWrapper`), so all twelve variants of a layout share one compiled graph.
Navigation, approach/orientation and move tie-breaking reuse `BaseAgent` (`_move_towards`,
`_get_target_orientation_action`); target selection is done here on the exact environment state because the
observation channels the older planners select from also mark carried items and non-interactable border walls.
These are frozen partners: they read the environment state like the existing planners and need nothing from the ego.

Decision procedure (every step; the first matching rule wins)
-----------------------------------------------------------
Own drops:  an item the agent put on a counter is left for the other agent: while it is still there the agent never
    targets it for pickup (so counter handoff cannot degenerate into re-taking its own onion).
Holding an onion:  put it in the nearest non-full pot, or on a free counter. The destination is drawn *once per
    onion* (committed until the onion leaves the hand): counter with probability `p_onion_on_counter`. A counter is
    used only if feasible, a pot only if feasible; with neither feasible the agent stays.
Holding a plate:   plate the nearest ready pot; else walk next to the cooking pot with the least time left and wait
    there (no interaction); else (nothing cooking or ready) put the plate on the nearest free counter, else stay.
Holding a dish:    deliver at the nearest delivery tile, else stay.
Empty-handed:      two subtasks.
    FILL  feasible: a non-full pot and an onion source (pile or onion on a counter) are reachable.
    SERVE feasible: a plate source and a delivery tile are reachable and (a pot is ready, or - when
          `prefetch_steps` > 0 - a pot is cooking with at most `prefetch_steps` steps left).
    One feasible: do it. Both: `priority` decides - FILL first, SERVE first, NEAREST (smaller Manhattan distance to
    the first target; tie FILL) or COMPLEMENT (the subtask the other agent covered *less* in the last
    `history_window` steps; tie FILL). None feasible: stay.
Everything that moves uses the layout's static floor map; "reachable" means adjacent to a floor tile in the agent's
connected floor component. A counter is a *handoff* counter if it is free and adjacent to floor of both agents'
components. Ties between equally good targets are broken by the lowest row-major tile id, never randomly.

Overlays (applied after the decision)
-------------------------------------
Blocked move:  the previous action was a translation onto a walkable tile and the position did not change. With
    `yield_wait` > 0 the agent then stays for up to `yield_wait` steps, stopping early as soon as the other agent has
    moved from where it stood when the block was detected; then it replans. With 0 it replans at once.
Deadlock/livelock recovery:  progress means the inventory changed or the Manhattan distance to the current target
    reached a new minimum since the inventory last changed (or since the decision last held steady for 3 steps).
    After `STALL_LIMIT` consecutive steps without progress the agent takes one random legal step (never onto the
    other agent) and restarts the count. This catches both frozen positions and two agents shuffling back and
    forth in a corridor. Steps with nothing to do, or deliberately waiting beside a cooking pot, never count.
Reset:  the wrapper clears all planner memory when `done` is set, before acting.

Other-agent activity (for COMPLEMENT): each step the other agent's inventory change is classified into one event -
FILL (it picked up or released an onion) or SERVE (it picked up/released a plate or dish, plated, delivered) - and
pushed on a fixed 32-entry history. The first step after a reset records nothing (no previous inventory known).
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from meal.env.overcooked import Actions
from meal.env.overcooked.common import OBJECT_TO_INDEX
from .base_agent import AgentState, BaseAgent, Goal, Holding

HISTORY_LEN = 32
STALL_LIMIT = 8
YIELD_NEVER = -1  # yield_wait sentinel: keep the planned move when blocked (no wait, no replan)

STAY = int(Actions.stay)
INTERACT = int(Actions.interact)

I_EMPTY, I_ONION, I_PLATE, I_DISH = (OBJECT_TO_INDEX[k] for k in ("empty", "onion", "plate", "dish"))
O_WALL, O_ONION, O_ONION_PILE, O_PLATE, O_PLATE_PILE, O_GOAL, O_POT, O_DISH = (
    OBJECT_TO_INDEX[k] for k in ("wall", "onion", "onion_pile", "plate", "plate_pile", "goal", "pot", "dish"))

FACE_DY = jnp.array([-1, 1, 0, 0])
FACE_DX = jnp.array([0, 0, 1, -1])
MOVE_DY = (-1, 1, 0, 0)
MOVE_DX = (0, 0, 1, -1)

FILL, SERVE = 0, 1  # subtasks and history event codes (-1: no event)

CLASSIC_PLANNER_IDS = tuple(f"P{i:02d}" for i in range(1, 13))


class Priority:
    FILL = 0
    SERVE = 1
    NEAREST = 2
    COMPLEMENT = 3


class Handoff:
    NEAREST = 0
    LOWEST = 1
    HIGHEST = 2


class Role:
    ALL = 0
    ONION = 1
    SERVE = 2


class PotPref:
    NEAREST = 0
    LEFT = 1
    RIGHT = 2


class Rotation:
    NONE = 0
    CLOCKWISE = 1
    COUNTERCLOCKWISE = 2


class Case:
    NONE, PICK_ONION, PICK_PLATE, ONION_TO_POT, ONION_TO_COUNTER, PLATE_TO_POT, PLATE_WAIT, PLATE_STASH, DELIVER, PARK = range(10)


@struct.dataclass
class PlannerParams:
    p_onion_on_counter: Any
    prefetch_steps: Any
    priority: Any
    yield_wait: Any
    handoff_rule: Any
    history_window: Any
    role: Any
    counter_only: Any
    pot_preference: Any
    rotation: Any
    idle_prob: Any


@struct.dataclass
class PlannerState(AgentState):
    prev_pos: Any  # (2,) [y, x] before the last action; -1 right after reset
    prev_inv: Any
    prev_case: Any
    case_run: Any  # consecutive steps the decision has been the same
    best_dist: Any
    prev_action: Any
    move_intended: Any  # last action was a translation onto a walkable tile
    other_prev_inv: Any  # -1: unknown
    hist: Any  # (HISTORY_LEN,) newest first: FILL, SERVE or -1
    dest_counter: Any  # committed onion destination
    dest_valid: Any
    wait_left: Any
    wait_ref: Any  # (2,) other agent's [y, x] when the block was detected
    stall: Any
    prev_faced: Any  # (2,) [y, x] of the tile the agent faced last step
    dropped: Any  # (H, W) items this agent put on counters that are still there


# ----------------------------------------------------------------------------------------------- registry

@dataclass(frozen=True)
class PlannerSpec:
    planner_id: str
    name: str
    description: str
    params: Dict[str, Any]
    activation: str = ""  # `act` info flag that counts the decisions this variant's distinguishing option affects

    @property
    def config(self) -> Dict[str, Any]:
        return {"planner_id": self.planner_id, "name": self.name, "params": dict(self.params)}

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()


# Fixed shared generalist. Every pair varies exactly one parameter around it with one value on each side, so no
# variant equals the generalist and all twelve resolved configurations differ (it is not itself a partner).
GENERALIST = dict(p_onion_on_counter=0.4, prefetch_steps=5, priority=Priority.NEAREST, yield_wait=1,
                  handoff_rule=Handoff.NEAREST, history_window=0)


def _spec(pid, name, description, activation, **override):
    return PlannerSpec(pid, name, description, {**GENERALIST, **override}, activation)


PLANNER_SPECS: Dict[str, PlannerSpec] = {s.planner_id: s for s in (
    _spec("P01", "supply_direct", "load pots directly (counter probability 0.0)", "pot_route",
          p_onion_on_counter=0.0),
    _spec("P02", "supply_handoff", "route onions via counters (counter probability 0.8)", "counter_route",
          p_onion_on_counter=0.8),
    _spec("P03", "plate_when_ready", "fetch a plate only once a soup is ready", "plate_ready", prefetch_steps=0),
    _spec("P04", "plate_prefetch", "fetch a plate when <=10 steps of cooking remain", "prefetch", prefetch_steps=10),
    _spec("P05", "fill_first", "when fill and serve are both feasible, fill", "both", priority=Priority.FILL),
    _spec("P06", "serve_first", "when fill and serve are both feasible, serve", "both", priority=Priority.SERVE),
    _spec("P07", "yield_replan", "replan immediately when a move is blocked", "blocked", yield_wait=0),
    _spec("P08", "yield_wait2", "wait up to 2 steps when a move is blocked", "waiting", yield_wait=2),
    _spec("P09", "handoff_lowest", "handoff counter: lowest row-major id", "multi_handoff",
          handoff_rule=Handoff.LOWEST),
    _spec("P10", "handoff_highest", "handoff counter: highest row-major id", "multi_handoff",
          handoff_rule=Handoff.HIGHEST),
    _spec("P11", "complement_8", "least-covered subtask over 8 steps", "complement_decided",
          priority=Priority.COMPLEMENT, history_window=8),
    _spec("P12", "complement_32", "least-covered subtask over 32 steps", "complement_decided",
          priority=Priority.COMPLEMENT, history_window=HISTORY_LEN),
    _spec("onion_left_cw", "onion_left_cw", "fills only, left pot, walks clockwise", "role_onion",
          role=Role.ONION, pot_preference=PotPref.LEFT, rotation=Rotation.CLOCKWISE),
    _spec("onion_right_ccw", "onion_right_ccw", "fills only, right pot, walks counter-clockwise", "role_onion",
          role=Role.ONION, pot_preference=PotPref.RIGHT, rotation=Rotation.COUNTERCLOCKWISE),
    _spec("server_cw", "server_cw", "serves only, walks clockwise", "role_serve",
          role=Role.SERVE, rotation=Rotation.CLOCKWISE),
    _spec("server_ccw", "server_ccw", "serves only, walks counter-clockwise", "role_serve",
          role=Role.SERVE, rotation=Rotation.COUNTERCLOCKWISE),
    _spec("counter_feeder", "counter_feeder", "fills only, every onion onto a handoff counter", "counter_only",
          role=Role.ONION, counter_only=True),
    _spec("generalist_left_cw", "generalist_left_cw", "does everything, left pot, walks clockwise", "pot_preference",
          pot_preference=PotPref.LEFT, rotation=Rotation.CLOCKWISE),
    _spec("generalist_right_ccw", "generalist_right_ccw",
          "does everything, right pot, walks counter-clockwise", "pot_preference",
          pot_preference=PotPref.RIGHT, rotation=Rotation.COUNTERCLOCKWISE),
    _spec("generalist_stubborn", "generalist_stubborn", "does everything, never gives way when blocked",
          "never_yield", yield_wait=YIELD_NEVER),
    _spec("generalist_slow", "generalist_slow", "does everything, idles half the steps", "idle", idle_prob=0.5),
    _spec("server_stubborn", "server_stubborn", "serves only, never gives way when blocked", "never_yield",
          role=Role.SERVE, yield_wait=YIELD_NEVER),
    _spec("onion_slow_ccw", "onion_slow_ccw",
          "fills only, idles half the steps, walks counter-clockwise", "idle",
          role=Role.ONION, idle_prob=0.5, rotation=Rotation.COUNTERCLOCKWISE),
    _spec("counter_feeder_right", "counter_feeder_right",
          "fills only via handoff counters, right pot when it must use a pot", "counter_only",
          role=Role.ONION, counter_only=True, pot_preference=PotPref.RIGHT),
    _spec("idle", "idle", "never acts; reference for the ego working alone", "idle",
          idle_prob=1.0),
    _spec('all_nearest_cw', 'all_nearest_cw', "", "", role=Role.ALL, pot_preference=PotPref.NEAREST, rotation=Rotation.CLOCKWISE, yield_wait=1),
    _spec('onion_nearest_none_stubborn', 'onion_nearest_none_stubborn', "", "", role=Role.ONION, pot_preference=PotPref.NEAREST, rotation=Rotation.NONE, yield_wait=YIELD_NEVER),
    _spec('onion_nearest_ccw', 'onion_nearest_ccw', "", "", role=Role.ONION, pot_preference=PotPref.NEAREST, rotation=Rotation.COUNTERCLOCKWISE, yield_wait=1),
    _spec('serve_nearest_none', 'serve_nearest_none', "", "", role=Role.SERVE, pot_preference=PotPref.NEAREST, rotation=Rotation.NONE, yield_wait=1),
)}

NEW_PLANNER_IDS = tuple(pid for pid in PLANNER_SPECS if pid not in CLASSIC_PLANNER_IDS)


def planner_params(spec) -> PlannerParams:
    """PlannerParams (scalar arrays) of a `PlannerSpec` or planner id."""
    p = (PLANNER_SPECS[spec] if isinstance(spec, str) else spec).params
    return PlannerParams(
        p_onion_on_counter=jnp.float32(p["p_onion_on_counter"]), prefetch_steps=jnp.int32(p["prefetch_steps"]),
        priority=jnp.int32(p["priority"]), yield_wait=jnp.int32(p["yield_wait"]),
        handoff_rule=jnp.int32(p["handoff_rule"]), history_window=jnp.int32(p["history_window"]),
        role=jnp.int32(p.get("role", Role.ALL)), counter_only=jnp.asarray(p.get("counter_only", False)),
        pot_preference=jnp.int32(p.get("pot_preference", PotPref.NEAREST)),
        rotation=jnp.int32(p.get("rotation", Rotation.NONE)), idle_prob=jnp.float32(p.get("idle_prob", 0.0)))


# ---------------------------------------------------------------------------------------------- geometry

def _geometry(layout):
    """Static floor map, connected floor components and per-component reachable tiles."""
    H, W = int(layout["height"]), int(layout["width"])
    solid = np.zeros(H * W, bool)
    solid[np.asarray(layout["wall_idx"])] = True
    special = np.zeros(H * W, bool)
    for key in ("pot_idx", "goal_idx", "onion_pile_idx", "plate_pile_idx"):
        special[np.asarray(layout[key])] = True
    solid, special = solid.reshape(H, W), special.reshape(H, W)
    floor = ~solid

    comp = -np.ones((H, W), np.int32)
    n = 0
    for y in range(H):
        for x in range(W):
            if floor[y, x] and comp[y, x] < 0:
                stack = [(y, x)]
                comp[y, x] = n
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < H and 0 <= nx < W and floor[ny, nx] and comp[ny, nx] < 0:
                            comp[ny, nx] = n
                            stack.append((ny, nx))
                n += 1
    reach = np.zeros((max(n, 1), H, W), bool)
    for y in range(H):
        for x in range(W):
            if floor[y, x]:
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < H and 0 <= nx < W and not floor[ny, nx]:
                        reach[comp[y, x], ny, nx] = True
    counters = solid & ~special & reach.any(0)
    return floor, comp, reach, counters


def _inner_counter_block(floor, counters):
    """Connected solid component that does not touch the map border and includes a counter, else None."""
    H, W = floor.shape
    solid = ~floor
    seen = np.zeros_like(solid)
    for y in range(H):
        for x in range(W):
            if not solid[y, x] or seen[y, x]:
                continue
            stack, cells, touches_border, has_counter = [(y, x)], [], False, False
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                touches_border |= cy in (0, H - 1) or cx in (0, W - 1)
                has_counter |= bool(counters[cy, cx])
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < H and 0 <= nx < W and solid[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if not touches_border and has_counter:
                return cells
    return None


def _ring_edges(floor, counters):
    """Directed floor edges that do not go against clockwise around an inner counter block.

    With y growing downward, a positive 2D cross product is clockwise. Moves with zero angular change
    (purely radial) are allowed. Returns (has_loop, center_y, center_x, edge_cw[H,W,4]).
    """
    H, W = floor.shape
    block = _inner_counter_block(floor, counters)
    edge_cw = np.zeros((H, W, 4), bool)
    if not block:
        return False, 0.0, 0.0, edge_cw
    cy = sum(y for y, _ in block) / len(block)
    cx = sum(x for _, x in block) / len(block)
    for y in range(H):
        for x in range(W):
            if not floor[y, x]:
                continue
            for a, (dy, dx) in enumerate(zip(MOVE_DY, MOVE_DX)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < H and 0 <= nx < W and floor[ny, nx]):
                    continue
                # (x right, y down): positive cross = clockwise
                cross = (x - cx) * (ny - cy) - (y - cy) * (nx - cx)
                edge_cw[y, x, a] = cross >= 0
    return True, cy, cx, edge_cw


def _next_hops(floor, edge_ok):
    """next/dist[sy,sx,ty,tx]: first action and length of a shortest directed path (-1 / big if none)."""
    H, W = floor.shape
    inf = H * W + 1
    nxt = -np.ones((H, W, H, W), np.int8)
    dist_all = np.full((H, W, H, W), inf, np.int32)
    for sy in range(H):
        for sx in range(W):
            if not floor[sy, sx]:
                continue
            dist = np.full((H, W), inf, np.int32)
            first = -np.ones((H, W), np.int8)
            dist[sy, sx] = 0
            first[sy, sx] = STAY
            queue = [(sy, sx)]
            head = 0
            while head < len(queue):
                y, x = queue[head]
                head += 1
                for a, (dy, dx) in enumerate(zip(MOVE_DY, MOVE_DX)):
                    if not edge_ok[y, x, a]:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < H and 0 <= nx < W and floor[ny, nx]):
                        continue
                    if dist[ny, nx] > dist[y, x] + 1:
                        dist[ny, nx] = dist[y, x] + 1
                        first[ny, nx] = np.int8(a if (y, x) == (sy, sx) else first[y, x])
                        queue.append((ny, nx))
            nxt[sy, sx] = first
            dist_all[sy, sx] = dist
    return nxt, dist_all


# ------------------------------------------------------------------------------------------------- agent

class PlannerAgent(BaseAgent):
    """See the module docstring. `act` is pure; `get_action` only exists to satisfy the BaseAgent interface."""

    def __init__(self, layout: Dict[str, Any]):
        super().__init__(layout)
        H, W = self.map_height, self.map_width
        self.floor, self.comp, self.reach, self.counters = _geometry(layout)
        self.flat = np.arange(H * W, dtype=np.int32).reshape(H, W)
        self.yy, self.xx = np.mgrid[0:H, 0:W].astype(np.int32)
        self.big = H * W * (H + W + 2) * 8
        self.has_loop, self.ring_cy, self.ring_cx, edge_cw = _ring_edges(self.floor, self.counters)
        edge_ccw = np.zeros_like(edge_cw)
        if self.has_loop:
            # zero angular change is in both; against-clockwise = not strictly clockwise
            for y in range(H):
                for x in range(W):
                    if not self.floor[y, x]:
                        continue
                    for a, (dy, dx) in enumerate(zip(MOVE_DY, MOVE_DX)):
                        ny, nx = y + dy, x + dx
                        if not (0 <= ny < H and 0 <= nx < W and self.floor[ny, nx]):
                            continue
                        cross = (x - self.ring_cx) * (ny - self.ring_cy) - (y - self.ring_cy) * (nx - self.ring_cx)
                        edge_ccw[y, x, a] = cross <= 0
            self.edge_cw, self.edge_ccw = edge_cw, edge_ccw
            self.next_cw, self.dist_cw = _next_hops(self.floor, edge_cw)
            self.next_ccw, self.dist_ccw = _next_hops(self.floor, edge_ccw)
        else:
            self.edge_cw = self.edge_ccw = np.zeros((H, W, 4), bool)
            self.next_cw = self.next_ccw = -np.ones((H, W, H, W), np.int8)
            self.dist_cw = self.dist_ccw = np.full((H, W, H, W), H * W + 1, np.int32)

    # state --------------------------------------------------------------------------------------------
    def init_agent_state(self, agent_id) -> PlannerState:
        i32 = lambda v: jnp.asarray(v, jnp.int32)
        return PlannerState(
            agent_id=i32(agent_id), holding=i32(Holding.nothing), goal=i32(Goal.get_onion),
            nonfull_pots=jnp.ones(self.num_pots, bool), soup_ready=jnp.asarray(False),
            rng_key=jax.random.PRNGKey(i32(agent_id)),
            prev_pos=-jnp.ones(2, jnp.int32), prev_inv=i32(-1), prev_case=i32(Case.NONE), case_run=i32(0),
            best_dist=i32(0), prev_action=i32(STAY),
            move_intended=jnp.asarray(False), other_prev_inv=i32(-1),
            hist=-jnp.ones(HISTORY_LEN, jnp.int8), dest_counter=jnp.asarray(False), dest_valid=jnp.asarray(False),
            wait_left=i32(0), wait_ref=-jnp.ones(2, jnp.int32), stall=i32(0),
            prev_faced=-jnp.ones(2, jnp.int32), dropped=jnp.zeros((self.map_height, self.map_width), bool))

    def get_action(self, obs, env_state, agent_state, params=None, rng=None):
        rng = jax.random.PRNGKey(0) if rng is None else rng
        action, state, _ = self.act(params, obs, env_state, agent_state, rng)
        return action, state

    # helpers ------------------------------------------------------------------------------------------
    def _pick(self, mask, key):
        """(y, x, valid) of the masked tile with the smallest key; ties go to the lowest row-major id."""
        W = self.map_width
        scored = jnp.where(mask, key.astype(jnp.int32) * (self.map_height * W) + self.flat, 2 ** 30)
        idx = jnp.argmin(scored.ravel())
        return idx // W, idx % W, mask.any()

    def _approach_tile(self, ty, tx, my_y, my_x, comp, oy, ox):
        """Floor tile next to the target, in my component, nearest to me; the other agent's tile only if no other."""
        H, W = self.map_height, self.map_width
        keys, ys, xs = [], [], []
        for order, (dy, dx) in enumerate(((-1, 0), (1, 0), (0, 1), (0, -1))):
            ny, nx = ty + dy, tx + dx
            inb = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            cy, cx = jnp.clip(ny, 0, H - 1), jnp.clip(nx, 0, W - 1)
            ok = inb & jnp.asarray(self.floor)[cy, cx] & (jnp.asarray(self.comp)[cy, cx] == comp)
            occupied = ((cy == oy) & (cx == ox)).astype(jnp.int32)
            keys.append(jnp.where(ok, occupied * 1000 + (jnp.abs(cy - my_y) + jnp.abs(cx - my_x)) * 4 + order, 10 ** 6))
            ys.append(cy)
            xs.append(cx)
        best = jnp.argmin(jnp.stack(keys))
        found = jnp.min(jnp.stack(keys)) < 10 ** 6
        return jnp.where(found, jnp.stack(ys)[best], my_y), jnp.where(found, jnp.stack(xs)[best], my_x)

    def _move_rotated(self, my_y, my_x, ay, ax, oy, ox, obs3, key, rotation):
        base, key = self._move_towards(my_y, my_x, ay, ax, obs3, key)
        use = self.has_loop & (rotation != Rotation.NONE)
        cw = rotation == Rotation.CLOCKWISE
        hop = jnp.where(cw, jnp.asarray(self.next_cw)[my_y, my_x, ay, ax],
                        jnp.asarray(self.next_ccw)[my_y, my_x, ay, ax]).astype(jnp.int32)
        # unreachable under the directed graph: fall back to today's step
        unreachable = hop < 0
        edge = jnp.where(cw, jnp.asarray(self.edge_cw)[my_y, my_x], jnp.asarray(self.edge_ccw)[my_y, my_x])
        dist = jnp.where(cw, jnp.asarray(self.dist_cw)[:, :, ay, ax], jnp.asarray(self.dist_ccw)[:, :, ay, ax])
        # if the shortest hop is occupied, take the directed neighbour that still gets closer
        scores = []
        for a, (dy, dx) in enumerate(zip(MOVE_DY, MOVE_DX)):
            ny, nx = my_y + dy, my_x + dx
            free = edge[a] & jnp.asarray(self.floor)[jnp.clip(ny, 0, self.map_height - 1),
                                                      jnp.clip(nx, 0, self.map_width - 1)]
            free = free & (ny >= 0) & (ny < self.map_height) & (nx >= 0) & (nx < self.map_width)
            free = free & ~((ny == oy) & (nx == ox))
            scores.append(jnp.where(free, dist[jnp.clip(ny, 0, self.map_height - 1),
                                               jnp.clip(nx, 0, self.map_width - 1)], 10 ** 6))
        scores = jnp.stack(scores)
        alt = jnp.argmin(scores).astype(jnp.int32)
        alt_ok = jnp.min(scores) < 10 ** 6
        ny = my_y + jnp.array(MOVE_DY)[jnp.clip(hop, 0, 3)]
        nx = my_x + jnp.array(MOVE_DX)[jnp.clip(hop, 0, 3)]
        hop_blocked = (hop < 4) & (ny == oy) & (nx == ox)
        directed = jnp.where(unreachable, base, jnp.where(hop_blocked, jnp.where(alt_ok, alt, STAY), hop))
        return jnp.where(use, directed, base).astype(jnp.int32), key

    def _go_interact(self, obs3, env_state, aid, my_y, my_x, ty, tx, comp, oy, ox, interact_ok, key, rotation):
        adjacent = (jnp.abs(my_y - ty) + jnp.abs(my_x - tx)) == 1
        face = self._get_target_orientation_action(my_y, my_x, ty, tx)
        facing = env_state.agent_dir_idx[aid]
        at_target = jnp.where(facing == face, jnp.where(interact_ok, INTERACT, STAY), face)
        ay, ax = self._approach_tile(ty, tx, my_y, my_x, comp, oy, ox)
        move, _ = self._move_rotated(my_y, my_x, ay, ax, oy, ox, obs3, key, rotation)
        return jnp.where(adjacent, at_target, move).astype(jnp.int32)

    def _sidestep(self, my_y, my_x, oy, ox, key):
        H, W = self.map_height, self.map_width
        ys = my_y + jnp.array([-1, 1, 0, 0])
        xs = my_x + jnp.array([0, 0, 1, -1])
        inb = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        cy, cx = jnp.clip(ys, 0, H - 1), jnp.clip(xs, 0, W - 1)
        ok = inb & jnp.asarray(self.floor)[cy, cx] & ~((cy == oy) & (cx == ox))
        action = jax.random.categorical(key, jnp.where(ok, 0.0, -jnp.inf))
        return jnp.where(ok.any(), action, STAY).astype(jnp.int32)

    # decision -----------------------------------------------------------------------------------------
    def act(self, params: PlannerParams, obs, env_state, state: PlannerState, rng) -> Tuple[Any, PlannerState, Dict]:
        env_state = getattr(env_state, "env_state", env_state)
        H, W = self.map_height, self.map_width
        obs3 = jnp.reshape(obs, self.obs_shape)
        aid = state.agent_id
        i32 = lambda v: jnp.asarray(v, jnp.int32)

        pos = env_state.agent_pos.astype(jnp.int32)  # rows are [x, y]
        my_y, my_x = pos[aid, 1], pos[aid, 0]
        oy, ox = pos[1 - aid, 1], pos[1 - aid, 0]
        inv, oinv = i32(env_state.agent_inv[aid]), i32(env_state.agent_inv[1 - aid])
        comp = jnp.asarray(self.comp)
        c_me, c_ot = jnp.maximum(comp[my_y, my_x], 0), jnp.maximum(comp[oy, ox], 0)
        reach = jnp.asarray(self.reach)
        reach_me, reach_ot = reach[c_me], reach[c_ot]

        obj = env_state.maze_map[..., 0].astype(jnp.int32)
        status = env_state.maze_map[..., 2].astype(jnp.int32)
        cook = i32(env_state.pot_full_status)
        pots = obj == O_POT
        nonfull = pots & (status > cook) & reach_me
        cooking = pots & (status > 0) & (status <= cook) & reach_me
        ready = pots & (status == 0) & reach_me
        item_here = (obj == O_ONION) | (obj == O_PLATE) | (obj == O_DISH)
        fy_p, fx_p = jnp.clip(state.prev_faced[0], 0, H - 1), jnp.clip(state.prev_faced[1], 0, W - 1)
        new_drop = (state.prev_action == INTERACT) & jnp.isin(state.prev_inv, jnp.array([I_ONION, I_PLATE, I_DISH])) \
            & (inv == I_EMPTY) & item_here[fy_p, fx_p] & (state.prev_faced[0] >= 0)
        dropped = (state.dropped & item_here).at[fy_p, fx_p].max(new_drop)
        onion_src = ((obj == O_ONION_PILE) | ((obj == O_ONION) & ~dropped)) & reach_me
        plate_src = ((obj == O_PLATE_PILE) | ((obj == O_PLATE) & ~dropped)) & reach_me
        goals = (obj == O_GOAL) & reach_me
        free_counter = jnp.asarray(self.counters) & (obj == O_WALL) & reach_me
        handoff = free_counter & reach_ot

        dist = jnp.abs(jnp.asarray(self.yy) - my_y) + jnp.abs(jnp.asarray(self.xx) - my_x)
        flat = jnp.asarray(self.flat)
        nearest_key = dist
        xx = jnp.asarray(self.xx)
        pot_key = jnp.where(params.pot_preference == PotPref.LEFT, xx,
                            jnp.where(params.pot_preference == PotPref.RIGHT, -xx, dist))
        pick_onion = self._pick(onion_src, nearest_key)
        pick_plate = self._pick(plate_src, nearest_key)
        pick_pot = self._pick(nonfull, pot_key)
        pick_ready = self._pick(ready, pot_key)
        pick_goal = self._pick(goals, nearest_key)
        pick_stash = self._pick(free_counter, nearest_key)
        pick_wait_pot = self._pick(cooking, status * (H + W + 2) + pot_key)
        rule = params.handoff_rule
        hand_key = jnp.where(rule == Handoff.LOWEST, flat, jnp.where(rule == Handoff.HIGHEST, H * W - 1 - flat, dist))
        handoff_ok = handoff.any()
        pick_counter = self._pick(jnp.where(handoff_ok, handoff, free_counter),
                                  jnp.where(handoff_ok, hand_key, dist))

        # other agent's activity ----------------------------------------------------------------------
        known, changed = state.other_prev_inv >= 0, state.other_prev_inv != oinv
        involves = lambda *items: jnp.isin(state.other_prev_inv, jnp.array(items)) | jnp.isin(oinv, jnp.array(items))
        code = jnp.where(known & changed & involves(I_ONION), FILL,
                         jnp.where(known & changed & involves(I_PLATE, I_DISH), SERVE, -1)).astype(jnp.int8)
        hist = jnp.concatenate([code[None], state.hist[:-1]])
        in_window = jnp.arange(HISTORY_LEN) < jnp.clip(params.history_window, 0, HISTORY_LEN)
        covered_fill = jnp.sum((hist == FILL) & in_window)
        covered_serve = jnp.sum((hist == SERVE) & in_window)

        # subtasks (empty-handed) --------------------------------------------------------------------
        fill_ok = nonfull.any() & onion_src.any()
        soon = cooking.any() & (params.prefetch_steps > 0) & (
                jnp.min(jnp.where(cooking, status, 10 ** 6)) <= params.prefetch_steps)
        serve_ok = (ready.any() | soon) & plate_src.any() & goals.any()
        d_fill = jnp.where(pick_onion[2], dist[pick_onion[0], pick_onion[1]], 10 ** 6)
        d_serve = jnp.where(pick_plate[2], dist[pick_plate[0], pick_plate[1]], 10 ** 6)
        prio = params.priority
        fill_over_serve = jnp.where(prio == Priority.FILL, True, jnp.where(
            prio == Priority.SERVE, False, jnp.where(prio == Priority.NEAREST, d_fill <= d_serve,
                                                     covered_fill <= covered_serve)))
        both = fill_ok & serve_ok
        do_fill = jnp.where(both, fill_over_serve, fill_ok)
        c_empty_all = jnp.where(do_fill, Case.PICK_ONION, jnp.where(serve_ok, Case.PICK_PLATE, Case.NONE))

        # role parking targets when the role has nothing it can act on yet
        park_onion_t = jnp.where(onion_src.any(), jnp.stack([pick_onion[0], pick_onion[1]]),
                                 jnp.where(nonfull.any(), jnp.stack([pick_pot[0], pick_pot[1]]),
                                           jnp.where(handoff_ok, jnp.stack([pick_counter[0], pick_counter[1]]),
                                                     jnp.stack([my_y, my_x]))))
        park_onion_ok = onion_src.any() | nonfull.any() | handoff_ok
        park_serve_t = jnp.where(plate_src.any(), jnp.stack([pick_plate[0], pick_plate[1]]),
                                 jnp.where(ready.any(), jnp.stack([pick_ready[0], pick_ready[1]]),
                                           jnp.where(cooking.any(), jnp.stack([pick_wait_pot[0], pick_wait_pot[1]]),
                                                     jnp.where(goals.any(), jnp.stack([pick_goal[0], pick_goal[1]]),
                                                               jnp.stack([my_y, my_x])))))
        park_serve_ok = plate_src.any() | ready.any() | cooking.any() | goals.any()
        c_empty = jnp.where(
            params.role == Role.ONION,
            jnp.where(fill_ok, Case.PICK_ONION, jnp.where(park_onion_ok, Case.PARK, Case.NONE)),
            jnp.where(params.role == Role.SERVE,
                      jnp.where(serve_ok, Case.PICK_PLATE, jnp.where(park_serve_ok, Case.PARK, Case.NONE)),
                      c_empty_all))

        # onion destination, committed once per held onion ----------------------------------------------
        holds_onion = inv == I_ONION
        draw = (jax.random.uniform(jax.random.fold_in(rng, 1)) < params.p_onion_on_counter) | params.counter_only
        want_counter = jnp.where(state.dest_valid, state.dest_counter, draw)
        pot_ok, stash_ok = nonfull.any(), free_counter.any()
        go_counter = (want_counter & handoff_ok) | (~pot_ok & stash_ok)
        c_onion = jnp.where(go_counter, Case.ONION_TO_COUNTER, jnp.where(pot_ok, Case.ONION_TO_POT, Case.NONE))

        c_plate = jnp.where(ready.any(), Case.PLATE_TO_POT, jnp.where(
            cooking.any(), Case.PLATE_WAIT, jnp.where(stash_ok, Case.PLATE_STASH, Case.NONE)))
        c_dish = jnp.where(goals.any(), Case.DELIVER, Case.NONE)
        # role filters: ONION never serves; SERVE never handles onions
        c_held_onion = jnp.where(params.role == Role.SERVE, jnp.where(park_serve_ok, Case.PARK, Case.NONE), c_onion)
        c_held_plate = jnp.where(params.role == Role.ONION, jnp.where(park_onion_ok, Case.PARK, Case.NONE), c_plate)
        c_held_dish = jnp.where(params.role == Role.ONION, jnp.where(park_onion_ok, Case.PARK, Case.NONE), c_dish)
        case = jnp.where(inv == I_EMPTY, c_empty, jnp.where(holds_onion, c_held_onion, jnp.where(
            inv == I_PLATE, c_held_plate, jnp.where(inv == I_DISH, c_held_dish, Case.NONE))))

        park_t = jnp.where(params.role == Role.SERVE, park_serve_t, park_onion_t)
        targets = jnp.stack([jnp.stack([my_y, my_x])] + [jnp.stack([p[0], p[1]]) for p in (
            pick_onion, pick_plate, pick_pot, pick_counter, pick_ready, pick_wait_pot, pick_stash, pick_goal)]
                            + [park_t])
        ty, tx = targets[case, 0], targets[case, 1]
        key_nav, key_side = jax.random.split(jax.random.fold_in(rng, 2))
        interact_ok = (case != Case.PLATE_WAIT) & (case != Case.PARK)
        planned = self._go_interact(obs3, env_state, aid, my_y, my_x, ty, tx, c_me, oy, ox,
                                    interact_ok, key_nav, params.rotation)
        planned = jnp.where(case == Case.NONE, STAY, planned)
        idle = jax.random.uniform(jax.random.fold_in(rng, 3)) < params.idle_prob
        planned = jnp.where(idle, STAY, planned)

        # blocked moves, yielding, deadlock recovery ----------------------------------------------------
        moved = jnp.any(jnp.stack([my_y, my_x]) != state.prev_pos)
        blocked = state.move_intended & ~moved
        # NEVER: keep pushing the blocked move instead of taking a freshly planned action
        planned = jnp.where(blocked & (params.yield_wait == YIELD_NEVER), state.prev_action, planned)
        target_dist = i32(jnp.abs(my_y - ty) + jnp.abs(my_x - tx))
        case_changed = case != state.prev_case
        fresh = (inv != state.prev_inv) | (case_changed & (state.case_run >= 3))
        improved = fresh | (target_dist < state.best_dist)
        best_dist = jnp.where(fresh, target_dist, jnp.minimum(state.best_dist, target_dist))
        counting = (case != Case.NONE) & (case != Case.PLATE_WAIT)
        stall = jnp.where(improved | ~counting | idle, 0, state.stall + 1)
        recover = stall >= STALL_LIMIT

        start_wait = blocked & (params.yield_wait > 0) & (state.wait_left == 0)
        other_pos = jnp.stack([oy, ox])
        wait_left = jnp.where(start_wait, params.yield_wait, state.wait_left)
        wait_ref = jnp.where(start_wait, other_pos, state.wait_ref)
        waiting = (wait_left > 0) & jnp.all(other_pos == wait_ref)

        side = self._sidestep(my_y, my_x, oy, ox, key_side)
        action = jnp.where(recover, side, jnp.where(waiting, STAY, planned)).astype(jnp.int32)

        # bookkeeping --------------------------------------------------------------------------------------
        dy = jnp.array([-1, 1, 0, 0])[jnp.minimum(action, 3)]
        dx = jnp.array([0, 0, 1, -1])[jnp.minimum(action, 3)]
        ny, nx = jnp.clip(my_y + dy, 0, H - 1), jnp.clip(my_x + dx, 0, W - 1)
        intended = (action < 4) & jnp.asarray(self.floor)[ny, nx] & ((my_y + dy >= 0) & (my_y + dy < H)) & (
                (my_x + dx >= 0) & (my_x + dx < W))

        dir_idx = jnp.clip(i32(env_state.agent_dir_idx[aid]), 0, 3)
        held = jnp.where(inv == I_ONION, Holding.onion, jnp.where(inv == I_PLATE, Holding.plate, jnp.where(
            inv == I_DISH, Holding.dish, Holding.nothing)))
        new_state = state.replace(
            holding=i32(held), goal=i32(case), soup_ready=ready.any(),
            prev_pos=jnp.stack([my_y, my_x]), prev_inv=inv, prev_case=i32(case),
            case_run=i32(jnp.where(case_changed, 1, state.case_run + 1)), best_dist=best_dist, prev_action=action,
            move_intended=intended, other_prev_inv=oinv, hist=hist,
            dest_counter=jnp.where(holds_onion, want_counter, False), dest_valid=holds_onion,
            wait_left=i32(jnp.where(waiting, wait_left - 1, 0)), wait_ref=wait_ref,
            stall=i32(jnp.where(recover, 0, stall)),
            prev_faced=jnp.stack([my_y + FACE_DY[dir_idx], my_x + FACE_DX[dir_idx]]), dropped=dropped)
        info = dict(case=case, target=jnp.stack([ty, tx]), pot_route=case == Case.ONION_TO_POT,
                    plate_ready=(case == Case.PICK_PLATE) & ready.any(),
                    multi_handoff=(case == Case.ONION_TO_COUNTER) & (jnp.sum(handoff) >= 2), both=both,
                    prefetch=(case == Case.PICK_PLATE) & ~ready.any(),
                    counter_route=case == Case.ONION_TO_COUNTER, handoff_ok=handoff_ok, blocked=blocked,
                    waiting=waiting, recovered=recover, covered_fill=covered_fill, covered_serve=covered_serve,
                    complement_decided=both & (prio == Priority.COMPLEMENT) & (covered_fill != covered_serve),
                    role_onion=(params.role == Role.ONION) & (case != Case.NONE),
                    role_serve=(params.role == Role.SERVE) & (case != Case.NONE),
                    counter_only=params.counter_only & (case == Case.ONION_TO_COUNTER),
                    pot_preference=(params.pot_preference != PotPref.NEAREST) & jnp.isin(
                        case, jnp.array([Case.ONION_TO_POT, Case.PLATE_TO_POT, Case.PLATE_WAIT])),
                    never_yield=blocked & (params.yield_wait == YIELD_NEVER), idle=idle)
        return action, new_state, info


def layout_collapses(layout: Dict[str, Any]) -> Dict[str, str]:
    """Distinctions a layout cannot express, from its static geometry (see `PLANNER_SPECS`).

    Returns {pair label: reason}. Reasons are structural: for example with one reachable pot the fill/serve choice
    never has two feasible options (a pot is either loadable or cooking/ready, not both; this also removes the
    complement priority's effect), without a shared counter there is no handoff, without two "lowest" and "highest"
    are the same tile, and agents on disjoint floors never block each other.
    """
    floor, comp, reach, counters = _geometry(layout)
    H, W = int(layout["height"]), int(layout["width"])
    pot = np.zeros(H * W, bool)
    pot[np.asarray(layout["pot_idx"])] = True
    pot = pot.reshape(H, W)
    spawn_comps = sorted({int(comp[int(i) // W, int(i) % W]) for i in np.asarray(layout["agent_idx"]).ravel()})
    out = {}
    pots_reachable = max(int((pot & reach[c]).sum()) for c in spawn_comps)
    if pots_reachable < 2:
        why = f"{pots_reachable} reachable pot: fill and serve are never both feasible"
        out["P05/P06"] = out["P11/P12"] = why
        why_pot = f"{pots_reachable} reachable pot: pot_preference LEFT/RIGHT has no effect"
        out["generalist_left_cw/generalist_right_ccw"] = why_pot
        out["onion_left_cw/onion_right_ccw"] = why_pot
        out["counter_feeder/counter_feeder_right"] = why_pot
    shared = counters & np.logical_and.reduce([reach[c] for c in spawn_comps])
    n_shared = int(shared.sum())
    if n_shared < 1:
        out["P01/P02"] = "no counter shared by both agents' floor: counter handoff cannot happen, onions go to pots"
        out["counter_feeder/counter_feeder_right"] = (
            "no counter shared by both agents' floor: counter_only falls back to pots")
    if n_shared < 2:
        out["P09/P10"] = f"{n_shared} counters shared by both agents' floor: lowest and highest coincide"
    if len(spawn_comps) > 1:
        out["P07/P08"] = "the agents' floors are disjoint: one can never block the other's move"
    has_loop, _, _, _ = _ring_edges(floor, counters)
    if not has_loop:
        why_rot = "no floor loop around an inner counter block: rotation has no effect"
        out["server_cw/server_ccw"] = why_rot
        out["onion_left_cw/onion_right_ccw"] = out.get("onion_left_cw/onion_right_ccw", why_rot)
        out["generalist_left_cw/generalist_right_ccw"] = out.get("generalist_left_cw/generalist_right_ccw", why_rot)
        out["onion_slow_ccw/server_ccw"] = why_rot
    return out
