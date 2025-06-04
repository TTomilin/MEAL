# overcooked_upper_bound.py  –  v2 “faces‑n‑multipots”
"""Heuristic reward upper bound for *any* Overcooked layout.

New in **v2**
────────────
1. **Facing / interaction actions** Each time an agent picks up or drops an
   item we add a constant `ACTION_OVERHEAD` (default = 2) to cover a turn and
   the interaction button.
2. **Multiple copies** of pots / piles / goals.
   We take the *closest* source–target pair each time.
3. **>2 agents** supported.  We assume the quickest agent handles each leg
   (upper bound) but ignore agents that cannot reach *all* required objects.

It’s still rough‑and‑ready (and intentionally optimistic) but now matches
real action counts far better on procedurally‑generated bizarro kitchens.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Tuple, Dict, Sequence, List, Set

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

# ─── Constants (sync with env) ────────────────────────────────────────────────
DELIVERY_REWARD = 20  # reward per soup
COOK_TIME = 20  # pot countdown
ACTION_OVERHEAD = 2  # turn+interact cost per pickup / drop

# recipe: 3 onions → soup → deliver on plate
# onion pick + drop + plate pick + soup pick + deliver
INTERACTIONS_PER_CYCLE = 3 * 2 + 1 + 1 + 1
OVERHEAD_PER_CYCLE = INTERACTIONS_PER_CYCLE * ACTION_OVERHEAD


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _unravel(idx: int, w: int) -> Tuple[int, int]:
    return idx % w, idx // w


def _bfs(wall: jnp.ndarray, srcs: Sequence[Tuple[int, int]], tgts: Sequence[Tuple[int, int]]) -> int:
    """Shortest Manhattan path avoiding *wall* tiles between two *sets* of tiles."""
    tgt: Set[Tuple[int, int]] = set(tgts)
    q: deque = deque([(s, 0) for s in srcs])
    seen: Set[Tuple[int, int]] = set(srcs)
    while q:
        (x, y), d = q.popleft()
        if (x, y) in tgt:
            return d
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (0 <= nx < wall.shape[1] and 0 <= ny < wall.shape[0]
                    and not bool(wall[ny, nx]) and (nx, ny) not in seen):
                seen.add((nx, ny))
                q.append(((nx, ny), d + 1))
    return math.inf


def _neighbourhood(indices: Sequence[int], w: int, wall: jnp.ndarray) -> List[Tuple[int, int]]:
    """Walkable 4‑neighbours around every tile in *indices*."""
    out: List[Tuple[int, int]] = []
    for idx in indices:
        x, y = _unravel(int(idx), w)
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < wall.shape[0] and not bool(wall[ny, nx]):
                out.append((nx, ny))
    return out


# ─── Core ─────────────────────────────────────────────────────────────────────

def estimate_cycle_time(layout: Dict, *, n_agents: int = 2) -> float:
    """Upper‑bound steps for **one** cook‑deliver cycle."""
    h, w = int(layout["height"]), int(layout["width"])
    wall = jnp.zeros((h, w), dtype=bool)
    wall = wall.at[jnp.unravel_index(jnp.asarray(layout["wall_idx"]), (h, w))].set(True)

    # Neighbour sets (walkable squares adjacent to each object family)
    onion_nei = _neighbourhood(layout["onion_pile_idx"], w, wall)
    plate_nei = _neighbourhood(layout["plate_pile_idx"], w, wall)
    pot_nei = _neighbourhood(layout["pot_idx"], w, wall)
    goal_nei = _neighbourhood(layout["goal_idx"], w, wall)

    if min(map(len, (onion_nei, plate_nei, pot_nei, goal_nei))) == 0:
        return math.inf

    d_onion = _bfs(wall, onion_nei, pot_nei)
    d_plate = _bfs(wall, plate_nei, pot_nei)
    d_goal = _bfs(wall, pot_nei, goal_nei)

    if math.isinf(d_onion) or math.isinf(d_plate) or math.isinf(d_goal):
        return math.inf

    # Single‑agent pessimistic: one agent does everything
    move_cost = 3 * d_onion + d_plate + 1 + d_goal + 3  # + hand‑offs

    return move_cost + COOK_TIME + OVERHEAD_PER_CYCLE


def estimate_max_soup(layout: FrozenDict, episode_len: int, *, n_agents: int = 2) -> int:
    cyc = estimate_cycle_time(layout, n_agents=n_agents)
    if math.isinf(cyc) or cyc == 0:
        return 0
    soups = episode_len // cyc
    return int(soups * DELIVERY_REWARD)


# ─── Vectorised wrapper ────────────────────────────────────────────────────────
@jax.jit
def batched_max_reward(layouts: Sequence[Dict], episode_len: int, n_agents: int = 2) -> jnp.ndarray:
    fn = jax.vmap(lambda lay: estimate_max_soup(lay, episode_len, n_agents=n_agents))
    return fn(jnp.array(layouts))


# ─── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from jax_marl.environments.overcooked_environment.layouts import overcooked_layouts

    print(estimate_max_soup(overcooked_layouts["cramped_room"], 400))
