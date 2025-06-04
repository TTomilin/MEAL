#!/usr/bin/env python
"""Validation logic for Overcooked layouts.

This module contains functions to validate if a grid layout is valid and solvable for Overcooked.
"""
from __future__ import annotations

from typing import List, Tuple

###############################################################################
# ─── Symbolic constants ──────────────────────────────────────────────────────
###############################################################################

FLOOR: str = " "
WALL: str = "W"
GOAL: str = "X"
ONION_PILE: str = "O"
PLATE_PILE: str = "B"
POT: str = "P"
AGENT: str = "A"

UNPASSABLE_TILES = {WALL, GOAL, ONION_PILE, PLATE_PILE, POT}
INTERACTIVE_TILES = {GOAL, ONION_PILE, PLATE_PILE, POT}

###############################################################################
# ─── Validation helpers ──────────────────────────────────────────────────────
###############################################################################

def _dfs(i: int, j: int, grid: List[List[str]], visited: List[List[bool]], acc: List[Tuple[int, int, str]]):
    """Depth‑first search used by :func:`evaluate_grid`."""
    if (
            i < 0
            or i >= len(grid)
            or j < 0
            or j >= len(grid[0])
            or visited[i][j]
    ):
        return
    visited[i][j] = True
    acc.append((i, j, grid[i][j]))

    if grid[i][j] not in (FLOOR, AGENT):
        return

    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        _dfs(i + dx, j + dy, grid, visited, acc)


def evaluate_grid(grid_str: str) -> tuple[bool, str]:
    """
    Evaluate if a grid layout is valid and solvable for Overcooked.

    Returns a tuple of (is_valid, reason) where:
    - is_valid: True if the layout is valid and solvable, False otherwise
    - reason: A string explaining why the layout is invalid, or empty if valid
    """
    rows = grid_str.strip().split("\n")
    width = len(rows[0])

    # 1. Rectangularity
    if any(len(r) != width for r in rows):
        return False, "Grid is not rectangular"

    # 2. Required tiles present
    required = [WALL, GOAL, ONION_PILE, PLATE_PILE, POT, AGENT]
    if any(grid_str.count(c) == 0 for c in required):
        missing = [c for c in required if grid_str.count(c) == 0]
        return False, f"Missing required tiles: {', '.join(missing)}"
    if grid_str.count(AGENT) != 2:
        return False, f"Expected 2 agents, found {grid_str.count(AGENT)}"

    # 3. Proper borders (walls or interactives)
    border_ok = {WALL, GOAL, PLATE_PILE, ONION_PILE, POT}
    for y, row in enumerate(rows):
        if y in (0, len(rows) - 1) and any(ch not in border_ok for ch in row):
            return False, f"Row {y} has invalid border tiles"
        if row[0] not in border_ok or row[-1] not in border_ok:
            return False, f"Row {y} has invalid border tiles at edges"

    # 4. Interactive tiles must have at least one adjacent free tile (FLOOR or AGENT)
    grid = [list(r) for r in rows]
    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch in (AGENT, GOAL, ONION_PILE, PLATE_PILE, POT):
                if all(
                        grid[i + dx][j + dy] not in (FLOOR, AGENT)
                        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
                ):
                    return False, f"Interactive tile at ({i}, {j}) has no adjacent free tiles"

    # 5. Find all agents, onions, pots, and delivery zones
    agents = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == AGENT]
    onions = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == ONION_PILE]
    pots = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == POT]
    delivery = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == GOAL]

    def reach(start: Tuple[int, int]):
        """Perform DFS from start position and return reachable positions and tiles"""
        visited = [[False] * width for _ in rows]
        acc: List[Tuple[int, int, str]] = []
        _dfs(start[0], start[1], grid, visited, acc)
        found = {tile: False for tile in INTERACTIVE_TILES}
        for _, _, c in acc:
            if c in found:
                found[c] = True
        return acc, found, visited

    # Get reachable positions for each agent
    acc1, found1, visited1 = reach(agents[0])
    acc2, found2, visited2 = reach(agents[1])

    # Extract positions reachable by each agent
    pos1 = {(x, y) for x, y, _ in acc1}
    pos2 = {(x, y) for x, y, _ in acc2}

    # 6. Check if agents can reach onions
    # If there are multiple onions, it's fine if some are unreachable as long as at least one is reachable
    reachable_onions = []
    for onion in onions:
        i, j = onion
        if visited1[i][j] or visited2[i][j]:
            reachable_onions.append(onion)

    if not reachable_onions:
        return False, "No onions are reachable by any agent. At least one onion must be reachable."

    # 7. Check if there's a path from onion to pot and if agents can reach both
    # If there are multiple pots, it's fine if some are unreachable as long as at least one is usable
    usable_pots = []
    for pot in pots:
        pot_i, pot_j = pot
        pot_reachable = visited1[pot_i][pot_j] or visited2[pot_i][pot_j]

        if not pot_reachable:
            continue

        # Check if there's a path from any reachable onion to this pot
        for onion in reachable_onions:
            onion_i, onion_j = onion
            # If an agent can reach both onion and pot, the pot is usable
            if (visited1[onion_i][onion_j] and visited1[pot_i][pot_j]) or \
               (visited2[onion_i][onion_j] and visited2[pot_i][pot_j]):
                usable_pots.append(pot)
                break

    if not usable_pots:
        return False, "No pots can be used for cooking (no path from onion to pot). At least one pot must be usable."

    # 8. Check if there's a path from pot to delivery and if agents can reach both
    # If there are multiple delivery zones, it's fine if some are unreachable as long as at least one is usable
    usable_delivery = []
    for del_pos in delivery:
        del_i, del_j = del_pos
        del_reachable = visited1[del_i][del_j] or visited2[del_i][del_j]

        if not del_reachable:
            continue

        # Check if there's a path from any usable pot to this delivery
        for pot in usable_pots:
            pot_i, pot_j = pot
            # If an agent can reach both pot and delivery, the delivery is usable
            if (visited1[pot_i][pot_j] and visited1[del_i][del_j]) or \
               (visited2[pot_i][pot_j] and visited2[del_i][del_j]):
                usable_delivery.append(del_pos)
                break

            # Check if one agent can reach the pot and another agent can reach the delivery
            # This allows for cooperative gameplay where one agent cooks and another delivers
            elif (visited1[pot_i][pot_j] and visited2[del_i][del_j]) or \
                 (visited2[pot_i][pot_j] and visited1[del_i][del_j]):
                # Check if there's a counter that both agents can reach for handoff
                pot_agent = 1 if visited1[pot_i][pot_j] else 2
                del_agent = 1 if visited1[del_i][del_j] else 2

                # Get positions reachable by each agent
                pot_agent_pos = pos1 if pot_agent == 1 else pos2
                del_agent_pos = pos1 if del_agent == 1 else pos2

                # Look for a common floor tile that both agents can reach
                common_floor = False
                for i, row in enumerate(grid):
                    for j, ch in enumerate(row):
                        if ch == FLOOR and (i, j) in pot_agent_pos and (i, j) in del_agent_pos:
                            common_floor = True
                            break
                    if common_floor:
                        break

                if common_floor:
                    usable_delivery.append(del_pos)
                    break

    if not usable_delivery:
        return False, "No delivery zones can be used (no path from pot to delivery). At least one delivery zone must be usable."

    # 9. Check if every agent can be useful
    # First, check if agents can reach functional elements directly
    agent1_useful = any(visited1[i][j] for i, j in reachable_onions) or \
                   any(visited1[i][j] for i, j in usable_pots) or \
                   any(visited1[i][j] for i, j in usable_delivery)

    agent2_useful = any(visited2[i][j] for i, j in reachable_onions) or \
                   any(visited2[i][j] for i, j in usable_pots) or \
                   any(visited2[i][j] for i, j in usable_delivery)

    # If agents are not directly useful, check additional conditions
    if not agent1_useful:
        # Check if agent 1 is next to agent 2 (could move if agent 2 moves away)
        agent1_i, agent1_j = agents[0]
        agent2_i, agent2_j = agents[1]
        agents_adjacent = abs(agent1_i - agent2_i) + abs(agent1_j - agent2_j) == 1

        # Check if agent 1 is adjacent to a counter that can be used for handoffs
        agent1_can_handoff = False
        for i, j in [(agent1_i+dx, agent1_j+dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]:
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == WALL:
                # Check if this counter is adjacent to a position reachable by agent 2
                for ni, nj in [(i+dx, j+dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]:
                    if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and (ni, nj) in pos2:
                        agent1_can_handoff = True
                        break

                # Check if this counter is adjacent to any interactive tiles
                for ni, nj in [(i+dx, j+dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]:
                    if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and grid[ni][nj] in INTERACTIVE_TILES:
                        agent1_can_handoff = True
                        break

                if agent1_can_handoff:
                    break

        # Update agent1_useful based on additional conditions
        agent1_useful = agents_adjacent or agent1_can_handoff

    if not agent2_useful:
        # Check if agent 2 is next to agent 1 (could move if agent 1 moves away)
        agent1_i, agent1_j = agents[0]
        agent2_i, agent2_j = agents[1]
        agents_adjacent = abs(agent1_i - agent2_i) + abs(agent1_j - agent2_j) == 1

        # Check if agent 2 is adjacent to a counter that can be used for handoffs
        agent2_can_handoff = False
        for i, j in [(agent2_i+dx, agent2_j+dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]:
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == WALL:
                # Check if this counter is adjacent to a position reachable by agent 1
                for ni, nj in [(i+dx, j+dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]:
                    if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and (ni, nj) in pos1:
                        agent2_can_handoff = True
                        break

                # Check if this counter is adjacent to any interactive tiles
                for ni, nj in [(i+dx, j+dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]:
                    if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and grid[ni][nj] in INTERACTIVE_TILES:
                        agent2_can_handoff = True
                        break

                if agent2_can_handoff:
                    break

        # Update agent2_useful based on additional conditions
        agent2_useful = agents_adjacent or agent2_can_handoff

    if not agent1_useful:
        return False, "Agent 1 cannot be useful (can't reach any functional elements or interact with agent 2)"

    if not agent2_useful:
        return False, "Agent 2 cannot be useful (can't reach any functional elements or interact with agent 1)"

    # 10. Check if agents can cooperate through a counter if they can't reach everything
    if all(found1.values()) and all(found2.values()):
        return True, ""  # both can independently reach everything

    # Otherwise at least one agent must be able to reach every interactive tile
    combined_reach = {k: found1[k] or found2[k] for k in found1}
    if not all(combined_reach.values()):
        return False, "Not all interactive tiles can be reached by any agent"

    # Look for a wall adjacent to both reachable regions (a potential hand‑off)
    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch == WALL:
                adj_to_1 = any((i + dx, j + dy) in pos1 for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)))
                adj_to_2 = any((i + dx, j + dy) in pos2 for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)))
                if adj_to_1 and adj_to_2:
                    return True, ""  # Found a counter for passing items

    return False, "Agents cannot cooperate (no shared counter for passing items)"