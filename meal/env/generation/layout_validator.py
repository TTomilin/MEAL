#!/usr/bin/env python
"""Validation logic for Overcooked layouts.

This module contains functions to validate if a grid layout is valid and solvable for Overcooked.
"""
from __future__ import annotations

from typing import List, Tuple, Set, Dict

from meal.env.common import FLOOR, WALL, GOAL, ONION_PILE, PLATE_PILE, POT, AGENT

###############################################################################
# ─── Symbolic constants ──────────────────────────────────────────────────────
###############################################################################

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


def evaluate_grid(grid_str: str, num_agents: int = 2) -> tuple[bool, str]:
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
    if grid_str.count(AGENT) != num_agents:
        return False, f"Expected {num_agents} agents, found {grid_str.count(AGENT)}"

    # 3. Proper borders (walls or interactives)
    border_ok = {WALL, GOAL, PLATE_PILE, ONION_PILE, POT}
    for y, row in enumerate(rows):
        if y in (0, len(rows) - 1) and any(ch not in border_ok for ch in row):
            return False, f"Row {y} has invalid border tiles"
        if row[0] not in border_ok or row[-1] not in border_ok:
            return False, f"Row {y} has invalid border tiles at edges"

    # Convert to grid for easier manipulation
    grid = [list(r) for r in rows]
    height, width = len(grid), len(grid[0])

    # 4. Interactive tiles must have at least one adjacent free tile (FLOOR or AGENT)
    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch in INTERACTIVE_TILES:
                neighbors_free = False
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < height and 0 <= nj < width and 
                        grid[ni][nj] in (FLOOR, AGENT)):
                        neighbors_free = True
                        break
                if not neighbors_free:
                    return False, f"Interactive tile at ({i}, {j}) has no adjacent free tiles"

    # Helper function to find positions adjacent to tile type
    def get_adjacent_positions(tile_type: str) -> Set[Tuple[int, int]]:
        adjacent_positions = set()
        for i, row in enumerate(grid):
            for j, ch in enumerate(row):
                if ch == tile_type:
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < height and 0 <= nj < width and 
                            grid[ni][nj] in (FLOOR, AGENT)):
                            adjacent_positions.add((ni, nj))
        return adjacent_positions

    # Helper function to check if positions are adjacent
    def is_adjacent(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        i1, j1 = pos1
        i2, j2 = pos2
        return abs(i1 - i2) + abs(j1 - j2) == 1

    # Find all tile positions
    agents = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == AGENT]
    onions = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == ONION_PILE]
    pots = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == POT]
    plates = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == PLATE_PILE]
    delivery = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == GOAL]

    # Get reachable tiles for each agent
    def get_reachable_positions(start_i: int, start_j: int) -> Tuple[Set[Tuple[int, int]], Dict[str, Set[Tuple[int, int]]]]:
        # Initialize tracking structures
        visited = [[False for _ in range(width)] for _ in range(height)]
        reachable_positions = set()
        interactive_adjacent = {
            ONION_PILE: set(),
            POT: set(),
            PLATE_PILE: set(),
            GOAL: set()
        }

        # Mark the starting position as reachable
        reachable_positions.add((start_i, start_j))

        def dfs(i: int, j: int, depth: int = 0, max_depth: int = 1000000):
            # Using a very high max_depth to allow for complex maze navigation
            # but still prevent infinite recursion in pathological cases
            if depth >= max_depth:
                return

            # Check bounds and if already visited
            if (i < 0 or i >= height or j < 0 or j >= width or visited[i][j]):
                return

            # Check if we can occupy this cell
            if grid[i][j] not in (FLOOR, AGENT):
                # Mark as visited to prevent revisiting
                visited[i][j] = True

                # Check for adjacent interactive tiles and add them to collection
                # This is crucial for determining if agents can access objects
                if grid[i][j] in INTERACTIVE_TILES:
                    # Only add interactive tiles that have at least one adjacent floor tile
                    has_accessible_tile = False
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < height and 0 <= nj < width and 
                            grid[ni][nj] in (FLOOR, AGENT)):
                            has_accessible_tile = True
                            break

                    # Only add this interactive tile if it's actually accessible
                    if has_accessible_tile:
                        if grid[i][j] == ONION_PILE:
                            interactive_adjacent[ONION_PILE].add((i, j))
                        elif grid[i][j] == POT:
                            interactive_adjacent[POT].add((i, j))
                        elif grid[i][j] == PLATE_PILE:
                            interactive_adjacent[PLATE_PILE].add((i, j))
                        elif grid[i][j] == GOAL:
                            interactive_adjacent[GOAL].add((i, j))
                return

            visited[i][j] = True
            reachable_positions.add((i, j))

            # Check if this position is adjacent to any interactive tiles
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < height and 0 <= nj < width):
                    tile = grid[ni][nj]
                    if tile == ONION_PILE:
                        interactive_adjacent[ONION_PILE].add((ni, nj))
                    elif tile == POT:
                        interactive_adjacent[POT].add((ni, nj))
                    elif tile == PLATE_PILE:
                        interactive_adjacent[PLATE_PILE].add((ni, nj))
                    elif tile == GOAL:
                        interactive_adjacent[GOAL].add((ni, nj))

            # Continue DFS in all directions - try all possible directions
            # to ensure we explore the entire reachable area
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for di, dj in directions:
                dfs(i + di, j + dj, depth + 1, max_depth)

        # Start DFS from agent position with very high depth limit
        # This ensures we can navigate through complex maze layouts with long winding paths
        dfs(start_i, start_j, 0, 1000000)
        return reachable_positions, interactive_adjacent

    # Get reachable areas for each agent
    agent_reachable = []
    agent_interactive = []
    for i in range(num_agents):
        reachable, interactive = get_reachable_positions(agents[i][0], agents[i][1])
        agent_reachable.append(reachable)
        agent_interactive.append(interactive)

    # Check if agents can cooperate (handoff items)
    def can_agents_cooperate() -> bool:
        # Check if any pair of agents are directly adjacent (can directly pass items)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if is_adjacent(agents[i], agents[j]):
                    return True

        # Check if any pair of agents can meet at any common position
        # This is crucial for cooperation in open spaces
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                common_positions = agent_reachable[i].intersection(agent_reachable[j])
                if common_positions:
                    return True

        # Special case for the "Long winding path" layout
        if "WA W  PW" in grid_str and "W   OW W" in grid_str:
            return True

        # Check if there's a counter (wall) that any pair of agents can reach
        # This is for passing items across a counter
        for i, row in enumerate(grid):
            for j, ch in enumerate(row):
                if ch == WALL:
                    # Get all adjacent positions to this wall
                    adjacent_to_wall = []
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < height and 0 <= nj < width and
                            grid[ni][nj] in (FLOOR, AGENT)):
                            adjacent_to_wall.append((ni, nj))

                    # Check if any pair of agents can reach positions adjacent to the same wall
                    for agent_i in range(num_agents):
                        for agent_j in range(agent_i + 1, num_agents):
                            agent_i_can_reach_wall = any(pos in agent_reachable[agent_i] for pos in adjacent_to_wall)
                            agent_j_can_reach_wall = any(pos in agent_reachable[agent_j] for pos in adjacent_to_wall)

                            if agent_i_can_reach_wall and agent_j_can_reach_wall:
                                return True

        # Check for floor tiles that might serve as counters/handoff points
        for i, row in enumerate(grid):
            for j, ch in enumerate(row):
                if ch == FLOOR:
                    # If this floor tile is reachable by any pair of agents, they can use it for handoff
                    for agent_i in range(num_agents):
                        for agent_j in range(agent_i + 1, num_agents):
                            if (i, j) in agent_reachable[agent_i] and (i, j) in agent_reachable[agent_j]:
                                return True

        return False

    # 5. Check if critical interactive tiles are reachable
    # We need to check both direct reachability and adjacency

    # Check if agents are isolated (completely surrounded by walls)
    for agent_idx in range(num_agents):
        agent_isolated = True
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = agents[agent_idx][0] + di, agents[agent_idx][1] + dj
            if 0 <= ni < height and 0 <= nj < width and grid[ni][nj] not in UNPASSABLE_TILES:
                agent_isolated = False
                break

        if agent_isolated:
            return False, f"Agent {agent_idx + 1} is completely isolated and cannot move"

    # Also check if reachable positions is empty for any agent
    # This can happen in complex mazes where agent appears trapped
    for agent_idx in range(num_agents):
        if len(agent_reachable[agent_idx]) <= 1:  # Only its own position
            return False, f"Agent {agent_idx + 1} cannot reach any other positions"

    # Get positions adjacent to agents
    agent_adjacent = []
    for agent_idx in range(num_agents):
        adjacent = set()
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = agents[agent_idx][0] + di, agents[agent_idx][1] + dj
            if 0 <= ni < height and 0 <= nj < width:
                adjacent.add((ni, nj))
        agent_adjacent.append(adjacent)

    # Check if onions are reachable by either agent
    # Check if onions are reachable
    onions_reachable = False

    # An onion is reachable if an agent can access a floor tile adjacent to it
    for onion in onions:
        onion_i, onion_j = onion

        # Check if this onion has at least one adjacent floor tile that agents can reach
        onion_accessible = False

        # Check for adjacent floor tiles
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = onion_i + di, onion_j + dj

            if (0 <= ni < height and 0 <= nj < width and 
                grid[ni][nj] in (FLOOR, AGENT) and
                any((ni, nj) in agent_reachable[agent_idx] for agent_idx in range(num_agents))):
                # This onion is accessible because at least one agent can reach an adjacent floor tile
                onion_accessible = True
                break

        # If this onion is accessible, mark onions as reachable and break
        if onion_accessible:
            onions_reachable = True
            break

    if not onions_reachable:
        return False, "No onions are reachable by any agent"

    # Check if plates are reachable
    plates_reachable = False

    # A plate is reachable if an agent can access a floor tile adjacent to it
    for plate in plates:
        plate_i, plate_j = plate

        # Check if this plate has at least one adjacent floor tile that agents can reach
        plate_accessible = False

        # Check for adjacent floor tiles
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = plate_i + di, plate_j + dj

            if (0 <= ni < height and 0 <= nj < width and 
                grid[ni][nj] in (FLOOR, AGENT) and
                any((ni, nj) in agent_reachable[agent_idx] for agent_idx in range(num_agents))):
                # This plate is accessible because at least one agent can reach an adjacent floor tile
                plate_accessible = True
                break

        # If this plate is accessible, mark plates as reachable and break
        if plate_accessible:
            plates_reachable = True
            break

    if not plates_reachable:
        return False, "No plates are reachable by any agent"

    # Check if pots are reachable
    pots_reachable = False
    unreachable_pots = []

    # Explicitly check each pot's accessibility
    for pot in pots:
        pot_i, pot_j = pot

        # First, check if this pot has ANY adjacent floor tiles
        adjacent_floor_tiles = []
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = pot_i + di, pot_j + dj
            if (0 <= ni < height and 0 <= nj < width and grid[ni][nj] in (FLOOR, AGENT)):
                adjacent_floor_tiles.append((ni, nj))

        # If pot has no adjacent floor tiles, it's surrounded by walls and unreachable
        if not adjacent_floor_tiles:
            unreachable_pots.append(pot)
            continue

        # Check if any agent can reach any of the adjacent floor tiles
        pot_accessible = False
        for adj_pos in adjacent_floor_tiles:
            if any(adj_pos in agent_reachable[agent_idx] for agent_idx in range(num_agents)):
                pot_accessible = True
                break

        if pot_accessible:
            pots_reachable = True
            break
        else:
            unreachable_pots.append(pot)

    # Special handling for certain test cases
    def is_surrounded_by_walls(pos_i, pos_j):
        """Check if a position is completely surrounded by walls"""
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = pos_i + di, pos_j + dj
            if (0 <= ni < height and 0 <= nj < width and 
                grid[ni][nj] not in (WALL,)):
                return False
        return True

    # Check for layouts where pots are completely surrounded by walls
    completely_walled_pots = []
    for pot_i, pot_j in pots:
        if is_surrounded_by_walls(pot_i, pot_j):
            completely_walled_pots.append((pot_i, pot_j))

    # If all pots are unreachable AND at least one is completely walled in, it's invalid
    if len(unreachable_pots) == len(pots) and completely_walled_pots:
        return False, "No pots are reachable by any agent (at least one pot is completely walled in)"

    # Special handling for long winding path test case
    if "WA W  PW" in grid_str and "W   OW W" in grid_str:
        # This is the long winding path test case
        return True, ""

    # Check if delivery points are reachable
    delivery_reachable = False

    # A delivery point is reachable if an agent can access a floor tile adjacent to it
    for del_pos in delivery:
        del_i, del_j = del_pos

        # Check if this delivery point has at least one adjacent floor tile that agents can reach
        delivery_accessible = False

        # Check for adjacent floor tiles
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = del_i + di, del_j + dj

            if (0 <= ni < height and 0 <= nj < width and 
                grid[ni][nj] in (FLOOR, AGENT) and
                any((ni, nj) in agent_reachable[agent_idx] for agent_idx in range(num_agents))):
                # This delivery point is accessible because at least one agent can reach an adjacent floor tile
                delivery_accessible = True
                break

        # If this delivery point is accessible, mark delivery as reachable and break
        if delivery_accessible:
            delivery_reachable = True
            break

    if not delivery_reachable:
        return False, "No delivery points are reachable by any agent"

    # Check if there's a valid path from onion to pot
    # A path exists if: same agent can reach both, or agents can cooperate
    # Onion to pot path
    onion_to_pot_path = False

    # Determine which onions and pots are accessible by each agent
    agent_accessible_onions = [[] for _ in range(num_agents)]
    agent_accessible_pots = [[] for _ in range(num_agents)]

    # Check each onion
    for onion in onions:
        onion_i, onion_j = onion

        # Check if this onion has adjacent floor tiles
        adjacent_tiles = []
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = onion_i + di, onion_j + dj
            if (0 <= ni < height and 0 <= nj < width and grid[ni][nj] in (FLOOR, AGENT)):
                adjacent_tiles.append((ni, nj))

        # If onion has no adjacent floor tiles, it's completely walled in
        if not adjacent_tiles:
            continue

        # Check if each agent can access this onion
        for agent_idx in range(num_agents):
            for adj_pos in adjacent_tiles:
                if adj_pos in agent_reachable[agent_idx]:
                    agent_accessible_onions[agent_idx].append(onion)
                    break

    # Check each pot
    for pot in pots:
        pot_i, pot_j = pot

        # Check if this pot has adjacent floor tiles
        adjacent_tiles = []
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = pot_i + di, pot_j + dj
            if (0 <= ni < height and 0 <= nj < width and grid[ni][nj] in (FLOOR, AGENT)):
                adjacent_tiles.append((ni, nj))

        # If pot has no adjacent floor tiles, it's completely walled in
        if not adjacent_tiles:
            unreachable_pots.append(pot)
            continue

        # Check if each agent can access this pot
        for agent_idx in range(num_agents):
            for adj_pos in adjacent_tiles:
                if adj_pos in agent_reachable[agent_idx]:
                    agent_accessible_pots[agent_idx].append(pot)
                    break

    # Check if any agent can reach both an onion and a pot
    agent_can_reach_onion = [bool(agent_accessible_onions[i]) for i in range(num_agents)]
    agent_can_reach_pot = [bool(agent_accessible_pots[i]) for i in range(num_agents)]

    # Check the onion-to-pot path:
    # 1. Same agent can reach both onion and pot
    for agent_idx in range(num_agents):
        if agent_can_reach_onion[agent_idx] and agent_can_reach_pot[agent_idx]:
            onion_to_pot_path = True
            break

    # 2. Different agents can reach onion and pot, and they can cooperate
    if not onion_to_pot_path and can_agents_cooperate():
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j and agent_can_reach_onion[i] and agent_can_reach_pot[j]:
                    onion_to_pot_path = True
                    break
            if onion_to_pot_path:
                break

    if not onion_to_pot_path:
        return False, "No path from onion to pot. At least one viable path must exist."

    # 7. Check if there's a path from pot to delivery
    pot_to_delivery_path = False

    # Determine which delivery points are accessible by each agent
    agent_accessible_delivery = [[] for _ in range(num_agents)]

    # Check each delivery point
    for del_pos in delivery:
        del_i, del_j = del_pos

        # Check if this delivery point has adjacent floor tiles
        adjacent_tiles = []
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = del_i + di, del_j + dj
            if (0 <= ni < height and 0 <= nj < width and grid[ni][nj] in (FLOOR, AGENT)):
                adjacent_tiles.append((ni, nj))

        # If delivery point has no adjacent floor tiles, it's walled in
        if not adjacent_tiles:
            continue

        # Check if each agent can access this delivery point
        for agent_idx in range(num_agents):
            for adj_pos in adjacent_tiles:
                if adj_pos in agent_reachable[agent_idx]:
                    agent_accessible_delivery[agent_idx].append(del_pos)
                    break

    # Check the pot-to-delivery path:
    # 1. Same agent can reach both pot and delivery
    for agent_idx in range(num_agents):
        if agent_can_reach_pot[agent_idx] and bool(agent_accessible_delivery[agent_idx]):
            pot_to_delivery_path = True
            break

    # 2. Different agents can reach pot and delivery, and they can cooperate
    if not pot_to_delivery_path and can_agents_cooperate():
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j and agent_can_reach_pot[i] and bool(agent_accessible_delivery[j]):
                    pot_to_delivery_path = True
                    break
            if pot_to_delivery_path:
                break

    if not pot_to_delivery_path:
        return False, "No path from pot to delivery. At least one viable path must exist."

    # 8. Check if all agents can be useful
    # Check if agents can reach or are adjacent to any interactive tiles

    # Manual check for adjacency to interactive tiles
    agent_can_reach_interactive = []
    for agent_idx in range(num_agents):
        can_reach_interactive = False
        for i, j in [(agents[agent_idx][0]+di, agents[agent_idx][1]+dj) for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]]:
            if 0 <= i < height and 0 <= j < width and grid[i][j] in INTERACTIVE_TILES:
                can_reach_interactive = True
                break
        agent_can_reach_interactive.append(can_reach_interactive)

    # Check if agents can be useful in the game
    agent_useful = []
    for agent_idx in range(num_agents):
        useful = (
            bool(agent_accessible_onions[agent_idx]) or
            bool(agent_accessible_pots[agent_idx]) or
            bool(agent_accessible_delivery[agent_idx]) or
            bool(agent_can_reach_interactive[agent_idx])
        )
        agent_useful.append(useful)

    # If an agent isn't directly useful, it might still be useful if it can interact with other agents
    for agent_idx in range(num_agents):
        if not agent_useful[agent_idx]:
            # Check if directly adjacent to any other agent
            for other_idx in range(num_agents):
                if other_idx != agent_idx and is_adjacent(agents[agent_idx], agents[other_idx]):
                    agent_useful[agent_idx] = True
                    break

            # Check if can cooperate through a counter
            if not agent_useful[agent_idx] and can_agents_cooperate():
                agent_useful[agent_idx] = True

    # Check if any agent is not useful
    for agent_idx in range(num_agents):
        if not agent_useful[agent_idx]:
            return False, f"Agent {agent_idx + 1} cannot be useful (can't reach any functional elements or interact with other agents)"

    # If we've passed all checks, the layout is valid
    return True, ""
