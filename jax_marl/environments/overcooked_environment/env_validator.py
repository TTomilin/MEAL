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

    # Only continue DFS through floor or agent tiles
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

    # Convert to grid of characters for easier manipulation
    grid = [list(r) for r in rows]
    height, width = len(grid), len(grid[0])

    # 4. Interactive tiles must have at least one adjacent free tile (FLOOR or AGENT)
    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch in (GOAL, ONION_PILE, PLATE_PILE, POT):
                if all(
                        i+dx < 0 or i+dx >= height or j+dy < 0 or j+dy >= width or
                        grid[i+dx][j+dy] not in (FLOOR, AGENT)
                        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
                ):
                    return False, f"Interactive tile at ({i}, {j}) has no adjacent free tiles"

    # 5. Find all agents, onions, pots, plates, and delivery zones
    agents = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == AGENT]
    onions = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == ONION_PILE]
    pots = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == POT]
    plates = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == PLATE_PILE]
    delivery = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == GOAL]

    # Helper function to check if positions are adjacent
    def is_adjacent(pos1, pos2):
        i1, j1 = pos1
        i2, j2 = pos2
        return abs(i1 - i2) + abs(j1 - j2) == 1

    # Compute reachable positions for each agent
    def get_reachable_tiles(start_i, start_j):
        visited = [[False for _ in range(width)] for _ in range(height)]
        reachable_positions = set()
        
        def dfs(i, j):
            if (i < 0 or i >= height or j < 0 or j >= width or 
                visited[i][j] or grid[i][j] in UNPASSABLE_TILES):
                return
            
            visited[i][j] = True
            reachable_positions.add((i, j))
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(i + di, j + dj)
        
        # Start DFS from agent position
        dfs(start_i, start_j)
        
        # Also mark interactive tiles adjacent to reachable positions as "reachable"
        # This represents tiles the agent can interact with
        interactive_reachable = set()
        for i, j in reachable_positions:
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < height and 0 <= nj < width and 
                    grid[ni][nj] in INTERACTIVE_TILES):
                    interactive_reachable.add((ni, nj))
        
        return reachable_positions, interactive_reachable, visited
    
    # Get reachable areas for each agent
    agent1_i, agent1_j = agents[0]
    agent2_i, agent2_j = agents[1]
    
    # Get positions reachable by each agent
    agent1_reachable, agent1_interactive, agent1_visited = get_reachable_tiles(agent1_i, agent1_j)
    agent2_reachable, agent2_interactive, agent2_visited = get_reachable_tiles(agent2_i, agent2_j)
    
    # Check if agents can cooperate via counters
    def can_agents_cooperate():
        # Check if agents are adjacent (can directly pass items)
        if is_adjacent(agents[0], agents[1]):
            return True
            
        # Check if agents can cooperate through a counter (wall)
        # Find walls that both agents can interact with
        for i, row in enumerate(grid):
            for j, ch in enumerate(row):
                if ch == WALL:
                    # Check if wall is adjacent to areas reachable by both agents
                    adjacent_to_agent1 = any(
                        is_adjacent((i, j), (ri, rj)) for ri, rj in agent1_reachable
                    )
                    adjacent_to_agent2 = any(
                        is_adjacent((i, j), (ri, rj)) for ri, rj in agent2_reachable
                    )
                    if adjacent_to_agent1 and adjacent_to_agent2:
                        return True
        
        # Check if agents have a common floor tile they can both reach
        common_floor = agent1_reachable.intersection(agent2_reachable)
        if common_floor:
            return True
            
        return False
    
    # Check if onions are reachable
    reachable_onions = []
    for onion_i, onion_j in onions:
        # Onion is reachable if either agent can interact with it
        if ((onion_i, onion_j) in agent1_interactive or 
            (onion_i, onion_j) in agent2_interactive):
            reachable_onions.append((onion_i, onion_j))
    
    if not reachable_onions:
        return False, "No onions are reachable by any agent"
    
    # Check if plates are reachable
    reachable_plates = []
    for plate_i, plate_j in plates:
        # Plate is reachable if either agent can interact with it
        if ((plate_i, plate_j) in agent1_interactive or 
            (plate_i, plate_j) in agent2_interactive):
            reachable_plates.append((plate_i, plate_j))
    
    if not reachable_plates:
        return False, "No plates are reachable by any agent"
    
    # Check if pots are reachable
    reachable_pots = []
    for pot_i, pot_j in pots:
        # Pot is reachable if either agent can interact with it
        if ((pot_i, pot_j) in agent1_interactive or 
            (pot_i, pot_j) in agent2_interactive):
            reachable_pots.append((pot_i, pot_j))
    
    if not reachable_pots:
        return False, "No pots are reachable by any agent"
    
    # Check if delivery points are reachable
    reachable_delivery = []
    for del_i, del_j in delivery:
        # Delivery is reachable if either agent can interact with it
        if ((del_i, del_j) in agent1_interactive or 
            (del_i, del_j) in agent2_interactive):
            reachable_delivery.append((del_i, del_j))
    
    if not reachable_delivery:
        return False, "No delivery points are reachable by any agent"
    
    # Check if both agents can be useful
    agent1_useful = bool(
        agent1_interactive.intersection(set(onions + pots + plates + delivery))
    )
    agent2_useful = bool(
        agent2_interactive.intersection(set(onions + pots + plates + delivery))
    )
    
    # If either agent cannot interact with any functional element, it's not useful
    if not agent1_useful and not agent2_useful:
        return False, "Neither agent can be useful (can't reach any functional elements)"
    
    if not agent1_useful and not is_adjacent(agents[0], agents[1]):
        return False, "Agent 1 cannot be useful (can't reach any functional elements or interact with agent 2)"
    
    if not agent2_useful and not is_adjacent(agents[0], agents[1]):
        return False, "Agent 2 cannot be useful (can't reach any functional elements or interact with agent 1)"
    
    # Check if there's a valid path from onion to pot
    onion_to_pot_path = False
    
    # Case 1: Single agent can reach both onion and pot
    if (any(o in agent1_interactive for o in onions) and 
        any(p in agent1_interactive for p in pots)):
        onion_to_pot_path = True
    
    if (any(o in agent2_interactive for o in onions) and 
        any(p in agent2_interactive for p in pots)):
        onion_to_pot_path = True
    
    # Case 2: Agents can cooperate - one gets onion, one uses pot
    if not onion_to_pot_path:
        if (any(o in agent1_interactive for o in onions) and 
            any(p in agent2_interactive for p in pots) and 
            can_agents_cooperate()):
            onion_to_pot_path = True
        
        if (any(o in agent2_interactive for o in onions) and 
            any(p in agent1_interactive for p in pots) and 
            can_agents_cooperate()):
            onion_to_pot_path = True
    
    if not onion_to_pot_path:
        return False, "No path from onion to pot. At least one viable path must exist."
    
    # Check if there's a valid path from pot to delivery
    pot_to_delivery_path = False
    
    # Case 1: Single agent can reach both pot and delivery
    if (any(p in agent1_interactive for p in pots) and 
        any(d in agent1_interactive for d in delivery)):
        pot_to_delivery_path = True
    
    if (any(p in agent2_interactive for p in pots) and 
        any(d in agent2_interactive for d in delivery)):
        pot_to_delivery_path = True
        
    # Case 2: Agents can cooperate - one uses pot, one delivers
    if not pot_to_delivery_path:
        if (any(p in agent1_interactive for p in pots) and 
            any(d in agent2_interactive for d in delivery) and 
            can_agents_cooperate()):
            pot_to_delivery_path = True
            
        if (any(p in agent2_interactive for p in pots) and 
            any(d in agent1_interactive for d in delivery) and 
            can_agents_cooperate()):
            pot_to_delivery_path = True
    
    if not pot_to_delivery_path:
        return False, "No path from pot to delivery. At least one viable path must exist."
    
    # If we've passed all checks, the layout is valid
    return True, ""