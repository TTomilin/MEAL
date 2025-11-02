"""
Centralized configuration for difficulty parameters used across the project.

This module defines the difficulty settings for environment generation,
including height/width ranges and wall density for each difficulty level.
"""

from typing import Dict

# Difficulty parameters for environment generation
DIFFICULTY_PARAMS = {
    "easy": {
        "height": 7,
        "width": 7,
        "wall_density": 0.15,
        "num_stations": 1,
        "num_pots": 2,
        "view_ahead": 1,
        "view_sides": 1,
        "view_behind": 0,
    },
    "medium": {
        "height": 9,
        "width": 9,
        "wall_density": 0.25,
        "num_stations": 1,
        "num_pots": 3,
        "view_ahead": 2,
        "view_sides": 1,
        "view_behind": 0,
    },
    "hard": {
        "height": 11,
        "width": 11,
        "wall_density": 0.35,
        "num_stations": 2,
        "num_pots": 4,
        "view_ahead": 3,
        "view_sides": 1,
        "view_behind": 1,
    }
}


def get_difficulty_params(difficulty: str) -> Dict:
    """
    Get difficulty parameters for a given difficulty level.

    Args:
        difficulty: The difficulty level ("easy", "medium", "hard")

    Returns:
        Dictionary containing the parameters for the specified difficulty

    Raises:
        ValueError: If the difficulty level is not recognized
    """
    difficulty_lower = difficulty.lower()

    # Handle alternative names
    if difficulty_lower == "med":
        difficulty_lower = "medium"

    if difficulty_lower not in DIFFICULTY_PARAMS:
        raise ValueError(f"Unknown difficulty level: {difficulty}. "
                         f"Available levels: {list(DIFFICULTY_PARAMS.keys())}")

    return DIFFICULTY_PARAMS[difficulty_lower].copy()
