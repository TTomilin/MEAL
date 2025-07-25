"""
Centralized configuration for difficulty parameters used across the project.

This module defines the difficulty settings for environment generation,
including height/width ranges and wall density for each difficulty level.
"""

from typing import Dict, Tuple


# Difficulty parameters for environment generation
DIFFICULTY_PARAMS = {
    "easy": {
        "height_min": 6,
        "height_max": 7,
        "width_min": 6,
        "width_max": 7,
        "wall_density": 0.15,
        "height_rng": (6, 7),
        "width_rng": (6, 7),
    },
    "medium": {
        "height_min": 8,
        "height_max": 9,
        "width_min": 8,
        "width_max": 9,
        "wall_density": 0.25,
        "height_rng": (8, 9),
        "width_rng": (8, 9),
    },
    "hard": {
        "height_min": 10,
        "height_max": 11,
        "width_min": 10,
        "width_max": 11,
        "wall_density": 0.35,
        "height_rng": (10, 11),
        "width_rng": (10, 11),
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


def apply_difficulty_to_config(config, difficulty: str) -> None:
    """
    Apply difficulty parameters to a configuration object.
    
    Args:
        config: Configuration object to modify
        difficulty: The difficulty level to apply
    """
    params = get_difficulty_params(difficulty)
    
    config.height_min = params["height_min"]
    config.height_max = params["height_max"]
    config.width_min = params["width_min"]
    config.width_max = params["width_max"]
    config.wall_density = params["wall_density"]