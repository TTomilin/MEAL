from meal.env import Overcooked, OvercookedPO
from meal.env.generation.layout_generator import generate_random_layout
from meal.env.generation.sequence_loader import create_sequence
from meal.env.layouts.presets import overcooked_layouts
from meal.env.utils.difficulty_config import DIFFICULTY_PARAMS

# Environment registry
registered_envs = ["overcooked", "overcooked_po"]


# Gym-style API
def make_env(env_id: str, **env_kwargs):
    """
    Create an environment following gym conventions.

    Args:
        env_id: Environment identifier (e.g., 'overcooked', 'overcooked_po')
        **env_kwargs: Additional environment arguments

    Returns:
        Environment instance

    Example:
        >>> import meal
        >>> env = meal.make_env('overcooked')
    """
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered.")
    cls = {"overcooked": Overcooked, "overcooked_po": OvercookedPO}[env_id]
    return cls(**env_kwargs)


def make_sequence(
        env_id: str = "overcooked",
        sequence_length: int = 10,
        strategy: str = "generate",
        num_agents: int = 2,
        difficulty: str = None,
        **env_kwargs
):
    """
    Generate a continual learning sequence of environments with full parameter support.

    Args:
        env_id: Base environment identifier
        sequence_length: Number of environments in the sequence
        strategy: Generation strategy ('random', 'ordered', 'generate', 'curriculum')
        num_agents: Number of agents in each environment
        difficulty: Difficulty level ('easy', 'medium', 'hard') - determines layout and view parameters
        **env_kwargs: Additional environment arguments

    Returns:
        List of environment instances for continual learning

    Example:
        >>> import meal
        >>> envs = meal.make_sequence(sequence_length=6, strategy='curriculum', num_agents=2)
        >>> # Returns 6 environments with increasing difficulty
        >>> 
        >>> envs = meal.make_sequence(sequence_length=3, difficulty='hard', num_agents=2)
        >>> # Returns 3 environments with hard difficulty parameters
    """

    # Generate sequence of environment configurations
    return create_sequence(
        env_id=env_id,
        sequence_length=sequence_length,
        strategy=strategy,
        num_agents=num_agents,
        difficulty=difficulty,
        **env_kwargs
    )


__all__ = ["make_env", "make_sequence", "registered_envs"]
