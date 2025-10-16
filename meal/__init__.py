from meal.registration import make, registered_envs
from meal.env.generation.sequence_loader import generate_sequence

# Gym-style API
def make_env(env_id: str, **env_kwargs):
    """
    Create an environment following gym conventions.

    Args:
        env_id: Environment identifier (e.g., 'overcooked', 'overcooked_po', 'overcooked_legacy')
        **env_kwargs: Additional environment arguments

    Returns:
        Environment instance

    Example:
        >>> import meal
        >>> env = meal.make_env('overcooked')
    """
    return make(env_id, **env_kwargs)


def make_sequence(
    sequence_length: int = 10,
    strategy: str = "generate",
    env_id: str = "overcooked",
    seed: int = None,
    **env_kwargs
):
    """
    Generate a continual learning sequence of environments.

    Args:
        sequence_length: Number of environments in the sequence
        strategy: Generation strategy ('random', 'ordered', 'generate', 'curriculum')
        env_id: Base environment identifier
        seed: Random seed for reproducibility
        **env_kwargs: Additional environment arguments

    Returns:
        List of environment instances for continual learning

    Strategies:
        - 'random': Sample from fixed layouts (no immediate repeats)
        - 'ordered': Deterministic slice through fixed layouts
        - 'generate': Create brand-new solvable kitchens on the fly
        - 'curriculum': Split tasks equally across difficulty levels (easy -> medium -> hard)

    Example:
        >>> import meal
        >>> envs = meal.make_sequence(sequence_length=6, strategy='curriculum')
        >>> # Returns 6 environments with increasing difficulty
    """
    env_kwargs_list, names = generate_sequence(
        sequence_length=sequence_length,
        strategy=strategy,
        seed=seed,
        **env_kwargs
    )

    # Create environment instances
    envs = []
    for i, kwargs in enumerate(env_kwargs_list):
        env = make(env_id, **kwargs)
        env.task_id = i
        env.task_name = names[i]
        envs.append(env)

    return envs


def list_envs():
    """
    List all available environment IDs.

    Returns:
        List of registered environment identifiers

    Example:
        >>> import meal
        >>> meal.list_envs()
        ['overcooked', 'overcooked_single', 'overcooked_po', 'overcooked_n_agent']
    """
    return registered_envs.copy()


__all__ = [
    "make", 
    "make_env",
    "make_sequence",
    "list_envs",
    "registered_envs"
]
__version__ = "0.1.0"
