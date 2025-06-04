import random
from typing import List, Dict, Any, Sequence, Tuple

from jax_marl.environments.overcooked_environment import hard_layouts, medium_layouts, easy_layouts, overcooked_layouts, \
    single_layouts
from jax_marl.environments.overcooked_environment.env_generator import generate_random_layout


def _resolve_pool(names: Sequence[str] | None) -> List[str]:
    """Turn user‐supplied `layout_names` into a concrete list of keys."""
    presets = {
        "hard_levels": list(hard_layouts),
        "medium_levels": list(medium_layouts),
        "easy_levels": list(easy_layouts),
        "single_levels": list(single_layouts),
    }

    if not names:  # None, [] or other falsy → all layouts
        return list(overcooked_layouts)

    if len(names) == 1 and names[0] in presets:  # the special “_levels” tokens
        return presets[names[0]]

    return list(names)  # custom list from caller


def _random_no_repeat(pool: List[str], k: int) -> List[str]:
    """Sample `k` items, allowing duplicates but never back-to-back repeats."""
    if k <= len(pool):
        return random.sample(pool, k)

    out, last = [], None
    for _ in range(k):
        choice = random.choice([x for x in pool if x != last] or pool)
        out.append(choice)
        last = choice
    return out


def generate_sequence(
        sequence_length: int | None = None,
        strategy: str = "random",
        layout_names: Sequence[str] | None = None,
        seed: int | None = None,
        num_agents: int = 2,
        height_rng: Tuple[int, int] = (5, 10),
        width_rng: Tuple[int, int] = (5, 10),
        wall_density: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Return a list of `env_kwargs` (what you feed to Overcooked) and
    a parallel list of pretty names.

    strategies
    ----------
    random   – sample from fixed layouts (no immediate repeats if len>pool)
    ordered  – deterministic slice through fixed layouts
    generate – create brand-new solvable kitchens on the fly
    """
    if seed is not None:
        random.seed(seed)

    pool = _resolve_pool(layout_names)
    if sequence_length is None:
        sequence_length = len(pool)

    env_kwargs: List[Dict[str, Any]] = []
    names: List[str] = []

    # ----------------------------------------------------------------– strategy
    if strategy == "random":
        selected = _random_no_repeat(pool, sequence_length)
        env_kwargs = [{"layout": overcooked_layouts[name]} for name in selected]
        names = selected

    elif strategy == "ordered":
        if sequence_length > len(pool):
            raise ValueError("ordered requires seq_length ≤ available layouts")
        selected = pool[:sequence_length]
        env_kwargs = [{"layout": overcooked_layouts[name]} for name in selected]
        names = selected

    elif strategy == "generate":
        base = seed if seed is not None else random.randrange(1 << 30)
        for i in range(sequence_length):
            _, layout = generate_random_layout(
                num_agents=num_agents,
                height_rng=height_rng,
                width_rng=width_rng,
                wall_density=wall_density,
                seed=base + i
            )
            env_kwargs.append({"layout": layout})  # already a FrozenDict
            names.append(f"gen_{i}")
            print(f"Generated layout {i}: {names[-1]}")

    else:
        raise NotImplementedError(f"Unknown strategy '{strategy}'")

    # prefix with index so logs stay ordered
    ordered_names = [f"{i}__{n}" for i, n in enumerate(names)]
    print("Selected layouts:", ordered_names)
    return env_kwargs, names
