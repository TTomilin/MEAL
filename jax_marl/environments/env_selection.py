import json
import random
from typing import List, Dict, Any, Sequence, Tuple
import ast
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jax_marl.environments.overcooked import hard_layouts, medium_layouts, easy_layouts, overcooked_layouts, \
    single_layouts
from jax_marl.environments.overcooked.env_generator import generate_random_layout


def _parse_layout_string(layout_str: str) -> FrozenDict:
    """Parse a string representation of a FrozenDict layout back to actual FrozenDict."""
    # Remove the "FrozenDict(" prefix and ")" suffix
    if layout_str.startswith("FrozenDict("):
        layout_str = layout_str[11:-1]

    # Parse the dictionary-like string
    # Replace Array(...) with list representation for parsing
    import re

    # Extract array contents and convert to lists
    def array_replacer(match):
        array_content = match.group(1)

        # Find the array values part (everything between [ and ], before dtype=)
        # Handle multi-line arrays
        bracket_match = re.search(r'\[(.*?)\]', array_content, re.DOTALL)
        if bracket_match:
            values_str = bracket_match.group(1)
            # Clean up whitespace and newlines
            values_str = re.sub(r'\s+', ' ', values_str).strip()
            # Split by comma and convert to integers
            try:
                values = [int(x.strip()) for x in values_str.split(',') if x.strip()]
                return str(values)
            except ValueError:
                # If parsing fails, return empty list
                return "[]"
        else:
            return "[]"

    # Replace Array(...) patterns with lists - use DOTALL flag for multi-line matching
    layout_str = re.sub(r'Array\(([^)]+(?:\([^)]*\))*[^)]*)\)', array_replacer, layout_str, flags=re.DOTALL)

    # Add quotes around dictionary keys to make it valid Python syntax
    key_pattern = r'\b(wall_idx|agent_idx|goal_idx|plate_pile_idx|onion_pile_idx|pot_idx|height|width)\b:'
    layout_str = re.sub(key_pattern, r'"\1":', layout_str)

    # Parse the cleaned string as a dictionary
    try:
        layout_dict = ast.literal_eval(layout_str)
    except (ValueError, SyntaxError) as e:
        print(f"Failed to parse layout string: {e}")
        print(f"Cleaned string: {layout_str[:200]}...")
        # If parsing fails, create a minimal valid layout
        layout_dict = {
            "height": 6,
            "width": 7,
            "wall_idx": [],
            "agent_idx": [],
            "goal_idx": [],
            "plate_pile_idx": [],
            "onion_pile_idx": [],
            "pot_idx": []
        }

    # Convert lists to JAX arrays with correct dtypes
    array_keys = ["wall_idx", "agent_idx", "goal_idx", "plate_pile_idx", "onion_pile_idx", "pot_idx"]
    for key in array_keys:
        if key in layout_dict:
            layout_dict[key] = jnp.array(layout_dict[key], dtype=jnp.int32)
        else:
            layout_dict[key] = jnp.array([], dtype=jnp.int32)

    return FrozenDict(layout_dict)


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
        layout_file: str | None = None,
        complementary_restrictions: bool = False,
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

    # ---- shortcut: load pre-baked layouts -----------------------------
    if layout_file is not None:
        with open(layout_file) as f:
            raw_data = json.load(f)

        # Convert string representations to actual FrozenDict objects
        env_kwargs = []
        for item in raw_data:
            if isinstance(item, dict) and "layout" in item:
                # Parse the layout string to FrozenDict
                layout_str = item["layout"]
                if isinstance(layout_str, str):
                    parsed_layout = _parse_layout_string(layout_str)
                    env_kwargs.append({"layout": parsed_layout})
                else:
                    # Already a proper layout object
                    env_kwargs.append(item)
            else:
                # Fallback for unexpected format
                env_kwargs.append(item)

        names = [f"file_{i}" for i in range(len(env_kwargs))]
        return env_kwargs, names
    # -------------------------------------------------------------------

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

    # Add agent restrictions if enabled
    if complementary_restrictions:
        for i, kwargs in enumerate(env_kwargs):
            # Randomly assign roles for each task: 0 = agent_0 can't pick onions, 1 = agent_0 can't pick plates
            role_assignment = random.randint(0, 1)
            kwargs["agent_restrictions"] = {
                "agent_0_cannot_pick_onions": role_assignment == 0,
                "agent_0_cannot_pick_plates": role_assignment == 1,
                "agent_1_cannot_pick_onions": role_assignment == 1,
                "agent_1_cannot_pick_plates": role_assignment == 0,
            }
            print(f"Task {i}: Agent 0 cannot pick {'onions' if role_assignment == 0 else 'plates'}, Agent 1 cannot pick {'plates' if role_assignment == 0 else 'onions'}")

    # prefix with index so logs stay ordered
    ordered_names = [f"{i}__{n}" for i, n in enumerate(names)]
    print("Selected layouts:", ordered_names)
    return env_kwargs, names
