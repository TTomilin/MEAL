import random
from typing import List, Dict, Any, Sequence, Tuple

from meal import Overcooked, OvercookedPO
from meal.env.layouts.presets import hard_layouts, medium_layouts, easy_layouts, overcooked_layouts, \
    get_layouts_by_difficulty
from meal.wrappers.observation import PadObsToMax


def _resolve_pool(names: Sequence[str] | None) -> List[str]:
    """Turn user‐supplied `layout_names` into a concrete list of keys."""
    presets = {
        "hard_levels": list(hard_layouts),
        "medium_levels": list(medium_layouts),
        "easy_levels": list(easy_layouts),
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


def _make_restrictions(enabled: bool, seed: int, num_agents: int) -> dict:
    # Only meaningful for 2 agents; extend if you support more
    if not enabled or num_agents != 2:
        return {}
    role = random.Random(seed).randint(0, 1)  # 0: a0 no onions / a1 no plates, 1: flipped
    return {
        "agent_0_cannot_pick_onions": role == 0,
        "agent_0_cannot_pick_plates": role == 1,
        "agent_1_cannot_pick_onions": role == 1,
        "agent_1_cannot_pick_plates": role == 0,
    }


def create_sequence(
        env_id: str = "overcooked",
        sequence_length: int | None = None,
        strategy: str = "generate",
        seed: int = 0,
        num_agents: int = 2,
        difficulty: str | None = None,
        complementary_restrictions: bool = False,
        repeat_sequence: int = 1,
        layout_names: Sequence[str] | None = None,
        **env_kwargs,
) -> list:
    """
    strategies
    ----------
    random     – sample from fixed layouts (no immediate repeats if len>pool)
    ordered    – deterministic slice through fixed layouts
    generate   – create brand-new solvable kitchens on the fly
    curriculum – split tasks equally across difficulty levels (easy -> medium -> hard)
    """
    if seed is not None:
        random.seed(seed)

    env_cls = {"overcooked": Overcooked, "overcooked_po": OvercookedPO}[env_id]

    # 1) Build a base list of specs for one sequence
    base_specs: list[dict] = []

    if strategy in ("random", "ordered"):
        pool_dict = get_layouts_by_difficulty(difficulty)  # dict[name] -> FrozenDict
        pool = list(pool_dict.keys()) if layout_names is None else _resolve_pool(layout_names)
        if sequence_length is None:
            sequence_length = len(pool)

        if strategy == "random":
            selected = _random_no_repeat(pool, sequence_length)
        else:  # ordered
            if sequence_length > len(pool):
                raise ValueError("ordered requires seq_length ≤ available layouts")
            selected = pool[:sequence_length]

        for name in selected:
            base_specs.append({"layout_name": name, "difficulty": difficulty})

    elif strategy == "generate":
        if sequence_length is None:
            raise ValueError("generate requires an explicit sequence_length")
        # “generate” doesn’t use preset names; store per-slot difficulty
        for _ in range(sequence_length):
            base_specs.append({"layout_name": None, "difficulty": difficulty})

    elif strategy == "curriculum":
        if sequence_length is None:
            raise ValueError("curriculum requires an explicit sequence_length")

        tasks_per_difficulty = sequence_length // 3
        remaining = sequence_length % 3
        counts = [
            tasks_per_difficulty + (1 if remaining > 0 else 0),  # easy
            tasks_per_difficulty + (1 if remaining > 1 else 0),  # medium
            tasks_per_difficulty,                                 # hard
        ]
        for diff, cnt in zip(["easy", "medium", "hard"], counts):
            for _ in range(cnt):
                base_specs.append({"layout_name": None, "difficulty": diff})
    else:
        raise NotImplementedError(f"Unknown strategy '{strategy}'")

    # 2) Repeat the base sequence
    full_specs = base_specs * max(1, int(repeat_sequence))

    # 3) Instantiate envs with increasing task_id and per-env restriction seed
    envs: list = []
    for i, spec in enumerate(full_specs):
        restr = _make_restrictions(complementary_restrictions, seed + i, num_agents)
        kwargs_common = dict(
            task_id=i,
            num_agents=num_agents,
            agent_restrictions=restr,
            **env_kwargs,
        )
        if spec["layout_name"] is not None:
            env = env_cls(layout_name=spec["layout_name"], **kwargs_common)
        else:
            # generate / curriculum
            env = env_cls(difficulty=spec["difficulty"], **kwargs_common, seed=seed + i)
        envs.append(env)

    # 4) Optional: pad observations to max grid across the set (only for fully-observed env)
    if env_id == "overcooked" and len(envs) > 1:
        max_w = max(int(env.layout["width"]) for env in envs)
        max_h = max(int(env.layout["height"]) for env in envs)
        envs = [PadObsToMax(env, max_h, max_w) for env in envs]

    return envs
