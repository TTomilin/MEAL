"""
Continual learning task sequence for SMAX.

Task diversity source: independently sampled ally and enemy unit compositions.
Each task is a 5v5 (or NvN) battle with a unique combination of unit types drawn
from the 6 available types: marine(m), marauder(M), stalker(s), zealot(Z),
zergling(z), hydralisk(h).

This creates near-infinite task variety because:
  - Each unit type has distinct stats (health, damage, range, speed)
  - Melee vs ranged, tank vs glass-cannon matchups require different strategies
  - 6^n_allies × 6^n_enemies possible compositions ≫ 100 tasks needed

All tasks share the same num_allies, num_enemies, and map dimensions, so
state/obs array shapes are identical across tasks — jax.lax.switch compatible.
"""

import numpy as np
import jax.numpy as jnp

from meal.env.smax.smax_env import Scenario
from meal.env.smax.heuristic_enemy_smax_env import HeuristicEnemySMAX

UNIT_SHORTHANDS = ["m", "M", "s", "Z", "z", "h"]


def _composition_id(types: np.ndarray) -> str:
    """Human-readable string for a unit type array, e.g. [0,0,2] → 'mms'."""
    return "".join(UNIT_SHORTHANDS[t] for t in types)


def make_smax_sequence(
    sequence_length: int,
    seed: int = 0,
    num_allies: int = 5,
    num_enemies: int = 5,
    max_steps: int = 100,
    enemy_shoots: bool = True,
) -> list:
    """Build a list of ``HeuristicEnemySMAX`` envs with varied unit compositions.

    Args:
        sequence_length: Number of tasks in the CL sequence.
        seed: Base RNG seed; task i uses seed + i for reproducibility.
        num_allies: Number of ally agents (fixed across sequence).
        num_enemies: Number of enemy agents (fixed across sequence).
        max_steps: Episode length in env steps.
        enemy_shoots: Whether the heuristic enemy actively attacks.

    Returns:
        List of ``HeuristicEnemySMAX`` instances, each with a ``.map_id``
        attribute describing the composition (e.g. ``"mMszZ_vs_zzhhm"``).
    """
    rng = np.random.default_rng(seed)
    envs = []

    for task_idx in range(sequence_length):
        # Sample unit types independently for each team
        ally_types = rng.integers(0, len(UNIT_SHORTHANDS), size=num_allies)
        enemy_types = rng.integers(0, len(UNIT_SHORTHANDS), size=num_enemies)

        unit_types = jnp.concatenate([
            jnp.array(ally_types, dtype=jnp.uint8),
            jnp.array(enemy_types, dtype=jnp.uint8),
        ])

        scenario = Scenario(
            unit_types=unit_types,
            num_allies=num_allies,
            num_enemies=num_enemies,
            smacv2_position_generation=False,
            smacv2_unit_type_generation=False,
        )

        map_id = (
            _composition_id(ally_types)
            + "_vs_"
            + _composition_id(enemy_types)
        )

        env = HeuristicEnemySMAX(
            scenario=scenario,
            max_steps=max_steps,
            enemy_shoots=enemy_shoots,
        )
        env.map_id = map_id
        envs.append(env)

    return envs
