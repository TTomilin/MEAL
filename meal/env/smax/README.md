# SMAX (StarCraft Multi-Agent Challenge, JAX port)

Team-battle environment: `num_allies` units under your policy's control fight `num_enemies`
enemy units controlled by a scripted heuristic AI (`HeuristicEnemySMAX`). Units have distinct
types (marine, marauder, stalker, zealot, zergling, hydralisk) with different health, damage,
range, and speed — melee vs. ranged and tank vs. glass-cannon matchups require different
strategies.

![SMAX gameplay](../../../docs/assets/gifs/SMAX.gif)

Note the asymmetry with the other 3 envs: `--num-agents` maps to `num_allies` (the units your
policy actually controls) — `num_enemies` is independent and always AI-controlled, never
trained.

## Observation / action space

Each ally observes nearby allies' and enemies' relative positions, health, unit type, weapon
cooldown, and previous actions within a limited field of view. Action space is discrete:
movement in a few directions plus "attack unit K" for each enemy in range.

## Reward

Standard SMAC-style reward: damage dealt, kills, and a bonus/penalty for winning/losing the
battle. `info["kill_fraction"]` (allies' fraction of enemies killed by episode end) is tracked
separately and used as the headline eval metric (`Evaluation/KillFraction`) alongside raw
return, since win/loss alone is a very sparse and high-variance signal.

## How a task sequence is built

Each task independently samples a unit-type composition for both the ally and enemy teams
from the 6 available unit types (`make_smax_sequence` in `smax_cl.py`). With `6^num_allies ×
6^num_enemies` possible compositions, task diversity is effectively unbounded well before you
run out of tasks. `num_allies`/`num_enemies`/map dimensions are fixed across the whole
sequence so every task shares identical observation/state shapes.

## Parameters (`--env.*` under `env:smax`)

| Flag | Default | Effect |
|---|---|---|
| `--num-agents` (top-level, shared) | *(SMAX's own `num_allies` default: 5)* | Maps directly to `num_allies` — only applied if you actually pass `--num-agents`; otherwise SMAX keeps its own default (5) instead of the shared default. |
| `--env.num-allies` | 5 | Size of your controlled team (overridden by `--num-agents` if given). |
| `--env.num-enemies` | 5 | Size of the AI-controlled opposing team. |
| `--env.enemy-shoots` | `True` | If `False`, the enemy heuristic AI never attacks (movement-only) — much easier, useful for debugging. |
| `--env.partial-observability` | `False` | Not yet implemented — raises `NotImplementedError`. |
| `--repeat-sequence` (top-level, shared) | 1 | Replay the whole task sequence back-to-back this many times. |
| `--max-episode-steps` (top-level, shared) | 100 | Episode length. |

## Example

```bash
python -m experiments.train vdn --cl-method packnet --seq-length 8 --num-agents 5 \
  env:smax --env.num-enemies 5
```
