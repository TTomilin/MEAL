# MPE (Multi-agent Particle Environment) — Spread + Obstacles

Continuous-space cooperative coverage task, extending the classic MPE `simple_spread`
scenario. `num_agents` particle agents must spread out to cover `num_landmarks` landmarks
while avoiding `num_obstacles` static circular obstacles, using continuous physics
(soft-contact collisions, momentum, damping).

![MPE gameplay](../../../docs/assets/gifs/MPE.gif)

## Observation / action space

Per-agent observation (dimension `4 + num_landmarks*2 + (num_agents-1)*4 + num_obstacles*2`):

- own velocity (2) and position (2)
- relative position of every landmark (`num_landmarks * 2`)
- relative position (`(num_agents-1) * 2`) and communication channel
  (`(num_agents-1) * 2`, always zero — agents are silent) of every other agent
- relative position of every obstacle (`num_obstacles * 2`)

Discrete action space (5 actions: no-op + 4 movement directions), applied as forces to
continuous-space particles.

Entity ordering inside the underlying physics state (`state.p_pos`) is
`[agents][landmarks][obstacles]`.

## Reward

Reward mixes a **global** term (negative of the sum of nearest-agent-to-landmark distances,
shared by all agents) and a **local** term (each agent's own distance-based penalty),
weighted by `local_ratio` (0 = fully global/shared, 1 = fully individual). `info` also
reports `coverage_fraction` (fraction of landmarks with an agent within a small radius) and
`num_covered`, used as the eval metrics (`Evaluation/CoverageFraction`, `.../NumCovered`)
since MPE has no natural single scalar success criterion like Overcooked's soup count.

## How a task sequence is built

Each task samples a fresh, independent obstacle field: `num_obstacles` circles at random
continuous positions with random radii, frozen for that task's episodes (same pattern as
Overcooked layouts / JaxNav maps). Because obstacle placement is continuous, task diversity
is effectively unbounded even with `num_obstacles` fixed at 3+. `num_agents`/`num_landmarks`/
`num_obstacles` are fixed across the whole sequence so every task shares identical
observation/state shapes (required for `jax.lax.switch`-based task selection).

## Parameters (`--env.*` under `env:mpe`)

| Flag | Default | Effect |
|---|---|---|
| `--num-agents` (top-level, shared) | 3 | Number of particle agents. |
| `--env.num-landmarks` | 3 | Number of landmarks agents must cover. |
| `--env.num-obstacles` | 4 | Number of static circular obstacles per task. |
| `--env.local-ratio` | 0.5 | 0 = fully shared/global reward, 1 = fully individual reward. |
| `--env.partial-observability` | `False` | Not yet implemented — raises `NotImplementedError`. |
| `--repeat-sequence` (top-level, shared) | 1 | Replay the whole task sequence back-to-back this many times. |
| `--max-episode-steps` (top-level, shared) | 100 | Episode length. |

## Example

```bash
python -m experiments.train ippo --cl-method agem --seq-length 10 --num-agents 4 \
  env:mpe --env.num-landmarks 4 --env.num-obstacles 6 --env.local-ratio 0.3
```
