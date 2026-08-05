# MEAL Examples

Small, self-contained scripts that each demonstrate one feature of MEAL. Every
script that renders an episode and most record a GIF of gameplay to `gifs/` (created next to
wherever you run the script from, e.g. `examples/gifs/` if run from this
directory).

Run any script from the repo root:

```bash
conda run -n meal python examples/<script>.py
```

## Overcooked

| Script | What it shows |
| --- | --- |
| [`basic_usage.py`](basic_usage.py) | Minimal gym-style loop: `meal.make_env('overcooked')`, reset, sample random actions, step, print rewards. No rendering — the smallest possible starting point. |
| [`four_agents.py`](four_agents.py) | Scaling to 4 agents at `difficulty='hard'`. Runs a 100-step random-action episode and renders it to `gifs/4_agents_hard.gif`. |
| [`cl_sequence.py`](cl_sequence.py) | Continual-learning sequences via `meal.make_sequence(strategy='curriculum')`: generates 6 Overcooked tasks of increasing difficulty, evaluates each with a few random-action episodes, and stitches all of them into one GIF. |
| [`partial_observability.py`](partial_observability.py) | `OvercookedPO` vs. full-observability `Overcooked` on the same layout: prints the (much smaller) PO observation shape, then renders a GIF with each agent's current visible window tinted in that agent's colour (`OvercookedVisualizerPO` + `env.get_agent_view_masks`). Higher `difficulty` → smaller field of view. |
| [`forced_coordination.py`](forced_coordination.py) | Two ways to force role specialisation: (1) the `forced_coord` preset layout, where a wall physically separates ingredients/pot from plates/goal, and (2) `agent_restrictions` on the ordinary `cramped_room` layout, which forbids one agent from touching plates and the other from touching onions even though nothing physically stops them. Renders one GIF per condition. |
| [`stochastic_transitions.py`](stochastic_transitions.py) | The two transition-noise wrappers, isolated on bare rooms with a single agent following a fixed scripted path (no cooking, no randomness in the actions themselves) so the effect is unmistakable: `SlipperyTiles` walking straight across a corridor (baseline vs. slippery, side by side, `stochastic_transitions_slippery.gif`) and `StickyActions` walking a closed square loop (baseline vs. sticky, side by side, `stochastic_transitions_sticky.gif` - sticky actions overshoot each turn). |
| [`mpe_spread.py`](mpe_spread.py) | MPE SimpleSpread with obstacles: agents (green) must cover landmarks (dark dots) while avoiding fixed, per-task obstacle fields (gray). Builds a 3-task sequence via `make_mpe_sequence` and renders one GIF per task with random-policy agents. |
| [`smax_battle.py`](smax_battle.py) | SMAX unit-composition battles: a trained-side team of allied units (blue) vs. a scripted heuristic enemy team (red), via `HeuristicEnemySMAX`. Task diversity comes from randomly sampled unit types per side (marine/marauder/stalker/zealot/zergling/hydralisk). Builds a 3-task 5v5 sequence and renders one GIF per task with random-policy allies. |
| [`jaxnav_navigation.py`](jaxnav_navigation.py) | JaxNav multi-robot goal navigation: agents must reach individual goals while avoiding walls and each other on a randomly-generated polygon map, frozen per task. Builds a 3-task sequence and renders one GIF per task, including lidar rays and travelled paths. |

## Notes

- All rollouts here use uniformly random actions purely to exercise the
  environment/renderer — none of these scripts train or load a policy. Reward
  numbers printed are therefore not meaningful as performance, only as a
  sanity check that `step()` is wired up correctly.
- If GIF creation fails with an import/display error, check that `pygame`
  and `imageio` are installed (`pip install -e ".[viz]"`) and that
  `meal/visualization/renderer_config.py` points at a valid renderer version.
- For more control (output format, task count, custom agent/landmark/unit
  counts, PNG-only output, etc.) see the dev-facing
  `scripts/visualize_env.py`, `scripts/visualize_mpe.py`, and
  `scripts/visualize_smax.py`, which these examples deliberately keep
  simpler than.
