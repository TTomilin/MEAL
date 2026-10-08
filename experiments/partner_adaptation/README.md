# Continual partner adaptation (CPA)

`run_br.py` trains an ego agent against a sequence of fixed partners (BRDiv population partners, then
heuristic planners) with a continual-learning method (FT, EWC, MAS, ...). See `scripts/partner_adaptation.sh`.

## Behaviour corrections

These change results relative to earlier runs; they are not optimizations.

- **Importance rollouts use the real partner.** EWC/MAS importance used to be computed with the partner
  replaced by action `0`, which is `up` (stay is `4`). The frozen partner of the stage now acts through the
  same `AgentPopulation.get_actions` interface used in training (planner state, done/reset handling,
  available actions, env state). The EWC and MAS estimators themselves are unchanged. FT runs no
  importance rollout.
- **Separate RNG streams per stage.** `make_stage_keys(seed, stage)` gives independent train, eval and
  importance keys. Previously every stage rebuilt `PRNGKey(seed)`, so all stages replayed the same random
  stream, and evaluation split the training key, so `eval_every` changed training draws. Evaluation keys now
  depend only on `(seed, stage, update)`.
- **Evaluation is logged only on actual evaluation events** (after update 1, every `eval_every` updates, and
  after the last update). Previously the last evaluation was carried forward and logged after every update.
  Each event records the stage id, partner ids, update, cumulative environment steps
  (`Eval/EnvSteps`, training interactions across all stages so far), training episodes completed in the stage,
  and per-partner episode counts.
- **No duplicate evaluation.** The partner being trained was evaluated both as the current population and as an
  entry of the explicit partner list. Partners are now evaluated once per event, keyed by partner id.
- **Only completed episodes are reported.** Evaluation and training means use episodes that finished; a
  partner/update without one is omitted rather than logged as 0.
- Train/eval soup metrics are raw team deliveries (sum over agents of the per-deliverer counts the env
  emits; shaping is never included). The env reports reward and soups only for the delivering agent, so this
  sums each delivery once.
- `Eval/EgoSoup*` headline keys describe the partner of the current stage. Per-partner keys
  (`Eval/EgoSoup_Partner{i}`, `..._scaled_Partner{i}`) are unchanged.

## Efficiency changes (no change to what is computed)

- Scan outputs are compact scalars per update instead of full `(num_steps, num_envs)` rollout info.
  PPO and replay-memory data are untouched.
- `run_br.py` passes one `compiled_cache` to every stage. Stages with the same partner policy class, CL method,
  and evaluation grouping reuse one compiled train function; ego params, partner params, CL state and the stage
  id are runtime arguments. Heuristic partners are different policy classes and still compile separately.
  `MLPActorCriticPolicyCL` no longer treats `env_id_idx` as a static argument.
- Evaluation partners that share a policy object are evaluated in one vmapped graph instead of one unrolled graph
  per partner.
- `run_br.py` imports visualization packages (pygame/imageio) only when `--record-video` is set.

## Not changed

Per-stage optimizer state and the per-stage reward-shaping schedule are unchanged. The ego training reward is
the ego's own delivery reward plus (annealed) shaping, as before.

## Partner banks

`run_br.py` takes its population partners from one of two sources.

- **Bundled (default).** The checkpoints in `partner_agents/BRDiv_population/<layout_name>`: the first
  `--num-population-partners` members (default 3). Only `params_seed{i}_agent{j}.pt` files count, ordered by
  `(i, j)`; a stray `params.pt` is ignored. Asking for more members than exist is an error (it used to silently
  train on fewer and mislabel the ego heads).
- **Explicit bank.** `--partner-bank bank.json`. The manifest declares `num_partners` and lists every member as
  `{partner_id, population, member_index, checkpoint}`; each population names the BRDiv config of the run that
  trained it. **Relative paths resolve against the directory containing the manifest.** Exactly the declared
  members are used, in order; `--num-population-partners`, if given, must equal the declared count. Schema and
  every enforced rule are in the docstring of `partner_bank.py`: unique resolved files, unique checkpoint payloads
  (content hash), no generic `params.pt`, `member_index` inside the population and consistent with the file name,
  layout of bank / population config agree.

A bank of 12 members is not a population of 12: each member keeps the architecture of the population that trained
it (`partner_pop_size` is the critic's teammate-id width). Populations with equal `(partner_pop_size, activation)`
share one policy object, hence one compiled graph and one vmapped evaluation group; others get their own policy
and are never stacked together. Every checkpoint is validated against the shapes its config implies, so a
checkpoint from another layout or population size fails with the mismatching parameter path. Layout identity is
otherwise only as good as the `layout_name` in the population config (the bundled `cramped_room` config has none,
so only the shape check applies there). `run_br.py` writes `partner_bank.json` (ids, labels, sizes, generation
seeds, payload hashes) next to its checkpoint.

### Generating and assembling populations

`scripts/run_teammate_generation.sh` (preview by default, `RUN=1` to submit) launches one BRDiv job per
`(layout, generation seed)`: default layouts `coord_ring cramped_room`, seeds `1001 1002 1003`, populations of 3,
`num_seeds=1`. `LAYOUTS=asymm_advantages` or `counter_circuit` select the other two layouts with the same code;
`--layout-name` is the only layout-specific input. Each job writes
`checkpoints/brdiv_<layout>_pop<size>_gseed<seed>/` (git-ignored) with the member files, `config.pckl` and
`generation.json`: resolved settings, the generation seed, population size, explicit member files and the
interaction accounting below. The generation seed selects the population (`split(PRNGKey(seed), num_seeds)`); it is
not an ego seed.

```bash
python -m experiments.partner_adaptation.partner_bank --layout coord_ring --out banks/coord_ring.json \
    --generation-dirs checkpoints/brdiv_coord_ring_pop3_gseed1001 checkpoints/brdiv_coord_ring_pop3_gseed1002 \
                      checkpoints/brdiv_coord_ring_pop3_gseed1003
```

This lists the three bundled members first (`--no-bundled` to omit), then each generated population, with paths
relative to `banks/coord_ring.json`, and re-loads the result as validation. Nothing is copied or renamed.

The BRDiv defaults of `partner_generation/run.py` are the reference and are unchanged (population 3, 32+32 envs,
400 steps, `total_timesteps=2.5e8`). `--total-timesteps` is a budget of **agent** transitions per seed; whole updates
only, so the run trains on `num_updates * num_steps * num_envs` joint environment transitions and
`num_agents` times as many agent transitions (`generation.json["interactions"]`; for the defaults 4882 updates =
124,979,200 joint = 249,958,400 agent transitions per population). Checkpoint-evaluation episodes are not counted.
A budget below one update is rejected.

### Behaviour changes in this stage

- `--num-population-partners` is now optional (`None` resolves to 3 for bundled populations or to the bank size);
  more than available is an error instead of silent truncation. Valid existing invocations
  (`--num-population-partners 3`) are unaffected.
- The bundled loader no longer opens any file whose name contains `param`; only `params_seed*_agent*.pt`.
- `partner_generation/run.py` no longer writes the generic `params.pt` (it duplicated the last member and was
  picked up by the old loader as an extra one), names its output directory by layout/size/seed when `--layout-name`
  is set, refuses to write into a directory that already has member files, and rejects unknown layouts and
  budgets below one update. `scripts/run_teammate_generation.sh` previously called a nonexistent module and is
  replaced; its positional arguments are gone.

## Planner partners P01-P12

Twelve frozen planning partners complete the 24-partner pilot bank next to the 12 BRDiv members. They are
candidate variants of one planner, not a claim of twelve distinct strategies. Code:
`partner_agents/overcooked/planner_variants.py` (`PlannerAgent`, registry `PLANNER_SPECS`); the policy wrapper is
`OvercookedPlannerPolicyWrapper` in `agent_policy_wrappers.py`. The existing planners (`Independent`, `Onion`,
`Plate`, `Static`, `Random`) and their constructors are untouched; random/static remain the legacy-mode controls
(`--num-heuristic-partners`) and are not among the twelve.

The configuration of a variant is `PlannerParams`, runtime data passed as the wrapper's `params`. All twelve share
one wrapper, hence one compiled train/importance function and one vmapped eval group per layout. The shared
generalist `G` is `p_onion_on_counter=0.4, prefetch_steps=5, priority=NEAREST, yield_wait=1, handoff=NEAREST,
history_window=0`; every ID changes exactly one option of `G`, one value on each side of it:

| ID | option (value) | ID | option (value) |
|---|---|---|---|
| P01 | counter probability 0.0 | P02 | counter probability 0.8 |
| P03 | prefetch 0 (plate only when soup ready) | P04 | prefetch 10 (plate when <=10 steps remain) |
| P05 | priority FILL | P06 | priority SERVE |
| P07 | yield 0 (replan at once) | P08 | yield 2 (wait up to 2 steps) |
| P09 | handoff: lowest row-major id | P10 | handoff: highest row-major id |
| P11 | complement, window 8 | P12 | complement, window 32 |

The bracketing values around `G` exist so that all twelve resolved configurations differ; they carry no scientific
claim. The registry hash of each configuration is stored in the manifest and re-checked on load.

**Decision rule** (exact environment state, not the observation channels, which also mark carried items and the
non-interactable border). "Reachable" = adjacent to a floor tile of the agent's connected floor component. A free
counter is an empty `wall` tile next to such floor; counters in the border corners are not counters. Ties between
equally good targets go to the lowest row-major tile id, never to a random draw.

- Holding an **onion**: nonfull pot or counter. The destination is drawn once per held onion (counter with
  probability `p`), then kept until the onion leaves the hand. A counter is only chosen if one is feasible: the
  *handoff* set is free counters reachable from both agents' floors (nearest, or lowest/highest id by P09/P10); if
  there is none, the onion goes to a pot. With no nonfull pot the onion goes to any free counter; with neither,
  the agent stays.
- Holding a **plate**: plate the nearest ready pot; else walk next to the pot with the least time left and wait
  (no interaction); else put the plate on the nearest free counter; else stay. Holding a **dish**: deliver at the
  nearest goal, else stay.
- **Empty-handed**: FILL is feasible if a nonfull pot and an onion source (pile or onion on a counter) are
  reachable; SERVE if a plate source and a goal are reachable and a pot is ready, or cooking with
  `<= prefetch_steps` left (`remaining` = pot status). One feasible: do it. Both: P05 fill, P06 serve, generalist
  the subtask whose first target is nearer (tie fill), P11/P12 the subtask the other agent covered less in the last
  `window` steps (tie fill). Neither: stay.
- **Own drops**: an item the agent put on a counter is left for the other agent; while it is still there it is never
  a pickup target. (Without this, "counter handoff" is the agent re-taking its own onion.)
- **Other agent's activity** (P11/P12): each step the other agent's inventory change is one event, FILL if an onion
  was picked up or released (including into a pot), SERVE if a plate or dish was (plate pickup, plating, delivery).
  The first step after a reset records nothing. A fixed 32-entry history is kept by every planner; the window only
  selects how much of it P11/P12 read. The planner reads the other agent's inventory from the env state (what the
  observation paints anyway); the learning ego needs nothing new.
- **Blocked move**: the previous action was a translation onto a floor tile and the position did not change. P07
  replans at once; P08 stays up to two steps, ending early as soon as the other agent has moved from where it stood
  when the block was detected. The generalist waits one step.
- **Deadlock and livelock recovery**: progress = inventory change, or the Manhattan distance to the current target
  reaching a new minimum since the inventory last changed (or since the decision was stable for 3 steps). After 8
  steps without progress the agent takes one random step to a free floor tile (never the other agent's) and starts
  counting again. Steps with nothing to do and waiting beside a cooking pot never count. This is a generic escape,
  not a guarantee: two agents in a one-tile corridor can still lock each other for long stretches.
- **Reset**: the wrapper clears all memory when `done` is set, before acting (the older wrappers clear it after).

### Bank manifest

Planners are ordinary partner records (`"kind": "planner", "planner_id": "P01"`); manifest position is training
order and `partner_id` must equal it. `--planners P01 ... | all` of `partner_bank.py` appends them after the BRDiv
members:

```bash
python -m experiments.partner_adaptation.partner_bank --layout coord_ring --out banks/coord_ring_24.json \
    --no-bundled --planners all --generation-dirs <four generation dirs of 3 members each>
```

The bank is validated (24 unique ids, planner ids unique, resolved configurations unique, hash match) and every
planner's configuration is resolved in `load_partners` before any ego training starts. `run_br.py` writes the
resolved planner configuration into `partner_bank.json`. In bank mode `--num-heuristic-partners` now defaults to
0 (it still defaults to 5 without `--partner-bank`); pass it explicitly to append the legacy partners.

### Quality check (a diagnostic, not a benchmark)

```bash
python -m experiments.partner_adaptation.partner_quality --layouts coord_ring cramped_room \
    asymm_advantages counter_circuit [--episodes 8 --steps 400 --seed 0 --counterparts independent onion plate]
```

Every planner plays agent_1 for `--episodes` episodes against each of the existing independent, onion and plate
planners (agent_0), with identical resets and per-step keys for all planners. It prints soups (team / by the
planner), idle share and action mix of the planner, items handed over through counters (put by one agent, taken by
the other), and how often the variant's distinguishing option mattered ("activation": P01 pot routing, P02 counter
routing, P03 plate pickup with a ready soup, P04 plate pickup before ready, P05/P06 both subtasks feasible, P07 a
blocked move, P08 a wait, P09/P10 counter choice with >=2 shared counters, P11/P12 a decided complement). Flags:
layout collapses known from the static geometry, variants that never activated (unless explained by a collapse),
and planners whose actions and positions are identical in every episode. A few episodes against three fixed
counterparts do not show diversity, and a low score with these counterparts does not make a partner unsolvable.

Layout limitations, derived from the geometry (`layout_collapses`) and visible in the quality output:

| layout | pairs the layout cannot distinguish |
|---|---|
| coord_ring, counter_circuit | none (two reachable pots, shared counters) |
| cramped_room | P05/P06 and P11/P12: one pot, so fill and serve are never both feasible |
| asymm_advantages | P01/P02 (no counter shared by the two rooms), P09/P10, P07/P08 (disjoint floors) |

These pairs are still in the bank (24 ids, no replacement); they behave as copies of `G` on that layout. On
`cramped_room` a 6-episode quality run also found P08 identical to P05/P06/P11/P12 (the waits it triggers end
after one step because the other agent moves), and on `coord_ring` P05 identical to P11 in sampled behaviour. Such
coincidences are reported, not fixed.
