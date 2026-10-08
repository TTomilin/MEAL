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
