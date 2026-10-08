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
