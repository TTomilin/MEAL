# Running experiments

Single entry point: `experiments/train.py`. The CLI is three layers, each a required or
optional subcommand/flag group, **in this order**:

```bash
python -m experiments.train <algo> [outer flags...] env:<env_name> [--env.* flags...]
```

```bash
python -m experiments.train ippo --cl-method ewc --seq-length 10 --num-agents 2 \
  env:overcooked --env.difficulty medium
```

- `<algo>`: one of `ippo`, `mappo`, `happo`, `vdn`, `qmix`. Optional — a bare
  `python -m experiments.train` (no args at all) defaults to `ippo`/`ft`/`overcooked`/`easy`, so
  it works out of the box for new users. **The moment you pass any other flag you must name the
  algo explicitly** — tyro only applies the default when literally nothing else is given.
- `[outer flags...]`: everything below in this doc — cl-method, network size, sequence/eval/
  logging params. Must come **before** the `env:` token.
- `env:<env_name>`: one of `env:overcooked`, `env:mpe`, `env:smax`, `env:jaxnav`. Optional —
  defaults to `env:overcooked`. Everything after it is that env's own `--env.*` flags; see
  `meal/env/<env_name>/README.md` for what each env exposes and how it works.

**Ordering matters.** tyro assigns trailing flags to the nearest preceding subcommand, so
`--env.difficulty easy --cl-method ewc` (env flag before the outer flag) fails, and outer flags
placed after `env:overcooked` get silently misrouted. Always: algo → outer flags → `env:...` →
`--env.*` flags.

Every run prints a `=== Run config ===` summary (algo, cl_method, resolved reg_coef, env +
full env settings, seq_length, seed, encoder) right after device init, so you can always see
exactly what a given invocation resolved to.

## Continual learning methods

`--cl-method {ewc, mas, l2, ft, agem, er_ace, packnet}` — required unless `--single-task-idx`
is set (which forces `ft`, i.e. train one task in isolation with no CL machinery).

- `--reg-coef <float>`: regularization strength for EWC/MAS/L2. Auto-set per method if
  omitted (EWC → `1e11`, MAS → `1e9`, L2 → `1e7`) — shown resolved in the run-config printout.
- EWC/MAS specific:
  - `--importance-mode {online, last, multi}` (default `online`)
  - `--importance-decay <float>` (default `0.9`) — online accumulation decay
  - `--importance-episodes`, `--importance-steps`, `--importance-stride` — control the
    rollout used to estimate Fisher/importance
  - `--normalize-importance` — normalize importance weights before penalizing
- AGEM specific: `--agem-memory-size` (default `100000`), `--agem-sample-size` (default `1024`)
- ER-ACE specific: `--er-ace-coef` (default `1.0`)
- PackNet specific: `--train-epochs` (default `8`), `--finetune-epochs` (default `2`),
  `--finetune-timesteps` (default `1e7`) — PackNet trains then prunes-and-finetunes each task
  in two phases; not supported with `--encoder cnn`, and not supported at all for VDN/QMIX.

## Network architecture

- `--encoder {mlp, cnn}` (default `mlp`)
- `--hidden-size <int>` (default `128`) / `--num-layers <int>` (default `2`) — on-policy
  algos (IPPO/MAPPO/HAPPO) only; controls actor/critic MLP width/depth (also applies to the
  CNN encoder's post-conv dense stack). VDN/QMIX have their own separate `--hidden-size`
  (default `256`, Q-network width) since they're a different config hierarchy.
- `--use-layer-norm` (default on)
- `--activation <str>` (default `relu`)
- `--use-task-id` / `--use-multihead` (both default on) — task-conditioning: append a
  task-id one-hot to observations, and/or give each task its own output head.
- `--shared-backbone` (IPPO only, default off) — share the actor/critic trunk instead of
  separate networks.
- `--regularize-critic` (IPPO/MAPPO/HAPPO, default off) — include critic params in the
  EWC/MAS/L2 penalty (default is actor-only).
- `--use-agent-id` (MAPPO default on, HAPPO default off) — append an agent one-hot to the
  actor's input so a shared-parameter policy can still specialize per agent.

## Sequence / task parameters

These are shared across all 4 envs (not env-specific, since they mean the same thing
everywhere):

- `--seq-length <int>` (default `20`) — number of tasks in the continual sequence.
- `--num-agents <int>` (default: each env's own default — see its README; SMAX maps this to
  `num_allies` and only overrides its own default of 5 if you actually pass the flag).
- `--repeat-sequence <int>` (default `1`) — replay the same base task sequence back-to-back
  this many times (updates `seq_length` accordingly); useful for forgetting/plasticity curves
  across repeated exposure to the same tasks.
- `--single-task-idx <int>` — skip the CL sequence entirely and train on just task `idx`
  (forces `cl_method=ft`).
- `--max-episode-steps <int>` (default: each env's own default — Overcooked/JaxNav `400`,
  MPE/SMAX `100`) — env episode length. Separate from `--num-steps`, which is the
  training-rollout collection size, not related to when an episode actually ends.

## Evaluation, video, logging

- `--evaluation` (default on) / `--no-evaluation` — periodic eval rollouts on all tasks seen
  so far, logged as `Evaluation/*` metrics.
- `--eval-deterministic` (default off) — fixed per-task RNG for eval instead of following the
  training RNG stream.
- `--record-video` (default **on**) / `--video-length <int>` (default `250`) — render an mp4 of
  the trained policy per task at the end of its training into `runs/<run_name>/`, and upload it
  to wandb if `--use-wandb` is on. VDN/QMIX video is Q-value-argmax based (verified only
  against Overcooked).
- `--log-interval <int>` (default `5`, matching paper Table 6) — how often (in update steps)
  to compute/log metrics; TensorBoard scalars only actually hit disk every `flush_secs`
  (tensorboardX default 120s) or on process exit — for continuous mid-run wandb sync on short
  runs you'd need to lower that in `experiments/algo_common.py`.
- `--use-wandb` (default on) / `--wandb-mode {online, offline, disabled}` / `--entity <str>` /
  `--project <str>` (default `MEAL`) / `--tags <str...>`.
- TensorBoard summaries + video always write to `runs/<run_name>/` regardless of wandb.
- Checkpoints (flax params + a JSON metadata sidecar with the network/env config) save to
  `checkpoints/<env>/<cl_method>/<run_name>/model_env_<task_idx>`.

`run_name` (and hence the wandb run id/name, `runs/` dir, and `checkpoints/` path) has the
shape `<algo>_<cl_method>_<env>_<env_tag>_<N>agents_<encoder>_seq<seq_length>_seed_<seed>_<timestamp>`,
e.g. `ippo_ewc_overcooked_medium_2agents_mlp_seq10_seed_1_2026-08-03_12-00-00_000`.

## Off-policy (VDN / QMIX) specifics

- `--eps-start`/`--eps-finish`/`--eps-decay` — epsilon-greedy exploration schedule
  (`eps_decay` is the fraction of `num_updates` over which epsilon anneals).
- `--target-update-interval <int>` — how often (in gradient steps) the target network syncs
  (VDN default `1` = every step; QMIX default `10`).
- QMIX-only: `--mixing-embed-dim <int>` (default `32`) — mixing-network hypernetwork size.

## Environments

`env:overcooked` / `env:mpe` / `env:smax` / `env:jaxnav` — each has its own `--env.*` flags,
fully documented in `meal/env/<name>/README.md` alongside how the environment itself works
(observation/action space, reward structure, how CL task diversity is generated).

## Downloading and plotting results

For downloading results from W&B and plotting figures, see
[experiments/results/README.MD](results/README.MD).
