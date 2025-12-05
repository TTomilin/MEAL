# Running experiments

## Continual Learning methods and key flags

- `--cl-method {ewc, mas, l2, ft, agem}` (required unless `--single-task-idx` is set)
- `--reg-coef <float>`: regularization strength (auto‑set per method if omitted)
- EWC/MAS specific:
    - `--importance-mode {online,last,multi}` (default: `online`)
    - `--importance-decay <float>` for online accumulation
- AGEM specific:
    - `--agem-memory-size`, `--agem-sample-size`, `--agem-gradient-scale`
- Architectural/heads:
    - `--use-multihead`, `--shared-backbone`, `--regularize-critic`, `--regularize-heads`
- Optimizer resets between tasks: `--reset-optimizer`

## Environment sequencing

- Sequence shape:
    - `--seq-length <int>`: number of tasks
    - `--repeat-sequence <int>`: repeat the sequence
- Generation strategy: `--strategy {generate, ordered, random, curriculum}`
- Difficulty preset: `--difficulty {easy, medium, hard}` (affects layout sampling)
- Layout list (optional): `--layouts [names...]`
- Agent placement/constraints:
    - `--random-agent-start`, `--separated-agents`, `--complementary-restrictions`
- Non‑stationarity toggles:
    - `--sticky-actions`, `--slippery-tiles`, `--random-pot-size`, `--random-cook-time`
    - Or enable all at once with `--non-stationary true`

## Rewards

- Default (dense shaped team reward): no extra flags
- Sparse team reward: `--sparse-rewards true`
- Individual rewards: `--individual-rewards true`
  Note: `--sparse-rewards` and `--individual-rewards` are mutually exclusive.

## Logging and outputs

- W&B: `--use-wandb true`, `--wandb-mode {online,offline,disabled}`, `--entity`, `--project`, `--tags ...`
- TensorBoard summaries are written to `runs/<run_name>` automatically.
- Checkpoints and config are saved under `runs/<run_name>`.

## Downloading and plotting results
For downloading results from W&B and plotting figures, please refer to [experiments/results/README.MD](results/README.MD). 