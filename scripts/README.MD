# Reproducing the MEAL paper's experiments

This directory holds one `.sh` script per experiment in [`MEAL.pdf`](MEAL.pdf) (the paper).
Each script is a parameter sweep over `experiments/train.py` (see
[`experiments/README.MD`](../experiments/README.MD) for the full CLI reference) that submits
one SLURM job per run, or runs them sequentially if SLURM isn't available.

## Usage

Every script sources [`_common.sh`](_common.sh), which gives them all the same interface:

```bash
./baseline_comparison.sh                       # preview: print every job + a count, submit nothing
RUN=1 ./baseline_comparison.sh                  # actually submit to SLURM (sbatch)
RUN=1 MEAL_LOCAL=1 ./baseline_comparison.sh     # actually run, sequentially, in this shell
```

Always preview first — these sweeps launch anywhere from ~10 to ~120 multi-hour, single-GPU
runs. Relevant env vars (all optional, see `_common.sh` for the full list): `CONDA_ENV`
(default `meal`), `SLURM_PARTITION` (default `gpu_h100`).

Every run is tagged (`--tags <SWEEP_NAME>`) so you can filter for it in W&B; see
[`experiments/results/README.MD`](../experiments/results/README.MD) for downloading and
plotting results afterward.

## Scripts → paper sections

| Script | Paper section | Reproduces |
|---|---|---|
| [`baseline_comparison.sh`](baseline_comparison.sh) | §5.1, Table 2, Figs. 3–4 | 8 CL methods × 3 difficulty levels, IPPO, 20-task sequences |
| [`n_agent_sweep.sh`](n_agent_sweep.sh) | §5.2, Table 3 | IPPO + Online EWC with 1–5 agents, Levels 1–2 |
| [`ablation_study.sh`](ablation_study.sh) | §5.3, Fig. 5 | Multi-head / task-id / critic-reg / layer-norm / CNN ablations for EWC, MAS, ER-ACE |
| [`sequence_length.sh`](sequence_length.sh) | §5.4, Table 5 | EWC vs. Online EWC on 10-task vs. 100-task sequences |
| [`plasticity_loss.sh`](plasticity_loss.sh) | §5.5, Table 4, Fig. 6 | FT on a 10-task sequence repeated 1×/3×/10× (`--repeat-sequence`) |
| [`partial_observability.sh`](partial_observability.sh) | §5.6, Fig. 7 | IPPO/MAPPO/HAPPO × full/partial observability × MLP/CNN, Levels 1–2 |
| [`partner_adaptation.sh`](partner_adaptation.sh) | §5.7, Fig. 8 | Ego agent adapting to 8 diverse partners across 4 classic layouts (uses `experiments/partner_adaptation/run_br.py`, not `train.py`) |
| [`algorithm_comparison.sh`](algorithm_comparison.sh) | App. J, Fig. 16 | All 8 CL methods × IPPO/MAPPO/HAPPO, Level 1 |
| [`non_stationary.sh`](non_stationary.sh) | App. K, Table 10 | Online EWC under pot-size/soup-timer/sticky-action/slippery-tile non-stationarity, isolated and combined |
| [`forced_coordination.sh`](forced_coordination.sh) | App. L.1, Table 11 | Online EWC with/without partitioned layouts requiring cross-counter hand-offs |
| [`designated_roles.sh`](designated_roles.sh) | App. L.2, Table 12 | Homogeneous vs. heterogeneous (chef/waiter) 2-agent teams, Level 1 |
| [`reward_settings.sh`](reward_settings.sh) | App. N, Table 14 | Dense shared vs. dense individual vs. sparse shared rewards, Level 1 |
| [`beyond_overcooked.sh`](beyond_overcooked.sh) | App. O.4, Fig. 20, Table 15 | Online EWC agent-count sweeps in JaxNav, SMAX, and MPE SimpleSpread |
| [`curriculum_learning.sh`](curriculum_learning.sh) | §4.3/App. F, Table 8 | Easy→medium→hard curriculum (`--env.curriculum`) vs. single-difficulty training, equal data budget |

### Prerequisites for `partner_adaptation.sh`

It reads population partners from a pre-generated BRDiv population under
`experiments/partner_adaptation/partner_agents/BRDiv_population/<layout>/`. If that's missing
for a layout, generate it first:

```bash
scripts/run_teammate_generation.sh <device> <difficulty> <layout_idx> <layout_name>
```

`scripts/run_br.sh` is a minimal single-invocation example of the same
`experiments.partner_adaptation.run_br` entry point `partner_adaptation.sh` sweeps over.