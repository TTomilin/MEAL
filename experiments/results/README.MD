## Results: download and plot

Pipeline: training logs live on Weights & Biases → **download** scripts export them to local
JSON/NPZ files in a fixed directory layout → **plotting**/**numerical** scripts read that
layout to produce figures and LaTeX tables. All three stages are run as `python -m` modules
from the repo root.

## 1. Downloading

Three download scripts (`experiments/results/download/{eval,train,layouts}.py`) share one CLI
parser (`download/common.py::cli()`), so every flag below works identically for all three —
only what gets fetched differs:

- **`download.eval`** — per-environment **evaluation** curves. Auto-detects the metric key per
  env from `config.env_name`: Overcooked → `Evaluation/Soup_Scaled`, JaxNav →
  `Evaluation/Success`, MPE → `Evaluation/CoverageFraction`, SMAX → `Evaluation/WinRate`. Only
  fetches keys not already saved on disk (unless `--overwrite`).
- **`download.train`** — dense **training-time** curves (Overcooked's `Soup/scaled`, logged
  every update). For SMAX/MPE, which don't log a dense per-step training metric, this falls
  back to the periodic evaluation metric for the task currently being trained.
- **`download.layouts`** — the *exact* Overcooked layouts used in each matched run (as JSON),
  so a run's tasks can be replayed later exactly as trained.

### Shared CLI flags

| Flag | Meaning |
|---|---|
| `--project <str>` (required) | W&B project to pull runs from. |
| `--output <path>` (default `data`) | Base output folder. |
| `--format {json, npz}` (default `json`) | Output file format. |
| `--algos <str...>` | Filter by `alg_name` (e.g. `ippo mappo vdn`). |
| `--cl_methods <str...>` | Filter by `cl_method` (e.g. `EWC MAS L2 FT AGEM`). |
| `--env_name <str>` | Filter by environment. |
| `--num_agents <int>` | Filter by team size. |
| `--seq_length <int>` | Filter by sequence length. |
| `--repeat_sequence <int>` | Filter by repeat-sequence value (multiplies effective seq length). |
| `--strategy {ordered, random, generate, curriculum}` | Filter by the sequence-generation strategy recorded in the run's saved config (historical — current training always uses `generate`; this only matters for filtering older runs). |
| `--difficulty <str...>` | Filter by difficulty level(s). |
| `--seeds <int...>` (default `1 2 3 4 5`) | Which seeds to fetch. |
| `--importance_mode {online, multi, last}` | Filter by EWC/MAS importance mode. |
| `--reward_settings {default, sparse, individual}` | Filter reward-setting variant experiments. |
| `--complementary_restrictions` | Only runs with complementary restrictions enabled. |
| `--layout_names <str...>` | Filter by fixed layout name(s) — presence implies partner-adaptation (BR) runs, not the CL benchmark. |
| `--wandb_tags <str...>` | Require at least one of these W&B tags. |
| `--include_runs <str...>` | Only include runs whose name contains one of these substrings. |
| `--recent_hours <float>` | Only runs started within the last N hours. |
| `--overwrite` | Re-download and overwrite files that already exist locally. |

```bash
# Evaluation curves
python -m experiments.results.download.eval \
  --project MEAL --output experiments/results/data \
  --algos ippo --cl_methods EWC MAS L2 FT AGEM \
  --seq_length 10 --difficulty medium --num_agents 2 --seeds 1 2 3

# Dense training curves
python -m experiments.results.download.train \
  --project MEAL --output experiments/results/data \
  --algos ippo --cl_methods EWC MAS L2 FT AGEM \
  --seq_length 10 --difficulty medium --num_agents 2 --seeds 1 2 3
```

### Resulting directory layout

```
experiments/results/data/
  <algo>/
    <METHOD>/
      level_<L>/
        agents_<N>/
          <strategy>_<seq_len>/
            seed_<k>/
              0_<env>_soup.json           # per-task eval curve
              1_<env>_soup.json
              ...
              training_soup.json          # optional: dense training curve
```

## 2. Plotting

`experiments/results/plotting/` — every script takes `--help` for its full flag list; below is
what each one is for, grouped by purpose.

**Core CL curves**
| Script | Purpose |
|---|---|
| `plot_tasks_progress` | Average performance or forgetting vs. #tasks seen. |
| `cumulative_eval` | Cumulative average performance over training, method or arch comparison. |
| `eval_per_task` | One subplot per CL method, one line per task/environment. |
| `training_curves` | Raw training curves from `training_soup` files (supports `repeat_sequence`). |
| `plasticity_curves` | Per-task-position curves aggregated across sequence repetitions (needs `--repeat_sequence`). |

**Comparisons / ablations**
| Script | Purpose |
|---|---|
| `algorithm_comparison` | Heatmap: algorithm × CL method, one panel per difficulty. |
| `compare_agents_histogram` | Bar chart comparing agent-count variants (Overcooked or JaxNav). |
| `forgetting_hist` / `final_performancel_hist` | Aggregated bar charts of forgetting / final performance. |
| `forward_transfer` / `forward_transfer_vs_forgetting` | Forward-transfer curves, and a forward-transfer-vs-forgetting scatter. |
| `mlp_vs_cnn_barchart` / `network_size` | Encoder-type and network-size comparisons. |
| `partial_obs` | Full- vs. partial-observability comparison (FO/PO × MLP/CNN). |
| `variations` | Reward-setting / restriction / ablation variation comparisons. |
| `ablation_hist` | General original-vs-ablated comparison bar chart, supports multiple ablations. |
| `dormant_ratio_curves` | Dormant-neuron-ratio over training. |

**Heterogeneity**
| Script | Purpose |
|---|---|
| `heterogeneity` | Role-heterogeneity index (agent action-distribution divergence) bar chart. |
| `heterogeneity_vs_performance` | Soup score vs. heterogeneity over a sequence, per focal task. |

**Partner adaptation** (`experiments/partner_adaptation/`, separate sub-project)
| Script | Purpose |
|---|---|
| `cumulative_eval_partners` / `eval_per_task_partners` | BR partner-adaptation equivalents of the core CL curve scripts. |
| `partner_adaptation_arch_hist` | Arch (multi-head vs. single-head) × CL method comparison, per layout. |

```bash
python -m experiments.results.plotting.plot_tasks_progress \
  --algo ippo --method Online_EWC --seq-len 10 --seeds 1 2 3 \
  --metric soup --levels 1 2 3 --agents 2 --y performance --ap-window 5 \
  --plot-name perf_online_ewc_L123

python -m experiments.results.plotting.compare_agents_histogram \
  --data_root data --algorithm ippo --method Online_EWC --env overcooked \
  --seq_len 10 --seeds 1 2 3 --num_agents 2 3 4 --end_window_evals 10 \
  --plot_name agents_hist
```

For JaxNav, pass `--env jaxnav` where a script takes one — it switches to success/return
metrics instead of soup.

## 3. Numerical tables

`experiments/results/numerical/` — same idea as plotting but producing LaTeX tables / printed
numbers instead of figures:

| Script | Purpose |
|---|---|
| `results_table` | Main results table: mean ± 95% CI per algo/method. |
| `algorithm_comparison` | Numeric counterpart of the plotting heatmap. |
| `agents_comparison` | Agent-count comparison table. |
| `curriculum_comparison`, `levels_vs_tasks` | Curriculum-strategy and level-vs-task-count comparisons. |
| `reward_settings` | Sparse/individual/default reward-setting comparison (extends `results_table`). |
| `restrictions` | Complementary-restrictions comparison. |
| `observable_comparison` | Full vs. partial observability comparison table. |
| `heterogeneity_table` | Mean JSD between agent action distributions, per method × difficulty. |
| `coordination_forgetting_table` | Heterogeneity-at-peak vs. forgetting, per focal task. |
| `lambda_ablation` | Ablation over the λ weighting parameter in the curve-based forgetting metric. |
| `compare_variants` | General variant-comparison table. |
| `partner_adaption`, `partner_adaption_arch_table` | Partner-adaptation (BR) result tables. |

## 4. Coordination matrix

[`coordination_matrix.py`](coordination_matrix.py) sits outside the download → plot/numerical
pipeline above: instead of reading downloaded W&B data, it loads a run's saved checkpoints
directly (flax params under `checkpoints/<env>/<cl_method>/<run_name>/`) and cross-plays every
task-`i`-trained policy against every task-`j` layout to build a coordination matrix, evaluating
actual trained behavior rather than logged metrics.

```bash
python experiments/results/coordination_matrix.py --run_dir checkpoints/overcooked/EWC/run_A
```
