## Results: download and plot

Training logs are stored on Weights & Biases (if enabled). Use the provided download scripts to export JSON/NPZ files in a layout that plotting scripts expect.

From the repo root:

```bash
# Download per‑environment evaluation curves
python -m experiments.results.download.eval \
  --project MEAL \
  --output experiments/results/data \
  --algos ippo \
  --cl_methods EWC MAS L2 FT AGEM \
  --strategy generate \
  --seq_length 10 \
  --difficulty medium \
  --num_agents 2 \
  --seeds 1 2 3

# (Optional) Download training curves only
python -m experiments.results.download.train \
  --project MEAL \
  --output experiments/results/data \
  --algos ippo \
  --cl_methods EWC MAS L2 FT AGEM \
  --strategy generate \
  --seq_length 10 \
  --difficulty medium \
  --num_agents 2 \
  --seeds 1 2 3
```

The resulting directory layout matches what the plotting code expects:

```
experiments/results/data/
  <algo>/
    <METHOD>/
      level_<L>/
        agents_<N>/
          <strategy>_<seq_len>/
            seed_<k>/
              0_<env>_soup.json           # per‑env eval
              1_<env>_soup.json
              ...
              training_soup.json          # optional: training aggregate
```
### Plot task‑progress (average performance or forgetting vs #tasks)

```bash
python -m experiments.results.plotting.plot_tasks_progress \
  --algo ippo \
  --method Online_EWC \
  --strategy generate \
  --seq-len 10 \
  --seeds 1 2 3 \
  --metric soup \
  --levels 1 2 3 \
  --agents 2 \
  --y performance \
  --ap-window 5 \
  --plot-name perf_online_ewc_L123
```

### Compare average performance/forgetting across numbers of agents

```bash
python -m experiments.results.plotting.compare_agents_histogram \
  --data_root data \
  --algorithm ippo \
  --method Online_EWC \
  --env overcooked \
  --strategy generate \
  --seq_len 10 \
  --seeds 1 2 3 \
  --num_agents 2 3 4 \
  --end_window_evals 10 \
  --plot_name agents_hist
```

For JaxNav experiments, set `--env jaxnav` (the script will use success/return metrics instead of soup).

### Compare reward variations (default vs sparse vs individual)

```bash
python -m experiments.results.plotting.variations \
  --data-root data \
  --algo ippo \
  --methods Online_EWC MAS FT \
  --strategy generate \
  --seq-len 10 \
  --seeds 1 2 3
```