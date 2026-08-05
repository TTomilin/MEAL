#!/bin/bash
# Reproduces the CL-method baseline comparison: paper Section 5.1, Table 2,
# Figures 3 & 4 (soup delivery / forgetting / forward transfer per method,
# per difficulty level).
#
# Sweeps 8 CL methods x 3 difficulty levels x 5 seeds = 120 IPPO runs on
# Overcooked, 20-task sequences. All PPO/network hyperparameters are left at
# their defaults, which already match paper Table 6 (steps_per_task=1e8,
# num_envs=2048, num_steps=400, update_epochs=8, num_minibatches=16, ...);
# reg_coef is likewise left unset so it auto-resolves to the paper's values
# (EWC 1e11, MAS 1e9, L2 1e7 -- see resolve_reg_coef in experiments/algo_common.py).
#
# "EWC"/"MAS" (Table 2) vs "Online EWC"/"Online MAS" differ only in
# --importance-mode: multi (cumulative Fisher/importance, the classic
# formulation) vs online (exponential running average, the default).
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

difficulties=(easy medium hard)
seeds=(1 2 3 4 5)

# name, extra CLI flags (reg_coef intentionally omitted -- auto-resolved)
methods=(
    "ft|--cl-method ft"
    "ewc|--cl-method ewc --importance-mode multi"
    "mas|--cl-method mas --importance-mode multi"
    "online_ewc|--cl-method ewc --importance-mode online"
    "online_mas|--cl-method mas --importance-mode online"
    "agem|--cl-method agem"
    "er_ace|--cl-method er_ace"
    "packnet|--cl-method packnet"
)

for method_spec in "${methods[@]}"; do
    method_name="${method_spec%%|*}"
    method_flags="${method_spec#*|}"
    for diff in "${difficulties[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="MEAL_baseline_${method_name}_${diff}_seed${seed}"
            cmd="python -m experiments.train ippo \
                ${method_flags} \
                --seq-length 20 \
                --seed ${seed} \
                --tags BASELINE_COMPARISON \
                env:overcooked --env.difficulty ${diff}"
            submit_job "${job_name}" "02:00:00" "${cmd}"
        done
    done
done

summarize
