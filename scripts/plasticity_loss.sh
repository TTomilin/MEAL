#!/bin/bash
# Reproduces the loss-of-plasticity experiment: paper Section 5.5, Table 4,
# Figure 6 (AUC-loss and dormant-neuron ratio on a Level 1 10-task sequence,
# repeated back-to-back 1x / 3x / 10x via --repeat-sequence).
#
# Sweeps 3 repetition counts x 10 seeds = 30 IPPO runs. Uses plain fine-tuning
# (--cl-method ft, no CL protection) since this experiment isolates network
# capacity/plasticity loss from repeated gradient updates, not catastrophic
# forgetting -- the paper's Section 5.5 text doesn't name a CL method, and FT
# is MEAL's naive "no CL mechanism" baseline used everywhere else in the paper.
#
# AUC-loss and dormant-ratio themselves aren't computed by this script -- they
# require post-hoc analysis of activations across training (see
# experiments/results/ for that tooling); this script only produces the runs.
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

repetitions=(1 3 10)
seeds=(1 2 3 4 5 6 7 8 9 10)

for reps in "${repetitions[@]}"; do
    # Base sequence is 10 tasks; wall-clock scales with repeat_sequence.
    time_budget="00:30:00"
    if [ "${reps}" -ge 10 ]; then
        time_budget="04:00:00"
    fi
    for seed in "${seeds[@]}"; do
        job_name="MEAL_plasticity_reps${reps}_seed${seed}"
        cmd="python -m experiments.train ippo \
            --cl-method ft \
            --seq-length 10 \
            --repeat-sequence ${reps} \
            --seed ${seed} \
            --tags PLASTICITY_LOSS \
            env:overcooked --env.difficulty easy"
        submit_job "${job_name}" "${time_budget}" "${cmd}"
    done
done

summarize
