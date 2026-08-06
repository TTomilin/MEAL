#!/bin/bash
# Reproduces the task-sequence-length comparison: paper Section 5.4, Table 5
# (standard EWC vs Online EWC on Level 1 sequences of length 10 vs 100 --
# short sequences make the two look similarly stable; only over 100 tasks
# does standard EWC's cumulative Fisher penalty over-regularize while Online
# EWC's decaying importance keeps plasticity).
#
# Sweeps 2 methods x 2 sequence lengths x 10 seeds = 40 IPPO runs on Level 1.
# Note: seq_length=100 is ~5x the default 20-task run, so budget wall-clock
# accordingly (see the --time budget below, generously sized off the paper's
# ~2 min / 1e8-step figure).
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

seq_lengths=(10 100)
seeds=(1 2 3 4 5 6 7 8 9 10)

methods=(
    "ewc|--cl-method ewc --importance-mode multi"
    "online_ewc|--cl-method ewc --importance-mode online"
)

for method_spec in "${methods[@]}"; do
    method_name="${method_spec%%|*}"
    method_flags="${method_spec#*|}"
    for seq_len in "${seq_lengths[@]}"; do
        # ~2 min per 1e8-step task on an H100 (paper Section 5), plus margin.
        time_budget="00:15:00"
        if [ "${seq_len}" -ge 100 ]; then
            time_budget="06:00:00"
        fi
        for seed in "${seeds[@]}"; do
            job_name="MEAL_seqlen_${method_name}_len${seq_len}_seed${seed}"
            cmd="python -m experiments.train ippo \
                ${method_flags} \
                --seq-length ${seq_len} \
                --seed ${seed} \
                --tags SEQUENCE_LENGTH \
                env:overcooked --env.difficulty easy"
            submit_job "${job_name}" "${time_budget}" "${cmd}"
        done
    done
done

summarize
