#!/bin/bash
# Reproduces the reward-setting comparison: paper Appendix N, Table 14
# (dense shared vs. dense individual vs. sparse shared rewards, EWC on
# 20-task Level 1 sequences).
#
# Sweeps 3 reward settings x 10 seeds = 30 IPPO + EWC runs. Dense shared is
# the default (no flags); dense individual and sparse shared are mutually
# exclusive (--env.individual-rewards / --env.sparse-rewards).
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

seeds=(1 2 3 4 5 6 7 8 9 10)

conditions=(
    "dense_shared|"
    "dense_individual|--env.individual-rewards"
    "sparse_shared|--env.sparse-rewards"
)

for cond_spec in "${conditions[@]}"; do
    cond_name="${cond_spec%%|*}"
    cond_flags="${cond_spec#*|}"
    for seed in "${seeds[@]}"; do
        job_name="MEAL_rewards_${cond_name}_seed${seed}"
        cmd="python -m experiments.train ippo \
            --cl-method ewc --importance-mode multi \
            --seq-length 20 \
            --seed ${seed} \
            --tags REWARD_SETTINGS \
            env:overcooked --env.difficulty easy ${cond_flags}"
        submit_job "${job_name}" "02:00:00" "${cmd}"
    done
done

summarize
