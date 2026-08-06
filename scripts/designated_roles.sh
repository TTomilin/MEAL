#!/bin/bash
# Reproduces the designated-roles comparison: paper Appendix L.2, Table 12
# (homogeneous vs. heterogeneous 2-agent teams over 20-task Level 1
# sequences, IPPO + EWC, shared rewards). In the heterogeneous condition,
# each task randomly assigns one agent the "chef" role (loads onions into
# the pot, can't pick up plates) and the other the "waiter" role (delivers
# soup, can't pick up onions) -- roles are re-sampled per task.
#
# Maps to --env.complementary-restrictions, which implements exactly this
# chef/waiter split (see meal/env/overcooked/generation/sequence_loader.py:
# _make_restrictions).
#
# Sweeps 2 conditions (homogeneous/heterogeneous) x 10 seeds = 20 runs, Level
# 1 only (that's all Table 12 covers).
#
# Note: paper Figure 17 (role-specialization heterogeneity index) and
# Table 13 (coordination forgetting) reuse baseline_comparison.sh's FT/MAS/
# EWC/Online-EWC runs with a different post-hoc metric computed over the
# same rollouts -- they're not a separate sweep, so there's no script for
# them here.
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

seeds=(1 2 3 4 5 6 7 8 9 10)

conditions=(
    "homogeneous|"
    "heterogeneous|--env.complementary-restrictions"
)

for cond_spec in "${conditions[@]}"; do
    cond_name="${cond_spec%%|*}"
    cond_flags="${cond_spec#*|}"
    for seed in "${seeds[@]}"; do
        job_name="MEAL_roles_${cond_name}_seed${seed}"
        cmd="python -m experiments.train ippo \
            --cl-method ewc --importance-mode multi \
            --num-agents 2 \
            --seq-length 20 \
            --seed ${seed} \
            --tags DESIGNATED_ROLES \
            env:overcooked --env.difficulty easy ${cond_flags}"
        submit_job "${job_name}" "02:00:00" "${cmd}"
    done
done

summarize
