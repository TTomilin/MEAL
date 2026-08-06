#!/bin/bash
# Reproduces the curriculum-learning comparison: paper Section 4.3 / Appendix
# F, Table 8 (does training on an easy->medium->hard curriculum improve
# performance on harder tasks vs. training on a single difficulty throughout,
# under an equal data budget).
#
# Curriculum sequence: 15 tasks, 5 each of easy/medium/hard in ascending
# order (--env.curriculum overrides --env.difficulty and splits the
# sequence accordingly -- see OvercookedEnvConfig.curriculum in
# experiments/envs/overcooked.py). Default baselines: a 15-task pure-medium
# sequence and a 15-task pure-hard sequence, matching the curriculum run's
# data budget so the same task-index windows are comparable.
#
# Table 8's columns read off specific windows of these runs, not separate
# launches: "Medium (6-10)" = tasks 6-10 of the curriculum run vs. tasks
# 6-10 of the pure-medium run; "Hard (11-15)" = tasks 11-15 of the
# curriculum run vs. tasks 11-15 of the pure-hard run. Slicing those windows
# out of the logged per-task curves is a post-processing step (see
# experiments/results/), not something this script does.
#
# Sweeps 3 conditions (curriculum, default-medium, default-hard) x 10 seeds =
# 30 IPPO + EWC runs.
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

seeds=(1 2 3 4 5 6 7 8 9 10)

# name, env flags
conditions=(
    "curriculum|--env.curriculum"
    "default_medium|--env.difficulty medium"
    "default_hard|--env.difficulty hard"
)

for cond_spec in "${conditions[@]}"; do
    cond_name="${cond_spec%%|*}"
    cond_flags="${cond_spec#*|}"
    for seed in "${seeds[@]}"; do
        job_name="MEAL_curriculum_${cond_name}_seed${seed}"
        cmd="python -m experiments.train ippo \
            --cl-method ewc \
            --seq-length 15 \
            --seed ${seed} \
            --tags CURRICULUM_LEARNING \
            env:overcooked ${cond_flags}"
        submit_job "${job_name}" "02:30:00" "${cmd}"
    done
done

summarize
