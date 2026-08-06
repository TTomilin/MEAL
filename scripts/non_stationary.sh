#!/bin/bash
# Reproduces the non-stationary-dynamics comparison: paper Appendix K,
# Table 10 (Online EWC on 20-task sequences under four additional sources of
# non-stationarity -- pot size, soup cook timer, sticky actions, slippery
# tiles -- each in isolation, then all four combined).
#
# Sweeps 6 dynamics conditions (default + 4 isolated + combined) x 3
# difficulty levels x 10 seeds = 180 runs. "Default" duplicates the online_ewc
# leg of baseline_comparison.sh, included here too so this script is
# standalone-runnable (same convention as ablation_study.sh's "original"
# variant).
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

difficulties=(easy medium hard)
seeds=(1 2 3 4 5 6 7 8 9 10)

# name, env flags (sticky_actions/slippery_tiles/random_pot_size/
# random_cook_time probabilities are difficulty-dependent, resolved inside
# the env itself -- see difficulty_config.py; "combined" == --env.non-stationary)
conditions=(
    "default|"
    "pot_size|--env.random-pot-size"
    "soup_timer|--env.random-cook-time"
    "sticky_actions|--env.sticky-actions"
    "slippery_tiles|--env.slippery-tiles"
    "combined|--env.non-stationary"
)

for diff in "${difficulties[@]}"; do
    for cond_spec in "${conditions[@]}"; do
        cond_name="${cond_spec%%|*}"
        cond_flags="${cond_spec#*|}"
        for seed in "${seeds[@]}"; do
            job_name="MEAL_nonstat_${diff}_${cond_name}_seed${seed}"
            cmd="python -m experiments.train ippo \
                --cl-method ewc --importance-mode online \
                --seq-length 20 \
                --seed ${seed} \
                --tags NON_STATIONARY \
                env:overcooked --env.difficulty ${diff} ${cond_flags}"
            submit_job "${job_name}" "02:00:00" "${cmd}"
        done
    done
done

summarize
