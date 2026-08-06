#!/bin/bash
# Reproduces the forced-coordination comparison: paper Appendix L.1, Table 11
# (Online EWC, with and without forced coordination, on 20-task sequences
# across all three difficulty levels -- forced coordination partitions each
# generated layout so no single agent's reachable region admits a full
# cook-deliver cycle, requiring cross-counter hand-offs).
#
# Maps to --env.separated-agents: the CLI's layout-generator constraint that
# agents start in different connected regions. This is the closest current
# equivalent to the paper's dedicated forced_coordination generator flag
# (which additionally guarantees no agent's region alone admits a full
# cook-deliver cycle, not just that agents start apart) -- see
# meal/env/overcooked/README.md.
#
# Sweeps 2 conditions (off/on) x 3 difficulty levels x 10 seeds = 60 runs.
# "Off" duplicates the online_ewc leg of baseline_comparison.sh, included
# here too so this script is standalone-runnable.
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

difficulties=(easy medium hard)
seeds=(1 2 3 4 5 6 7 8 9 10)

conditions=(
    "off|"
    "on|--env.separated-agents"
)

for diff in "${difficulties[@]}"; do
    for cond_spec in "${conditions[@]}"; do
        cond_name="${cond_spec%%|*}"
        cond_flags="${cond_spec#*|}"
        for seed in "${seeds[@]}"; do
            job_name="MEAL_forcedcoord_${diff}_${cond_name}_seed${seed}"
            cmd="python -m experiments.train ippo \
                --cl-method ewc --importance-mode online \
                --seq-length 20 \
                --seed ${seed} \
                --tags FORCED_COORDINATION \
                env:overcooked --env.difficulty ${diff} ${cond_flags}"
            submit_job "${job_name}" "02:00:00" "${cmd}"
        done
    done
done

summarize
