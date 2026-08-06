#!/bin/bash
# Reproduces the N-agent team-size sweep: paper Section 5.2, Table 3
# (IPPO + Online EWC with 1-5 agents on Levels 1 and 2).
#
# Sweeps 5 team sizes x 2 difficulty levels x 10 seeds = 100 IPPO + Online EWC
# runs, 20-task sequences. All other hyperparameters default (match Table 6).
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

difficulties=(easy medium)
agent_counts=(1 2 3 4 5)
seeds=(1 2 3 4 5 6 7 8 9 10)

for diff in "${difficulties[@]}"; do
    for num_agents in "${agent_counts[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="MEAL_nagent_${diff}_${num_agents}agents_seed${seed}"
            cmd="python -m experiments.train ippo \
                --cl-method ewc --importance-mode online \
                --seq-length 20 \
                --num-agents ${num_agents} \
                --seed ${seed} \
                --tags N_AGENT_SWEEP \
                env:overcooked --env.difficulty ${diff}"
            submit_job "${job_name}" "02:00:00" "${cmd}"
        done
    done
done

summarize
