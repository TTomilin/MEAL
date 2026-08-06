#!/bin/bash
# Reproduces the MARL-algorithm comparison: paper Appendix J, Figure 16
# (every CL method paired with each of IPPO / MAPPO / HAPPO on Level 1, to
# check whether the CL-method rankings from Section 5.1 are specific to
# IPPO).
#
# Sweeps 3 algorithms x 8 CL methods x 10 seeds = 240 runs, 20-task Level 1
# sequences. Same method/flag mapping as baseline_comparison.sh (see there
# for why "Online EWC"/"Online MAS" are just --importance-mode online).
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

algos=(ippo mappo happo)
seeds=(1 2 3 4 5 6 7 8 9 10)

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

for algo in "${algos[@]}"; do
    for method_spec in "${methods[@]}"; do
        method_name="${method_spec%%|*}"
        method_flags="${method_spec#*|}"
        for seed in "${seeds[@]}"; do
            job_name="MEAL_algocmp_${algo}_${method_name}_seed${seed}"
            cmd="python -m experiments.train ${algo} \
                ${method_flags} \
                --seq-length 20 \
                --seed ${seed} \
                --tags ALGORITHM_COMPARISON \
                env:overcooked --env.difficulty easy"
            submit_job "${job_name}" "02:00:00" "${cmd}"
        done
    done
done

summarize
