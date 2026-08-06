#!/bin/bash
# Reproduces the component ablation study: paper Section 5.3, Figure 5
# (removing multi-head outputs, task-id input, critic regularization, layer
# norm, and swapping the MLP encoder for a CNN, for EWC/MAS/ER-ACE).
#
# Sweeps 3 CL methods x 6 variants (original + 5 single-component ablations)
# x 10 seeds = 180 IPPO runs, 20-task sequences on Level 1 (easy)
# Override DIFFICULTY below to reproduce at another level.
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

DIFFICULTY="${DIFFICULTY:-easy}"
seeds=(1 2 3 4 5 6 7 8 9 10)

# name, extra CLI flags (each is the "original"/default config with exactly
# one component flipped off, per Figure 5's bar groups)
variants=(
    "original|"
    "reg_critic|--regularize-critic"
    "cnn|--encoder cnn"
    "no_multihead|--no-use-multihead"
    "no_task_id|--no-use-task-id"
    "no_layer_norm|--no-use-layer-norm"
)

methods=(
    "ewc|--cl-method ewc --importance-mode online"
    "mas|--cl-method mas --importance-mode online"
    "er_ace|--cl-method er_ace"
)

for method_spec in "${methods[@]}"; do
    method_name="${method_spec%%|*}"
    method_flags="${method_spec#*|}"
    for variant_spec in "${variants[@]}"; do
        variant_name="${variant_spec%%|*}"
        variant_flags="${variant_spec#*|}"
        for seed in "${seeds[@]}"; do
            job_name="MEAL_ablation_${method_name}_${variant_name}_seed${seed}"
            cmd="python -m experiments.train ippo \
                ${method_flags} \
                ${variant_flags} \
                --seq-length 20 \
                --seed ${seed} \
                --tags ABLATION_STUDY \
                env:overcooked --env.difficulty ${DIFFICULTY}"
            submit_job "${job_name}" "02:00:00" "${cmd}"
        done
    done
done

summarize
