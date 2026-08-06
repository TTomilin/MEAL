#!/bin/bash
# Reproduces the partial-observability comparison: paper Section 5.6,
# Figure 7 (IPPO/MAPPO/HAPPO, full vs partial observability, MLP vs CNN
# encoder, on Levels 1 and 2, with Online EWC over 20-task sequences).
#
# Sweeps 3 algorithms x 3 variants (FO-mlp, PO-mlp, PO-cnn) x 2 difficulty
# levels x 10 seeds = 180 runs. FO+CNN is intentionally omitted -- Figure 7
# doesn't include it (the paper's Section 5.3 ablation already shows CNNs
# hurt on small fully-observable grids; PO is where the CNN's spatial
# inductive bias pays off).
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

algos=(ippo mappo happo)
difficulties=(easy medium)
seeds=(1 2 3 4 5 6 7 8 9 10)

# name, env flags, encoder flags
variants=(
    "fo_mlp|--env.no-partial-observability|--encoder mlp"
    "po_mlp|--env.partial-observability|--encoder mlp"
    "po_cnn|--env.partial-observability|--encoder cnn"
)

for algo in "${algos[@]}"; do
    for diff in "${difficulties[@]}"; do
        for variant_spec in "${variants[@]}"; do
            IFS='|' read -r variant_name env_flag encoder_flag <<< "${variant_spec}"
            for seed in "${seeds[@]}"; do
                job_name="MEAL_po_${algo}_${diff}_${variant_name}_seed${seed}"
                cmd="python -m experiments.train ${algo} \
                    --cl-method ewc --importance-mode online \
                    ${encoder_flag} \
                    --seq-length 20 \
                    --seed ${seed} \
                    --tags PARTIAL_OBSERVABILITY \
                    env:overcooked --env.difficulty ${diff} ${env_flag}"
                submit_job "${job_name}" "02:00:00" "${cmd}"
            done
        done
    done
done

summarize
