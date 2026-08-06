#!/bin/bash
# Reproduces the continual partner-adaptation experiment: paper Section 5.7,
# Figure 8 (an ego agent sequentially adapts to 8 diverse partners -- 5
# heuristic + 3 population -- across 4 original Overcooked layouts, comparing
# FT / Online MAS / Online EWC and multi-head vs single-head policies).
#
# This uses experiments/partner_adaptation/run_br.py (best-response training
# against a fixed partner sequence), not experiments/train.py -- it's a
# different pipeline from the rest of these scripts. See scripts/run_br.sh
# for a single-invocation example of the same entry point.
#
# Sweeps 4 layouts x 3 methods x 2 head configs x 10 seeds = 240 runs.
# num_heuristic_partners=5 + num_population_partners=3 = 8 partners per run,
# 1e8 steps each (both default).
#
# Prerequisite: population partners are read from a pre-generated BRDiv
# population under experiments/partner_adaptation/partner_agents/BRDiv_population/<layout>/
# -- see scripts/run_teammate_generation.sh to generate one if it's missing.
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

layouts=(cramped_room asymm_advantages coord_ring counter_circuit)
seeds=(1 2 3 4 5 6 7 8 9 10)

# name, extra CLI flags (reg_coef intentionally omitted -- auto-resolved)
methods=(
    "ft|--cl-method ft"
    "online_mas|--cl-method mas --importance-mode online"
    "online_ewc|--cl-method ewc --importance-mode online"
)

head_configs=(
    "multihead|--use-multihead"
    "singlehead|--no-use-multihead"
)

for layout in "${layouts[@]}"; do
    for method_spec in "${methods[@]}"; do
        method_name="${method_spec%%|*}"
        method_flags="${method_spec#*|}"
        for head_spec in "${head_configs[@]}"; do
            head_name="${head_spec%%|*}"
            head_flags="${head_spec#*|}"
            for seed in "${seeds[@]}"; do
                job_name="MEAL_partner_${layout}_${method_name}_${head_name}_seed${seed}"
                cmd="python -m experiments.partner_adaptation.run_br \
                    --layout-name ${layout} \
                    ${method_flags} \
                    ${head_flags} \
                    --num-heuristic-partners 5 \
                    --num-population-partners 3 \
                    --seed ${seed} \
                    --tags PARTNER_ADAPTATION"
                submit_job "${job_name}" "03:00:00" "${cmd}"
            done
        done
    done
done

summarize
