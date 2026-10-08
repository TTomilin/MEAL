#!/bin/bash
# Generates BRDiv partner populations for the continual partner-adaptation (CPA) pipeline, one job per
# (layout, generation seed). Each job trains one population of PARTNER_POP_SIZE members with num_seeds=1;
# the generation seed (not an ego seed) is what makes populations differ.
#
# Output per job: checkpoints/brdiv_<layout>_pop<size>_gseed<seed>/ with params_seed0_agent<j>.pt,
# config.pckl and generation.json (resolved settings, interaction accounting, explicit member list).
# Nothing is written to the repository's bundled populations. After the jobs finish, build the bank:
#   python -m experiments.partner_adaptation.partner_bank --layout <layout> --out <bank.json> \
#       --generation-dirs checkpoints/brdiv_<layout>_pop3_gseed1001 ...
#
# Env vars (all optional), on top of those in scripts/_common.sh:
#   LAYOUTS="coord_ring cramped_room"   layouts to generate; asymm_advantages and counter_circuit are
#                                       supported but not part of the default
#   GEN_SEEDS="1001 1002 1003"          generation seeds, one population each
#   PARTNER_POP_SIZE=3                  members per population
#   TOTAL_TIMESTEPS=<n>                 agent-transition budget per population. Unset keeps the BRDiv
#                                       default of partner_generation/run.py (2.5e8); every run reports the
#                                       joint-environment vs agent transitions it actually trained on.
#   WANDB_MODE=online                   online | offline | disabled
#   TIME_BUDGET=24:00:00                SLURM time limit per job (a placeholder: not measured)
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to skip SLURM; default is a preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

read -r -a layouts <<< "${LAYOUTS:-coord_ring cramped_room}"
read -r -a seeds <<< "${GEN_SEEDS:-1001 1002 1003}"
pop_size="${PARTNER_POP_SIZE:-3}"
budget_flag=""
if [ -n "${TOTAL_TIMESTEPS:-}" ]; then
    budget_flag="--total-timesteps ${TOTAL_TIMESTEPS}"
fi

for layout in "${layouts[@]}"; do
    for seed in "${seeds[@]}"; do
        job_name="MEAL_brdiv_${layout}_pop${pop_size}_gseed${seed}"
        cmd="python -m experiments.partner_adaptation.partner_generation.run \
            --layout-name ${layout} \
            --seed ${seed} \
            --num-seeds 1 \
            --partner-pop-size ${pop_size} \
            --mode ${WANDB_MODE:-online} \
            ${budget_flag}"
        submit_job "${job_name}" "${TIME_BUDGET:-24:00:00}" "${cmd}"
    done
done

summarize
