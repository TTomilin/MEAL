#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu HF_HUB_DISABLE_PROGRESS_BARS=1
: "${MEAL_CPA_ROOT:=${PWD}/cpa_root}"
export MEAL_CPA_ROOT

LAYOUTS="${LAYOUTS:-cramped_room asymm_advantages coord_ring counter_circuit forced_coord}"
# shellcheck disable=SC2086
python -m experiments.partner_adaptation.jax_aht_bridge.fetch --layouts ${LAYOUTS} \
    --root "${MEAL_CPA_ROOT}/partners" ${FETCH_ARGS:-}
ls -l "${MEAL_CPA_ROOT}/partners"/bank_*.json
