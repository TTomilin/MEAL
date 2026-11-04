#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
: "${MEAL_CPA_ROOT:=${PWD}/cpa_root}"

ALG="${ALG:-comedi}"
LAYOUT="${LAYOUT:-coord_ring}"
SEED="${SEED:-0}"
if [[ "${ALG}" == "comedi" ]]; then POP_SIZE="${POP_SIZE:-10}"; else POP_SIZE="${POP_SIZE:-20}"; fi
NUM_ENVS="${NUM_ENVS:-1024}"  # jax-aht configs use 64; more envs = fewer updates at a fixed budget

args=(--alg "${ALG}" --layout-name "${LAYOUT}" --seed "${SEED}" --partner-pop-size "${POP_SIZE}"
      --aht-num-envs "${NUM_ENVS}" --checkpoint-path "${MEAL_CPA_ROOT}/partners/generated" --mode disabled)
[[ -n "${TOTAL_TIMESTEPS:-}" ]] && args+=(--aht-total-timesteps "${TOTAL_TIMESTEPS}")
# shellcheck disable=SC2086
python -m experiments.partner_adaptation.partner_generation.run "${args[@]}" ${EXTRA_ARGS:-}
