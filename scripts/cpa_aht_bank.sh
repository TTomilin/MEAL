#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
# 16384 envs peak ~11 GB on a 20 GB gpu_mig slice; ~24k envs is the ceiling
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.95}"
: "${MEAL_CPA_ROOT:=${PWD}/cpa_root}"
# Concord's CNN rollout needs a long one-off ptxas compile; cache it across jobs
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${MEAL_CPA_ROOT}/jax_cache}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-5}"

LAYOUT="${LAYOUT:-coord_ring}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-2048}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1e7}"
# a quarter of TOTAL_TIMESTEPS so shaping reaches 0 within every stage
REWARD_SHAPING_HORIZON="${REWARD_SHAPING_HORIZON:-$(awk -v t="${TOTAL_TIMESTEPS}" 'BEGIN { printf "%.6g", t / 4 }')}"
# optional override for the partner-bank manifest path
BANK="${BANK:-${MEAL_CPA_ROOT}/partners/bank_${LAYOUT}${BANK_SUFFIX:-}.json}"
[[ -f "${BANK}" ]] || { echo "${BANK} not found: run scripts/aht_fetch_partners.sh first" >&2; exit 1; }
# optional sub-directory under runs/aht_${LAYOUT} (e.g. r1/ft/seed0)
RUN_DIR="${MEAL_CPA_ROOT}/runs/aht_${LAYOUT}${RUN_TAG:+/${RUN_TAG}}"

args=(--layout-name "${LAYOUT}" --partner-bank "${BANK}" --seed "${SEED}"
      --num-envs "${NUM_ENVS}" --num-steps 400 --total-timesteps "${TOTAL_TIMESTEPS}"
      --reward-shaping-horizon "${REWARD_SHAPING_HORIZON}"
      --no-use-task-id --no-use-multihead
      --eval-schedule pilot --checkpoint-path "${RUN_DIR}"
      --group "cpa_aht_${LAYOUT}" --tags cpa aht "${LAYOUT}")
[[ -n "${CL_METHOD:-}" ]] && args+=(--cl-method "${CL_METHOD}")
# optional space-separated partner indices that set stage order
[[ -n "${STAGE_ORDER:-}" ]] && args+=(--stage-order ${STAGE_ORDER})
if command -v nvidia-smi >/dev/null; then
    mkdir -p "${RUN_DIR}"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 30 \
        > "${RUN_DIR}/gpu_util_${SLURM_JOB_ID:-local}.csv" 2>&1 &
    trap 'kill $! 2>/dev/null || true' EXIT
fi
# shellcheck disable=SC2086
python -m experiments.partner_adaptation.run_br "${args[@]}" ${EXTRA_ARGS:-}
