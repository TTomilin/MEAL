#!/bin/bash
# Train event-reward IPPO partners with jax-aht (one shaping draw per setting).
#
#   LAYOUT=coord_ring ./scripts/cpa_event_partners.sh
#   LAYOUT=cramped_room N_SETTINGS=4 PARALLEL=2 ./scripts/cpa_event_partners.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="${PWD}"
: "${MEAL_CPA_ROOT:=${REPO_ROOT}/cpa_root}"
: "${LAYOUT:?LAYOUT is required}"
N_SETTINGS="${N_SETTINGS:-12}"
SETTING_SEED="${SETTING_SEED:-0}"
SEEDS="${SEEDS:-1}"
PARALLEL="${PARALLEL:-3}"
PYTHON="${PYTHON:-python}"
AHT_DIR="${AHT_DIR:-third_party/jax-aht}"
# jax-aht imports brax through jaxmarl; the bridge stubs it out.
RUN_WITH_BRIDGE='import runpy, sys, experiments.partner_adaptation.jax_aht_bridge; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")'
[[ -d "${AHT_DIR}" ]] || { echo "${AHT_DIR} not found; run scripts/aht_setup.sh" >&2; exit 1; }

mkdir -p "${MEAL_CPA_ROOT}"
MEAL_CPA_ROOT="$(cd "${MEAL_CPA_ROOT}" && pwd)"
OUT_ROOT="${MEAL_CPA_ROOT}/event_partners/${LAYOUT}"
mkdir -p "${OUT_ROOT}"

draw_and_launch() {
    local k="$1"
    local run_dir="${OUT_ROOT}/event_${k}"
    if [[ -d "${run_dir}/saved_train_run" ]]; then
        echo "skip event_${k}: saved_train_run exists"
        return 0
    fi
    mkdir -p "${run_dir}"
    local payload
    payload="$(PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" \
        -m experiments.partner_adaptation.jax_aht_bridge.event_shaping \
        --k "${k}" --n-settings "${N_SETTINGS}" --seed "${SETTING_SEED}")"
    PATH_OUT="${run_dir}/weights.json" PAYLOAD="${payload}" "${PYTHON}" -c \
        "import json,os,pathlib; pathlib.Path(os.environ['PATH_OUT']).write_text(json.dumps(json.loads(os.environ['PAYLOAD'])['weights'], indent=2)+'\n')"
    local -a overrides=()
    local line
    while IFS= read -r line; do
        [[ -n "${line}" ]] && overrides+=("${line}")
    done < <(PAYLOAD="${payload}" "${PYTHON}" -c \
        "import json,os; print('\n'.join(json.loads(os.environ['PAYLOAD'])['overrides']))")

    (
        cd "${AHT_DIR}"
        export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}"
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" -c "${RUN_WITH_BRIDGE}" marl/run.py task=overcooked-v1/"${LAYOUT}" label="event_${k}" \
            algorithm.NUM_SEEDS="${SEEDS}" logger.mode=disabled \
            hydra.run.dir="${run_dir}" \
            "${overrides[@]}"
    )
}

has_wait_n=0
if (( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )); then
    has_wait_n=1
fi

failed=0
pids=()
for k in $(seq 1 "${N_SETTINGS}"); do
    while (( $(jobs -pr | wc -l) >= PARALLEL )); do
        if (( has_wait_n )); then
            wait -n || failed=1
        else
            sleep 0.2
        fi
    done
    draw_and_launch "${k}" &
    pids+=("$!")
done

if (( has_wait_n )); then
    while (( $(jobs -pr | wc -l) > 0 )); do
        wait -n || failed=1
    done
else
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
fi
(( failed == 0 )) || exit 1
echo "event partners under ${OUT_ROOT}"
