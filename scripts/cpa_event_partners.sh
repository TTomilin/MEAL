#!/bin/bash
# Train event-reward IPPO partners with jax-aht (one shaping draw per setting).
#
#   LAYOUT=coord_ring ./scripts/cpa_event_partners.sh
#   LAYOUT=cramped_room N_SETTINGS=4 PARALLEL=2 ./scripts/cpa_event_partners.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MEAL_CPA_ROOT:=${PWD}/cpa_root}"
: "${LAYOUT:?LAYOUT is required}"
N_SETTINGS="${N_SETTINGS:-12}"
SETTING_SEED="${SETTING_SEED:-0}"
SEEDS="${SEEDS:-2}"
PARALLEL="${PARALLEL:-3}"
AHT_DIR="${AHT_DIR:-third_party/jax-aht}"
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
    payload="$(SETTING_SEED="${SETTING_SEED}" K="${k}" python - <<'PY'
import json, os, random
k = int(os.environ["K"])
rng = random.Random(int(os.environ["SETTING_SEED"]) + k)
choices = {
    "PLACEMENT_IN_POT_REW": (-3, 0, 3),
    "ONION_PICKUP_REWARD": (-1, 0, 1),
    "PLATE_PICKUP_REWARD": (-3, 0, 3),
    "SOUP_PICKUP_REWARD": (-3, 0, 5),
    "COUNTER_DROP_REWARD": (-1, 0, 1),
    "COUNTER_PICKUP_REWARD": (-1, 0, 1),
}
while True:
    weights = {name: rng.choice(vals) for name, vals in choices.items()}
    if any(weights.values()):
        break
overrides = [f"task.ENV_KWARGS.reward_shaping_params.{n}=[{w},{w}]" for n, w in sorted(weights.items())]
print(json.dumps({"weights": weights, "overrides": overrides}))
PY
)"
    PATH_OUT="${run_dir}/weights.json" PAYLOAD="${payload}" python -c \
        "import json,os,pathlib; pathlib.Path(os.environ['PATH_OUT']).write_text(json.dumps(json.loads(os.environ['PAYLOAD'])['weights'], indent=2)+'\n')"
    local -a overrides
    mapfile -t overrides < <(PAYLOAD="${payload}" python -c \
        "import json,os; print('\n'.join(json.loads(os.environ['PAYLOAD'])['overrides']))")

    (
        cd "${AHT_DIR}"
        export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}"
        python marl/run.py task=overcooked-v1/"${LAYOUT}" label="event_${k}" \
            algorithm.NUM_SEEDS="${SEEDS}" logger.mode=disabled \
            hydra.run.dir="${run_dir}" \
            "${overrides[@]}"
    )
}

for k in $(seq 1 "${N_SETTINGS}"); do
    while (( $(jobs -pr | wc -l) >= PARALLEL )); do
        wait -n || true
    done
    draw_and_launch "${k}" &
done
wait
echo "event partners under ${OUT_ROOT}"
