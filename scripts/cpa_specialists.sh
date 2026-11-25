#!/bin/bash
# Train one specialist ego per bank partner (single-stage BR via cpa_aht_bank.sh).
#
#   LAYOUT=coord_ring ./scripts/cpa_specialists.sh
#   LAYOUT=coord_ring PARTNERS="0 2" DRY_RUN=1 ./scripts/cpa_specialists.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
: "${MEAL_CPA_ROOT:=${PWD}/cpa_root}"

: "${LAYOUT:?LAYOUT is required}"
BANK="${BANK:-${MEAL_CPA_ROOT}/partners/bank_${LAYOUT}.json}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-1024}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-4915200}"
GROUP="${GROUP:-cpa_specialists_${LAYOUT}}"
ENV_DYNAMICS="${ENV_DYNAMICS:-aht}"
PYTHON="${PYTHON:-python}"
USER_EXTRA="${EXTRA_ARGS:-}"

[[ -f "${BANK}" ]] || { echo "${BANK} not found" >&2; exit 1; }

# One line per partner: index<TAB>label<TAB>payload_or_config_sha256
PARTNER_ROWS="$("${PYTHON}" -c "
import sys
from experiments.partner_adaptation.partner_bank import load_partner_bank, partner_identity
bank = load_partner_bank(sys.argv[1], sys.argv[2])
for r in bank.records:
    ident = partner_identity(r)
    digest = ident.get('payload_sha256') or ident.get('config_sha256') or ''
    print('%s\t%s\t%s' % (r.partner_id, ident['label'], digest))
" "${BANK}" "${LAYOUT}")"

if [[ -n "${PARTNERS:-}" ]]; then
    # shellcheck disable=SC2206
    SELECTED=(${PARTNERS})
else
    SELECTED=()
    while IFS=$'\t' read -r index _label _digest; do
        SELECTED+=("${index}")
    done <<< "${PARTNER_ROWS}"
fi

partner_status() {
    local index="$1" expected="$2" tag_root="$3" seed="$4"
    "${PYTHON}" -c "
import json, sys
from pathlib import Path
root, index, expected, seed = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
if not root.is_dir():
    print('missing')
    raise SystemExit
matched = mismatched = incomplete = False
for run in sorted(p for p in root.iterdir() if p.is_dir()):
    status_path = run / 'status.json'
    if not status_path.is_file():
        continue
    state = json.loads(status_path.read_text()).get('state')
    params = run / ('params_seed%s.pt' % seed)
    done = state in ('complete', 'profile') and params.is_file()
    if not done:
        incomplete = True
        continue
    bank = json.loads((run / 'partner_bank.json').read_text())
    ident = bank[index]
    digest = ident.get('payload_sha256') or ident.get('config_sha256') or ''
    if digest == expected:
        matched = True
    else:
        mismatched = True
if matched:
    print('skipped')
elif mismatched:
    print('stale')
elif incomplete:
    print('incomplete')
else:
    print('missing')
" "${tag_root}" "${index}" "${expected}" "${seed}"
}

for index in "${SELECTED[@]}"; do
    row=""
    while IFS=$'\t' read -r cand_index cand_label cand_digest; do
        if [[ "${cand_index}" == "${index}" ]]; then
            row="${cand_index}"
            label="${cand_label}"
            digest="${cand_digest}"
            break
        fi
    done <<< "${PARTNER_ROWS}"
    [[ -n "${row}" ]] || { echo "partner index ${index} not in ${BANK}" >&2; exit 1; }
    RUN_TAG="specialists/${label//\//__}"
    tag_root="${MEAL_CPA_ROOT}/runs/aht_${LAYOUT}/${RUN_TAG}"
    status="$(partner_status "${index}" "${digest}" "${tag_root}" "${SEED}")"
    if [[ "${status}" == "skipped" ]]; then
        echo "${index} ${label} skipped"
        continue
    fi
    outcome="trained"
    [[ "${status}" == "stale" || "${status}" == "incomplete" ]] && outcome="${status}"
    if [[ " ${USER_EXTRA} " =~ [[:space:]]--env[-_]dynamics([[:space:]=]) ]]; then
        EXTRA_ARGS="--only-stages ${index} --no-record-video ${USER_EXTRA}"
    else
        EXTRA_ARGS="--only-stages ${index} --env-dynamics ${ENV_DYNAMICS} --no-record-video ${USER_EXTRA}"
    fi
    export MEAL_CPA_ROOT PYTHON LAYOUT BANK SEED NUM_ENVS TOTAL_TIMESTEPS ENV_DYNAMICS RUN_TAG GROUP EXTRA_ARGS
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "LAYOUT=${LAYOUT} BANK=${BANK} SEED=${SEED} NUM_ENVS=${NUM_ENVS} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} RUN_TAG=${RUN_TAG} GROUP=${GROUP} EXTRA_ARGS=${EXTRA_ARGS} scripts/cpa_aht_bank.sh"
    else
        bash scripts/cpa_aht_bank.sh
    fi
    echo "${index} ${label} ${outcome}"
done
