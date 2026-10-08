#!/bin/bash
# One direct run of experiments/partner_adaptation/run_br.py (the entry point partner_adaptation.sh sweeps over).
# For the 24-partner CPA pilot (generation, banks, profiling, resumable training) use scripts/cpa_pilot.sh instead.
#
#   scripts/run_br.sh [gpu_index] [layout_name] [extra run_br options ...]
#   scripts/run_br.sh 0 cramped_room --cl-method ewc --total-timesteps 4915200
DEFAULTVALUE=0
device="${1:-$DEFAULTVALUE}"
layout_name="${2:-"cramped_room"}"
shift $(( $# < 2 ? $# : 2 ))

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
CUDA_VISIBLE_DEVICES=${device} LD_LIBRARY_PATH="" nice -n 5 python -m experiments.partner_adaptation.run_br \
    --layout-name "${layout_name}" "$@"
