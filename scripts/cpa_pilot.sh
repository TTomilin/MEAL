#!/bin/bash
# Launcher for the 24-partner continual partner adaptation (CPA) pilot. Thin wrapper around
#   python -m experiments.partner_adaptation.pilot <action> [options]
# which also works directly. `scripts/cpa_pilot.sh --help` lists the actions; `<action> --help` its options.
#
# Actions: smoke, preflight, profile, profile-brdiv, plan, generate, bank, train, resume.
# `generate`, `train` and `resume` only print their plan unless --run is given or RUN=1 (as in _common.sh).
# Choose a persistent output root with --root or MEAL_CPA_ROOT (a network volume on a pod).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m experiments.partner_adaptation.pilot "$@"
