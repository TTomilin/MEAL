#!/bin/bash
# Shared helpers for the experiment-reproduction scripts in this directory.
# Not meant to be run directly -- sourced by baseline_comparison.sh,
# n_agent_sweep.sh, ablation_study.sh, sequence_length.sh, plasticity_loss.sh,
# partial_observability.sh, and partner_adaptation.sh.
#
# Env vars that control every script that sources this file:
#   RUN=1               Actually submit/execute jobs. Without it, every script
#                       only *previews* the commands it would run and how many
#                       -- these sweeps launch dozens to low-hundreds of
#                       multi-hour GPU runs, so preview before committing.
#   MEAL_LOCAL=1        Run jobs sequentially in this shell instead of
#                       `sbatch`-ing them (auto-selected anyway if `sbatch`
#                       isn't on PATH -- set this to force local mode on a
#                       machine that has SLURM but where you don't want to use it).
#   CONDA_ENV           Conda env to activate (default: meal).
#   SLURM_PARTITION     SLURM partition (default: gpu_h100).
#
# Example:
#   ./baseline_comparison.sh              # preview: prints every job + a count
#   RUN=1 ./baseline_comparison.sh         # actually submits to SLURM
#   RUN=1 MEAL_LOCAL=1 ./baseline_comparison.sh   # actually runs, sequentially, no SLURM

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-meal}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu_h100}"
LOG_DIR="${REPO_ROOT}/scripts/logs"

JOB_COUNT=0

# submit_job <job_name> <time_budget> <command string>
#
# Preview-only unless RUN=1. When RUN=1: either sbatch's `<command string>` as
# a single-GPU job, or (MEAL_LOCAL=1 / no sbatch on PATH) runs it sequentially
# in this shell.
submit_job() {
    local job_name="$1"
    local time_budget="$2"
    local cmd="$3"
    JOB_COUNT=$((JOB_COUNT + 1))

    if [ -z "${RUN:-}" ]; then
        echo "[preview ${JOB_COUNT}] ${job_name}: ${cmd}"
        return
    fi

    mkdir -p "${LOG_DIR}"

    if [ -z "${MEAL_LOCAL:-}" ] && command -v sbatch >/dev/null 2>&1; then
        cat <<EOF | sbatch
#!/bin/bash
#SBATCH -p ${SLURM_PARTITION}
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time ${time_budget}
#SBATCH --gres gpu:1
#SBATCH --job-name=${job_name}
#SBATCH -o ${LOG_DIR}/%j_${job_name}.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${REPO_ROOT}
${cmd}
EOF
        sleep 0.1
    else
        echo ">>> [local $(date +%H:%M:%S)] ${job_name}"
        ( cd "${REPO_ROOT}" && eval "${cmd}" ) 2>&1 | tee -a "${LOG_DIR}/${job_name}.out"
    fi
}

# Call once at the very end of each script, after all submit_job calls.
summarize() {
    if [ -z "${RUN:-}" ]; then
        echo
        echo "Preview only: ${JOB_COUNT} job(s) listed above, nothing submitted."
        echo "Re-run with RUN=1 to actually submit/execute them."
    else
        echo
        echo "${JOB_COUNT} job(s) submitted/executed."
    fi
}
