#!/bin/bash
# Reproduces the "MEAL beyond Overcooked" agent-count sweeps: paper
# Appendix O.4, Figure 20 and Table 15 (Online EWC on 20-task sequences in
# JaxNav, SMAX, and MPE SimpleSpread, varying team size, to check whether
# the "more agents makes CL harder" trend from Overcooked (Section 5.2)
# holds in other domains).
#
# Sweeps:
#   JaxNav: 2, 3, 4 agents            (Figure 20)
#   SMAX:   5, 6, 7, 8 agents         (Table 15 -- "evaluated from five
#                                       agents upward"; ally and enemy team
#                                       kept equal-sized, i.e. NvN, since the
#                                       paper doesn't vary them independently)
#   MPE:    3, 4, 5, 6, 7, 8 agents   (Table 15; num_landmarks/num_obstacles
#                                       left at their CLI defaults, 3/4, since
#                                       the paper doesn't state values for
#                                       this specific sweep)
# x 10 seeds = (3 + 4 + 6) * 5 = 130 runs total.
#
# Caveats:
#   - JaxNav's obstacle fill ratio isn't exposed on the CLI (see
#     meal/env/jaxnav/README.md); it uses the env's own default (0.3).
#     Map size is (7x7).
#
# Usage: see scripts/_common.sh (RUN=1 to actually submit; MEAL_LOCAL=1 to
# skip SLURM; default is a dry preview).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

seeds=(1 2 3 4 5 6 7 8 9 10)

# --- JaxNav ---
jaxnav_agent_counts=(2 3 4)
for num_agents in "${jaxnav_agent_counts[@]}"; do
    for seed in "${seeds[@]}"; do
        job_name="MEAL_beyond_jaxnav_${num_agents}agents_seed${seed}"
        cmd="python -m experiments.train ippo \
            --cl-method ewc --importance-mode online \
            --num-agents ${num_agents} \
            --seq-length 20 \
            --seed ${seed} \
            --tags BEYOND_OVERCOOKED \
            env:jaxnav --env.map-dim 7"
        submit_job "${job_name}" "02:00:00" "${cmd}"
    done
done

# --- SMAX ---
smax_agent_counts=(5 6 7 8)
for num_agents in "${smax_agent_counts[@]}"; do
    for seed in "${seeds[@]}"; do
        job_name="MEAL_beyond_smax_${num_agents}agents_seed${seed}"
        cmd="python -m experiments.train ippo \
            --cl-method ewc --importance-mode online \
            --num-agents ${num_agents} \
            --seq-length 20 \
            --seed ${seed} \
            --tags BEYOND_OVERCOOKED \
            env:smax --env.num-enemies ${num_agents}"
        submit_job "${job_name}" "02:00:00" "${cmd}"
    done
done

# --- MPE SimpleSpread ---
mpe_agent_counts=(3 4 5 6 7 8)
for num_agents in "${mpe_agent_counts[@]}"; do
    for seed in "${seeds[@]}"; do
        job_name="MEAL_beyond_mpe_${num_agents}agents_seed${seed}"
        cmd="python -m experiments.train ippo \
            --cl-method ewc --importance-mode online \
            --num-agents ${num_agents} \
            --seq-length 20 \
            --seed ${seed} \
            --tags BEYOND_OVERCOOKED \
            env:mpe"
        submit_job "${job_name}" "02:00:00" "${cmd}"
    done
done

summarize
