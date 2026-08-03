"""
Visualize SMAX unit-composition tasks with random-policy allies vs heuristic
enemies, saving one GIF per task.

Blue circles = ally units (label = unit type shorthand).
Red circles  = enemy units (label = unit type shorthand).
Unit types: m=marine, M=marauder, s=stalker, Z=zealot, z=zergling, h=hydralisk

Usage (from repo root, conda env meal):
    conda run -n meal python scripts/visualize_smax.py

    # Customize:
    conda run -n meal python scripts/visualize_smax.py \\
        --num_allies 5 --num_enemies 5 --num_tasks 8 \\
        --num_steps 60 --out_dir gifs/smax --seed 0

Output:
    One GIF per task in out_dir, named task_<i>_<map_id>.gif
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import matplotlib
matplotlib.use("Agg")

from meal.env.smax import HeuristicEnemySMAX, SMAXVisualizer, make_smax_sequence


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_random_allies(env: HeuristicEnemySMAX, num_steps: int, seed: int = 0) -> list:
    """Run an episode with random ally actions, return list of inner SMAX states."""
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    # state is EnemySMAX State(state=smax_state, enemy_policy_state=...)
    smax_states = [state.state]

    action_n = env.action_space(env.agents[0]).n
    for _ in range(num_steps - 1):
        key, act_key, step_key = jax.random.split(key, 3)
        act_keys = jax.random.split(act_key, env.num_agents)
        actions = {
            a: jax.random.randint(act_keys[i], shape=(), minval=0, maxval=action_n)
            for i, a in enumerate(env.agents)
        }
        obs, state, reward, done, info = env.step(step_key, state, actions)
        smax_states.append(state.state)
        if done.get("__all__", False):
            break

    return smax_states



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_allies",  type=int, default=5)
    parser.add_argument("--num_enemies", type=int, default=5)
    parser.add_argument("--num_tasks",   type=int, default=8,
                        help="Number of tasks to visualise")
    parser.add_argument("--num_steps",   type=int, default=60,
                        help="Steps per rollout")
    parser.add_argument("--out_dir",     type=str, default="gifs/smax")
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--fps",         type=int, default=10)
    parser.add_argument("--max_steps",   type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"Building {args.num_tasks}-task sequence: "
        f"{args.num_allies}v{args.num_enemies}, "
        f"{args.num_steps} steps per rollout"
    )

    envs = make_smax_sequence(
        sequence_length=args.num_tasks,
        seed=args.seed,
        num_allies=args.num_allies,
        num_enemies=args.num_enemies,
        max_steps=args.max_steps,
    )

    for task_idx, env in enumerate(envs):
        print(f"  Task {task_idx:02d}: {env.map_id}")
        smax_states = rollout_random_allies(
            env, num_steps=args.num_steps, seed=args.seed + task_idx
        )

        viz = SMAXVisualizer(env=env, state_seq=smax_states, map_id=env.map_id)
        stem = os.path.join(args.out_dir, f"task_{task_idx:02d}_{env.map_id}")
        viz.save_frame(stem + ".png")
        print(f"    → {stem}.png")
        viz.animate(save_fname=stem + ".gif", fps=args.fps)
        print(f"    → {stem}.gif")

    print("Done.")


if __name__ == "__main__":
    main()
