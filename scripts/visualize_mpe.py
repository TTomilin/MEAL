"""
Visualize MPE obstacle-layout tasks with random-policy agents and save GIFs.

Agents (green circles) navigate around static gray obstacles to cover dark
landmarks.  Each task has a unique procedurally-generated obstacle field.

Usage (from repo root, conda env meal):
    conda run -n meal python scripts/visualize_mpe.py

    # Customize:
    conda run -n meal python scripts/visualize_mpe.py \\
        --num_agents 3 --num_landmarks 3 --num_obstacles 4 \\
        --num_tasks 8 --num_steps 100 --out_dir gifs/mpe --seed 0

Output:
    One GIF per task in out_dir, named task_<i>_<map_id>.gif
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

from meal.env.mpe import MPESpreadEnv, MPEObstacleVisualizer, make_mpe_sequence


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_random(env: MPESpreadEnv, num_steps: int, seed: int = 0) -> list:
    """Run a random-policy episode, return list of states."""
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    state_seq = [state]

    action_dim = env.action_space().n
    for _ in range(num_steps - 1):
        key, act_key, step_key = jax.random.split(key, 3)
        act_keys = jax.random.split(act_key, env.num_agents)
        actions = {
            a: jax.random.randint(act_keys[i], shape=(), minval=0, maxval=action_dim)
            for i, a in enumerate(env.agents)
        }
        obs, state, reward, done, info = env.step(step_key, state, actions)
        state_seq.append(state)
        if done.get("__all__", False):
            break

    return state_seq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_obstacles", type=int, default=4)
    parser.add_argument("--num_tasks", type=int, default=8,
                        help="Number of tasks to visualise")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Steps per rollout")
    parser.add_argument("--local_ratio", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="gifs/mpe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"Building {args.num_tasks}-task sequence: "
        f"{args.num_agents} agents, {args.num_landmarks} landmarks, "
        f"{args.num_obstacles} obstacles, {args.num_steps} steps"
    )

    envs = make_mpe_sequence(
        sequence_length=args.num_tasks,
        seed=args.seed,
        num_agents=args.num_agents,
        num_landmarks=args.num_landmarks,
        num_obstacles=args.num_obstacles,
        max_steps=args.num_steps,
        local_ratio=args.local_ratio,
    )

    for task_idx, env in enumerate(envs):
        print(f"  Task {task_idx:02d}: {env.map_id}")
        state_seq = rollout_random(env, num_steps=args.num_steps, seed=args.seed + task_idx)

        viz = MPEObstacleVisualizer(env=env, state_seq=state_seq)
        stem = os.path.join(args.out_dir, f"task_{task_idx:02d}_{env.map_id}")
        viz.save_frame(stem + ".png")
        print(f"    → {stem}.png")
        viz.animate(save_fname=stem + ".gif", fps=args.fps)
        print(f"    → {stem}.gif")

    print("Done.")


if __name__ == "__main__":
    main()
