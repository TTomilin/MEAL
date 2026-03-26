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
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from meal.env.mpe import MPESpreadEnv, make_mpe_sequence


# ---------------------------------------------------------------------------
# Custom visualiser (handles obstacle colouring correctly)
# ---------------------------------------------------------------------------

class MPEObstacleVisualizer:
    """Matplotlib-based visualiser for MPESpreadEnv.

    Renders agents (solid circles), landmarks (X markers), and obstacles
    (gray filled circles with outline) at each step.
    """

    AGENT_COLORS = [
        "#4CAF50",  # green
        "#2196F3",  # blue
        "#FF5722",  # orange-red
        "#9C27B0",  # purple
        "#00BCD4",  # cyan
    ]
    LANDMARK_COLOR = "#424242"   # dark grey
    OBSTACLE_COLOR = "#9E9E9E"   # medium grey
    BG_COLOR = "#FAFAFA"

    def __init__(self, env: MPESpreadEnv, state_seq: list):
        self.env = env
        self.state_seq = state_seq
        self._init_figure()

    def _init_figure(self):
        from matplotlib.patches import Circle

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.patch.set_facecolor(self.BG_COLOR)
        self.ax.set_facecolor(self.BG_COLOR)
        lim = 1.3
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        state = self.state_seq[0]
        A = self.env.num_agents
        L = self.env._num_goals
        K = self.env._num_obstacles

        # Draw obstacles (static — drawn once, never updated)
        for i in range(K):
            idx = A + L + i
            pos = np.array(state.p_pos[idx])
            r = float(self.env.rad[idx])
            circ = Circle(pos, r, facecolor=self.OBSTACLE_COLOR, edgecolor="#616161",
                          linewidth=1.5, zorder=1)
            self.ax.add_patch(circ)

        # Landmark markers (animated)
        self._landmark_patches = []
        for i in range(L):
            idx = A + i
            pos = np.array(state.p_pos[idx])
            r = float(self.env.rad[idx])
            circ = Circle(pos, r, color=self.LANDMARK_COLOR, zorder=2)
            self.ax.add_patch(circ)
            self._landmark_patches.append(circ)

        # Agent circles (animated)
        self._agent_patches = []
        for i in range(A):
            pos = np.array(state.p_pos[i])
            r = float(self.env.rad[i])
            color = self.AGENT_COLORS[i % len(self.AGENT_COLORS)]
            circ = Circle(pos, r, color=color, zorder=3)
            self.ax.add_patch(circ)
            self._agent_patches.append(circ)

        self._step_text = self.ax.text(
            -1.25, 1.18, f"Step 0 | {self.env.map_id}",
            fontsize=7, va="top", color="#333333"
        )

    def _update(self, frame: int):
        state = self.state_seq[frame]
        A = self.env.num_agents
        L = self.env._num_goals

        for i, patch in enumerate(self._agent_patches):
            patch.center = tuple(np.array(state.p_pos[i]))

        for i, patch in enumerate(self._landmark_patches):
            patch.center = tuple(np.array(state.p_pos[A + i]))

        self._step_text.set_text(f"Step {frame} | {self.env.map_id}")

    def animate(self, save_fname: str, fps: int = 10):
        ani = animation.FuncAnimation(
            self.fig, self._update,
            frames=len(self.state_seq),
            interval=1000 // fps,
            blit=False,
        )
        ani.save(save_fname, writer="pillow", fps=fps)
        plt.close(self.fig)


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
        fname = os.path.join(args.out_dir, f"task_{task_idx:02d}_{env.map_id}.gif")
        viz.animate(save_fname=fname, fps=args.fps)
        print(f"    → {fname}")

    print("Done.")


if __name__ == "__main__":
    main()
