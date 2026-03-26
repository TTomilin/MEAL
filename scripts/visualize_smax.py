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
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from meal.env.smax import HeuristicEnemySMAX, make_smax_sequence


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
# Visualizer
# ---------------------------------------------------------------------------

class SMAXVisualizer:
    """Animate a sequence of SMAX inner states (unit positions, health, alive)."""

    ALLY_COLOR   = "#4fc3f7"  # light blue
    ENEMY_COLOR  = "#ef9a9a"  # light red
    DEAD_ALPHA   = 0.15
    BG_COLOR     = "#1a1a2e"

    def __init__(self, env: HeuristicEnemySMAX, state_seq: list, map_id: str):
        self._inner_env = env._env   # underlying SMAX instance
        self.state_seq = state_seq
        self.map_id = map_id
        self._init_figure()

    def _init_figure(self):
        from matplotlib.patches import Circle

        smax = self._inner_env
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.patch.set_facecolor(self.BG_COLOR)
        ax = self.ax
        ax.set_facecolor(self.BG_COLOR)
        ax.set_xlim(0, smax.map_width)
        ax.set_ylim(0, smax.map_height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        state0 = self.state_seq[0]

        # Build patches for all units
        self._patches = []
        self._labels  = []

        for i in range(smax.num_allies):
            pos = np.array(state0.unit_positions[i])
            r   = float(smax.unit_type_radiuses[state0.unit_types[i]])
            sh  = smax.unit_type_shorthands[int(state0.unit_types[i])]
            c = Circle(pos, r, facecolor=self.ALLY_COLOR, edgecolor="white",
                       linewidth=0.8, zorder=2)
            ax.add_patch(c)
            txt = ax.text(pos[0], pos[1], sh, ha="center", va="center",
                          fontsize=5, color="black", fontweight="bold", zorder=3)
            self._patches.append(c)
            self._labels.append(txt)

        for i in range(smax.num_enemies):
            idx = i + smax.num_allies
            pos = np.array(state0.unit_positions[idx])
            r   = float(smax.unit_type_radiuses[state0.unit_types[idx]])
            sh  = smax.unit_type_shorthands[int(state0.unit_types[idx])]
            c = Circle(pos, r, facecolor=self.ENEMY_COLOR, edgecolor="white",
                       linewidth=0.8, zorder=2)
            ax.add_patch(c)
            txt = ax.text(pos[0], pos[1], sh, ha="center", va="center",
                          fontsize=5, color="black", fontweight="bold", zorder=3)
            self._patches.append(c)
            self._labels.append(txt)

        # Title shows map_id
        self._step_text = ax.text(
            0.5, 1.02, f"Step 0 | {self.map_id}",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=7, color="white",
        )

    def _update(self, frame: int):
        smax  = self._inner_env
        state = self.state_seq[frame]

        for i in range(smax.num_allies):
            pos   = np.array(state.unit_positions[i])
            alive = bool(state.unit_alive[i])
            p     = self._patches[i]
            t     = self._labels[i]
            p.center = tuple(pos)
            t.set_position(pos)
            p.set_alpha(1.0 if alive else self.DEAD_ALPHA)
            t.set_alpha(1.0 if alive else 0.0)

        for i in range(smax.num_enemies):
            idx   = i + smax.num_allies
            pos   = np.array(state.unit_positions[idx])
            alive = bool(state.unit_alive[idx])
            pidx  = smax.num_allies + i
            p     = self._patches[pidx]
            t     = self._labels[pidx]
            p.center = tuple(pos)
            t.set_position(pos)
            p.set_alpha(1.0 if alive else self.DEAD_ALPHA)
            t.set_alpha(1.0 if alive else 0.0)

        self._step_text.set_text(f"Step {frame} | {self.map_id}")

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
        fname = os.path.join(args.out_dir, f"task_{task_idx:02d}_{env.map_id}.gif")
        viz.animate(save_fname=fname, fps=args.fps)
        print(f"    → {fname}")

    print("Done.")


if __name__ == "__main__":
    main()
