"""
Unified environment visualizer for MEAL CL benchmarks.

Supported environments: overcooked, mpe, smax, jaxnav

Usage (from repo root):
    python scripts/visualize_env.py overcooked
    python scripts/visualize_env.py mpe --output both --num_agents 3
    python scripts/visualize_env.py smax --num_allies 4 --num_enemies 4
    python scripts/visualize_env.py jaxnav --map_dim 9 --output image

Output (--output):
    gif   – animated GIF per task (default)
    image – single PNG per task
    both  – GIF + PNG per task
"""

import argparse
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Overcooked
# ---------------------------------------------------------------------------

def run_overcooked(args):
    import pygame
    pygame.init()
    from meal import make_sequence
    from meal.visualization.visualizer import OvercookedVisualizer
    from PIL import Image as PILImage

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"Building {args.num_tasks}-task Overcooked sequence: "
        f"{args.num_agents} agents, {args.num_steps} steps"
    )

    envs = make_sequence(
        env_id="overcooked",
        sequence_length=args.num_tasks,
        num_agents=args.num_agents,
        seed=args.seed,
        difficulty=args.difficulty,
    )
    viz = OvercookedVisualizer(num_agents=args.num_agents)

    for task_idx, env in enumerate(envs):
        env_name = getattr(env, "layout_name", f"task_{task_idx:02d}")
        print(f"  Task {task_idx:02d}: {env_name}")

        key = jax.random.PRNGKey(args.seed + task_idx)
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        state_seq = [state]

        action_dim = env.action_space().n
        for _ in range(args.num_steps - 1):
            key, act_key, step_key = jax.random.split(key, 3)
            act_keys = jax.random.split(act_key, env.num_agents)
            actions = {
                a: jax.random.randint(act_keys[i], shape=(), minval=0, maxval=action_dim)
                for i, a in enumerate(env.agents)
            }
            obs, state, _, done, _ = env.step(step_key, state, actions)
            state_seq.append(state)
            if done.get("__all__", False):
                break

        stem = os.path.join(args.out_dir, f"task_{task_idx:02d}_{env_name}")

        if args.output in ("gif", "both"):
            out = stem + ".gif"
            viz.animate(state_seq=state_seq, out_path=out, task_idx=task_idx, fps=args.fps)
            print(f"    → {out}")

        if args.output in ("image", "both"):
            out = stem + ".png"
            frame = viz.render(state_seq[0])
            PILImage.fromarray(frame).save(out)
            print(f"    → {out}")


# ---------------------------------------------------------------------------
# MPE
# ---------------------------------------------------------------------------

def run_mpe(args):
    from visualize_mpe import MPEObstacleVisualizer, rollout_random
    from meal.env.mpe import make_mpe_sequence

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"Building {args.num_tasks}-task MPE sequence: "
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

        if args.output in ("image", "both"):
            viz.save_frame(stem + ".png")
            print(f"    → {stem}.png")

        if args.output in ("gif", "both"):
            viz.animate(stem + ".gif", fps=args.fps)
            print(f"    → {stem}.gif")
        else:
            plt.close(viz.fig)


# ---------------------------------------------------------------------------
# SMAX
# ---------------------------------------------------------------------------

def run_smax(args):
    from visualize_smax import SMAXVisualizer, rollout_random_allies
    from meal.env.smax import make_smax_sequence

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"Building {args.num_tasks}-task SMAX sequence: "
        f"{args.num_allies}v{args.num_enemies}, {args.num_steps} steps per rollout"
    )

    envs = make_smax_sequence(
        sequence_length=args.num_tasks,
        seed=args.seed,
        num_allies=args.num_allies,
        num_enemies=args.num_enemies,
        max_steps=args.smax_max_steps,
    )

    for task_idx, env in enumerate(envs):
        print(f"  Task {task_idx:02d}: {env.map_id}")
        smax_states = rollout_random_allies(env, num_steps=args.num_steps, seed=args.seed + task_idx)

        viz = SMAXVisualizer(env=env, state_seq=smax_states, map_id=env.map_id)
        stem = os.path.join(args.out_dir, f"task_{task_idx:02d}_{env.map_id}")

        if args.output in ("image", "both"):
            viz.save_frame(stem + ".png")
            print(f"    → {stem}.png")

        if args.output in ("gif", "both"):
            viz.animate(stem + ".gif", fps=args.fps)
            print(f"    → {stem}.gif")
        else:
            plt.close(viz.fig)


# ---------------------------------------------------------------------------
# JaxNav
# ---------------------------------------------------------------------------

def run_jaxnav(args):
    from meal.env.jaxnav.jaxnav_env import JaxNav
    from meal.env.jaxnav.jaxnav_viz import JaxNavVisualizer

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"Building {args.num_tasks}-task JaxNav sequence: "
        f"{args.num_agents} agents, map_dim={args.map_dim}, {args.num_steps} steps"
    )

    key = jax.random.PRNGKey(args.seed)
    envs = []
    for _ in range(args.num_tasks):
        env = JaxNav(
            num_agents=args.num_agents,
            act_type="Discrete",
            max_steps=args.num_steps,
            map_id="Grid-Rand-Poly",
            # valid_path_check rejects start/goal samples that aren't actually
            # connected by free space (otherwise an obstacle layout can wall an
            # agent's start off from its goal entirely).
            map_params={"map_size": (args.map_dim, args.map_dim), "valid_path_check": True},
        )
        key, k_layout = jax.random.split(key)
        env.map_obj._fixed_map = env.map_obj.sample_map(k_layout)
        envs.append(env)

    for task_idx, env in enumerate(envs):
        print(f"  Task {task_idx:02d}")

        rkey = jax.random.PRNGKey(args.seed + task_idx)
        rkey, reset_key = jax.random.split(rkey)
        obs, state = env.reset(reset_key)
        obs_seq = [obs]
        state_seq = [state]
        reward_seq = []
        done_frames = jnp.full((env.num_agents,), args.num_steps - 1, dtype=jnp.int32)
        action_dim = env.action_space().n

        for t in range(args.num_steps):
            rkey, act_key, step_key = jax.random.split(rkey, 3)
            acts = jax.random.randint(act_key, (env.num_agents,), minval=0, maxval=action_dim)
            actions = {a: acts[i] for i, a in enumerate(env.agents)}
            obs, state, reward, done, _ = env.step(step_key, state, actions)
            obs_seq.append(obs)
            state_seq.append(state)
            r = sum(float(v) for v in reward.values()) if isinstance(reward, dict) else float(jnp.sum(jnp.asarray(reward)))
            reward_seq.append(r)

            if isinstance(done, dict):
                if "__all__" in done:
                    done_arr = jnp.full((env.num_agents,), bool(done["__all__"]))
                else:
                    done_arr = jnp.array([bool(done[a]) for a in env.agents])
            else:
                done_arr = jnp.asarray(done, jnp.bool_)
                if done_arr.ndim == 0:
                    done_arr = jnp.full((env.num_agents,), done_arr)

            done_frames = jnp.where(
                done_arr & (done_frames == (args.num_steps - 1)),
                jnp.full_like(done_frames, t),
                done_frames,
            )
            if bool(jnp.all(done_arr)):
                break

        viz = JaxNavVisualizer(
            env=env,
            obs_seq=obs_seq,
            state_seq=state_seq,
            reward_seq=reward_seq,
            done_frames=done_frames,
            title_text=f"task_{task_idx:02d}",
            plot_lidar=True,
            plot_path=True,
            plot_agent=True,
            plot_reward=True,
        )
        stem = os.path.join(args.out_dir, f"task_{task_idx:02d}")

        if args.output in ("image", "both"):
            viz.init()
            viz.update(len(state_seq) // 2)
            viz.fig.savefig(stem + ".png", dpi=150)
            print(f"    → {stem}.png")

        if args.output in ("gif", "both"):
            viz.animate(save_fname=stem + ".gif")
            print(f"    → {stem}.gif")
        else:
            plt.close(viz.fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DISPATCH = {
    "overcooked": run_overcooked,
    "mpe": run_mpe,
    "smax": run_smax,
    "jaxnav": run_jaxnav,
}


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MEAL environments with random-policy rollouts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "env", choices=list(_DISPATCH),
        help="Environment to visualize",
    )
    parser.add_argument(
        "--output", choices=["gif", "image", "both"], default="gif",
        help="Output format",
    )

    # Common
    parser.add_argument("--num_tasks",  type=int, default=3,  help="Number of tasks")
    parser.add_argument("--num_steps",  type=int, default=100, help="Rollout steps per task")
    parser.add_argument("--out_dir",    type=str, default=None, help="Output directory (default: gifs/<env>)")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--fps",        type=int, default=10)

    # Overcooked / JaxNav
    parser.add_argument("--num_agents", type=int, default=4, help="[overcooked/jaxnav] Number of agents")
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard", "extreme"],
                        help="[overcooked] Layout difficulty for generated tasks")

    # MPE
    parser.add_argument("--num_landmarks", type=int, default=4,  help="[mpe] Number of landmarks")
    parser.add_argument("--num_obstacles", type=int, default=5,  help="[mpe] Number of obstacles")
    parser.add_argument("--local_ratio",   type=float, default=0.5, help="[mpe] Local reward ratio")

    # SMAX
    parser.add_argument("--num_allies",    type=int, default=7,   help="[smax] Number of allies")
    parser.add_argument("--num_enemies",   type=int, default=7,   help="[smax] Number of enemies")
    parser.add_argument("--smax_max_steps", type=int, default=100, help="[smax] Max steps for env construction")

    # JaxNav
    parser.add_argument("--map_dim", type=int, default=7, help="[jaxnav] Map grid dimension")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = f"gifs/{args.env}"

    _DISPATCH[args.env](args)
    print("Done.")


if __name__ == "__main__":
    main()
