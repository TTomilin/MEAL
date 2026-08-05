"""
MPE Spread (with Obstacles) Example

MEAL's third CL benchmark: a discrete-action variant of MPE SimpleSpread.
Agents (green circles) must spread out to cover landmarks (dark dots) while
avoiding fixed, per-task obstacle fields (gray circles). Obstacles are frozen
per task (same every episode reset of that task); agent/landmark start
positions are randomised each episode.

This example builds a small continual-learning sequence of tasks - each with
a unique procedurally-generated obstacle layout - and renders one GIF per
task with random-policy agents.
"""

import os

import jax

from meal.env.mpe import MPEObstacleVisualizer, make_mpe_sequence


def rollout_random(env, num_steps, seed=0):
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    state_sequence = [state]

    action_dim = env.action_space().n
    for _ in range(num_steps - 1):
        key, action_key, step_key = jax.random.split(key, 3)
        act_keys = jax.random.split(action_key, env.num_agents)
        actions = {
            agent: jax.random.randint(act_keys[i], shape=(), minval=0, maxval=action_dim)
            for i, agent in enumerate(env.agents)
        }
        obs, state, rewards, dones, info = env.step(step_key, state, actions)
        state_sequence.append(state)
        if dones.get("__all__", False):
            break

    return state_sequence


def main():
    print("MPE Spread (with Obstacles) Example")
    print("=" * 60)

    num_tasks = 3
    num_steps = 100
    seed = 0
    output_dir = "gifs"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Building {num_tasks}-task MPE sequence (3 agents, 3 landmarks, 4 obstacles)...")
    envs = make_mpe_sequence(
        sequence_length=num_tasks,
        seed=seed,
        num_agents=3,
        num_landmarks=3,
        num_obstacles=4,
        max_steps=num_steps,
        local_ratio=0.5,
    )

    for task_idx, env in enumerate(envs):
        print(f"Task {task_idx}: {env.map_id}")
        state_sequence = rollout_random(env, num_steps=num_steps, seed=seed + task_idx)
        print(f"  Collected {len(state_sequence)} states")

        viz = MPEObstacleVisualizer(env=env, state_seq=state_sequence)
        gif_path = os.path.join(output_dir, f"mpe_spread_task{task_idx}_{env.map_id}.gif")
        viz.animate(gif_path, fps=10)
        print(f"  ✓ {gif_path}")

    print()
    print("Example completed!")


if __name__ == "__main__":
    main()
