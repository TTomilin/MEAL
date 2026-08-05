"""
SMAX Battle Example

MEAL's fourth CL benchmark: JAX StarCraft Multi-Agent Challenge (SMAX) with
`HeuristicEnemySMAX` - a fixed team of allied units (trained agents, blue)
fights a fixed team of enemy units controlled by a scripted heuristic policy
(move toward the enemy, attack whoever is closest in range), red.

Task diversity comes from unit composition: each task independently samples
ally and enemy unit types from {marine, marauder, stalker, zealot, zergling,
hydralisk}, giving 6^n_allies x 6^n_enemies strategically distinct match-ups.

This example builds a small sequence of 5v5 tasks and renders one GIF per
task with random-policy allies (so units mill about rather than fighting
well - swap in a trained policy to see actual combat behaviour).
"""

import os

import jax

from meal.env.smax import HeuristicEnemySMAX, SMAXVisualizer, make_smax_sequence


def rollout_random_allies(env: HeuristicEnemySMAX, num_steps, seed=0):
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    smax_states = [state.state]  # state is EnemySMAX State(state=smax_state, ...)

    action_dim = env.action_space(env.agents[0]).n
    for _ in range(num_steps - 1):
        key, action_key, step_key = jax.random.split(key, 3)
        act_keys = jax.random.split(action_key, env.num_agents)
        actions = {
            agent: jax.random.randint(act_keys[i], shape=(), minval=0, maxval=action_dim)
            for i, agent in enumerate(env.agents)
        }
        obs, state, rewards, dones, info = env.step(step_key, state, actions)
        smax_states.append(state.state)
        if dones.get("__all__", False):
            break

    return smax_states


def main():
    print("SMAX Battle Example")
    print("=" * 60)

    num_tasks = 3
    num_steps = 60
    seed = 0
    output_dir = "gifs"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Building {num_tasks}-task SMAX sequence (5v5)...")
    envs = make_smax_sequence(
        sequence_length=num_tasks,
        seed=seed,
        num_allies=5,
        num_enemies=5,
        max_steps=100,
    )

    for task_idx, env in enumerate(envs):
        print(f"Task {task_idx}: {env.map_id}")
        smax_states = rollout_random_allies(env, num_steps=num_steps, seed=seed + task_idx)
        print(f"  Collected {len(smax_states)} states")

        viz = SMAXVisualizer(env=env, state_seq=smax_states, map_id=env.map_id)
        gif_path = os.path.join(output_dir, f"smax_battle_task{task_idx}_{env.map_id}.gif")
        viz.animate(save_fname=gif_path, fps=10)
        print(f"  ✓ {gif_path}")

    print()
    print("Example completed!")


if __name__ == "__main__":
    main()
