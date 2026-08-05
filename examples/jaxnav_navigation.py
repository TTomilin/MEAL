"""
JaxNav Navigation Example

MEAL also includes JaxNav, a multi-robot goal-navigation benchmark: agents
must reach individual goal positions while avoiding walls and each other on a
randomly-generated polygon map (`Grid-Rand-Poly`). Task diversity comes from
the map layout, which is re-sampled per task and frozen for that task's
lifetime (analogous to Overcooked layouts / MPE obstacle fields).

This example builds a small sequence of navigation tasks and renders one GIF
per task with random-policy agents, including lidar rays and travelled paths.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless rendering, must be set before pyplot import
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from meal.env.jaxnav import JaxNav, JaxNavVisualizer


def build_task(seed, num_agents, map_dim):
    env = JaxNav(
        num_agents=num_agents,
        act_type="Discrete",
        max_steps=100,
        map_id="Grid-Rand-Poly",
        # valid_path_check rejects start/goal samples that aren't actually connected
        # by free space. Without it, obstacles can wall an agent's start off from its
        # goal entirely, making the task unsolvable no matter what the agent does.
        map_params={"map_size": (map_dim, map_dim), "valid_path_check": True},
    )
    # Freeze a random map layout for this task.
    env.map_obj._fixed_map = env.map_obj.sample_map(jax.random.PRNGKey(seed))
    return env


def rollout_random(env, num_steps, seed=0):
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    obs_sequence = [obs]
    state_sequence = [state]
    reward_sequence = []
    done_frames = jnp.full((env.num_agents,), num_steps - 1, dtype=jnp.int32)

    action_dim = env.action_space().n
    for t in range(num_steps):
        key, action_key, step_key = jax.random.split(key, 3)
        acts = jax.random.randint(action_key, (env.num_agents,), minval=0, maxval=action_dim)
        actions = {agent: acts[i] for i, agent in enumerate(env.agents)}

        obs, state, rewards, dones, info = env.step(step_key, state, actions)
        obs_sequence.append(obs)
        state_sequence.append(state)
        reward_sequence.append(float(jnp.sum(jnp.array(list(rewards.values())))))

        done_arr = jnp.array([bool(dones[a]) for a in env.agents])
        done_frames = jnp.where(done_arr & (done_frames == num_steps - 1), t, done_frames)
        if bool(jnp.all(done_arr)):
            break

    return obs_sequence, state_sequence, reward_sequence, done_frames


def main():
    print("JaxNav Navigation Example")
    print("=" * 60)

    num_tasks = 3
    num_agents = 3
    map_dim = 7
    num_steps = 80
    seed = 0
    output_dir = "gifs"
    os.makedirs(output_dir, exist_ok=True)

    for task_idx in range(num_tasks):
        print(f"Task {task_idx}: {num_agents} agents, {map_dim}x{map_dim} random polygon map")
        env = build_task(seed=seed + task_idx, num_agents=num_agents, map_dim=map_dim)
        obs_sequence, state_sequence, reward_sequence, done_frames = rollout_random(
            env, num_steps=num_steps, seed=seed + task_idx
        )
        print(f"  Collected {len(state_sequence)} states")

        viz = JaxNavVisualizer(
            env=env,
            obs_seq=obs_sequence,
            state_seq=state_sequence,
            reward_seq=reward_sequence,
            done_frames=done_frames,
            title_text=f"task_{task_idx}",
            plot_lidar=True,
            plot_path=True,
            plot_agent=True,
            plot_reward=True,
        )
        gif_path = os.path.join(output_dir, f"jaxnav_task{task_idx}.gif")
        viz.animate(save_fname=gif_path)
        print(f"  ✓ {gif_path}")
        plt.close(viz.fig)

    print()
    print("Example completed!")


if __name__ == "__main__":
    main()
