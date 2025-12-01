#!/usr/bin/env python3
"""Render example JaxNav layouts using the full env renderer (agents, lidar, etc.)."""

from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from meal.env.jaxnav.jaxnav_env import JaxNav, State
from meal.env.jaxnav.jaxnav_viz import JaxNavVisualizer


# -------------------------------------------------------------------------
# Rollout helper (multi-agent, dict actions, matches your training API)
# -------------------------------------------------------------------------
def rollout_random(
        env: JaxNav,
        num_steps: int,
        seed: int = 0,
) -> Tuple[List, List[State], List[float], None]:
    """
    Simple random-policy rollout to debug JaxNav maps.

    Returns:
        obs_seq:     list of observations
        state_seq:   list of State objects
        reward_seq:  list of per-step total reward (sum over agents)
        done_frames: None (we don't bother with per-agent done frame here)
    """
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)

    # NOTE: your env.reset returns (obs, state)
    obs, state = env.reset(reset_key)

    obs_seq: List = [obs]
    state_seq: List[State] = [state]
    reward_seq: List[float] = []

    num_agents = env.num_agents
    action_dim = env.action_space().n
    agents = env.agents  # e.g. ["agent_0", "agent_1", ...]

    for _t in range(num_steps):
        key, act_key, step_key = jax.random.split(key, 3)

        # sample one discrete action per agent
        acts_vec = jax.random.randint(
            act_key,
            (num_agents,),
            minval=0,
            maxval=action_dim,
        )
        actions = {a: acts_vec[i] for i, a in enumerate(agents)}

        # NOTE: your env.step signature: obs, state, reward, done, info
        obs, state, reward, done, info = env.step(step_key, state, actions)

        obs_seq.append(obs)
        state_seq.append(state)

        # reward can be dict or array; sum to scalar
        if isinstance(reward, dict):
            r = 0.0
            for v in reward.values():
                r += float(v)
        else:
            r = float(jnp.sum(jnp.asarray(reward)))
        reward_seq.append(r)

    # We don't use done_frames for this static screenshot use-case
    done_frames = None
    return obs_seq, state_seq, reward_seq, done_frames


# -------------------------------------------------------------------------
# Single-frame render helper using your JaxNavVisualizer
# -------------------------------------------------------------------------
def render_single_frame_png(
        env: JaxNav,
        obs_seq: List,
        state_seq: List[State],
        reward_seq: List[float],
        done_frames,
        name: str,
        frame: int,
        out_dir: Path,
) -> None:
    """Use JaxNavVisualizer to draw a single frame and save as PNG."""

    viz = JaxNavVisualizer(
        env=env,
        obs_seq=obs_seq,
        state_seq=state_seq,
        reward_seq=reward_seq,
        done_frames=done_frames,
        title_text=name,
        plot_lidar=True,
        plot_path=False,
        plot_agent=True,
        plot_reward=False,
        plot_line_to_goal=True,
    )

    viz.init()
    viz.update(frame)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_frame{frame}.png"
    viz.fig.savefig(out_path, dpi=150)
    plt.close(viz.fig)
    print(f"  -> saved {out_path}")


# -------------------------------------------------------------------------
# Env builders – same style as make_jaxnav_sequence
# -------------------------------------------------------------------------
def make_env(map_dim: int, num_agents: int, max_steps: int, seed: int, name: str) -> JaxNav:
    """
    Build a JaxNav env with a fixed underlying map, like in make_jaxnav_sequence.
    For now we just use Grid-Rand-Poly; tweak map_id/map_params if you want barn/circle variants.
    """
    env = JaxNav(
        num_agents=num_agents,
        act_type="Discrete",
        max_steps=max_steps,
        map_id="Grid-Rand-Poly",
        map_params={"map_size": (map_dim, map_dim)},
    )

    # Fix the map so rerenders use the same layout
    key = jax.random.PRNGKey(seed)
    key, k_layout = jax.random.split(key)
    env.map_obj._fixed_map = env.map_obj.sample_map(k_layout)

    print(f"[JaxNav] Created env '{name}' with map_dim={map_dim}, num_agents={num_agents}")
    return env


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    base_key = jax.random.PRNGKey(0)
    out_dir = Path.cwd() / "assets" / "screenshots" / "jaxnav"
    print(f"[JaxNav] Saving example layouts (full render) to: {out_dir}")

    # (name, map_dim, num_agents, max_steps, rollout_steps)
    specs = [
        ("jaxnav_dim7_2agents", 7, 2, 128, 128),
        ("jaxnav_dim7_3agents", 7, 3, 128, 128),
        ("jaxnav_dim7_4agents", 7, 4, 128, 128),
    ]

    for i, (name, map_dim, num_agents, max_steps, rollout_steps) in enumerate(specs):
        key, base_key = jax.random.split(base_key)

        env = make_env(map_dim=map_dim,
                       num_agents=num_agents,
                       max_steps=max_steps,
                       seed=i,
                       name=name)

        obs_seq, state_seq, reward_seq, done_frames = rollout_random(
            env,
            num_steps=rollout_steps,
            seed=i,
        )

        # middle of the trajectory usually looks OK
        frame = rollout_steps // 2
        render_single_frame_png(
            env=env,
            obs_seq=obs_seq,
            state_seq=state_seq,
            reward_seq=reward_seq,
            done_frames=done_frames,
            name=name,
            frame=frame,
            out_dir=out_dir,
        )

    print("[JaxNav] Done.")
