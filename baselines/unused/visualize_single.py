#!/usr/bin/env python3
"""
Make quick GIFs of single-chef Overcooked layouts.

This script does **no learning** – it just rolls a random policy so you
can eyeball the layouts produced by jax-marl’s generator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import flax
import jax
import jax.numpy as jnp
import numpy as np

from jax_marl import make
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.environments.overcooked_environment.common import OBJECT_TO_INDEX
from jax_marl.environments.overcooked_environment.overcooked_single import (
    OvercookedSingle,
)
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer


# --------------------------------------------------------------------------- #
# helper: pad a layout (so width/height are equal for a whole batch of envs)  #
# --------------------------------------------------------------------------- #
def pad_layout(l, max_h, max_w):
    l = flax.core.unfreeze(l)
    dh, dw = max_h - l["height"], max_w - l["width"]
    top, lef = dh // 2, dw // 2
    old_w = l["width"]

    def shift(idxs):
        rows = idxs // old_w + top
        cols = idxs % old_w + lef
        return rows * max_w + cols

    for k in ["wall_idx", "agent_idx", "goal_idx",
              "plate_pile_idx", "onion_pile_idx", "pot_idx"]:
        l[k] = jnp.asarray([shift(i) for i in l[k]], dtype=jnp.uint32)

    # -------- new: close the padded frame with walls ------------------
    border = []
    for r in range(max_h):
        for c in range(max_w):
            if r < top or r >= top + l["height"] or c < lef or c >= lef + l["width"]:
                border.append(r * max_w + c)
    l["wall_idx"] = jnp.concatenate([l["wall_idx"],
                                     jnp.asarray(border, dtype=jnp.uint32)])
    # ------------------------------------------------------------------
    l["height"], l["width"] = max_h, max_w
    return flax.core.freeze(l)


# --------------------------------------------------------------------------- #
# visualiser shim: convert single-chef State → something the visualiser likes #
# --------------------------------------------------------------------------- #
_EMPTY = OBJECT_TO_INDEX["empty"]


def to_vis_state(state):
    """
    Wrap the env State so that the two-agent visualiser is happy with only
    one agent present.
    """
    return SimpleNamespace(
        maze_map=np.asarray(state.maze_map),
        agent_inv=jnp.asarray([state.agent_inv], dtype=jnp.uint8),
        agent_dir_idx=jnp.asarray([state.agent_dir_idx], dtype=jnp.uint8),
        agent_pos=np.asarray(state.agent_pos),
        agent_dir=np.asarray(state.agent_dir),
    )


# --------------------------------------------------------------------------- #
# one random episode -------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def record_episode(env: OvercookedSingle, rng, max_steps=300):
    rng, key_r = jax.random.split(rng)
    obs_dict, state = env.reset(key_r)
    obs = obs_dict["agent_0"]

    frames = [SimpleNamespace(env_state=to_vis_state(state))]

    for _ in range(max_steps):
        # ­­­– random action –––––––––––––––––––––––––––––––––––––––––––––– #
        rng, key_a, key_s = jax.random.split(rng, 3)
        act = int(jax.random.randint(key_a, (), 0, env.action_space().n))
        obs_dict, state, _, done, _ = env.step(key_s, state, {"agent_0": act})
        frames.append(SimpleNamespace(env_state=to_vis_state(state)))
        if done["__all__"]:
            break
        obs = obs_dict["agent_0"]

    return frames


# --------------------------------------------------------------------------- #
# main ---------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--layout",
        type=str,
        default=None,
        help="Specific layout key (e.g. cramped_room). "
             "If omitted, a random layout is sampled.",
    )
    p.add_argument("--n", type=int, default=1, help="How many layouts to sample")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = jax.random.PRNGKey(args.seed)

    # ---- get layout(s) ---------------------------------------------------- #
    if args.layout is None:
        env_kwargs, layout_names = generate_sequence(
            sequence_length=args.n,
            strategy="random",
            seed=args.seed,
        )
        layouts = [kw["layout"] for kw in env_kwargs]
    else:
        from jax_marl.environments.overcooked_environment.layouts import (
            single_layouts,
        )

        layouts = [single_layouts[args.layout]]
        layout_names = [args.layout]

    # pad so every env has identical H×W (needed if you batch later / use CNN)
    max_h = max(l["height"] for l in layouts)
    max_w = max(l["width"] for l in layouts)
    layouts = [pad_layout(l, max_h, max_w) for l in layouts]

    viz = OvercookedVisualizer(num_agents=1)
    out_dir = Path(__file__).resolve().parent.parent / "runs" / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for layout, name in zip(layouts, layout_names):
        env = make("overcooked_single", layout=layout)
        frames = record_episode(env, rng, max_steps=args.max_steps)
        gif_path = out_dir / f"{name}.gif"
        viz.animate(frames, agent_view_size=5, task_idx=0, task_name=name, exp_dir=out_dir)
        print(f"Saved {gif_path}")

    print("Done.")


if __name__ == "__main__":
    main()
