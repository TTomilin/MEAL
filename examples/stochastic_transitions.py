"""
Stochastic Transitions Example (Sticky Actions & Slippery Tiles)

MEAL ships two transition wrappers that inject stochasticity into an
otherwise-deterministic Overcooked environment, useful for testing policy
robustness:

- `SlipperyTiles`: 25% of floor tiles are marked slippery; stepping onto one
  arms a random-direction override on the agent's *next* action (simulates
  an icy kitchen floor).
- `StickyActions`: with probability p, the agent's previous action is
  repeated instead of the one it just chose (simulates lag / momentum).

Both wrappers are also available as CL sequence options in `meal.make_sequence`
(`sticky_actions=True` / `slippery_tiles=True`), tied to per-difficulty
probabilities in `difficulty_config.py`.

A cluttered kitchen makes the wrappers' effect hard to see (cooking objects
draw the eye, and there's nowhere to walk in a straight line for long). So
instead this example builds two bare, obstacle-free rooms and drives a single
agent through a fixed, scripted action sequence - no policy, no randomness in
the actions themselves - so any deviation you see is purely the wrapper:

1. `stochastic_transitions_slippery.gif`: a long corridor, agent walks
   straight across it. Baseline (left) vs. `SlipperyTiles` (right), rendered
   side by side.
2. `stochastic_transitions_sticky.gif`: a square room, agent walks a closed
   square loop (right, down, left, up). Baseline (left) vs. `StickyActions`
   (right), side by side - sticky actions overshoot each leg of the square
   before turning, visibly warping the loop.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

import meal
from meal.env.overcooked.layouts.presets import layout_grid_to_dict
from meal.env.overcooked.overcooked_env import Actions
from meal.visualization.visualizer import OvercookedVisualizer
from meal.wrappers.slippery_tiles import SlipperyTiles
from meal.wrappers.sticky_actions import StickyActions


def empty_room_layout(height: int, width: int, agent_row: int, agent_col: int) -> dict:
    """A bordered, (almost) obstacle-free room with a single agent - no piles
    or goal, just floor - so scripted movement is all there is to look at.

    One pot tile is tucked into the top-right corner: the engine's maze-map
    construction indexes into a per-pot status array sized to the layout's
    actual pot count, so a truly pot-less layout hits an out-of-bounds index
    at reset. The agent's scripted path never goes near it or presses
    interact, so it's inert - just there to satisfy that invariant.
    """
    rows = []
    for r in range(height):
        if r == 0 or r == height - 1:
            rows.append("W" * width)
        else:
            row = ["W"] + [" "] * (width - 2) + ["W"]
            if r == agent_row:
                row[agent_col] = "A"
            rows.append("".join(row))
    rows[1] = rows[1][:-2] + "P" + rows[1][-1]
    return layout_grid_to_dict("\n".join(rows))


def rollout_scripted(env, action_seq, key):
    """Step through a fixed sequence of single-agent actions (no randomness
    beyond whatever the wrapper itself injects)."""
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    state_sequence = [state]
    for a in action_seq:
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(step_key, state, {"agent_0": jnp.array(a, dtype=jnp.uint32)})
        state_sequence.append(state)
    return state_sequence


def render_side_by_side(viz, left_states, right_states, left_label, right_label, out_path, fps=8):
    """Render two state sequences to one GIF, frame-by-frame, side by side."""
    left_frames = [viz.render(s) for s in left_states]
    right_frames = [viz.render(s) for s in right_states]

    # Hold the last frame of whichever sequence finishes first so both halves
    # stay the same length.
    n = max(len(left_frames), len(right_frames))
    left_frames += [left_frames[-1]] * (n - len(left_frames))
    right_frames += [right_frames[-1]] * (n - len(right_frames))

    gap, header = 12, 28
    h = max(left_frames[0].shape[0], right_frames[0].shape[0])
    w = left_frames[0].shape[1] + gap + right_frames[0].shape[1]

    composed = []
    for lf, rf in zip(left_frames, right_frames):
        canvas = np.full((h + header, w, 3), 30, dtype=np.uint8)
        canvas[header:header + lf.shape[0], :lf.shape[1]] = lf
        canvas[header:header + rf.shape[0], lf.shape[1] + gap:] = rf
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        draw.text((8, 6), left_label, fill=(255, 255, 255))
        draw.text((lf.shape[1] + gap + 8, 6), right_label, fill=(255, 255, 255))
        composed.append(np.array(img))

    frames = np.stack(composed, axis=0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    viz._save_frames(frames, out_path, fps)


def main():
    print("Stochastic Transitions Example (sticky actions / slippery tiles)")
    print("=" * 60)

    output_dir = "gifs"
    os.makedirs(output_dir, exist_ok=True)
    viz = OvercookedVisualizer(num_agents=1)

    # --- Slippery tiles: walk straight across an open corridor ---------------
    # Only ~25% of tiles are slippery, and the wrapper only *arms* a slip on the
    # step after the agent lands on one (see SlipperyTiles docstring) - so most
    # seeds happen to place too few slippery tiles under this particular path to
    # show much. seed=6 with a high slip_prob reliably lands on enough of them.
    print("Slippery tiles: baseline straight walk vs. slippery-tiles walk")
    corridor = empty_room_layout(height=5, width=19, agent_row=2, agent_col=1)
    walk_right = [Actions.right] * 18
    slip_seed = 6

    base_env = meal.make_env("overcooked", layout=corridor, layout_name="corridor", num_agents=1)
    slip_env = SlipperyTiles(
        meal.make_env("overcooked", layout=corridor, layout_name="corridor", num_agents=1), slip_prob=0.8
    )

    base_states = rollout_scripted(base_env, walk_right, jax.random.PRNGKey(slip_seed))
    slip_states = rollout_scripted(slip_env, walk_right, jax.random.PRNGKey(slip_seed))
    print(f"  Collected {len(base_states)} baseline / {len(slip_states)} slippery states")

    gif_path = os.path.join(output_dir, "stochastic_transitions_slippery.gif")
    render_side_by_side(viz, base_states, slip_states, "baseline", "slippery_tiles", gif_path)
    print(f"  ✓ {gif_path}")
    print()

    # --- Sticky actions: walk a closed square loop ----------------------------
    # Stickiness only changes behaviour at the four corners, where it re-plays
    # the *previous* leg's direction instead of turning - mid-leg it just repeats
    # whatever direction was already happening, which is invisible. seed=1 lands
    # a sticky repeat right on a corner, so the loop visibly overshoots before
    # turning.
    print("Sticky actions: baseline square loop vs. sticky-actions loop")
    room = empty_room_layout(height=13, width=13, agent_row=3, agent_col=3)
    walk_square = [Actions.right] * 6 + [Actions.down] * 6 + [Actions.left] * 6 + [Actions.up] * 6
    sticky_seed = 1

    base_env2 = meal.make_env("overcooked", layout=room, layout_name="room", num_agents=1)
    sticky_env = StickyActions(
        meal.make_env("overcooked", layout=room, layout_name="room", num_agents=1), p=0.6
    )

    base_states2 = rollout_scripted(base_env2, walk_square, jax.random.PRNGKey(sticky_seed))
    sticky_states2 = rollout_scripted(sticky_env, walk_square, jax.random.PRNGKey(sticky_seed))
    print(f"  Collected {len(base_states2)} baseline / {len(sticky_states2)} sticky states")

    gif_path2 = os.path.join(output_dir, "stochastic_transitions_sticky.gif")
    render_side_by_side(viz, base_states2, sticky_states2, "baseline", "sticky_actions", gif_path2)
    print(f"  ✓ {gif_path2}")

    print()
    print("Example completed! In the slippery GIF, watch the right-hand agent")
    print("drift off its row; in the sticky GIF, watch the right-hand agent")
    print("overshoot each turn of the square.")


if __name__ == "__main__":
    main()
