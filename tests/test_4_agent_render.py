from pathlib import Path

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze

from meal.env.overcooked import Overcooked
from meal.visualization.visualizer import OvercookedVisualizer


def flat_idx(x, y, W):
    return y * W + x


def test_make_soup_and_render_gif(tmp_path: Path = None, num_steps: int = 150):
    """
    Render a short episode of 4 agents taking random actions and save a GIF.

    Layout (W=7, H=7):
      - Counter strip on y=3
      - Onion pile at (1,3); Pot at (3,3); Plate pile at (5,3)
      - Serve tile at (3,1)
      - Agents: A0 at (1,2), A1 at (3,2), A2 at (5,4), A3 at (2,5)
    """
    W, H = 7, 7

    wall_idx = [flat_idx(x, 3, W) for x in range(W)]

    layout = freeze({
        "width": W,
        "height": H,
        "wall_idx": jnp.array(wall_idx, dtype=jnp.uint32),
        "agent_idx": jnp.array([
            flat_idx(1, 2, W),  # A0
            flat_idx(3, 2, W),  # A1
            flat_idx(5, 4, W),  # A2
            flat_idx(2, 5, W),  # A3
        ], dtype=jnp.uint32),
        "goal_idx":      jnp.array([flat_idx(3, 1, W)], dtype=jnp.uint32),
        "onion_pile_idx": jnp.array([flat_idx(1, 3, W)], dtype=jnp.uint32),
        "plate_pile_idx": jnp.array([flat_idx(5, 3, W)], dtype=jnp.uint32),
        "pot_idx":        jnp.array([flat_idx(3, 3, W)], dtype=jnp.uint32),
    })

    env = Overcooked(
        layout=layout,
        layout_name="test_4_agents_render",
        random_reset=False,
        random_agent_start=False,
        max_steps=num_steps,
        num_agents=4,
        task_id=0,
        cook_time=5,
    )

    key = jax.random.PRNGKey(42)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    num_actions = env.num_actions
    states = [state]

    for _ in range(num_steps):
        key, step_key, act_key = jax.random.split(key, 3)
        raw = jax.random.randint(act_key, shape=(env.num_agents,), minval=0, maxval=num_actions)
        actions = {f"agent_{i}": jnp.array(raw[i], dtype=jnp.uint8) for i in range(env.num_agents)}
        obs, state, _rew, done, _info = env.step(step_key, state, actions)
        states.append(state)
        if done["__all__"]:
            break

    out_dir = Path("gifs")
    out_dir.mkdir(parents=True, exist_ok=True)

    viz = OvercookedVisualizer(num_agents=4)
    gif_path = out_dir / "4_agents_render.gif"
    viz.animate(states, out_path=str(gif_path))

    print(f"✓ GIF saved to: {gif_path}  ({len(states)} frames)")


if __name__ == "__main__":
    test_make_soup_and_render_gif()
