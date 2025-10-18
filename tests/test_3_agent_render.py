from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import freeze

from meal.env.overcooked import Actions, Overcooked, POT_FULL_STATUS
from meal.visualization.visualizer import OvercookedVisualizer

# Aliases
U, D, R, L, S, I = Actions.up, Actions.down, Actions.right, Actions.left, Actions.stay, Actions.interact

def flat_idx(x, y, W):
    return y * W + x

def test_make_soup_and_render_gif(tmp_path: Path = None):
    """
    Layout (W=7, H=7):
      - A wall "counter" strip on y=3
      - Onion pile at (1,3); Pot at (3,3); Plate pile at (5,3)
      - Goal at (3,1) (not used, just there)
      - Agents:
          A0 at (1,2)  (above onion pile)
          A1 at (3,2)  (above pot)
          A2 at (5,4)  (below plate pile)
    All moves/orientations are scripted and deterministic.
    """

    W, H = 7, 7

    # --- Counters (walls) line at y=3
    wall_idx = [flat_idx(x, 3, W) for x in range(W)]

    onion_pile = (1, 3)
    pot = (3, 3)
    plate_pile = (5, 3)
    goal = (3, 1)

    # Replace the counters at those positions with the specific station types (env code handles this)
    onion_pile_idx = jnp.array([flat_idx(*onion_pile, W)], dtype=jnp.uint32)
    pot_idx = jnp.array([flat_idx(*pot, W)], dtype=jnp.uint32)
    plate_pile_idx = jnp.array([flat_idx(*plate_pile, W)], dtype=jnp.uint32)
    goal_idx = jnp.array([flat_idx(*goal, W)], dtype=jnp.uint32)

    layout = freeze({
        "width": W,
        "height": H,
        "wall_idx": jnp.array(wall_idx, dtype=jnp.uint32),
        "agent_idx": jnp.array([
            flat_idx(1, 2, W),  # A0 above onion
            flat_idx(3, 2, W),  # A1 above pot
            flat_idx(5, 4, W),  # A2 below plate pile
        ], dtype=jnp.uint32),
        "goal_idx": goal_idx,
        "onion_pile_idx": onion_pile_idx,
        "plate_pile_idx": plate_pile_idx,
        "pot_idx": pot_idx,
    })

    # --- Build env (deterministic reset, 3 agents)
    env = Overcooked(layout=layout, layout_name="test_3_agents_make_soup",
                     random_reset=False, random_agent_start=False,
                     max_steps=400, num_agents=3, task_id=0)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # === Scripted joint actions ===
    # Plan:
    # - Orient agents towards counters: A0, A1 face SOUTH; A2 face NORTH.
    # - A0 and A1 each place one onion into the pot; A0 returns to place the 3rd.
    # - A2 picks a plate and parks below pot; when pot ready, interact to pick soup.
    #
    # NOTE: Interact uses the tile *in front* of the agent.

    joint_actions = []

    # Set orientations (blocked by wall → only orientation changes)
    joint_actions.append([D, D, U])   # face counters

    # A0 pick onion, A2 pick plate
    joint_actions.append([I, S, I])   # A0 grabs onion; A2 grabs plate

    # A0 to above pot: (1,2) -> (3,2)
    joint_actions += [[R, S, S], [R, S, L]]  # A2 starts moving left towards (3,4)

    # A0 drop onion into pot; A1 move left towards onion
    joint_actions.append([I, L, L])

    # A1 move left again; A0 start back to onion
    joint_actions.append([S, L, S])

    # A1 face SOUTH (already), interact to pick onion; A0 go left
    joint_actions.append([L, I, S])

    # A1 go right twice back to pot; A0 go left again to onion
    joint_actions += [[L, R, S], [S, R, S]]  # A0 now at (1,2), A1 at (3,2)

    # A1 drop onion into pot
    joint_actions.append([S, I, S])

    # A0 pick onion again
    joint_actions += [[I, S, S]]

    # A0 to pot (two rights)
    joint_actions += [[R, S, S], [R, S, S]]

    # A0 drops the third onion
    joint_actions.append([I, S, S])

    # A2 finish moving to (3,4) if not there yet & face NORTH (already), then wait for cook time
    # Move A2 one more left if needed so it sits at (3,4)
    joint_actions.append([S, S, L])

    joint_actions.append([U, U, D])
    joint_actions.append([R, R, L])
    joint_actions.append([U, U, D])
    joint_actions.append([R, R, L])

    # Wait until pot cooks from FULL→0 (POT_FULL_STATUS steps)
    # We already spent ~ some steps; to be safe, just wait POT_FULL_STATUS + 2
    wait_steps = int(POT_FULL_STATUS) + 2
    joint_actions += [[S, S, S] for _ in range(wait_steps)]

    # A2 interact to pick soup (ready)
    joint_actions.append([S, S, I])

    # Collect states for GIF
    states_for_gif = []
    states_for_gif.append(state)

    # Step through the plan
    def to_env_dict(act_vec):
        return {f"agent_{i}": jnp.array(act_vec[i], dtype=jnp.uint8) for i in range(env.num_agents)}

    step_key = key
    for acts in joint_actions:
        step_key, use_key = jax.random.split(step_key)
        obs, state, rew, done, info = env.step(use_key, state, to_env_dict(acts))
        states_for_gif.append(state)
        if done["__all__"]:
            break

    # --- Render GIF
    out_dir = Path("gifs")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path("images")
    frames_dir.mkdir(parents=True, exist_ok=True)

    task_name = "3_agents_render"
    viz = OvercookedVisualizer(num_agents=3)
    # viz.animate(states_for_gif, agent_view_size=5, task_idx=0, task_name=task_name, exp_dir=str(out_dir))
    viz.animate(states_for_gif, agent_view_size=5, out_path=str(out_dir / f"{task_name}.gif"))

    gif_path = out_dir / f"{task_name}.gif"

    print(f"✓ GIF saved to: {gif_path}")

    padding = 5 - 1  # agent_view_size - 1 (matches what animate() uses)

    num_saved = 0
    states_for_gif = []
    for t, st in enumerate(states_for_gif):
        # unwrap LogEnvState if present
        env_state = st.env_state if hasattr(st, "env_state") else st

        # crop to the same inner map region used by animate()
        grid = np.asarray(env_state.maze_map[padding:-padding, padding:-padding, :])

        # pass per-agent headings / inventories so hats & held objects render
        agent_dir_idx = np.atleast_1d(env_state.agent_dir_idx)
        agent_inv = np.atleast_1d(env_state.agent_inv)

        print()
        print(f"Step {t}")

        # render a frame (returns HxWx3 uint8)
        frame_img = viz.render_grid(
            grid,
            tile_size=64,                     # keep consistent with viz
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            title=f"t={t}"
        )

        # save PNG
        frame_path = frames_dir / f"frame_{t:04d}.png"
        iio.imwrite(str(frame_path), frame_img)
        num_saved += 1

    assert num_saved == len(states_for_gif), "Mismatch: not all frames were saved"
    print(f"✓ Saved {num_saved} PNG frames to: {frames_dir}")

if __name__ == "__main__":
    test_make_soup_and_render_gif()