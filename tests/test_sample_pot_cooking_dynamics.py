#!/usr/bin/env python
import os
import sys

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from meal.env.common import OBJECT_TO_INDEX, OBJECT_INDEX_TO_VEC
from meal.env.overcooked import Overcooked, POT_READY_STATUS
from meal.env.layouts.presets import cramped_room

# Add project root to path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


# ======================================================================
# Pot cooking dynamics tests
# ======================================================================

def make_env(sample_pot_cooking_dynamics: bool) -> Overcooked:
    """Helper to build a standard Overcooked env on cramped_room."""
    return Overcooked(
        layout=FrozenDict(cramped_room),
        num_agents=2,
        random_reset=False,
        max_steps=400,
        sample_pot_cooking_dynamics=sample_pot_cooking_dynamics,
    )


def test_pot_encoding_consistency_with_sampled_dynamics():
    """
    For sample_pot_cooking_dynamics=True, the encoding should satisfy:
        pot_empty_status - pot_full_status == max_onions_in_pot
    for every reset.

    For sample_pot_cooking_dynamics=False, the same equality should hold
    and the values should match the legacy constants implicitly.
    """
    # sampled case
    env = make_env(sample_pot_cooking_dynamics=True)
    key = jax.random.PRNGKey(0)

    for i in range(5):
        key, subkey = jax.random.split(key)
        _, state = env.reset(subkey)

        empty = int(state.pot_empty_status)
        full = int(state.pot_full_status)
        max_onions = int(state.max_onions_in_pot)

        assert empty > full > POT_READY_STATUS
        assert max_onions > 0
        assert empty - full == max_onions, (
            f"Encoding mismatch at iter {i}: "
            f"empty={empty}, full={full}, max_onions={max_onions}"
        )

    env = make_env(sample_pot_cooking_dynamics=False)
    key = jax.random.PRNGKey(123)
    _, state = env.reset(key)

    empty = int(state.pot_empty_status)
    full = int(state.pot_full_status)
    max_onions = int(state.max_onions_in_pot)

    assert empty - full == max_onions
    # legacy defaults were 23, 20, 3 — this checks we haven't drifted
    assert empty == 23
    assert full == 20
    assert max_onions == 3


def test_pot_cook_time_countdown_matches_sampled_cook_time():
    """
    If we manually set a pot tile to 'full' (status == pot_full_status),
    then stepping the env for exactly pot_full_status timesteps with no
    interactions should drive that status down to POT_READY_STATUS (0).
    """
    env = make_env(sample_pot_cooking_dynamics=True)
    key = jax.random.PRNGKey(1)

    # Reset and grab state + encoding params
    _, state = env.reset(key)
    cook_time = int(state.pot_full_status)
    assert cook_time > 0

    # Find first real pot index
    pot_indices = jnp.where(state.pot_mask)[0]
    assert pot_indices.size > 0
    pot_idx = int(pot_indices[0])

    pos = state.pot_pos[pot_idx]  # (2,)
    x, y = int(pos[0]), int(pos[1])

    # Manually set this pot's status to "full" (start of cooking countdown)
    maze_map = state.maze_map
    maze_map = maze_map.at[y, x, 2].set(state.pot_full_status.astype(maze_map.dtype))

    state = state.replace(maze_map=maze_map)

    # Step env with "stay" actions so no interactions happen
    actions = {
        "agent_0": jnp.array(4, dtype=jnp.int32),  # index for Actions.stay
        "agent_1": jnp.array(4, dtype=jnp.int32),
    }

    key = jax.random.PRNGKey(2)
    for _ in range(cook_time):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions)

    # After exactly cook_time steps, this pot should be ready
    final_status = int(state.maze_map[y, x, 2])
    assert final_status == POT_READY_STATUS, (
        f"Expected pot status 0 after {cook_time} steps, got {final_status}"
    )


def test_observation_soup_channel_matches_max_onions_in_pot():
    """
    The onions-in-soup observation channel (18) should equal
    state.max_onions_in_pot for tiles containing a 'dish'.
    """
    env = make_env(sample_pot_cooking_dynamics=True)
    key = jax.random.PRNGKey(3)
    _, state = env.reset(key)

    max_onions = int(state.max_onions_in_pot)
    assert max_onions > 0

    # Take the first goal position as a free tile and overwrite it with a dish
    gx, gy = int(state.goal_pos[0, 0]), int(state.goal_pos[0, 1])

    dish_idx = OBJECT_TO_INDEX["dish"]
    dish_vec = OBJECT_INDEX_TO_VEC[dish_idx]

    maze_map = state.maze_map
    maze_map = maze_map.at[gy, gx, :].set(dish_vec)
    state = state.replace(maze_map=maze_map)

    # Compute observations for this modified state
    obs = env.get_obs(state)

    # Channel index 18 is "number of onions in soup"
    soup_channel = 18

    for agent in env.agents:
        val = int(obs[agent][gy, gx, soup_channel])
        assert val == max_onions, (
            f"Agent {agent} sees {val} onions in soup, "
            f"expected {max_onions}"
        )
