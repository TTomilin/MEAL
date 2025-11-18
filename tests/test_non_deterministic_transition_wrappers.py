#!/usr/bin/env python
from meal.env.overcooked import Overcooked
from meal.env.layouts.presets import cramped_room
from meal.wrappers.randomized_actions import RandomizedActions
from meal.wrappers.sticky_actions import StickyActions
from flax.core import FrozenDict
import jax.numpy as jnp
import jax
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

# ======================================================================
# StickyActions tests
# ======================================================================


def test_sticky_actions_p0_is_transparent():
    """
    For p = 0.0, StickyActions should never override the given action.
    We check that env sees exactly the action we pass in.
    """
    base_env = Overcooked(layout=FrozenDict(
        cramped_room), num_agents=2, random_reset=False, max_steps=400, soup_cook_time=5)
    wrapper = StickyActions(base_env, p=0.0)

    rng = jax.random.PRNGKey(0)
    obs, state = wrapper.reset(rng)

    actions_seq = [
        {"agent_0": jnp.array(1, dtype=jnp.int32),
         "agent_1": jnp.array(2, dtype=jnp.int32)},
        {"agent_0": jnp.array(3, dtype=jnp.int32),
         "agent_1": jnp.array(4, dtype=jnp.int32)},
        {"agent_0": jnp.array(5, dtype=jnp.int32),
         "agent_1": jnp.array(6, dtype=jnp.int32)},
    ]

    for a in actions_seq:
        rng, step_key = jax.random.split(rng)
        obs, state, rew, done, info = wrapper.step(step_key, state, a)

        applied = info["applied_action"]
        assert int(applied["agent_0"]) == int(a["agent_0"])
        assert int(applied["agent_1"]) == int(a["agent_1"])

        # last_action in wrapper state should be the same as applied action
        assert int(state.last_action["agent_0"]) == int(a["agent_0"])
        assert int(state.last_action["agent_1"]) == int(a["agent_1"])


def test_sticky_actions_p1_always_repeats_last():
    """
    For p = 1.0, the wrapper should *always* use the previous action
    (starting from zeros after reset).
    """
    base_env = Overcooked(layout=FrozenDict(
        cramped_room), num_agents=2, random_reset=False, max_steps=400, soup_cook_time=5)
    wrapper = StickyActions(base_env, p=1.0)

    rng = jax.random.PRNGKey(0)
    obs, state = wrapper.reset(rng)

    # First step: last_action initialized to 0, so effective_action must be 0
    action_1 = {"agent_0": jnp.array(
        5, dtype=jnp.int32), "agent_1": jnp.array(7, dtype=jnp.int32)}
    rng, step_key = jax.random.split(rng)
    obs, state, rew, done, info = wrapper.step(step_key, state, action_1)

    applied_1 = info["applied_action"]
    assert int(applied_1["agent_0"]) == 0
    assert int(applied_1["agent_1"]) == 0
    assert int(state.last_action["agent_0"]) == 0
    assert int(state.last_action["agent_1"]) == 0

    # Second step: last_action is still 0, so even with new action, env still sees 0
    action_2 = {"agent_0": jnp.array(
        9, dtype=jnp.int32), "agent_1": jnp.array(11, dtype=jnp.int32)}
    rng, step_key = jax.random.split(rng)
    obs, state, rew, done, info = wrapper.step(step_key, state, action_2)

    applied_2 = info["applied_action"]
    assert int(applied_2["agent_0"]) == 0
    assert int(applied_2["agent_1"]) == 0
    assert int(state.last_action["agent_0"]) == 0
    assert int(state.last_action["agent_1"]) == 0


def test_sticky_actions_updates_last_action_to_effective_action():
    """
    Regardless of p, last_action in the wrapper state must always equal
    the *effective* action actually sent to env (not the raw input).
    We check this by comparing with info["applied_action"].
    """
    base_env = Overcooked(layout=FrozenDict(
        cramped_room), num_agents=2, random_reset=False, max_steps=400, soup_cook_time=5)
    wrapper = StickyActions(base_env, p=0.7)

    rng = jax.random.PRNGKey(123)
    obs, state = wrapper.reset(rng)

    action = {"agent_0": jnp.array(
        1, dtype=jnp.int32), "agent_1": jnp.array(2, dtype=jnp.int32)}

    for _ in range(5):
        rng, step_key = jax.random.split(rng)
        obs, state, rew, done, info = wrapper.step(step_key, state, action)

        applied = info["applied_action"]
        assert int(state.last_action["agent_0"]) == int(applied["agent_0"])
        assert int(state.last_action["agent_1"]) == int(applied["agent_1"])


# ======================================================================
# RandomizedActions tests
# ======================================================================

def test_randomized_actions_p0_is_transparent():
    """
    For p = 0.0, RandomizedActions should never replace the action.
    """
    base_env = Overcooked(layout=FrozenDict(
        cramped_room), num_agents=2, random_reset=False, max_steps=400, soup_cook_time=5)
    wrapper = RandomizedActions(base_env, p_replace=0.0)

    rng = jax.random.PRNGKey(0)
    obs, state = wrapper.reset(rng)

    actions_seq = [
        {"agent_0": jnp.array(0, dtype=jnp.int32),
         "agent_1": jnp.array(1, dtype=jnp.int32)},
        {"agent_0": jnp.array(2, dtype=jnp.int32),
         "agent_1": jnp.array(3, dtype=jnp.int32)},
        {"agent_0": jnp.array(1, dtype=jnp.int32),
         "agent_1": jnp.array(0, dtype=jnp.int32)},
    ]

    for a in actions_seq:
        rng, step_key = jax.random.split(rng)
        obs, state, rew, done, info = wrapper.step(step_key, state, a)

        applied = info["applied_action"]
        assert int(applied["agent_0"]) == int(a["agent_0"])
        assert int(applied["agent_1"]) == int(a["agent_1"])


def test_randomized_actions_p1_matches_rng_logic():
    """
    For p = 1.0, every element should be replaced by a random action.

    We make this deterministic by reproducing the exact RNG usage of the wrapper:
        key, subkey = jax.random.split(key)
        random_action = randint(subkey, ...)
    and checking that env sees exactly this random_action.
    """
    base_env = Overcooked(layout=FrozenDict(
        cramped_room), num_agents=2, random_reset=False, max_steps=400, soup_cook_time=5)
    wrapper = RandomizedActions(base_env, p_replace=1.0)

    rng = jax.random.PRNGKey(42)
    obs, state = wrapper.reset(rng)

    # We always feed the same user action so we can distinguish it from random ones
    user_action = {
        "agent_0": jnp.array(0, dtype=jnp.int32),
        "agent_1": jnp.array(0, dtype=jnp.int32),
    }

    for _ in range(5):
        rng, step_key = jax.random.split(rng)

        # Reproduce wrapper's randomness:
        #   key, subkey = jax.random.split(key)
        _, subkey = jax.random.split(step_key)

        expected_rand = jax.random.randint(
            subkey, shape=(), minval=0, maxval=wrapper.n_actions
        )

        obs, state, rew, done, info = wrapper.step(
            step_key, state, user_action)
        applied = info["applied_action"]
        assert int(applied["agent_0"]) == int(expected_rand)
        assert int(applied["agent_1"]) == int(expected_rand)


def test_randomized_actions_mixture_of_user_and_random():
    """
    Smoke test for 0 < p_replace < 1. We don't do probabilistic checks,
    just verify that the wrapper runs, produces legal actions, and the
    values are always within [0, n_actions).
    """
    base_env = Overcooked(layout=FrozenDict(
        cramped_room), num_agents=2, random_reset=False, max_steps=400, soup_cook_time=5)
    wrapper = RandomizedActions(base_env, p_replace=0.5)

    rng = jax.random.PRNGKey(7)
    obs, state = wrapper.reset(rng)

    user_action = {
        "agent_0": jnp.array(1, dtype=jnp.int32),
        "agent_1": jnp.array(2, dtype=jnp.int32),
    }

    for _ in range(20):
        rng, step_key = jax.random.split(rng)
        obs, state, rew, done, info = wrapper.step(
            step_key, state, user_action)

        applied = info["applied_action"]
        a0 = int(applied["agent_0"])
        a1 = int(applied["agent_1"])

        assert 0 <= a0 < wrapper.n_actions
        assert 0 <= a1 < wrapper.n_actions
