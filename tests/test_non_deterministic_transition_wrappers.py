import os
import sys

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from meal.env.layouts.presets import cramped_room
from meal.env.overcooked import Overcooked
from meal.wrappers.slippery_tiles import SlipperyTiles
from meal.wrappers.sticky_actions import StickyActions

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# ======================================================================
# StickyActions tests
# ======================================================================


def test_sticky_actions_p0_is_transparent():
    """
    For p = 0.0, StickyActions should never override the given action.
    We check that env sees exactly the action we pass in.
    """
    base_env = Overcooked(layout=FrozenDict(cramped_room), layout_name="Cramped Room")
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
    base_env = Overcooked(layout=FrozenDict(cramped_room), layout_name="Cramped Room")
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
    base_env = Overcooked(layout=FrozenDict(cramped_room), layout_name="Cramped Room")
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
# SlipperyTiles tests
# ======================================================================

def test_slippery_tiles_p0_is_transparent():
    """
    For p_replace = 0.0, SlipperyTiles should NEVER replace the given action,
    regardless of which tiles are slippery or who is armed to slip.
    """
    base_env = Overcooked(layout=FrozenDict(cramped_room), layout_name="Cramped Room")
    wrapper = SlipperyTiles(base_env, slip_prob=0.0)

    rng = jax.random.PRNGKey(0)
    obs, state = wrapper.reset(rng)

    actions_seq = [
        {"agent_0": jnp.array(0, dtype=jnp.int32),
         "agent_1": jnp.array(1, dtype=jnp.int32)},
        {"agent_0": jnp.array(2, dtype=jnp.int32),
         "agent_1": jnp.array(3, dtype=jnp.int32)},
        {"agent_0": jnp.array(5, dtype=jnp.int32),
         "agent_1": jnp.array(5, dtype=jnp.int32)},
    ]

    for a in actions_seq:
        rng, step_key = jax.random.split(rng)
        obs, state, rew, done, info = wrapper.step(step_key, state, a)

        applied = info["applied_action"]
        slipped = info["slipped"]

        # No replacement: applied == user action
        assert int(applied["agent_0"]) == int(a["agent_0"])
        assert int(applied["agent_1"]) == int(a["agent_1"])

        # And no slip is ever reported
        assert int(slipped[0]) == 0
        assert int(slipped[1]) == 0


def test_slippery_tiles_p1_slips_when_armed():
    """
    For p_replace = 1.0, any agent that is *armed* to slip (will_slip_next=True)
    MUST slip on the next step, and its action must be replaced by a move action
    in {0,1,2,3}.
    """
    base_env = Overcooked(layout=FrozenDict(cramped_room), layout_name="Cramped Room")
    wrapper = SlipperyTiles(base_env, slip_prob=1.0)

    rng = jax.random.PRNGKey(1)
    obs, state = wrapper.reset(rng)

    # Force both agents to be "armed" to slip on the next step
    prev_state = state
    state = state.replace(will_slip_next=jnp.array([True, True]))

    # Use a non-move action (5 = interact) so replacement is obvious
    user_action = {
        "agent_0": jnp.array(5, dtype=jnp.int32),
        "agent_1": jnp.array(5, dtype=jnp.int32),
    }

    rng, step_key = jax.random.split(rng)
    obs, state, rew, done, info = wrapper.step(step_key, state, user_action)

    applied = info["applied_action"]
    slipped = info["slipped"]

    # Both agents must have slipped
    assert bool(slipped[0])
    assert bool(slipped[1])

    # Slip mask should match previous will_slip_next
    assert int(slipped[0]) == int(prev_state.will_slip_next[0])
    assert int(slipped[1]) == int(prev_state.will_slip_next[1])

    # And the resulting actions must be *move* actions: 0,1,2,3
    a0 = int(applied["agent_0"])
    a1 = int(applied["agent_1"])
    assert a0 in (0, 1, 2, 3)
    assert a1 in (0, 1, 2, 3)


def test_slippery_tiles_partial_prob_behaviour():
    """
    Smoke test for 0 < p_replace < 1.

    We periodically force agents to be armed (will_slip_next=True) and check:
      - slipped[i] == True  → applied action in {0,1,2,3}
      - slipped[i] == False → applied action equals user action
    """
    base_env = Overcooked(layout=FrozenDict(cramped_room), layout_name="Cramped Room")
    wrapper = SlipperyTiles(base_env, slip_prob=0.5)

    rng = jax.random.PRNGKey(7)
    obs, state = wrapper.reset(rng)

    # Again use a non-move action so replacement is detectable
    user_action = {
        "agent_0": jnp.array(5, dtype=jnp.int32),
        "agent_1": jnp.array(5, dtype=jnp.int32),
    }

    for t in range(20):
        # Every other step, forcibly arm both agents so we actually
        # exercise the slip branch regardless of the random mask.
        if t % 2 == 0:
            state = state.replace(will_slip_next=jnp.array([True, True]))

        rng, step_key = jax.random.split(rng)
        obs, state, rew, done, info = wrapper.step(step_key, state, user_action)

        applied = info["applied_action"]
        slipped = info["slipped"]

        a0 = int(applied["agent_0"])
        a1 = int(applied["agent_1"])

        # Basic range sanity: all actions must be legal discrete actions
        assert 0 <= a0 < wrapper._env.num_actions
        assert 0 <= a1 < wrapper._env.num_actions

        # If slipped → must be a move action 0..3
        if bool(slipped[0]):
            assert a0 in (0, 1, 2, 3)
        else:
            # No slip → must equal user action
            assert a0 == int(user_action["agent_0"])

        if bool(slipped[1]):
            assert a1 in (0, 1, 2, 3)
        else:
            assert a1 == int(user_action["agent_1"])


if __name__ == "__main__":
    test_sticky_actions_p0_is_transparent()
    test_sticky_actions_p1_always_repeats_last()
    test_sticky_actions_updates_last_action_to_effective_action()
    test_slippery_tiles_p0_is_transparent()
    test_slippery_tiles_p1_slips_when_armed()
    test_slippery_tiles_partial_prob_behaviour()
    print("All tests passed.")