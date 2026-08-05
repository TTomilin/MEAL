"""Agent-level pickup restrictions (`agent_restrictions`): an agent barred from
picking up onions/plates must never succeed at it, while the complementary,
unrestricted agent must still be able to pick up that same item type."""
import jax
import jax.numpy as jnp
import pytest
from flax.core import FrozenDict

from meal.env.overcooked import Overcooked, OBJECT_TO_INDEX
from meal.env.overcooked.layouts.presets import cramped_room

A = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5}

# Agent 0 walks: to the onion pile and tries to grab one, then to the plate
# pile and tries to grab one. Agent 1 does the same in its own two-step probe.
_AGENT_0_PROBES = [
    (A['L'], A['S']), (A['I'], A['S']),                    # try onion
    (A['R'], A['S']), (A['D'], A['S']), (A['L'], A['S']), (A['D'], A['S']), (A['I'], A['S']),  # try plate
    (A['U'], A['S']), (A['R'], A['S']), (A['U'], A['S']),  # reset position
]
_AGENT_1_PROBES = [
    (A['S'], A['R']), (A['S'], A['I']),                    # try onion
    # Route to the plate pile via row 2 (not row 1, where agent 0 ends up
    # parked after its own probe above) to avoid blocking on agent 0.
    (A['S'], A['D']), (A['S'], A['L']), (A['S'], A['L']), (A['S'], A['D']), (A['S'], A['I']),  # try plate
]
ALL_PROBES = _AGENT_0_PROBES + _AGENT_1_PROBES
# Indices (into ALL_PROBES) of the four "attempt" steps we check pickups on.
STEP_AGENT_0_ONION = 1
STEP_AGENT_0_PLATE = 6
STEP_AGENT_1_ONION = len(_AGENT_0_PROBES) + 1
STEP_AGENT_1_PLATE = len(_AGENT_0_PROBES) + 6


@pytest.mark.parametrize("restrictions", [
    pytest.param(
        {"agent_0_cannot_pick_onions": True, "agent_0_cannot_pick_plates": False,
         "agent_1_cannot_pick_onions": False, "agent_1_cannot_pick_plates": True},
        id="agent_0_no_onions_agent_1_no_plates",
    ),
    pytest.param(
        {"agent_0_cannot_pick_onions": False, "agent_0_cannot_pick_plates": True,
         "agent_1_cannot_pick_onions": True, "agent_1_cannot_pick_plates": False},
        id="agent_0_no_plates_agent_1_no_onions",
    ),
])
def test_complementary_restrictions(restrictions):
    env = Overcooked(
        layout=FrozenDict(cramped_room), num_agents=2, random_reset=False,
        max_steps=400, agent_restrictions=restrictions,
    )
    rng = jax.random.PRNGKey(42)
    _, state = env.reset(rng)

    invs_by_step = []
    for action_0, action_1 in ALL_PROBES:
        rng, step_key = jax.random.split(rng)
        _, state, _, _, _ = env.step_env(
            step_key, state, {"agent_0": jnp.uint32(action_0), "agent_1": jnp.uint32(action_1)}
        )
        invs_by_step.append((int(state.agent_inv[0]), int(state.agent_inv[1])))

    def picked_up(step_idx, agent_idx, item):
        prev_inv = invs_by_step[step_idx - 1][agent_idx] if step_idx > 0 else OBJECT_TO_INDEX["empty"]
        curr_inv = invs_by_step[step_idx][agent_idx]
        return prev_inv == OBJECT_TO_INDEX["empty"] and curr_inv == OBJECT_TO_INDEX[item]

    agent_0_got_onion = picked_up(STEP_AGENT_0_ONION, 0, "onion")
    agent_0_got_plate = picked_up(STEP_AGENT_0_PLATE, 0, "plate")
    agent_1_got_onion = picked_up(STEP_AGENT_1_ONION, 1, "onion")
    agent_1_got_plate = picked_up(STEP_AGENT_1_PLATE, 1, "plate")

    # A restricted agent must never succeed at the restricted pickup.
    if restrictions["agent_0_cannot_pick_onions"]:
        assert not agent_0_got_onion
    else:
        assert agent_0_got_onion
    if restrictions["agent_0_cannot_pick_plates"]:
        assert not agent_0_got_plate
    else:
        assert agent_0_got_plate
    if restrictions["agent_1_cannot_pick_onions"]:
        assert not agent_1_got_onion
    else:
        assert agent_1_got_onion
    if restrictions["agent_1_cannot_pick_plates"]:
        assert not agent_1_got_plate
    else:
        assert agent_1_got_plate
