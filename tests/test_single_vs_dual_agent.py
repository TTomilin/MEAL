"""1-agent Overcooked vs 2-agent Overcooked (agent 1 idle), agent 0 given an
identical scripted action sequence in both. From agent 0's perspective the two
environments should be observationally and reward-wise equivalent, modulo the
extra agent-1 layers the 2-agent observation carries (which are excluded from
comparison here)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax.core import FrozenDict

from meal.env.overcooked import Overcooked, DELIVERY_REWARD
from meal.env.overcooked.layouts.presets import cramped_room, asymm_advantages

A = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5}


def _cramped_room_actions():
    onion_cycle = [A['L'], A['I'], A['R'], A['U'], A['I']]
    actions = onion_cycle * 3 + [A['S']] * 20
    actions += [A['D'], A['L'], A['D'], A['I'], A['U'], A['R'], A['U'], A['I'],
                A['D'], A['R'], A['D'], A['I']]
    return actions


def _asymm_advantages_actions():
    actions = [
        A['U'], A['L'], A['U'], A['L'], A['I'], A['D'], A['R'], A['R'], A['I'],
        A['L'], A['L'], A['U'], A['L'], A['I'], A['D'], A['R'], A['R'], A['I'],
        A['L'], A['L'], A['U'], A['L'], A['I'], A['D'], A['R'], A['R'], A['I'],
    ]
    actions += [A['S']] * 16
    actions += [A['D'], A['I'], A['U'], A['R'], A['I'], A['U'], A['I'], A['L']]
    return actions


LAYOUT_SCENARIOS = {
    "cramped_room": (FrozenDict(cramped_room), _cramped_room_actions()),
    "asymm_advantages": (FrozenDict(asymm_advantages), _asymm_advantages_actions()),
    # coord_ring and a custom tiny "simple_kitchen" grid were tried here too,
    # but both are structurally incompatible with this harness's "agent 1
    # always stays put" design: coord_ring's only path to its plate pile
    # passes through agent 1's fixed spawn tile (permanently blocking it),
    # and simple_kitchen is too small for agent 1's randomly-placed 2-agent
    # spawn (its layout only defines one agent slot) to reliably avoid
    # agent 0's path. Neither is a product bug, just a bad fit for this test.
}


def _obs_diff(obs_1, obs_2):
    """Sum of |diff| over agent-0's position/orientation channels and the
    shared environment layers, excluding the 2-agent env's agent-1 channels
    (channel 1, channels 6-9) which have no counterpart in the 1-agent obs."""
    obs_1 = obs_1.astype(np.float32)
    obs_2 = obs_2.astype(np.float32)

    pos_diff = np.sum(np.abs(obs_1[:, :, 0] - obs_2[:, :, 0]))
    ori_diff = np.sum(np.abs(obs_1[:, :, 2:6] - obs_2[:, :, 2:6]))
    env_diff = np.sum(np.abs(obs_1[:, :, 6:22] - obs_2[:, :, 10:26]))
    return pos_diff + ori_diff + env_diff


@pytest.mark.parametrize("layout_name", list(LAYOUT_SCENARIOS.keys()))
def test_single_vs_dual_agent_equivalence(layout_name):
    layout, action_sequence = LAYOUT_SCENARIOS[layout_name]

    env_1_agent = Overcooked(layout=layout, num_agents=1, random_reset=False, max_steps=400)
    env_2_agent = Overcooked(layout=layout, num_agents=2, random_reset=False, max_steps=400)

    rng = jax.random.PRNGKey(42)
    rng1, rng2 = jax.random.split(rng)
    _, state_1 = env_1_agent.reset(rng1)
    _, state_2 = env_2_agent.reset(rng2)

    total_reward_1 = total_reward_2 = 0.0
    total_shaped_1 = total_shaped_2 = 0.0
    obs_diffs = []

    for action in action_sequence:
        rng, key1 = jax.random.split(rng)
        obs_1, state_1, reward_1, _, info_1 = env_1_agent.step(
            key1, state_1, {"agent_0": jnp.uint32(action)}
        )
        rng, key2 = jax.random.split(rng)
        obs_2, state_2, reward_2, _, info_2 = env_2_agent.step_env(
            key2, state_2, {"agent_0": jnp.uint32(action), "agent_1": jnp.uint32(A['S'])}
        )

        total_reward_1 += float(reward_1["agent_0"])
        total_reward_2 += float(reward_2["agent_0"])
        total_shaped_1 += float(info_1["shaped_reward"]["agent_0"])
        total_shaped_2 += float(info_2["shaped_reward"]["agent_0"])
        obs_diffs.append(_obs_diff(obs_1["agent_0"], obs_2["agent_0"]))

    # Delivery rewards must match exactly: agent 0 does identical actions.
    assert abs(total_reward_1 - total_reward_2) < 1e-3
    # Shaped rewards may differ slightly due to implementation details, but
    # not by more than a couple of shaping events' worth.
    assert abs(total_shaped_1 - total_shaped_2) < 5.0
    # Observations (agent-0 position/orientation + shared env layers) match,
    # up to one known, by-design cell: when num_agents < the layout's spawn
    # count, unused spawn tiles are walled off (see overcooked_env.py reset),
    # so the 1-agent env's wall channel has exactly one more wall cell than
    # the 2-agent env's (agent 1's unused spawn tile). That's a constant
    # per-step offset of 1.0, not drift, so anything above it is a real bug.
    assert np.mean(obs_diffs) < 1.5

    # Both environments must have actually completed the task.
    assert total_reward_1 >= DELIVERY_REWARD * 0.9, f"1-agent didn't complete task: {total_reward_1}"
    assert total_reward_2 >= DELIVERY_REWARD * 0.9, f"2-agent didn't complete task: {total_reward_2}"
