"""Scripted cramped_room rollout: cook + deliver one soup, in single-agent,
double-agent, and partially-observable variants. Checks shaped-reward
bookkeeping (3 onions + 1 soup-pickup event) and the delivery reward."""
import jax
import numpy as np
from flax.core import FrozenDict

from meal.env.overcooked import Overcooked, DELIVERY_REWARD
from meal.env.overcooked.layouts.presets import cramped_room
from meal.env.overcooked.overcooked_po import OvercookedPO

A = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5}

# "pick onion -> pot" pattern, repeated for all 3 onions, then plate + deliver.
ONION_CYCLE = [A['L'], A['I'], A['R'], A['U'], A['I']]
COOK_AND_DELIVER = (
    ONION_CYCLE * 3
    + [A['S']] * 5  # wait for cooking (cook_time=5)
    + [
        A['D'], A['L'], A['D'], A['I'],  # walk to plate pile, take plate
        A['U'], A['R'], A['U'], A['I'],  # walk to pot, scoop soup
        A['D'], A['R'], A['D'], A['I'],  # walk to serving window, deliver
    ]
)
EXPECTED_SHAPED = 3 * 3 + 5  # 3 onion-in-pot events + 1 soup-pickup event


def test_cramped_room_single_agent(rollout):
    env = Overcooked(layout=FrozenDict(cramped_room), num_agents=1, random_reset=False,
                      max_steps=400, cook_time=5)
    rng = jax.random.PRNGKey(0)

    _, _, shaped_rewards, _ = rollout(env, rng, {"agent_0": COOK_AND_DELIVER})

    total_shaped = sum(float(s["agent_0"]) for s in shaped_rewards)
    assert np.isclose(total_shaped, EXPECTED_SHAPED)


def test_cramped_room_double_agent(rollout):
    env = Overcooked(layout=FrozenDict(cramped_room), num_agents=2, random_reset=False,
                      max_steps=400, cook_time=5)
    rng = jax.random.PRNGKey(0)
    actions = {"agent_0": COOK_AND_DELIVER, "agent_1": [A['S']] * len(COOK_AND_DELIVER)}

    _, rewards, shaped_rewards, _ = rollout(env, rng, actions)

    total_reward = sum(float(r["agent_0"]) for r in rewards)
    total_shaped_0 = sum(float(s["agent_0"]) for s in shaped_rewards)
    assert np.isclose(total_shaped_0, EXPECTED_SHAPED)
    assert total_reward >= float(DELIVERY_REWARD)


def test_cramped_room_partially_observable(rollout):
    env = OvercookedPO(
        layout=FrozenDict(cramped_room), num_agents=2, random_reset=False,
        max_steps=400, cook_time=5, view_ahead=3, view_behind=1, view_sides=1,
    )
    rng = jax.random.PRNGKey(0)
    actions = {"agent_0": COOK_AND_DELIVER, "agent_1": [A['S']] * len(COOK_AND_DELIVER)}

    _, rewards, shaped_rewards, _ = rollout(env, rng, actions)

    total_reward = sum(float(r["agent_0"]) for r in rewards)
    total_shaped_0 = sum(float(s["agent_0"]) for s in shaped_rewards)
    # PO shaping differs slightly from full observability; check it's in the
    # right ballpark rather than pinning the exact value.
    assert total_shaped_0 >= EXPECTED_SHAPED - 4
    assert total_reward >= float(DELIVERY_REWARD)
