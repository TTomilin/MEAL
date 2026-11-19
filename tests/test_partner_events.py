"""Per-agent task-event counts in Overcooked (partner can contribute without delivering)."""
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from meal import make_env
from meal.env.overcooked.common import EVENT_NAMES
from meal.env.overcooked.layouts.presets import cramped_room
from meal.env.overcooked.overcooked_env import BASE_REW_SHAPING_PARAMS

A = {"U": 0, "D": 1, "R": 2, "L": 3, "S": 4, "I": 5}
ONION_CYCLE = [A["L"], A["I"], A["R"], A["U"], A["I"]]
PICK_ONION_POT_PLATE = ONION_CYCLE + [A["D"], A["L"], A["D"], A["I"]]


def rollout(actions_0, actions_1):
    env = make_env("overcooked", layout=FrozenDict(cramped_room), max_steps=len(actions_0), cook_time=5)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    rewards, shaped, events = [], [], []
    for t in range(len(actions_0)):
        key, step_key = jax.random.split(key)
        act = {"agent_0": jnp.uint32(actions_0[t]), "agent_1": jnp.uint32(actions_1[t])}
        _, state, reward, _, info = env.step(step_key, state, act)
        rewards.append([float(reward["agent_0"]), float(reward["agent_1"])])
        shaped.append([float(info["shaped_reward"]["agent_0"]), float(info["shaped_reward"]["agent_1"])])
        events.append(np.asarray(info["events"]["agent_0"], dtype=np.float32))
    return np.array(rewards), np.array(shaped), np.array(events)


def test_onion_pot_plate_events_and_unchanged_rewards():
    actions_1 = [A["S"]] * len(PICK_ONION_POT_PLATE)
    rewards, shaped, events = rollout(PICK_ONION_POT_PLATE, actions_1)
    totals = events.sum(axis=0)
    by_name = dict(zip(EVENT_NAMES, totals))

    assert by_name["onion_pickups"] == 1
    assert by_name["onions_into_pot"] == 1
    assert by_name["plate_pickups"] == 1
    assert by_name["soup_pickups"] == 0
    assert by_name["deliveries"] == 0
    assert by_name["counter_drops"] == 0
    assert by_name["counter_pickups"] == 0

    # Same delivery + shaping as before event tracking (no reward path changes).
    assert rewards.sum() == 0.0
    assert shaped[:, 1].sum() == 0.0
    assert shaped[:, 0].sum() == BASE_REW_SHAPING_PARAMS["PLACEMENT_IN_POT_REW"]
