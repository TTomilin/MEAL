"""Reward-attribution tests: for three cook/deliver scenarios (one agent doing
everything, onion contributions split across agents, plating/delivery split
across agents), verify that 'default' (shared delivery + individual shaping),
'sparse' (shared delivery only), and 'individual' (individual delivery +
individual shaping) reward settings all attribute reward exactly as the
reward-processing logic specifies."""
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from meal.env.overcooked import Overcooked, DELIVERY_REWARD
from meal.env.overcooked.layouts.presets import cramped_room

A = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5}

ONION_CYCLE = [A['L'], A['I'], A['R'], A['U'], A['I']]
COOK_AND_DELIVER = (
        ONION_CYCLE * 3
        + [A['S']] * 5
        + [
            A['D'], A['L'], A['D'], A['I'],
            A['U'], A['R'], A['U'], A['I'],
            A['D'], A['R'], A['D'], A['I'],
        ]
)


def process_reward(reward, shaped_reward, sparse_rewards=False, individual_rewards=False, annealing_factor=1.0):
    """Reproduces the reward-processing logic used by IPPO_CL for the three
    reward settings ('default' is neither sparse nor individual)."""
    if sparse_rewards:
        # Both agents get the total delivery reward, no shaping.
        total_delivery = reward["agent_0"] + reward["agent_1"]
        return {"agent_0": total_delivery, "agent_1": total_delivery}
    elif individual_rewards:
        # Each agent gets only its own delivery reward plus its own shaping.
        return {"agent_0": reward["agent_0"] + shaped_reward["agent_0"] * annealing_factor,
                "agent_1": reward["agent_1"] + shaped_reward["agent_1"] * annealing_factor}
    else:
        # Shared delivery reward + individual shaping.
        total_delivery = reward["agent_0"] + reward["agent_1"]
        return {"agent_0": total_delivery + shaped_reward["agent_0"] * annealing_factor,
                "agent_1": total_delivery + shaped_reward["agent_1"] * annealing_factor}


def run_scenario(actions_agent_0, actions_agent_1):
    """Roll out `actions_agent_0`/`actions_agent_1` against a fresh cramped_room
    env once per reward setting, returning summed reward/shaping/soups per agent."""
    max_len = max(len(actions_agent_0), len(actions_agent_1))
    actions_agent_0 = actions_agent_0 + [A['S']] * (max_len - len(actions_agent_0))
    actions_agent_1 = actions_agent_1 + [A['S']] * (max_len - len(actions_agent_1))

    env = Overcooked(layout=FrozenDict(cramped_room), num_agents=2, random_reset=False,
                     max_steps=400, cook_time=5)

    results = {}
    for setting in ('default', 'sparse', 'individual'):
        total_reward = {"agent_0": 0.0, "agent_1": 0.0}
        total_shaped = {"agent_0": 0.0, "agent_1": 0.0}
        total_soups = {"agent_0": 0.0, "agent_1": 0.0}

        rng = jax.random.PRNGKey(0)
        _, state = env.reset(rng)

        for t in range(max_len):
            rng, step_key = jax.random.split(rng)
            _, state, rew, done, info = env.step_env(
                step_key, state,
                {"agent_0": jnp.uint32(actions_agent_0[t]), "agent_1": jnp.uint32(actions_agent_1[t])}
            )
            processed = process_reward(
                rew, info["shaped_reward"],
                sparse_rewards=(setting == 'sparse'),
                individual_rewards=(setting == 'individual'),
            )
            for agent in ("agent_0", "agent_1"):
                total_reward[agent] += float(processed[agent])
                total_shaped[agent] += float(info["shaped_reward"][agent])
                total_soups[agent] += float(info["soups"][agent])

        results[setting] = {"reward": total_reward, "shaped": total_shaped, "soups": total_soups}

    return results


def assert_reward_settings_consistent(results):
    """Shared assertions: given the per-setting totals from `run_scenario`,
    check that all three reward settings attribute reward as specified."""
    default, sparse, individual = results['default'], results['sparse'], results['individual']

    total_soups = default['soups']['agent_0'] + default['soups']['agent_1']
    total_delivery = total_soups * DELIVERY_REWARD

    # Default: shared delivery reward + individual shaping.
    for agent in ("agent_0", "agent_1"):
        expected = total_delivery + default['shaped'][agent]
        assert np.isclose(default['reward'][agent], expected, atol=1e-6)

    # Sparse: shared delivery reward only.
    for agent in ("agent_0", "agent_1"):
        assert np.isclose(sparse['reward'][agent], total_delivery, atol=1e-6)

    # Individual: individual delivery reward + individual shaping.
    for agent in ("agent_0", "agent_1"):
        expected = default['soups'][agent] * DELIVERY_REWARD + default['shaped'][agent]
        assert np.isclose(individual['reward'][agent], expected, atol=1e-6)


def test_scenario_agent_0_does_everything():
    """Agent 0 collects all 3 onions, plates, and delivers alone."""
    actions_agent_1 = [A['S']] * 50
    results = run_scenario(list(COOK_AND_DELIVER), actions_agent_1)
    assert_reward_settings_consistent(results)


def test_scenario_shared_onion_contribution():
    """Agent 0 contributes 2 onions and delivers; agent 1 contributes 1 onion."""
    actions_agent_0 = (
            ONION_CYCLE * 2
            + [A['L']]  # step out of agent 1's way
            + [A['S']] * 10  # wait for agent 1's onion
            + [A['S']] * 5  # wait for cooking
            + [
                A['D'], A['L'], A['D'], A['I'],
                A['U'], A['R'], A['U'], A['I'],
                A['D'], A['R'], A['D'], A['I'],
            ]
    )
    actions_agent_1 = (
            [A['S']] * 15  # wait for agent 0 to clear the onion pile
            + [A['R'], A['I'], A['L'], A['L'], A['U'], A['I']]  # 1 onion
            + [A['S']] * 100
    )
    results = run_scenario(actions_agent_0, actions_agent_1)
    assert_reward_settings_consistent(results)


def test_scenario_one_plates_other_delivers():
    """Agent 0 collects all 3 onions and plates, drops the dish on a counter;
    agent 1 picks it up from the counter and delivers."""
    actions_agent_0 = (
            ONION_CYCLE * 3
            + [A['S']] * 5  # wait for cooking
            + [
                A['D'], A['L'], A['D'], A['I'],  # take plate
                A['U'], A['R'], A['U'], A['I'],  # scoop soup
                A['D'], A['I'],  # drop dish on counter
                A['L'],  # step out of agent 1's way
            ]
            + [A['S']] * 20
    )
    wait_time = len(ONION_CYCLE) * 3 + 5 + 10
    actions_agent_1 = (
            [A['S']] * wait_time
            + [A['L'], A['D'], A['I'], A['R'], A['D'], A['I']]  # pick up dish, deliver
            + [A['S']] * 20
    )
    results = run_scenario(actions_agent_0, actions_agent_1)
    assert_reward_settings_consistent(results)
