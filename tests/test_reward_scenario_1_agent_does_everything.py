#!/usr/bin/env python
import os
import sys
from os import makedirs

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from flax.core import FrozenDict

from meal.env.layouts.presets import cramped_room
from meal.env.overcooked import Overcooked, DELIVERY_REWARD
from meal.visualization.visualizer import OvercookedVisualizer


def simulate_reward_processing(reward, shaped_reward, sparse_rewards=False, individual_rewards=False,
                               annealing_factor=1.0):
    """
    Simulate the reward processing logic from IPPO_CL.py (updated for new reward structure)
    """
    if sparse_rewards:
        # Sparse rewards: only shared delivery rewards (no shaped rewards)
        # Both agents get the total delivery reward from all agents
        total_delivery_reward = reward["agent_0"] + reward["agent_1"]
        return {"agent_0": total_delivery_reward, "agent_1": total_delivery_reward}
    elif individual_rewards:
        # Individual rewards: individual delivery rewards + individual shaped rewards
        # Each agent gets only their own delivery reward plus their shaped rewards
        return {"agent_0": reward["agent_0"] + shaped_reward["agent_0"] * annealing_factor,
                "agent_1": reward["agent_1"] + shaped_reward["agent_1"] * annealing_factor}
    else:
        # Default behavior: shared delivery rewards + individual shaped rewards
        # Convert individual delivery rewards to shared rewards (both agents get total)
        total_delivery_reward = reward["agent_0"] + reward["agent_1"]
        shared_delivery_rewards = {"agent_0": total_delivery_reward, "agent_1": total_delivery_reward}

        return {"agent_0": shared_delivery_rewards["agent_0"] + shaped_reward["agent_0"] * annealing_factor,
                "agent_1": shared_delivery_rewards["agent_1"] + shaped_reward["agent_1"] * annealing_factor}


def test_scenario_1_agent_0_does_everything():
    """Test scenario where agent 0 does everything with GIF recording"""

    print("=== SCENARIO 1: Agent 0 does everything ===")

    # Set up env (deterministic reset -> we know the spawn)
    # Use shorter soup cooking time for faster tests
    env = Overcooked(layout=FrozenDict(cramped_room), num_agents=2, random_reset=False, max_steps=400, soup_cook_time=5)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    # Set up GIF recording
    frames = []
    viz = OvercookedVisualizer(pot_full_status=5, pot_empty_status=8)

    def add_frame(st):
        # Use the visualizer's render method to get the frame
        surface = viz.render(env.agent_view_size, st)
        # Convert pygame surface to numpy array
        frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        frames.append(frame)

    # Add initial frame
    add_frame(state)

    # Action aliases
    A = {
        'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5,
    }

    # "pick onion ➜ pot" pattern (same as original test)
    onion_cycle = [A['L'], A['I'], A['R'], A['U'], A['I']]
    actions = onion_cycle * 3  # 3 onions
    actions += [A['S']] * 5  # wait for cooking (5→0 with shorter cook time)
    actions += [
        A['D'],  # step down
        A['L'],  # step down
        A['D'],  # step down, now facing plate-pile
        A['I'],  # take plate
        A['U'],  # back up
        A['R'],  # step right
        A['U'],  # turn up to pot, face pot
        A['I'],  # scoop soup (now holding dish)
        A['D'],  # step down toward goal line
        A['R'],  # step right
        A['D'],  # turn down, facing the serving window
        A['I'],  # deliver!
    ]

    actions_agent_2 = [A['S']] * 50

    # Run test scenario and collect results
    results = {}

    # Test all three reward settings
    for reward_setting in ['default', 'sparse', 'individual']:
        print(f"\n{reward_setting.capitalize()} rewards for Agent 0 does everything...")

        total_reward = {"agent_0": 0.0, "agent_1": 0.0}
        total_shaped = {"agent_0": 0.0, "agent_1": 0.0}
        total_soups = {"agent_0": 0.0, "agent_1": 0.0}

        # Reset environment
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        # Only record frames for the default setting to avoid redundancy
        if reward_setting == 'default':
            frames.clear()
            add_frame(state)

        for t in range(len(actions)):
            rng, step_key = jax.random.split(rng)
            obs, state, rew, done, info = env.step_env(
                step_key, state, {"agent_0": jnp.uint32(actions[t]), "agent_1": jnp.uint32(actions_agent_2[t])}
            )

            # Apply reward processing based on setting
            if reward_setting == 'sparse':
                processed_reward = simulate_reward_processing(rew, info["shaped_reward"],
                                                              sparse_rewards=True, individual_rewards=False)
            elif reward_setting == 'individual':
                processed_reward = simulate_reward_processing(rew, info["shaped_reward"],
                                                              sparse_rewards=False, individual_rewards=True)
            else:  # default
                processed_reward = simulate_reward_processing(rew, info["shaped_reward"],
                                                              sparse_rewards=False, individual_rewards=False)

            total_reward["agent_0"] += float(processed_reward["agent_0"])
            total_reward["agent_1"] += float(processed_reward["agent_1"])
            total_shaped["agent_0"] += float(info["shaped_reward"]["agent_0"])
            total_shaped["agent_1"] += float(info["shaped_reward"]["agent_1"])
            total_soups["agent_0"] += float(info["soups"]["agent_0"])
            total_soups["agent_1"] += float(info["soups"]["agent_1"])

            # Record frame for default setting
            if reward_setting == 'default':
                add_frame(state)

        results[reward_setting] = {
            'total_reward': total_reward,
            'total_shaped': total_shaped,
            'total_soups': total_soups
        }

        print(
            f"  Agent 0: {total_reward['agent_0']:.1f} total ({total_shaped['agent_0']:.1f} shaped, {total_soups['agent_0']:.0f} soups)")
        print(
            f"  Agent 1: {total_reward['agent_1']:.1f} total ({total_shaped['agent_1']:.1f} shaped, {total_soups['agent_1']:.0f} soups)")

    # Save GIF (slower fps for easier viewing)
    gif_path = "gifs/test_reward_scenario_1_agent_does_everything.gif"
    makedirs("gifs", exist_ok=True)
    iio.imwrite(gif_path, frames, loop=0, fps=6)
    print(f"\nGIF saved to {gif_path}")

    # Validate results with assertions
    print("\n=== VALIDATING SCENARIO 1 ===")

    default_results = results['default']
    sparse_results = results['sparse']
    individual_results = results['individual']

    # Extract values for easier reference
    def_r0, def_r1 = default_results['total_reward']['agent_0'], default_results['total_reward']['agent_1']
    def_s0, def_s1 = default_results['total_shaped']['agent_0'], default_results['total_shaped']['agent_1']
    def_soups0, def_soups1 = default_results['total_soups']['agent_0'], default_results['total_soups']['agent_1']

    sparse_r0, sparse_r1 = sparse_results['total_reward']['agent_0'], sparse_results['total_reward']['agent_1']

    ind_r0, ind_r1 = individual_results['total_reward']['agent_0'], individual_results['total_reward']['agent_1']

    total_soups_delivered = def_soups0 + def_soups1
    total_delivery_reward = total_soups_delivered * DELIVERY_REWARD

    print(f"  Total soups delivered: {total_soups_delivered}")
    print(f"  Agent 0 shaped rewards: {def_s0}")
    print(f"  Agent 1 shaped rewards: {def_s1}")
    print(f"  Agent 0 soups delivered: {def_soups0}")
    print(f"  Agent 1 soups delivered: {def_soups1}")

    # ASSERTION 1: Default rewards = shared delivery reward + individual shaped rewards
    expected_default_r0 = total_delivery_reward + def_s0
    expected_default_r1 = total_delivery_reward + def_s1

    assert np.isclose(def_r0, expected_default_r0, atol=1e-6), \
        f"Default Agent 0: expected {expected_default_r0}, got {def_r0}"
    assert np.isclose(def_r1, expected_default_r1, atol=1e-6), \
        f"Default Agent 1: expected {expected_default_r1}, got {def_r1}"
    print(f"  ✓ Default rewards correct: A0={def_r0:.1f}, A1={def_r1:.1f}")

    # ASSERTION 2: Sparse rewards = only shared delivery reward (no shaped rewards)
    expected_sparse_r0 = total_delivery_reward
    expected_sparse_r1 = total_delivery_reward

    assert np.isclose(sparse_r0, expected_sparse_r0, atol=1e-6), \
        f"Sparse Agent 0: expected {expected_sparse_r0}, got {sparse_r0}"
    assert np.isclose(sparse_r1, expected_sparse_r1, atol=1e-6), \
        f"Sparse Agent 1: expected {expected_sparse_r1}, got {sparse_r1}"
    print(f"  ✓ Sparse rewards correct: A0={sparse_r0:.1f}, A1={sparse_r1:.1f}")

    # ASSERTION 3: Individual rewards = individual delivery reward + individual shaped rewards
    expected_individual_r0 = def_soups0 * DELIVERY_REWARD + def_s0
    expected_individual_r1 = def_soups1 * DELIVERY_REWARD + def_s1

    assert np.isclose(ind_r0, expected_individual_r0, atol=1e-6), \
        f"Individual Agent 0: expected {expected_individual_r0}, got {ind_r0}"
    assert np.isclose(ind_r1, expected_individual_r1, atol=1e-6), \
        f"Individual Agent 1: expected {expected_individual_r1}, got {ind_r1}"
    print(f"  ✓ Individual rewards correct: A0={ind_r0:.1f}, A1={ind_r1:.1f}")

    print("\n🎉 SCENARIO 1 TEST PASSED! 🎉")
    print("✅ Agent 0 does everything: All reward settings work correctly")

    return results


if __name__ == "__main__":
    test_scenario_1_agent_0_does_everything()
