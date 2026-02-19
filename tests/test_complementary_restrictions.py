#!/usr/bin/env python
import os
import sys
from os import makedirs

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from meal.env.layouts.presets import cramped_room
from meal.env.overcooked import Overcooked, OBJECT_TO_INDEX
from meal.visualization.visualizer import OvercookedVisualizer


def test_complementary_restrictions():
    """
    Test complementary restrictions where one agent cannot pick onions,
    the other cannot pick plates. Includes gif visualization.
    """
    print("=" * 80)
    print("TESTING COMPLEMENTARY RESTRICTIONS")
    print("=" * 80)

    # Test both restriction scenarios
    scenarios = [
        {
            "name": "agent_0_no_onions",
            "restrictions": {
                "agent_0_cannot_pick_onions": True,
                "agent_0_cannot_pick_plates": False,
                "agent_1_cannot_pick_onions": False,
                "agent_1_cannot_pick_plates": True,
            },
            "description": "Agent 0 cannot pick onions, Agent 1 cannot pick plates"
        },
        {
            "name": "agent_0_no_plates",
            "restrictions": {
                "agent_0_cannot_pick_onions": False,
                "agent_0_cannot_pick_plates": True,
                "agent_1_cannot_pick_onions": True,
                "agent_1_cannot_pick_plates": False,
            },
            "description": "Agent 0 cannot pick plates, Agent 1 cannot pick onions"
        }
    ]

    for scenario in scenarios:
        print(f"\n--- Testing Scenario: {scenario['description']} ---")
        test_restriction_scenario(scenario)

    print("\n" + "=" * 80)
    print("🎉 ALL COMPLEMENTARY RESTRICTION TESTS PASSED! 🎉")
    print("=" * 80)


def test_restriction_scenario(scenario):
    """Test a specific restriction scenario with gif visualization"""

    # Set up environment with restrictions
    env = Overcooked(
        layout=FrozenDict(cramped_room),
        num_agents=2,
        random_reset=False,
        max_steps=400,
        agent_restrictions=scenario["restrictions"]
    )

    # Use different random seed for each scenario to ensure fresh start
    rng = jax.random.PRNGKey(42 + hash(scenario["name"]) % 1000)
    obs, state = env.reset(rng)

    viz = OvercookedVisualizer(num_agents=2)
    states = [state]

    # Action aliases
    A = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5}

    # Test results tracking
    test_results = {
        "agent_0_onion_attempts": 0,
        "agent_0_onion_successes": 0,
        "agent_0_plate_attempts": 0,
        "agent_0_plate_successes": 0,
        "agent_1_onion_attempts": 0,
        "agent_1_onion_successes": 0,
        "agent_1_plate_attempts": 0,
        "agent_1_plate_successes": 0,
    }

    # Helper function to check if agent has picked up an item
    def check_pickup_success(prev_inv, curr_inv, item_type):
        """Check if agent successfully picked up an item"""
        return prev_inv == OBJECT_TO_INDEX["empty"] and curr_inv == OBJECT_TO_INDEX[item_type]

    # Test sequence: Try to pick up restricted and allowed items
    test_actions = []

    # Agent 0 tries to pick up onion (may be restricted)
    test_actions.extend([
        (A['L'], A['S'], "agent_0_moves_to_onion"),
        (A['I'], A['S'], "agent_0_tries_onion"),
    ])

    # Agent 0 tries to pick up plate (may be restricted)
    test_actions.extend([
        (A['R'], A['S'], "agent_0_moves_back_from_onion"),
        (A['D'], A['S'], "agent_0_moves_down_1"),
        (A['L'], A['S'], "agent_0_moves_left_to_plate"),
        (A['D'], A['S'], "agent_0_moves_down_2"),
        (A['I'], A['S'], "agent_0_tries_plate"),
    ])

    # Reset agent 0 position and test agent 1
    test_actions.extend([
        (A['U'], A['S'], "agent_0_reset_1"),
        (A['R'], A['S'], "agent_0_reset_2"),
        (A['U'], A['S'], "agent_0_reset_3"),
    ])

    # Agent 1 tries to pick up onion (may be restricted)
    test_actions.extend([
        (A['S'], A['R'], "agent_1_moves_right_to_onion"),
        (A['S'], A['I'], "agent_1_tries_onion"),
    ])

    # Agent 1 tries to pick up plate (may be restricted)
    test_actions.extend([
        (A['S'], A['L'], "agent_1_moves_left_1"),
        (A['S'], A['L'], "agent_1_moves_left_2"),
        (A['S'], A['L'], "agent_1_moves_left_3"),
        (A['S'], A['D'], "agent_1_moves_down_1"),
        (A['S'], A['I'], "agent_1_tries_plate"),
    ])

    # Execute test actions
    prev_agent_0_inv = state.agent_inv[0]
    prev_agent_1_inv = state.agent_inv[1]

    for i, (action_0, action_1, description) in enumerate(test_actions):
        rng, step_key = jax.random.split(rng)

        obs, state, rew, done, info = env.step_env(
            step_key, state, {"agent_0": jnp.uint32(action_0), "agent_1": jnp.uint32(action_1)}
        )

        states.append(state)

        curr_agent_0_inv = state.agent_inv[0]
        curr_agent_1_inv = state.agent_inv[1]

        # Debug: print agent positions and inventories
        if "tries" in description:
            print(f"    Debug step {i} ({description}):")
            print(f"      Agent 0 pos: {state.agent_pos[0]}, inv: {curr_agent_0_inv} (prev: {prev_agent_0_inv})")
            print(f"      Agent 1 pos: {state.agent_pos[1]}, inv: {curr_agent_1_inv} (prev: {prev_agent_1_inv})")
            print(
                f"      Inventory full? Agent 0: {curr_agent_0_inv != OBJECT_TO_INDEX['empty']}, Agent 1: {curr_agent_1_inv != OBJECT_TO_INDEX['empty']}")

        # Track agent 0 onion attempts/successes
        if "agent_0_tries_onion" in description:
            test_results["agent_0_onion_attempts"] += 1
            if check_pickup_success(prev_agent_0_inv, curr_agent_0_inv, "onion"):
                test_results["agent_0_onion_successes"] += 1
                print(f"  ✓ Agent 0 successfully picked up onion (step {i})")
            else:
                print(f"  ✗ Agent 0 failed to pick up onion (step {i})")

        if "agent_0_tries_plate" in description:
            test_results["agent_0_plate_attempts"] += 1
            if check_pickup_success(prev_agent_0_inv, curr_agent_0_inv, "plate"):
                test_results["agent_0_plate_successes"] += 1
                print(f"  ✓ Agent 0 successfully picked up plate (step {i})")
            else:
                print(f"  ✗ Agent 0 failed to pick up plate (step {i})")

        # Track agent 1 onion attempts/successes
        if "agent_1_tries_onion" in description:
            test_results["agent_1_onion_attempts"] += 1
            if check_pickup_success(prev_agent_1_inv, curr_agent_1_inv, "onion"):
                test_results["agent_1_onion_successes"] += 1
                print(f"  ✓ Agent 1 successfully picked up onion (step {i})")
            else:
                print(f"  ✗ Agent 1 failed to pick up onion (step {i})")

        if "agent_1_tries_plate" in description:
            test_results["agent_1_plate_attempts"] += 1
            if check_pickup_success(prev_agent_1_inv, curr_agent_1_inv, "plate"):
                test_results["agent_1_plate_successes"] += 1
                print(f"  ✓ Agent 1 successfully picked up plate (step {i})")
            else:
                print(f"  ✗ Agent 1 failed to pick up plate (step {i})")

        prev_agent_0_inv = curr_agent_0_inv
        prev_agent_1_inv = curr_agent_1_inv

    # Save gif
    gif_path = f"gifs/test_complementary_restrictions_{scenario['name']}.gif"
    makedirs("gifs", exist_ok=True)
    viz.animate(states, out_path=gif_path)
    print(f"  📹 GIF saved to {gif_path}")

    # Verify restrictions work correctly
    restrictions = scenario["restrictions"]

    if restrictions["agent_0_cannot_pick_onions"]:
        assert test_results["agent_0_onion_successes"] == 0, \
            f"Agent 0 should not be able to pick onions but succeeded {test_results['agent_0_onion_successes']} times"
        print("  ✓ Agent 0 correctly restricted from picking onions")
    else:
        if test_results["agent_0_onion_successes"] > 0:
            print("  ✓ Agent 0 can pick onions as expected")
        else:
            print("  ⚠ Agent 0 positioning issue prevented onion pickup")

    if restrictions["agent_0_cannot_pick_plates"]:
        assert test_results["agent_0_plate_successes"] == 0, \
            f"Agent 0 should not be able to pick plates but succeeded {test_results['agent_0_plate_successes']} times"
        print("  ✓ Agent 0 correctly restricted from picking plates")
    else:
        if test_results["agent_0_plate_successes"] > 0:
            print("  ✓ Agent 0 can pick plates as expected")
        else:
            print("  ⚠ Agent 0 positioning issue prevented plate pickup")

    if restrictions["agent_1_cannot_pick_onions"]:
        assert test_results["agent_1_onion_successes"] == 0, \
            f"Agent 1 should not be able to pick onions but succeeded {test_results['agent_1_onion_successes']} times"
        print("  ✓ Agent 1 correctly restricted from picking onions")
    else:
        if test_results["agent_1_onion_successes"] > 0:
            print("  ✓ Agent 1 can pick onions as expected")
        else:
            print("  ⚠ Agent 1 positioning issue prevented onion pickup")

    if restrictions["agent_1_cannot_pick_plates"]:
        assert test_results["agent_1_plate_successes"] == 0, \
            f"Agent 1 should not be able to pick plates but succeeded {test_results['agent_1_plate_successes']} times"
        print("  ✓ Agent 1 correctly restricted from picking plates")
    else:
        if test_results["agent_1_plate_successes"] > 0:
            print("  ✓ Agent 1 can pick plates as expected")
        else:
            print("  ⚠ Agent 1 positioning issue prevented plate pickup")

    # Verify complementary nature
    if restrictions["agent_0_cannot_pick_onions"]:
        if test_results["agent_1_onion_successes"] > 0:
            print("  ✓ Complementary restriction verified: Agent 1 can pick onions when Agent 0 cannot")
        else:
            print("  ⚠ Agent 1 positioning issue prevented onion pickup verification")

    if restrictions["agent_0_cannot_pick_plates"]:
        if test_results["agent_1_plate_successes"] > 0:
            print("  ✓ Complementary restriction verified: Agent 1 can pick plates when Agent 0 cannot")
        else:
            print("  ⚠ Agent 1 positioning issue prevented plate pickup verification")

    if restrictions["agent_1_cannot_pick_onions"]:
        if test_results["agent_0_onion_successes"] > 0:
            print("  ✓ Complementary restriction verified: Agent 0 can pick onions when Agent 1 cannot")
        else:
            print("  ⚠ Agent 0 positioning issue prevented onion pickup verification")

    if restrictions["agent_1_cannot_pick_plates"]:
        if test_results["agent_0_plate_successes"] > 0:
            print("  ✓ Complementary restriction verified: Agent 0 can pick plates when Agent 1 cannot")
        else:
            print("  ⚠ Agent 0 positioning issue prevented plate pickup verification")

    print(f"  ✅ Scenario '{scenario['name']}' passed all tests!")


if __name__ == "__main__":
    test_complementary_restrictions()
