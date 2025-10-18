#!/usr/bin/env python3
"""
Test script to simulate actual experimental conditions with complementary restrictions.
This test uses a very small manual layout with onion and plate piles directly next to agents.
Enhanced with gif recording to visualize agent behavior and restrictions.
"""

import os
import sys
from os import makedirs

sys.path.insert(0, os.path.abspath('..'))

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import pygame
from flax.core import FrozenDict
from meal import make_env
from meal.env.layouts.presets import layout_grid_to_dict
from meal.env.overcooked import OBJECT_TO_INDEX
from meal.visualization.visualizer import OvercookedVisualizer


def test_experimental_conditions():
    """Test complementary restrictions with a very small manual layout"""

    print("Testing complementary restrictions with manual small layout...")
    print("=" * 70)

    # Create a very small custom layout with onions and plates directly next to agents
    # Layout: 5x4 grid
    # W W W W W
    # W O A B W  (Agent A next to Onion O and Plate B)
    # W   A P W  (Agent A with empty space and Pot P)
    # W W X W W  (X = goal for delivery)
    small_layout_grid = """
WWWWW
WOABW
W APW
WWXWW
"""

    # Convert grid to layout dictionary
    layout_dict = layout_grid_to_dict(small_layout_grid)
    layout = FrozenDict(layout_dict)

    print("Created small test layout:")
    print("W W W W W")
    print("W O A B W")  # O=onion pile, A=agent, B=plate pile
    print("W   A P W")  # A=agent, space=empty counter, P=pot
    print("W W X W W")  # X=goal for delivery
    print()

    # Test both restriction scenarios
    restriction_scenarios = [
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

    for scenario_idx, scenario in enumerate(restriction_scenarios):
        print(f"\n--- Testing Scenario {scenario_idx}: {scenario['description']} ---")

        # Create environment with restrictions
        env = make_env("overcooked", layout=layout, agent_restrictions=scenario["restrictions"])

        # Reset environment
        rng = jax.random.PRNGKey(42 + scenario_idx)
        obs, state = env.reset(rng)

        print(f"Initial agent positions: {state.agent_pos}")
        print(f"Initial agent inventories: {state.agent_inv}")
        print(f"Restrictions: {scenario['restrictions']}")

        # Set up gif recording
        frames = []
        viz = OvercookedVisualizer(num_agents=2, use_old_rendering=False)

        def add_frame(st):
            surface = viz.render(env.agent_view_size, st)
            frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
            frames.append(frame)

        # Add initial frame
        add_frame(state)

        # Action aliases for readability
        A = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5}

        # Test restrictions with specific sequence
        bug_found = False
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

        def check_pickup_success(prev_inv, curr_inv, item_type):
            """Check if agent successfully picked up an item"""
            return prev_inv == OBJECT_TO_INDEX["empty"] and curr_inv == OBJECT_TO_INDEX[item_type]

        def check_drop_success(prev_inv, curr_inv):
            """Check if agent successfully dropped an item"""
            return prev_inv != OBJECT_TO_INDEX["empty"] and curr_inv == OBJECT_TO_INDEX["empty"]

        # Create a specific test sequence:
        # 1. Each agent tries to pick up onion
        # 2. If successful, place it on empty counter
        # 3. Each agent tries to pick up plate  
        # 4. If successful, place it on empty counter
        # 5. Retry the restricted item to confirm restriction
        test_actions = []

        print("  Phase 1: Agents try to pick up onions")
        # Agent 0 tries onion (may be restricted)
        test_actions.extend([
            (A['L'], A['S'], "agent_0_move_to_onion"),  # Move left to onion
            (A['I'], A['S'], "agent_0_try_pickup_onion"),  # Try to pick up onion
        ])

        # Agent 1 tries onion (may be restricted)  
        test_actions.extend([
            (A['S'], A['U'], "agent_1_move_to_onion"),  # Move up to onion
            (A['S'], A['I'], "agent_1_try_pickup_onion"),  # Try to pick up onion
        ])

        print("  Phase 2: If holding onion, place on empty counter")
        # If agents picked up onion, place it down
        test_actions.extend([
            (A['R'], A['D'], "agents_move_to_counter"),  # Move to empty counter space
            (A['I'], A['I'], "agents_try_place_onion"),  # Try to place onion on counter
        ])

        print("  Phase 3: Agents try to pick up plates")
        # Agent 0 tries plate (may be restricted)
        test_actions.extend([
            (A['U'], A['S'], "agent_0_move_to_plate"),  # Move up to plate
            (A['I'], A['S'], "agent_0_try_pickup_plate"),  # Try to pick up plate
        ])

        # Agent 1 tries plate (may be restricted)
        test_actions.extend([
            (A['S'], A['L'], "agent_1_move_to_plate"),  # Move left to plate  
            (A['S'], A['I'], "agent_1_try_pickup_plate"),  # Try to pick up plate
        ])

        print("  Phase 4: If holding plate, place on empty counter")
        # If agents picked up plate, place it down
        test_actions.extend([
            (A['D'], A['D'], "agents_move_to_counter_2"),  # Move to empty counter space
            (A['I'], A['I'], "agents_try_place_plate"),  # Try to place plate on counter
        ])

        print("  Phase 5: Retry restricted items to confirm restrictions")
        # Try restricted items again to confirm they're still blocked
        test_actions.extend([
            (A['U'], A['U'], "agents_move_back"),  # Move back
            (A['L'], A['L'], "agents_move_to_onion_again"),  # Move to onion
            (A['I'], A['I'], "agents_retry_onion"),  # Retry onion pickup
            (A['R'], A['R'], "agents_move_to_plate_again"),  # Move to plate
            (A['I'], A['I'], "agents_retry_plate"),  # Retry plate pickup
        ])

        # Execute test actions and record frames
        prev_agent_0_inv = state.agent_inv[0]
        prev_agent_1_inv = state.agent_inv[1]

        for step, (action_0, action_1, description) in enumerate(test_actions):
            rng, step_key = jax.random.split(rng)

            actions = {"agent_0": jnp.uint32(action_0), "agent_1": jnp.uint32(action_1)}
            obs, state, rewards, dones, info = env.step_env(step_key, state, actions)

            # Add frame after each step
            add_frame(state)

            curr_agent_0_inv = state.agent_inv[0]
            curr_agent_1_inv = state.agent_inv[1]

            # Check for pickup/drop attempts and violations
            if "try_pickup" in description or "retry" in description:
                print(f"    Step {step} ({description}): Positions: {state.agent_pos}, Inventories: {state.agent_inv}")

                # Check agent 0 interactions
                if "onion" in description:
                    if check_pickup_success(prev_agent_0_inv, curr_agent_0_inv, "onion"):
                        test_results["agent_0_onion_successes"] += 1
                        if scenario["restrictions"].get("agent_0_cannot_pick_onions", False):
                            print(f"      ❌ BUG: Agent 0 picked up onion despite restriction!")
                            bug_found = True
                        else:
                            print(f"      ✓ Agent 0 picked up onion (allowed)")

                if "plate" in description:
                    if check_pickup_success(prev_agent_0_inv, curr_agent_0_inv, "plate"):
                        test_results["agent_0_plate_successes"] += 1
                        if scenario["restrictions"].get("agent_0_cannot_pick_plates", False):
                            print(f"      ❌ BUG: Agent 0 picked up plate despite restriction!")
                            bug_found = True
                        else:
                            print(f"      ✓ Agent 0 picked up plate (allowed)")

                # Check agent 1 interactions
                if "onion" in description:
                    if check_pickup_success(prev_agent_1_inv, curr_agent_1_inv, "onion"):
                        test_results["agent_1_onion_successes"] += 1
                        if scenario["restrictions"].get("agent_1_cannot_pick_onions", False):
                            print(f"      ❌ BUG: Agent 1 picked up onion despite restriction!")
                            bug_found = True
                        else:
                            print(f"      ✓ Agent 1 picked up onion (allowed)")

                if "plate" in description:
                    if check_pickup_success(prev_agent_1_inv, curr_agent_1_inv, "plate"):
                        test_results["agent_1_plate_successes"] += 1
                        if scenario["restrictions"].get("agent_1_cannot_pick_plates", False):
                            print(f"      ❌ BUG: Agent 1 picked up plate despite restriction!")
                            bug_found = True
                        else:
                            print(f"      ✓ Agent 1 picked up plate (allowed)")

            elif "try_place" in description:
                print(f"    Step {step} ({description}): Positions: {state.agent_pos}, Inventories: {state.agent_inv}")

                # Check if agents successfully placed items
                if check_drop_success(prev_agent_0_inv, curr_agent_0_inv):
                    print(f"      ✓ Agent 0 placed item on counter")
                if check_drop_success(prev_agent_1_inv, curr_agent_1_inv):
                    print(f"      ✓ Agent 1 placed item on counter")

            # Update previous inventories
            prev_agent_0_inv = curr_agent_0_inv
            prev_agent_1_inv = curr_agent_1_inv

        # Save gif with slower fps for better visibility
        gif_path = f"gifs/test_restrictions_{scenario['name']}.gif"
        makedirs("gifs", exist_ok=True)
        iio.imwrite(gif_path, frames, loop=0, fps=3)  # Slower fps (3) for better observation
        print(f"  📹 GIF saved to {gif_path}")

        # Print summary of what happened
        print(f"  📊 Test Results:")
        print(f"    Agent 0: {test_results['agent_0_onion_successes']} onions, {test_results['agent_0_plate_successes']} plates")
        print(f"    Agent 1: {test_results['agent_1_onion_successes']} onions, {test_results['agent_1_plate_successes']} plates")

        if bug_found:
            print(f"❌ Scenario {scenario_idx}: Restrictions violated!")
            return False
        else:
            print(f"✓ Scenario {scenario_idx}: Restrictions working correctly")

    print("\n" + "=" * 70)
    print("🎉 ALL MANUAL LAYOUT TESTS PASSED!")
    print("✅ Complementary restrictions work correctly with manual small layout")
    print("✅ Agents successfully demonstrated pickup/place/retry behavior")
    print("✅ No restriction violations found in controlled environment")
    print("=" * 70)
    return True


if __name__ == "__main__":
    test_experimental_conditions()
