#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from baselines.IPPO_CL import Config
from jax_marl.environments.env_selection import generate_sequence

def test_ippo_cl_with_restrictions():
    """Test IPPO_CL with complementary restrictions enabled"""

    print("Testing IPPO_CL with complementary restrictions...")

    # Create config with complementary restrictions enabled
    config = Config()
    config.complementary_restrictions = True
    config.seq_length = 3
    config.strategy = "generate"
    config.seed = 42

    print(f"Complementary restrictions enabled: {config.complementary_restrictions}")

    # Test generate_sequence with restrictions
    env_kwargs, layout_names = generate_sequence(
        sequence_length=config.seq_length,
        strategy=config.strategy,
        layout_names=config.layouts,
        seed=config.seed,
        height_rng=(config.height_min, config.height_max),
        width_rng=(config.width_min, config.width_max),
        wall_density=config.wall_density,
        layout_file=config.layout_file,
        complementary_restrictions=config.complementary_restrictions,
    )

    print(f"Generated {len(env_kwargs)} environments")

    # Check that each environment has agent restrictions
    for i, kwargs in enumerate(env_kwargs):
        print(f"\nEnvironment {i}:")
        print(f"  Layout: {layout_names[i]}")

        if "agent_restrictions" in kwargs:
            restrictions = kwargs["agent_restrictions"]
            print(f"  Agent restrictions: {restrictions}")

            # Verify complementary restrictions
            agent_0_onions = restrictions.get("agent_0_cannot_pick_onions", False)
            agent_0_plates = restrictions.get("agent_0_cannot_pick_plates", False)
            agent_1_onions = restrictions.get("agent_1_cannot_pick_onions", False)
            agent_1_plates = restrictions.get("agent_1_cannot_pick_plates", False)

            # Check that restrictions are complementary
            assert agent_0_onions != agent_0_plates, "Agent 0 should have exactly one restriction"
            assert agent_1_onions != agent_1_plates, "Agent 1 should have exactly one restriction"
            assert agent_0_onions != agent_1_onions, "Agents should have different onion restrictions"
            assert agent_0_plates != agent_1_plates, "Agents should have different plate restrictions"

            print("  ✓ Restrictions are complementary")
        else:
            raise AssertionError(f"Environment {i} missing agent_restrictions")

    print("\n✓ All environments have proper complementary restrictions")
    print("✓ IPPO_CL integration test passed!")

if __name__ == "__main__":
    test_ippo_cl_with_restrictions()
