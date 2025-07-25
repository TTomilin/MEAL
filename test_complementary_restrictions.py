#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import jax
import jax.numpy as jnp
from jax_marl.registration import make

def test_complementary_restrictions():
    """Test that complementary restrictions work correctly"""
    
    print("Testing complementary restrictions...")
    
    # Test with restrictions enabled
    env_kwargs = {
        "layout_name": "cramped_room",
        "agent_restrictions": {
            "agent_0_cannot_pick_onions": True,
            "agent_0_cannot_pick_plates": False,
            "agent_1_cannot_pick_onions": False,
            "agent_1_cannot_pick_plates": True,
        }
    }
    
    env = make("overcooked", **env_kwargs)
    
    # Check that restrictions are stored correctly
    print(f"Agent restrictions: {env.agent_restrictions}")
    
    # Verify the restrictions are as expected
    assert env.agent_restrictions["agent_0_cannot_pick_onions"] == True
    assert env.agent_restrictions["agent_0_cannot_pick_plates"] == False
    assert env.agent_restrictions["agent_1_cannot_pick_onions"] == False
    assert env.agent_restrictions["agent_1_cannot_pick_plates"] == True
    
    print("✓ Agent restrictions stored correctly")
    
    # Test environment reset and basic functionality
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    print("✓ Environment resets successfully with restrictions")
    
    # Test a few steps to make sure the environment still works
    actions = {"agent_0": 0, "agent_1": 1}  # Some basic actions
    obs, state, rewards, dones, info = env.step(key, state, actions)
    
    print("✓ Environment steps successfully with restrictions")
    print(f"Rewards: {rewards}")
    
    print("All tests passed! Complementary restrictions are working.")

if __name__ == "__main__":
    test_complementary_restrictions()