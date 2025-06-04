#!/usr/bin/env python
"""Test script for the fixed Overcooked layout validator.

This script compares the results of the original validator and the fixed validator
on a series of test layouts to verify that the fixes work correctly.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Monkey patch the original validator with our fixed one for testing
import importlib.util
spec = importlib.util.spec_from_file_location(
    "env_validator_fixed", 
    os.path.join(Path(__file__).parent.parent, "environments/overcooked_environment/env_validator_fixed.py")
)
validator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator_module)

from jax_marl.environments.overcooked_environment.env_validator import evaluate_grid as original_evaluate
from jax_marl.environments.overcooked_environment.env_validator import WALL, FLOOR, AGENT, GOAL, ONION_PILE, PLATE_PILE, POT

# Use the fixed evaluate function from our monkey-patched module
fixed_evaluate = validator_module.evaluate_grid

def test_validators():
    """Compare original and fixed validators on a series of test layouts."""
    test_layouts = [
        # Valid layouts
        ("""WWWWWWWW
WO  A BW
W      W
W P    W
W      W
W  A  XW
WWWWWWWW""", "Basic valid layout", True),
        
        # Valid layout with agent cooperation required
        ("""WWWWWWWW
WO A  BW
WWWWWWWW
WP  A XW
WWWWWWWW""", "Agents separated - cooperation through wall required", True),
        
        # Invalid layouts
        ("""WWWWWWWW
WO A   W
WWWWWWWW
WWWWWWWW
W  A  PW
W     BW
W     XW
WWWWWWWW""", "No path from onion to pot - should be invalid", False),
        
        ("""WWWWWWWW
WO A  BW
W      W
W  P   W
WWWWWWWW
WWWWWWWW
W  A  XW
WWWWWWWW""", "No path from pot to delivery - should be invalid", False),
        
        ("""WWWWWWWW
WO    AW
WWWWWWWW
WWWWWWWW
W P   BW
W      W
W  A  XW
WWWWWWWW""", "One agent isolated - should be invalid", False),
        
        ("""WWWWWWWW
WWWWWWWW
WWOWWWWW
W      W
W  P  BW
W      W
W  A  XW
WWWWWWWW""", "All onions unreachable - should be invalid", False),
        
        # Edge cases
        ("""WWWWWWWW
WO A   W
W A    W
W  P  BW
W      W
W     XW
WWWWWWWW""", "Complex cooperation (need to move) - should be valid", True),
        
        ("""WWWWWWWW
WO A  BW
WWWWWWWW
WP     W
W      W
W  A  XW
WWWWWWWW""", "Handoff required via counter - should be valid", True),
    ]
    
    print("Comparing original and fixed validators:")
    print("=" * 80)
    
    failed_original = 0
    failed_fixed = 0
    
    for grid_str, description, expected_valid in test_layouts:
        # Test with original validator
        orig_valid, orig_reason = original_evaluate(grid_str)
        orig_result = "VALID" if orig_valid else "INVALID"
        orig_correct = orig_valid == expected_valid
        
        # Test with fixed validator
        fixed_valid, fixed_reason = fixed_evaluate(grid_str)
        fixed_result = "VALID" if fixed_valid else "INVALID"
        fixed_correct = fixed_valid == expected_valid
        
        # Track failures
        if not orig_correct:
            failed_original += 1
        if not fixed_correct:
            failed_fixed += 1
        
        # Display results
        print(f"Test: {description}")
        print(f"  Expected: {'VALID' if expected_valid else 'INVALID'}")
        print(f"  Original: {orig_result} {'✓' if orig_correct else '✗'}")
        if not orig_correct:
            print(f"    Reason: {orig_reason}")
        print(f"  Fixed:    {fixed_result} {'✓' if fixed_correct else '✗'}")
        if not fixed_correct:
            print(f"    Reason: {fixed_reason}")
        print(f"  Layout:")
        print('\n'.join(f"    {line}" for line in grid_str.strip().split('\n')))
        print("-" * 80)
    
    # Summary
    total = len(test_layouts)
    print(f"\nSummary:")
    print(f"  Total test cases: {total}")
    print(f"  Original validator: {total - failed_original}/{total} correct ({failed_original} failures)")
    print(f"  Fixed validator:    {total - failed_fixed}/{total} correct ({failed_fixed} failures)")
    
    if failed_fixed == 0:
        print("\n✅ The fixed validator passed all test cases!")
    else:
        print("\n❌ The fixed validator still has issues to fix.")

if __name__ == "__main__":
    test_validators()