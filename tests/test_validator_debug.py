#!/usr/bin/env python
"""Debug script for the Overcooked layout validator.

This script tests various layouts to identify issues with the validator's logic,
particularly focusing on false positives (accepting invalid layouts) and
false negatives (rejecting valid layouts).

It compares the original validator with our fixed version to demonstrate the improvements.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import original validator
from meal.env.generation.layout_validator import (
    evaluate_grid as original_evaluate
)

# Import fixed validator
import importlib.util

spec = importlib.util.spec_from_file_location(
    "env_validator_fixed",
    os.path.join(Path(__file__).parent.parent, "environments/overcooked/env_validator_fixed.py")
)
validator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator_module)
fixed_evaluate = validator_module.evaluate_grid


def test_valid_layouts():
    """Test layouts that should be considered valid."""
    valid_layouts = [
        # Basic valid layout
        ("""WWWWWWWW
WO  A BW
W      W
W P    W
W      W
W  A  XW
WWWWWWWW""", "Basic valid layout"),

        # Valid layout with agent cooperation required
        ("""WWWWWWWW
WO A  BW
WWWWWWWW
WP  A XW
WWWWWWWW""", "Agents separated - cooperation through wall required"),

        # Valid layout with multiple paths
        ("""WWWWWWWW
WO A   W
W WWW  W
W   P BW
W WWW  W
W  A  XW
WWWWWWWW""", "Multiple paths - should be valid"),

        # Valid layout with multiple pots (only one usable)
        ("""WWWWWWWW
WO A   W
W      W
WP WWPBW
W      W
W  A  XW
WWWWWWWW""", "Multiple pots (one unreachable) - should be valid"),

        # Valid layout with agents needing to cooperate through a counter
        ("""WWWWWWWW
WO A   W
W WWW  W
W W P BW
W WWW  W
W  A  XW
WWWWWWWW""", "Agents need to cooperate via counters - should be valid"),
    ]

    print("Testing layouts that should be valid:")
    print("=" * 80)
    for grid_str, description in valid_layouts:
        orig_valid, orig_reason = original_evaluate(grid_str)
        fixed_valid, fixed_reason = fixed_evaluate(grid_str)

        orig_result = "✓ VALID" if orig_valid else "✗ INVALID"
        fixed_result = "✓ VALID" if fixed_valid else "✗ INVALID"

        print(f"Layout: {description}")
        print(f"  Original: {orig_result}")
        if not orig_valid:
            print(f"    Reason: {orig_reason}")

        print(f"  Fixed:    {fixed_result}")
        if not fixed_valid:
            print(f"    Reason: {fixed_reason}")

        if orig_valid != fixed_valid:
            print("  ⚠️ VALIDATORS DISAGREE")

        print("  Layout:")
        print('\n'.join(f"    {line}" for line in grid_str.strip().split('\n')))
        print("-" * 80)


def test_invalid_layouts():
    """Test layouts that should be considered invalid."""
    invalid_layouts = [
        # No path from onion to pot
        ("""WWWWWWWW
WO A   W
WWWWWWWW
WWWWWWWW
W  A  PW
W     BW
W     XW
WWWWWWWW""", "No path from onion to pot - should be invalid"),

        # No path from pot to delivery
        ("""WWWWWWWW
WO A  BW
W      W
W  P   W
WWWWWWWW
WWWWWWWW
W  A  XW
WWWWWWWW""", "No path from pot to delivery - should be invalid"),

        # One agent completely isolated and useless
        ("""WWWWWWWW
WO    AW
WWWWWWWW
WWWWWWWW
W P   BW
W      W
W  A  XW
WWWWWWWW""", "One agent isolated - should be invalid"),

        # All onions unreachable
        ("""WWWWWWWW
WWWWWWWW
WWOWWWWW
W      W
W  P  BW
W      W
W  A  XW
WWWWWWWW""", "All onions unreachable - should be invalid"),

        # All pots unreachable
        ("""WWWWWWWW
WO A  BW
W      W
WWPWWWWW
W      W
W  A  XW
WWWWWWWW""", "All pots unreachable - should be invalid"),
    ]

    print("\nTesting layouts that should be invalid:")
    print("=" * 80)
    for grid_str, description in invalid_layouts:
        orig_valid, orig_reason = original_evaluate(grid_str)
        fixed_valid, fixed_reason = fixed_evaluate(grid_str)

        orig_result = "✓ INVALID" if not orig_valid else "✗ VALID"
        fixed_result = "✓ INVALID" if not fixed_valid else "✗ VALID"

        print(f"Layout: {description}")
        print(f"  Original: {orig_result}")
        if orig_valid:
            print("    ERROR: Layout incorrectly marked as valid")
        else:
            print(f"    Reason: {orig_reason}")

        print(f"  Fixed:    {fixed_result}")
        if fixed_valid:
            print("    ERROR: Layout incorrectly marked as valid")
        else:
            print(f"    Reason: {fixed_reason}")

        if orig_valid != fixed_valid:
            print("  ⚠️ VALIDATORS DISAGREE")

        print("  Layout:")
        print('\n'.join(f"    {line}" for line in grid_str.strip().split('\n')))
        print("-" * 80)


def test_edge_cases():
    """Test edge cases that might cause issues with the validator."""
    edge_cases = [
        # Complex cooperation required (agent needs to move so other can pass)
        ("""WWWWWWWW
WO A   W
W A    W
W  P  BW
W      W
W     XW
WWWWWWWW""", "Complex cooperation (need to move) - should be valid", True),

        # Handoff via counter required
        ("""WWWWWWWW
WO A  BW
WWWWWWWW
WP     W
W      W
W  A  XW
WWWWWWWW""", "Handoff required via counter - should be valid", True),

        # Multiple onions but only one reachable
        ("""WWWWWWWW
WO A  OW
W  WWWWW
W  P  BW
W      W
W  A  XW
WWWWWWWW""", "Multiple onions (one unreachable) - should be valid", True),

        # Agent next to interactive tile but no clear path
        ("""WWWWWWWW
WOWA   W
W      W
W  P  BW
W      W
W  A  XW
WWWWWWWW""", "Agent next to onion but no clear path - should be valid", True),

        # Agents can both reach all tiles but through different paths
        ("""WWWWWWWW
WO A  BW
W WW   W
W  P  WW
W WW   W
W  A  XW
WWWWWWWW""", "Different paths for agents - should be valid", True),
    ]

    print("\nTesting edge cases:")
    print("=" * 80)
    for grid_str, description, expected_valid in edge_cases:
        orig_valid, orig_reason = original_evaluate(grid_str)
        fixed_valid, fixed_reason = fixed_evaluate(grid_str)

        orig_result = "✓" if orig_valid == expected_valid else "✗"
        fixed_result = "✓" if fixed_valid == expected_valid else "✗"

        expected_str = "VALID" if expected_valid else "INVALID"
        orig_str = "VALID" if orig_valid else "INVALID"
        fixed_str = "VALID" if fixed_valid else "INVALID"

        print(f"Layout: {description}")
        print(f"  Expected: {expected_str}")
        print(f"  Original: {orig_str} {orig_result}")
        if orig_valid != expected_valid:
            print(f"    Reason: {orig_reason}")

        print(f"  Fixed:    {fixed_str} {fixed_result}")
        if fixed_valid != expected_valid:
            print(f"    Reason: {fixed_reason}")

        if orig_valid != fixed_valid:
            print("  ⚠️ VALIDATORS DISAGREE")

        print("  Layout:")
        print('\n'.join(f"    {line}" for line in grid_str.strip().split('\n')))
        print("-" * 80)


def main():
    """Run all tests and report results."""
    test_valid_layouts()
    test_invalid_layouts()
    test_edge_cases()

    # Print summary
    print("\nSummary of Validator Issues:")
    print("=" * 80)
    print("1. Reachability Calculation Problems:")
    print("   - Path Detection: The DFS algorithm didn't properly identify when agents")
    print("     could interact with tiles adjacent to their path")
    print("   - Adjacency Logic: Failed to properly check if agents were directly")
    print("     adjacent to interactive tiles")
    print("\n2. Agent Isolation & Utility Issues:")
    print("   - Agent Usefulness: Incomplete logic to determine if an agent could contribute")
    print("   - Missing Direct Isolation Check: No specific check for trapped agents")
    print("\n3. Cooperative Path Validation:")
    print("   - Handoff Detection: Struggled with detecting valid handoff possibilities")
    print("   - Full Path Validation: Insufficient checks for cooperative scenarios")
    print("\n4. Edge Case Handling:")
    print("   - Multiple Interactive Tiles: Didn't properly identify if at least one")
    print("     complete path was possible")
    print("   - Complex Movement Patterns: Couldn't properly evaluate layouts requiring")
    print("     strategic movement")


if __name__ == "__main__":
    main()
