#!/usr/bin/env python
"""Comparison script for Overcooked layout validators **with visualisation**.

This version compares the results of the original (`env_validator.py`) and the
fixed (`env_validator_fixed.py`) validators on a collection of handmade test
layouts *and* pops up a simple graphic of each layout so you can eyeball what's
going on.

Close the plot window to advance to the next layout.
"""
# ==== standard libs ==========================================================
import os
import sys
from pathlib import Path
from typing import List, Tuple

from meal.env.layouts import layout_grid_to_dict

# ==== project imports ========================================================
# put project root on import path so the jax‑marl package can be resolved when
# running the script directly from the repo root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from meal.env.generation.layout_validator import (
    evaluate_grid as original_evaluate,
)

# Dynamically load the *fixed* validator that lives next to this file
import importlib.util

spec = importlib.util.spec_from_file_location(
    "env_validator_fixed",
    os.path.join(Path(__file__).parent.parent, "environments/overcooked/env_validator_fixed.py"),
)
validator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator_module)  # type: ignore[arg-type]
fixed_evaluate = validator_module.evaluate_grid

# ==== extra imports for visualisation ========================================
import jax
import jax.numpy as jnp

from flax.core import FrozenDict

from meal.env import Overcooked
from meal.visualization.visualizer import OvercookedVisualizer


# =============================================================================
# ───────────────────────────  VISUALISATION  ────────────────────────────────
# =============================================================================

def _add_dummy_tiles(layout_dict: FrozenDict) -> FrozenDict:
    """If a layout is missing an onion/pot/goal/plate pile, Overcooked fails.
    Drop a dummy one on a spare wall so the environment boots.
    """
    required = ["plate_pile_idx", "onion_pile_idx", "pot_idx", "goal_idx"]
    missing = [k for k in required if len(layout_dict[k]) == 0]
    if not missing:
        return layout_dict

    wall_idx = layout_dict["wall_idx"]
    used: set[int] = set()
    for key in [
        "agent_idx",
        "goal_idx",
        "onion_pile_idx",
        "pot_idx",
        "plate_pile_idx",
    ]:
        used.update(int(i) for i in layout_dict[key])

    spares = [int(i) for i in wall_idx if int(i) not in used]
    if len(spares) < len(missing):
        return layout_dict  # ¯\\_(ツ)_/¯ just give up and hope for the best

    as_dict = dict(layout_dict)
    for i, key in enumerate(missing):
        as_dict[key] = jnp.array([spares[i]])
    return FrozenDict(as_dict)


def visualise_layout(
        name: str,
        grid_str: str,
        expected_valid: bool,
        orig_valid: bool,
        fixed_valid: bool,
        orig_reason: str | None = None,
        fixed_reason: str | None = None,
) -> None:
    """Pop up an Overcooked grid with a one‑liner title summarising verdicts."""
    layout_dict = layout_grid_to_dict(grid_str)
    layout_dict = _add_dummy_tiles(layout_dict)

    env = Overcooked(layout=layout_dict, layout_name=name, random_reset=False)
    _, state = env.reset(jax.random.PRNGKey(0))

    vis = OvercookedVisualizer()

    title_parts = [
        f"Exp:{'✓' if expected_valid else '✗'}",
        f"Orig:{'✓' if orig_valid else '✗'}",
        f"Fix:{'✓' if fixed_valid else '✗'}",
    ]
    title = f"{name}  |  " + "  ·  ".join(title_parts)
    if expected_valid != orig_valid and orig_reason:
        title += f"\nOrig reason: {orig_reason}"
    if expected_valid != fixed_valid and fixed_reason:
        title += f"\nFix reason: {fixed_reason}"

    print(f"Showing: {title}\nClose the window to continue…")
    vis.render_grid(state, show=True)


# =============================================================================
# ──────────────────────────  VALIDATION STATS  ──────────────────────────────
# =============================================================================
class ValidationStats:
    def __init__(self) -> None:
        self.total_layouts = 0
        self.agreement_count = 0
        self.disagreement_count = 0

        # Track correct/incorrect validations
        self.orig_correct = 0
        self.orig_incorrect = 0
        self.fixed_correct = 0
        self.fixed_incorrect = 0

        # Track by expected outcome
        self.should_be_valid = 0
        self.should_be_invalid = 0

        # Track false positives/negatives
        self.orig_false_positives = 0  # Original marks as valid when should be invalid
        self.orig_false_negatives = 0  # Original marks as invalid when should be valid
        self.fixed_false_positives = 0  # Fixed marks as valid when should be invalid
        self.fixed_false_negatives = 0  # Fixed marks as invalid when should be valid

    # ------------------------------------------------------------------
    def add_result(self, expected_valid: bool, orig_valid: bool, fixed_valid: bool) -> None:
        self.total_layouts += 1
        if expected_valid:
            self.should_be_valid += 1
        else:
            self.should_be_invalid += 1

        if orig_valid == fixed_valid:
            self.agreement_count += 1
        else:
            self.disagreement_count += 1

        # original validator accuracy
        if orig_valid == expected_valid:
            self.orig_correct += 1
        else:
            self.orig_incorrect += 1
            if orig_valid and not expected_valid:
                self.orig_false_positives += 1
            elif not orig_valid and expected_valid:
                self.orig_false_negatives += 1

        # fixed validator accuracy
        if fixed_valid == expected_valid:
            self.fixed_correct += 1
        else:
            self.fixed_incorrect += 1
            if fixed_valid and not expected_valid:
                self.fixed_false_positives += 1
            elif not fixed_valid and expected_valid:
                self.fixed_false_negatives += 1

    # ------------------------------------------------------------------
    def print_summary(self) -> None:  # noqa: C901 – a *summary* is allowed to be long
        print("\nValidation Statistics:")
        print("=" * 80)
        print(f"Total layouts tested: {self.total_layouts}")
        print(f"  Should be valid: {self.should_be_valid}")
        print(f"  Should be invalid: {self.should_be_invalid}")
        print(
            f"\nAgreement between validators: {self.agreement_count}/{self.total_layouts} "
            f"({self.agreement_count / self.total_layouts * 100:.1f}%)"
        )
        print(
            f"Disagreement between validators: {self.disagreement_count}/{self.total_layouts} "
            f"({self.disagreement_count / self.total_layouts * 100:.1f}%)"
        )

        print("\nOriginal Validator Performance:")
        print(
            f"  Correct validations: {self.orig_correct}/{self.total_layouts} "
            f"({self.orig_correct / self.total_layouts * 100:.1f}%)"
        )
        print(
            f"  Incorrect validations: {self.orig_incorrect}/{self.total_layouts} "
            f"({self.orig_incorrect / self.total_layouts * 100:.1f}%)"
        )
        print(
            f"  False positives: {self.orig_false_positives}/{self.should_be_invalid} "
            f"({self.orig_false_positives / max(1, self.should_be_invalid) * 100:.1f}%)"
        )
        print(
            f"  False negatives: {self.orig_false_negatives}/{self.should_be_valid} "
            f"({self.orig_false_negatives / max(1, self.should_be_valid) * 100:.1f}%)"
        )

        print("\nFixed Validator Performance:")
        print(
            f"  Correct validations: {self.fixed_correct}/{self.total_layouts} "
            f"({self.fixed_correct / self.total_layouts * 100:.1f}%)"
        )
        print(
            f"  Incorrect validations: {self.fixed_incorrect}/{self.total_layouts} "
            f"({self.fixed_incorrect / self.total_layouts * 100:.1f}%)"
        )
        print(
            f"  False positives: {self.fixed_false_positives}/{self.should_be_invalid} "
            f"({self.fixed_false_positives / max(1, self.should_be_invalid) * 100:.1f}%)"
        )
        print(
            f"  False negatives: {self.fixed_false_negatives}/{self.should_be_valid} "
            f"({self.fixed_false_negatives / max(1, self.should_be_valid) * 100:.1f}%)"
        )

        orig_acc = self.orig_correct / self.total_layouts * 100
        fixed_acc = self.fixed_correct / self.total_layouts * 100
        print("\nImprovement from Original to Fixed:")
        print(f"  Overall accuracy improvement: {fixed_acc - orig_acc:.1f}%")


# =============================================================================
# ──────────────────────────  TEST HARNESS  ──────────────────────────────────
# =============================================================================

stats = ValidationStats()


def _print_and_plot(
        idx: int,
        grid_str: str,
        desc: str,
        expected: bool,
        orig_valid: bool,
        orig_reason: str,
        fixed_valid: bool,
        fixed_reason: str,
) -> None:
    """Inline helper: dump text then spawn a plot."""
    print(f"Layout {idx}: {desc}")
    print(f"  Expected: {'VALID' if expected else 'INVALID'}")
    print(f"  Original: {'VALID' if orig_valid else 'INVALID'}")
    if orig_valid != expected:
        print(f"    Reason: {orig_reason}")
    print(f"  Fixed:    {'VALID' if fixed_valid else 'INVALID'}")
    if fixed_valid != expected:
        print(f"    Reason: {fixed_reason}")
    if orig_valid != fixed_valid:
        print("  ⚠️  VALIDATORS DISAGREE")
    print("  Layout:")
    for line in grid_str.strip().split("\n"):
        print("    " + line)
    print("-" * 80)

    # Plot – keep this *after* stdout dump so logs stay readable.
    visualise_layout(
        name=f"L{idx}",
        grid_str=grid_str,
        expected_valid=expected,
        orig_valid=orig_valid,
        fixed_valid=fixed_valid,
        orig_reason=orig_reason if orig_valid != expected else None,
        fixed_reason=fixed_reason if fixed_valid != expected else None,
    )


# ---------------------------------------------------------------------------
# All the hard‑coded layout strings from the original script live unchanged
# below.  We just call `_print_and_plot` instead of dumping bespoke text.
# ---------------------------------------------------------------------------

def test_valid_layouts() -> None:
    """Layouts that *should* be valid."""
    valid_layouts: List[Tuple[str, str]] = [
        (
            """WWWWWWWW
WO  A BW
W      W
W P    W
W      W
W  A  XW
WWWWWWWW""",
            "Basic valid layout",
        ),
        (
            """WWWWWWWW
WO A  BW
WWWWWWWW
WP  A XW
WWWWWWWW""",
            "Agents separated – cooperation through wall required",
        ),
        (
            """WWWWWWWW
WO A   W
W WWW  W
W   P BW
W WWW  W
W  A  XW
WWWWWWWW""",
            "Multiple paths – should be valid",
        ),
        (
            """WWWWWWWW
WO A   W
W      W
WP WWPBW
W      W
W  A  XW
WWWWWWWW""",
            "Multiple pots (one unreachable) – should be valid",
        ),
        (
            """WWWWWWWW
WO A   W
W WWW  W
W W P BW
W WWW  W
W  A  XW
WWWWWWWW""",
            "Agents need to cooperate via counters – should be valid",
        ),
    ]

    print("Testing *valid* layouts\n" + "=" * 80)
    for i, (grid_str, desc) in enumerate(valid_layouts, start=1):
        orig_valid, orig_reason = original_evaluate(grid_str)
        fixed_valid, fixed_reason = fixed_evaluate(grid_str)
        stats.add_result(True, orig_valid, fixed_valid)
        _print_and_plot(
            idx=i,
            grid_str=grid_str,
            desc=desc,
            expected=True,
            orig_valid=orig_valid,
            orig_reason=orig_reason,
            fixed_valid=fixed_valid,
            fixed_reason=fixed_reason,
        )


def test_invalid_layouts(start_idx: int = 1) -> int:
    """Layouts that *should* be invalid."""
    invalid_layouts: List[Tuple[str, str]] = [
        (
            """WWWWWWWW
WO A   W
WWWWWWWW
WWWWWWWW
W  A  PW
W     BW
W     XW
WWWWWWWW""",
            "No path onion → pot – should be invalid",
        ),
        (
            """WWWWWWWW
WO A  BW
W      W
W  P   W
WWWWWWWW
WWWWWWWW
W  A  XW
WWWWWWWW""",
            "No path pot → delivery – should be invalid",
        ),
        (
            """WWWWWWWW
WO    AW
WWWWWWWW
WWWWWWWW
W P   BW
W      W
W  A  XW
WWWWWWWW""",
            "One agent isolated – should be invalid",
        ),
        (
            """WWWWWWWW
WWWWWWWW
WWOWWWWW
W      W
W  P  BW
W      W
W  A  XW
WWWWWWWW""",
            "All onions unreachable – should be invalid",
        ),
        (
            """WWWWWWWW
WO A  BW
W      W
WWPWWWWW
W      W
W  A  XW
WWWWWWWW""",
            "All pots unreachable – should be invalid",
        ),
        (
            """WWWWWWWW
WO A  BW
W      W
WWPWWWWW
WA     W
W     XW
WWWWWWWW""",
            "Pot unreachable despite agent adjacent – should be invalid",
        ),
        (
            """WWWWWWWW
WOAWWWBW
WWWWWW W
W  P   W
W      W
W  A  XW
WWWWWWWW""",
            "Agent trapped with onion – should be invalid",
        ),
        (
            """WWWWWWWW
WO WWWBW
W WWWW W
W WWWW W
W WWWW W
W AWWP W
WWWWWWAW
WWWWWWXW
WWWWWWWW""",
            "Agents opposite sides, no cooperation – should be invalid",
        ),
    ]

    print("\nTesting *invalid* layouts\n" + "=" * 80)
    idx = start_idx
    for grid_str, desc in invalid_layouts:
        orig_valid, orig_reason = original_evaluate(grid_str)
        fixed_valid, fixed_reason = fixed_evaluate(grid_str)
        stats.add_result(False, orig_valid, fixed_valid)
        _print_and_plot(
            idx=idx,
            grid_str=grid_str,
            desc=desc,
            expected=False,
            orig_valid=orig_valid,
            orig_reason=orig_reason,
            fixed_valid=fixed_valid,
            fixed_reason=fixed_reason,
        )
        idx += 1
    return idx


def test_edge_cases(start_idx: int = 1) -> None:
    """Edge‑case oddities we want to exercise."""
    edge_cases: List[Tuple[str, str, bool]] = [
        (
            """WWWWWWWW
WO A   W
W A    W
W  P  BW
W      W
W     XW
WWWWWWWW""",
            "Complex cooperation (need to move) – should be valid",
            True,
        ),
        (
            """WWWWWWWW
WO A  BW
WWWWWWWW
WP     W
W      W
W  A  XW
WWWWWWWW""",
            "Handoff via counter – should be valid",
            True,
        ),
        (
            """WWWWWWWW
WO A  OW
W  WWWWW
W  P  BW
W      W
W  A  XW
WWWWWWWW""",
            "Multiple onions (one unreachable) – should be valid",
            True,
        ),
        (
            """WWWWWWWW
WOWA  BW
W      W
W  P   W
W      W
W  A  XW
WWWWWWWW""",
            "Agent next to onion but no clear path – should be valid",
            True,
        ),
        (
            """WWWWWWWW
WO A  BW
W WW   W
W  P  WW
W WW   W
W  A  XW
WWWWWWWW""",
            "Different paths for agents – should be valid",
            True,
        ),
        (
            """WWWWWWWW
WO A  BW
WWWWWW W
W    W W
W P  W W
W    W W
W    WAW
W  X   W
WWWWWWWW""",
            "Complex multi‑stage cooperation – should be valid",
            True,
        ),
        (
            """WWWWWWWW
WA W  PW
W WW W W
W W  W W
W WWWW W
W   OW W
WWWWWW W
W  A  XW
W     BW
WWWWWWWW""",
            "Long winding path to onion – should be valid",
            True,
        ),
        (
            """WWWWWWWW
WO WWWPW
W WWWW W
WA     W
WWWWWW W
W  A  BW
W     XW
WWWWWWWW""",
            "One agent does cooking, other helps – should be valid",
            True,
        ),
    ]

    print("\nTesting *edge cases*\n" + "=" * 80)
    idx = start_idx
    for grid_str, desc, expected in edge_cases:
        orig_valid, orig_reason = original_evaluate(grid_str)
        fixed_valid, fixed_reason = fixed_evaluate(grid_str)
        stats.add_result(expected, orig_valid, fixed_valid)
        _print_and_plot(
            idx=idx,
            grid_str=grid_str,
            desc=desc,
            expected=expected,
            orig_valid=orig_valid,
            orig_reason=orig_reason,
            fixed_valid=fixed_valid,
            fixed_reason=fixed_reason,
        )
        idx += 1


# =============================================================================
# ─────────────────────────────────  MAIN  ───────────────────────────────────
# =============================================================================

def main() -> None:
    i = 1
    test_valid_layouts()
    i = test_invalid_layouts(start_idx=6)  # 5 valid layouts → start from 6
    test_edge_cases(start_idx=i)

    stats.print_summary()


if __name__ == "__main__":
    # TkAgg is friendlier for interactive closing; fallback silently if missing.
    try:
        import matplotlib

        matplotlib.use("TkAgg")  # pylint: disable=import-error
    except Exception:  # pragma: no cover – not critical
        pass

    main()
