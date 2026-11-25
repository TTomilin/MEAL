"""Unit and real-data tests for pure-NumPy Vendi diversity, elbow selection, and specialist health."""
import csv
from pathlib import Path

import numpy as np
import pytest

from experiments.partner_adaptation import diversity as div
from experiments.results.plotting import partner_diversity as pdiversity

JAXAHT_XP_DIR = Path(__file__).resolve().parents[1] / "third_party" / "jax-aht" / "evaluation" / "xp_matrices"


def test_vendi_score_identities_and_duplicate_non_monotonicity():
    assert div.vendi_score(np.zeros((0, 0))) == 0.0
    assert div.vendi_score(np.ones((1, 1))) == pytest.approx(1.0)
    # N identical clones -> Vendi = 1.0
    assert div.vendi_score(np.ones((5, 5))) == pytest.approx(1.0)
    # N orthogonal partners -> Vendi = N
    assert div.vendi_score(np.eye(5)) == pytest.approx(5.0)
    # Adding 4 clones of partner 0 to 5 orthogonal partners strictly lowers Vendi below 5.0
    xp_orth = np.eye(5) * 10.0
    v_orth = div.vendi_score(div.response_kernel(xp_orth))
    idx = [0, 1, 2, 3, 4, 0, 0, 0, 0]
    xp_dup = xp_orth[np.ix_(idx, idx)]
    v_dup = div.vendi_score(div.response_kernel(xp_dup))
    assert v_orth == pytest.approx(5.0)
    assert v_dup < v_orth


def test_behaviour_kernel_and_vendi():
    # Three completely disjoint spatial occupancy vectors -> Vendi = 3.0
    vis = np.eye(3, 9) * 100.0
    K = div.behaviour_kernel(vis)
    np.testing.assert_allclose(K, np.eye(3))
    assert div.vendi_score(K) == pytest.approx(3.0)
    # Identical distributions -> Vendi = 1.0
    same = np.tile(np.array([1.0, 2.0, 3.0]), (4, 1))
    assert div.vendi_score(div.behaviour_kernel(same)) == pytest.approx(1.0)


def test_vendi_bootstrap_ci_deterministic_and_bounded():
    xp = np.eye(4) * 10.0 + 1.0
    std = np.full((4, 4), 1.5)
    ci1 = div.vendi_bootstrap_ci(xp, std, episodes=20, n_boot=200, seed=7)
    ci2 = div.vendi_bootstrap_ci(xp, std, episodes=20, n_boot=200, seed=7)
    assert ci1 == ci2
    assert ci1["ci_low"] <= ci1["mean"] <= ci1["ci_high"]
    zero_ci = div.vendi_bootstrap_ci(xp, np.zeros((4, 4)), episodes=20, n_boot=50, seed=0)
    point = div.vendi_score(div.response_kernel(xp))
    assert zero_ci["ci_low"] == pytest.approx(point)
    assert zero_ci["ci_high"] == pytest.approx(point)


def test_diagnose_specialists_flags_defects_vs_healthy_weak_partner():
    labels = ["p_ok", "p_zero", "p_weak_br", "p_still_learning", "p_low_but_converged"]
    xp = np.array([
        [10.0, 3.5, 9.0, 1.0, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 7.0, 1.0, 0.1],
        [1.0, 0.0, 1.0, 6.0, 0.1],
        [0.1, 0.0, 0.1, 0.1, 0.4],
    ])
    learn = [
        {"label": "p_ok", "final_train_soups": 10.0, "last_quarter_gain": 0.2},
        {"label": "p_zero", "final_train_soups": 0.0, "last_quarter_gain": 0.0},
        {"label": "p_weak_br", "final_train_soups": 7.0, "last_quarter_gain": 0.1},
        {"label": "p_still_learning", "final_train_soups": 6.0, "last_quarter_gain": 2.0},
        {"label": "p_low_but_converged", "final_train_soups": 0.4, "last_quarter_gain": 0.0},
    ]
    health = div.diagnose_specialists(xp, labels, learn, min_own_score=0.5)
    by = {p["label"]: p for p in health["partners"]}
    assert not health["healthy"]
    assert by["p_ok"]["flags"] == [] and not by["p_ok"]["defect"]
    assert set(by["p_zero"]["flags"]) == {"ZERO_OWN_SCORE", "WEAK_SPECIALIST"} and by["p_zero"]["defect"]
    assert by["p_weak_br"]["flags"] == ["WEAK_SPECIALIST"] and by["p_weak_br"]["defect"]
    assert by["p_still_learning"]["flags"] == ["STILL_LEARNING"] and by["p_still_learning"]["defect"]
    assert by["p_low_but_converged"]["flags"] == ["LOW_OWN_SCORE"] and not by["p_low_but_converged"]["defect"]
    assert health["defect_indices"] == [1, 2, 3]


def test_greedy_vendi_stops_at_five_when_pool_has_five_conventions_duplicated():
    # 5 orthogonal conventions, each duplicated with a weaker clone (10 partners total).
    base = np.eye(5) * 10.0 + 0.5
    scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8])
    idx = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    xp = base[np.ix_(idx, idx)] * scales[np.newaxis, :]
    selected = div.select_by_vendi(xp, range(10), min_gain=0.15, max_off_diag=0.7)
    assert selected == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("layout,expected_k", [
    ("asymm_advantages", 4),
    ("counter_circuit", 6),
    ("forced_coord", 6),
    ("coord_ring", 9),
    ("cramped_room", 10),
])
def test_vendi_elbow_on_official_jaxaht_xp_matrices(layout, expected_k):
    path = JAXAHT_XP_DIR / f"{layout}.csv"
    if not path.is_file():
        pytest.skip(f"{path} not present")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r["metric_name"] == "base_return" and r["aggregate_stat"] == "mean" and r["seating"] == "new"]
    labels = sorted({r["row_label"] for r in rows})
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    xp = np.zeros((n, n), dtype=np.float64)
    for r in rows:
        xp[idx[r["br_for_label"]], idx[r["row_label"]]] = float(r["mean"])
    curve = div.greedy_vendi_curve(xp, range(n))
    assert curve["recommended_k"] == expected_k
    accepted_steps = [s for s in curve["steps"] if s["accepted"]]
    assert accepted_steps[-1]["efficiency"] >= 0.70
    assert accepted_steps[-1]["max_off_diag"] < div.DEFAULT_MAX_OFF_DIAG


def test_partner_diversity_lazy_sklearn_and_own_specialist_features():
    # Importing partner_diversity and using relative_xp / behaviour_features must not require scikit-learn.
    xp = np.array([[10.0, 2.0], [3.0, 8.0]])
    np.testing.assert_allclose(pdiversity.relative_xp(xp), div.relative_xp(xp))

    acts = np.zeros((2, 2, 6))
    acts[0, 0] = [10, 0, 0, 0, 0, 0]
    acts[1, 0] = [0, 10, 0, 0, 0, 0]  # off-diagonal mismatch should not pollute own_only=True
    acts[1, 1] = [0, 0, 10, 0, 0, 0]
    acts[0, 1] = [0, 0, 0, 10, 0, 0]
    visits = np.ones((2, 2, 4))
    pots = np.ones((2, 2, 2))
    own = pdiversity.behaviour_features(acts, visits, pots, own_only=True)
    pooled = pdiversity.behaviour_features(acts, visits, pots, own_only=False)
    np.testing.assert_allclose(own[0, :6], [1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(own[1, :6], [0, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(pooled[0, :6], [0.5, 0.5, 0, 0, 0, 0])


def test_diversity_edge_cases_and_validation():
    # n_valid = 0 and n_valid = 1
    zero_xp = np.zeros((3, 3))
    s0 = div.diversity_summary(zero_xp, ["a", "b", "c"], ["planner"] * 3)
    assert s0["n_valid"] == 0 and s0["vendi_resp"] == 0.0
    one_xp = np.diag([5.0, 0.0])
    s1 = div.diversity_summary(one_xp, ["a", "b"], ["planner", "planner"])
    assert s1["n_valid"] == 1 and s1["vendi_resp"] == pytest.approx(1.0)
    assert div.greedy_vendi_curve(one_xp, [0, 1], max_k=0)["recommended_k"] == 0
    assert div.greedy_vendi_curve(one_xp, [0, 1], max_k=1)["recommended_indices"] == [0]

    # kind_caps without kinds must fail deterministically instead of silently ignoring caps
    xp = np.eye(4) * 10.0
    with pytest.raises(ValueError, match="kind_caps requires kinds"):
        div.greedy_vendi_curve(xp, range(4), kind_caps={"planner": 1})
    capped = div.select_by_vendi(xp, range(4), kinds=["planner", "planner", "jaxaht", "jaxaht"],
                                 kind_caps={"planner": 1})
    assert sum(1 for i in capped if i < 2) == 1 and sum(1 for i in capped if i >= 2) == 2

    # negative/NaN xp_std must be rejected
    with pytest.raises(ValueError, match="non-negative"):
        div.vendi_bootstrap_ci(xp, np.full((4, 4), -1.0))

    # NaN own_score, negative own_score, and NaN off-diagonal must never mask defects
    bad_xp = np.array([
        [np.nan, 1.0, 12.0],
        [1.0, -2.0, np.nan],
        [0.0, 0.0, 8.0],
    ])
    health = div.diagnose_specialists(bad_xp, ["p_nan", "p_neg", "p_weak_with_nan"])
    by = {p["label"]: p for p in health["partners"]}
    assert "NON_FINITE_OWN_SCORE" in by["p_nan"]["flags"] and by["p_nan"]["defect"]
    assert "ZERO_OWN_SCORE" in by["p_neg"]["flags"] and by["p_neg"]["defect"]
    assert "WEAK_SPECIALIST" in by["p_weak_with_nan"]["flags"] and by["p_weak_with_nan"]["defect"]
    assert health["defect_indices"] == [0, 1, 2]
