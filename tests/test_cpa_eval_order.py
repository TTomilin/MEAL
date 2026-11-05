"""stage_order / recovery handling in experiments.results.cpa_eval (CPU only)."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.results.cpa_eval import load_run, main, partner_metrics, recovery

MAX_SOUP = 10.0
NUM_ENVS, NUM_STEPS = 2, 5
SPU = NUM_ENVS * NUM_STEPS
EVAL_COLS = ["event_id", "row_id", "scope", "env_steps", "stage", "update", "train_episodes",
             "eval_partner_id", "eval_partner", "eval_episode", "episode_key", "soups", "ego_soups",
             "partner_soups", "ego_return", "completed", "attempt"]
TRAIN_COLS = ["env_steps", "stage", "partner_id", "partner", "update", "episodes", "soups", "ego_return",
              "sparse_reward", "shaped_reward", "shaping_coef", "actor_loss", "value_loss", "entropy",
              "cl_penalty", "grad_norm", "attempt"]


def write_run(root: Path, name: str, order, full, current=(), train=(), stage_order=None):
    """full: list of (env_steps, {partner_id: soups}); current: (stage, update, soups); train: (stage, update, soups)."""
    d = root / name
    d.mkdir(parents=True)
    bank = [{"partner_id": i, "kind": "brdiv", "label": f"P{i}"} for i in range(2)]
    config = {"schedule": "sequential", "num_envs": NUM_ENVS, "num_steps": NUM_STEPS}
    if stage_order is not None:
        config["stage_order"] = stage_order
    (d / "config.json").write_text(json.dumps(config))
    (d / "run.json").write_text(json.dumps({"normalization": {"max_soup": MAX_SOUP}}))
    (d / "partner_bank.json").write_text(json.dumps(bank))
    rows = []
    for e, (steps, scores) in enumerate(full):
        stage = -1 if e == 0 else e - 1
        update = 0 if e == 0 else 4
        for pid, soups in scores.items():
            rows.append({"event_id": f"ev{e}", "row_id": f"ev{e}-p{pid}-e0", "scope": "full", "env_steps": steps,
                         "stage": stage, "update": update, "eval_partner_id": pid, "eval_partner": f"P{pid}",
                         "eval_episode": 0, "soups": soups, "ego_soups": soups / 2, "partner_soups": soups / 2,
                         "completed": 1, "attempt": 0})
    for stage, update, soups in current:
        rows.append({"event_id": f"cur{stage}-{update}", "row_id": f"cur{stage}-{update}-e0", "scope": "current",
                     "env_steps": (stage * 4 + update) * SPU, "stage": stage, "update": update,
                     "eval_partner_id": order[stage], "eval_partner": f"P{order[stage]}", "eval_episode": 0,
                     "soups": soups, "ego_soups": soups / 2, "partner_soups": soups / 2, "completed": 1, "attempt": 0})
    with open(d / "eval_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_COLS, restval="")
        w.writeheader()
        w.writerows(rows)
    with open(d / "train_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAIN_COLS, restval="")
        w.writeheader()
        for stage, update, soups in train:
            w.writerow({"env_steps": (stage * 4 + update) * SPU, "stage": stage, "partner_id": order[stage],
                        "partner": f"P{order[stage]}", "update": update, "episodes": 1, "soups": soups,
                        "cl_penalty": 0.0, "attempt": 0})
    return d


def test_bank_order_matches_hand_values(tmp_path):
    # R (normalised): row0 [0.2, 0.1]; after p0 [0.8, 0.1]; after p1 [0.6, 0.9]
    d = write_run(tmp_path, "bank", [0, 1], [
        (0, {0: 2.0, 1: 1.0}),
        (40, {0: 8.0, 1: 1.0}),
        (80, {0: 6.0, 1: 9.0}),
    ], train=[(0, 1, 5.0), (0, 4, 8.0), (1, 1, 4.0), (1, 4, 9.0)])
    run = load_run("bank", d)
    assert run.order == [0, 1] and run.steps_per_update == SPU
    m = partner_metrics(run)
    np.testing.assert_allclose(m["acq"], [0.8, 0.9])
    np.testing.assert_allclose(m["peak"], [0.8, 0.9])
    np.testing.assert_allclose(m["final"], [0.6, 0.9])
    np.testing.assert_allclose(m["forgetting"][0], 0.2)
    assert np.isnan(m["forgetting"][1])
    np.testing.assert_allclose(m["first_stage"], [0.0, 1.0])
    assert recovery(run) == []


def test_returning_partner_acq_forgetting_and_recovery(tmp_path):
    # order [0, 1, 0]; R rows: init, after0, after1, after2(return 0)
    # partner 0 acq from row 1 (=0.8); trained last -> no forgetting
    # stage 2: previous_end=0.8, before=0.5, target=0.72; current update 3 hits 0.75
    d = write_run(tmp_path, "ret", [0, 1, 0], [
        (0, {0: 2.0, 1: 1.0}),
        (40, {0: 8.0, 1: 1.0}),
        (80, {0: 5.0, 1: 9.0}),
        (120, {0: 7.5, 1: 8.0}),
    ], current=[(2, 1, 4.0), (2, 3, 7.5)], train=[(2, 1, 4.0), (2, 3, 7.0), (2, 4, 7.5)],
       stage_order=[0, 1, 0])
    run = load_run("ret", d)
    m = partner_metrics(run)
    assert run.order == [0, 1, 0]
    np.testing.assert_allclose(m["acq"][0], 0.8)
    np.testing.assert_allclose(m["acq"][1], 0.9)
    assert np.isnan(m["forgetting"][0])  # trained last
    np.testing.assert_allclose(m["forgetting"][1], 0.9 - 0.8)
    rows = recovery(run)
    assert len(rows) == 1 and rows[0]["stage"] == 2 and rows[0]["partner_id"] == 0
    assert rows[0]["previous_end"] == pytest.approx(0.8)
    assert rows[0]["before"] == pytest.approx(0.5)
    assert rows[0]["target"] == pytest.approx(0.72)
    assert rows[0]["recovery_update"] == 3
    assert rows[0]["recovery_env_steps"] == 3 * SPU

    # same schedule but current/end never reach 90% of previous_end
    d2 = write_run(tmp_path, "miss", [0, 1, 0], [
        (0, {0: 2.0, 1: 1.0}),
        (40, {0: 8.0, 1: 1.0}),
        (80, {0: 5.0, 1: 9.0}),
        (120, {0: 6.0, 1: 8.0}),  # end 0.6 < 0.72
    ], current=[(2, 1, 4.0), (2, 3, 5.0)], train=[(2, 1, 4.0), (2, 4, 6.0)], stage_order=[0, 1, 0])
    miss = recovery(load_run("miss", d2))
    assert len(miss) == 1 and miss[0]["recovery_update"] is None and miss[0]["recovery_env_steps"] is None


def test_main_writes_four_files(tmp_path, monkeypatch):
    a = write_run(tmp_path, "a", [0, 1], [
        (0, {0: 2.0, 1: 1.0}), (40, {0: 8.0, 1: 1.0}), (80, {0: 6.0, 1: 9.0}),
    ], train=[(0, 4, 8.0), (1, 4, 9.0)])
    b = write_run(tmp_path, "b", [0, 1, 0], [
        (0, {0: 2.0, 1: 1.0}), (40, {0: 8.0, 1: 1.0}), (80, {0: 5.0, 1: 9.0}), (120, {0: 7.5, 1: 8.0}),
    ], current=[(2, 3, 7.5)], train=[(2, 4, 7.5)], stage_order=[0, 1, 0])
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["cpa_eval", "--run", f"ft/seed0={a}", "--run", f"ft/seed1={b}",
                                     "--out", str(out), "--prefix", "t", "--no-figures"])
    main()
    for name in ("t_summary.json", "t_partners.csv", "t_stages.csv", "t_recovery.csv"):
        assert (out / name).exists()
    partners = list(csv.DictReader((out / "t_partners.csv").open()))
    assert {r["run"] for r in partners} == {"ft/seed0", "ft/seed1"} and len(partners) == 4
    recovery_rows = list(csv.DictReader((out / "t_recovery.csv").open()))
    assert len(recovery_rows) == 1 and recovery_rows[0]["recovery_update"] == "3"
    summary = json.loads((out / "t_summary.json").read_text())
    assert "ft" in summary and "all" in summary["ft"]
