import csv
import json
from pathlib import Path

import jax
import numpy as np
import optax
import pytest

from experiments.partner_adaptation import run_br
from experiments.partner_adaptation.run_br import TrainConfig, execute, resolve_run
from experiments.partner_adaptation.run_outputs import read_checkpoint

POP_DIR = (Path(__file__).resolve().parent.parent
           / "experiments/partner_adaptation/partner_agents/BRDiv_population/cramped_room")
SPU = 4 * 16


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    path = tmp_path_factory.mktemp("bank") / "bank.json"
    path.write_text(json.dumps({
        "format_version": 1, "name": "tiny", "layout": "cramped_room", "num_partners": 2,
        "populations": {"pop": {"config": str(POP_DIR / "config.pckl"), "seed_index": 0}},
        "partners": [{"partner_id": 0, "population": "pop", "member_index": 0,
                      "checkpoint": str(POP_DIR / "params_seed0_agent0.pt")},
                     {"partner_id": 1, "kind": "planner", "planner_id": "P01"}]}))
    return str(path)


def config(root, bank, **overrides):
    settings = dict(
        mode="disabled", checkpoint_path=str(root), layout_name="cramped_room", partner_bank=bank,
        num_heuristic_partners=0, num_envs=4, num_steps=16, total_timesteps=2 * SPU, update_epochs=1,
        num_minibatches=2, num_eval_episodes=1, hidden_size=16, num_layers=1, cl_method="ft", seed=0,
        use_task_id=False, use_multihead=False, eval_schedule="pilot", reward_shaping_horizon=200,
        schedule="interleaved", interleave_updates=1, aht_eval=False)
    settings.update(overrides)
    return TrainConfig(**settings)


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def adam_count(opt_state):
    adam = lambda x: isinstance(x, optax.ScaleByAdamState)
    states = [s for s in jax.tree.leaves(opt_state, is_leaf=adam) if adam(s)]
    assert len(states) == 1
    return int(states[0].count)


@pytest.fixture(scope="module")
def reference(bank, tmp_path_factory):
    calls = []
    real = run_br.run_br_training

    def spy(*args, **kwargs):
        calls.append({k: kwargs.get(k) for k in ("env_id_idx", "key_index", "update_base", "step_base",
                                                 "shaping_base", "final_eval", "init_eval")}
                     | {"opt_count_in": adam_count(kwargs["init_opt_state"])})
        out = real(*args, **kwargs)
        calls[-1]["opt_count_out"] = adam_count(kwargs["carry"]["final_opt_state"])
        return out

    patch = pytest.MonkeyPatch()
    patch.setattr(run_br, "run_br_training", spy)
    try:
        result = execute(config(tmp_path_factory.mktemp("reference"), bank))
    finally:
        patch.undo()
    return result, calls


def test_micro_stage_order_and_bookkeeping(reference):
    result, calls = reference
    assert [(c["env_id_idx"], c["key_index"], c["update_base"], c["step_base"], c["shaping_base"]) for c in calls] == [
        (0, 0, 0, 0, 0), (1, 1, 0, 1, 1), (0, 2, 1, 2, 2), (1, 3, 1, 3, 3)]
    assert [c["final_eval"] for c in calls] == [False, True, False, True]
    assert [c["init_eval"] for c in calls] == [True, False, False, False]

    train = rows(result.run_dir / "train_metrics.csv")
    assert [(int(r["env_steps"]), int(r["stage"]), int(r["update"])) for r in train] == [
        (SPU, 0, 1), (2 * SPU, 1, 1), (3 * SPU, 0, 2), (4 * SPU, 1, 2)]
    status = json.loads((result.run_dir / "status.json").read_text())
    assert status["state"] == "complete" and status["stages_completed"] == 4
    assert result.counters["train_env_steps"] == 4 * SPU and result.counters["eval_full_events"] == 3
    budget = json.loads((result.run_dir / "run.json").read_text())["budget"]
    assert (budget["cycles"], budget["micro_stages"], budget["planned_full_evaluations"]) == (2, 4, 3)
    meta, _ = read_checkpoint(result.run_dir / "latest.ckpt")
    assert (meta["stages_completed"], meta["cycle"], meta["partner"]) == (4, 1, 1)


def test_full_bank_evaluation_only_before_training_and_after_each_cycle(reference):
    result, _ = reference
    ev = rows(result.run_dir / "eval_metrics.csv")
    events = sorted({(int(r["env_steps"]), r["scope"], int(r["stage"]), int(r["update"])) for r in ev})
    assert events == [(0, "full", 0, 0), (2 * SPU, "full", 1, 1), (4 * SPU, "full", 1, 2)]
    for steps in (0, 2 * SPU, 4 * SPU):
        assert sorted(int(r["eval_partner_id"]) for r in ev if int(r["env_steps"]) == steps) == [0, 1]
    assert len({r["row_id"] for r in ev}) == len(ev)


def test_shaping_anneal_continues_over_the_run(reference):
    result, _ = reference
    coefs = [float(r["shaping_coef"]) for r in rows(result.run_dir / "train_metrics.csv")]
    np.testing.assert_allclose(coefs, [1 - m * SPU / 200 for m in range(4)])


def test_optimizer_state_is_carried_across_micro_stages(reference):
    _, calls = reference
    per_micro = 1 * 1 * 2
    assert [c["opt_count_in"] for c in calls] == [0, per_micro, 2 * per_micro, 3 * per_micro]
    assert [c["opt_count_out"] for c in calls] == [per_micro, 2 * per_micro, 3 * per_micro, 4 * per_micro]


@pytest.mark.parametrize("interruption", ["before_micro_2", "micro_2_rows_written"])
def test_boundary_resume_matches_uninterrupted(interruption, bank, reference, tmp_path):
    ref, _ = reference
    patch = pytest.MonkeyPatch()
    if interruption == "before_micro_2":
        real = run_br.run_br_training

        def failing(*args, **kwargs):
            if kwargs["key_index"] == 2:
                raise RuntimeError("simulated crash")
            return real(*args, **kwargs)
        patch.setattr(run_br, "run_br_training", failing)
    else:
        real_save = run_br.save_checkpoint

        def failing_save(run_dir, params, state, meta):
            if meta["stages_completed"] == 3:
                raise RuntimeError("simulated crash")
            return real_save(run_dir, params, state, meta)
        patch.setattr(run_br, "save_checkpoint", failing_save)
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            execute(config(tmp_path, bank))
    finally:
        patch.undo()
    run_dir, = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert read_checkpoint(run_dir / "latest.ckpt")[0]["stages_completed"] == 2

    resumed = execute(config(tmp_path, bank, resume=str(run_dir)))

    for a, b in zip(jax.tree.leaves(resumed.ego_params), jax.tree.leaves(ref.ego_params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)
    assert resumed.counters == ref.counters
    for name in ("train_metrics.csv", "eval_metrics.csv"):
        got, want = rows(run_dir / name), rows(ref.run_dir / name)
        assert [(r["env_steps"], r["stage"], r["update"]) for r in got] == \
               [(r["env_steps"], r["stage"], r["update"]) for r in want]
        for g, w in zip(got, want):
            for key in ("soups", "ego_return", "actor_loss", "shaping_coef"):
                if key in g and g[key] not in ("", None):
                    np.testing.assert_allclose(float(g[key]), float(w[key]), rtol=1e-5, atol=1e-6, err_msg=key)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "complete" and status["attempts"][1]["resumed_from_stage"] == 2
    dropped = status["attempts"][1]["discarded_rows_of_uncommitted_stages"]
    assert dropped == (0 if interruption == "before_micro_2" else 1 + 1)


@pytest.mark.parametrize("overrides, message", [
    (dict(cl_method="ewc"), "supports cl_method ft"),
    (dict(interleave_updates=3, total_timesteps=4 * SPU), "must be positive and divide"),
    (dict(anneal_lr=True), "constant learning rate"),
    (dict(reward_shaping_horizon=5 * SPU), "exceeds the whole interleaved run"),
    (dict(schedule="random"), "schedule must be one of"),
])
def test_configuration_guards(overrides, message, bank, tmp_path):
    with pytest.raises(ValueError, match=message):
        resolve_run(config(tmp_path, bank, **overrides))


def test_sequential_fingerprint_does_not_mention_schedules(bank, tmp_path):
    sequential = resolve_run(config(tmp_path, bank, schedule="sequential")).fingerprint["config"]
    interleaved = resolve_run(config(tmp_path, bank)).fingerprint["config"]
    assert "schedule" not in sequential and "interleave_updates" not in sequential
    assert interleaved["schedule"] == "interleaved" and interleaved["interleave_updates"] == 1
