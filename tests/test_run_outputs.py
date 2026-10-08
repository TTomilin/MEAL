"""Run outputs, boundary checkpoints and resume for `run_br` (CPU only).

Training here is tiny (4 envs, 16-step episodes, 2 updates per partner, two real bundled BRDiv partners of
`cramped_room`). The checks are about records and recovery, not about learning or performance.
"""
import csv
import json
import pickle
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from experiments.continual.agem import init_agem_memory
from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.l2 import L2
from experiments.continual.mas import MAS
from experiments.model.mlp import ActorCritic
from experiments.partner_adaptation import run_br, run_outputs
from experiments.partner_adaptation.partner_agents.agent_interface import MLPActorCriticPolicyCL
from experiments.partner_adaptation.partner_bank import PartnerBankError
from experiments.partner_adaptation.run_br import TrainConfig, execute
from experiments.partner_adaptation.run_outputs import (
    CheckpointError, ResumeError, RunOutputError, RunRecorder, allocate_run_dir, read_checkpoint, restore_state,
    save_checkpoint, truncate_csv,
)
from experiments.partner_adaptation.train_ego import (
    build_eval_event, build_train_log, build_train_record, shaping_coefficient,
)
from experiments.utils import init_cl_state

POP_DIR = (Path(__file__).resolve().parent.parent
           / "experiments/partner_adaptation/partner_agents/BRDiv_population/cramped_room")


def write_bank(path: Path, members):
    manifest = {
        "format_version": 1, "name": "tiny", "layout": "cramped_room", "num_partners": len(members),
        "populations": {"pop": {"config": str(POP_DIR / "config.pckl"), "seed_index": 0}},
        "partners": [{"partner_id": i, "population": "pop", "member_index": m,
                      "checkpoint": str(POP_DIR / f"params_seed0_agent{m}.pt")} for i, m in enumerate(members)]}
    path.write_text(json.dumps(manifest))
    return str(path)


def tiny_config(root, bank, cl_method="ft", **overrides):
    settings = dict(
        mode="disabled", checkpoint_path=str(root), layout_name="cramped_room", partner_bank=bank,
        num_heuristic_partners=0, num_envs=4, num_steps=16, total_timesteps=2 * 4 * 16, update_epochs=1,
        num_minibatches=2, num_eval_episodes=2, eval_every=2, hidden_size=16, num_layers=1, importance_episodes=1,
        importance_steps=8, cl_method=cl_method, reg_coef=1.0, seed=0, reward_shaping_horizon=1e3)
    settings.update(overrides)
    return TrainConfig(**settings)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def assert_rows_match(a, b, ignore=("attempt",)):
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        for key in ra:
            if key in ignore:
                continue
            try:
                np.testing.assert_allclose(float(ra[key]), float(rb[key]), rtol=1e-5, atol=1e-6, err_msg=key)
            except ValueError:
                assert ra[key] == rb[key], key


def assert_trees_close(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-6)


class FakeWandb:
    """Stands in for a wandb run; records every log call."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.logged = []
        self.finished = False

    def log(self, data, step=None):
        self.logged.append((step, data))

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch):
    runs = []

    def init(**kwargs):
        runs.append(FakeWandb(**kwargs))
        return runs[-1]

    def login(*args, **kwargs):
        raise AssertionError("wandb.login must not be called unless mode is online")

    monkeypatch.setattr(run_br.wandb, "init", init)
    monkeypatch.setattr(run_br.wandb, "login", login)
    return runs


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    return write_bank(tmp_path_factory.mktemp("bank") / "bank.json", [0, 1])


@pytest.fixture(scope="module")
def references(bank, tmp_path_factory):
    """One uninterrupted two-partner run per CL method, shared by the tests that compare against it."""
    cache = {}

    def get(cl_method):
        if cl_method not in cache:
            root = tmp_path_factory.mktemp(f"reference_{cl_method}")
            cache[cl_method] = execute(tiny_config(root, bank, cl_method))
        return cache[cl_method]
    return get


# ----------------------------------------------------------------------------------------------------------
# resume equals uninterrupted
# ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cl_method, interruption", [("ft", "before_partner_1"), ("ewc", "partner_1_rows_written")])
def test_boundary_resume_matches_uninterrupted(cl_method, interruption, bank, references, tmp_path, fake_wandb):
    reference = references(cl_method)
    config = tiny_config(tmp_path, bank, cl_method)

    crash = pytest.MonkeyPatch()
    if interruption == "before_partner_1":
        real = run_br.run_br_training

        def failing_training(*args, **kwargs):
            if kwargs["env_id_idx"] == 1:
                raise RuntimeError("simulated crash")
            return real(*args, **kwargs)
        crash.setattr(run_br, "run_br_training", failing_training)
    else:  # partner 1 is fully recorded, then the process dies before its checkpoint lands
        real_save = run_br.save_checkpoint

        def failing_save(run_dir, params, cl_state, meta):
            if meta["stages_completed"] == 2:
                raise RuntimeError("simulated crash")
            return real_save(run_dir, params, cl_state, meta)
        crash.setattr(run_br, "save_checkpoint", failing_save)

    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            execute(config)
    finally:
        crash.undo()
    run_dir, = [p for p in tmp_path.iterdir() if p.is_dir()]
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "failed" and status["stages_completed"] == 1
    meta, _ = read_checkpoint(run_dir / "latest.ckpt")
    assert meta["stages_completed"] == 1
    rows_before = len(read_rows(run_dir / "train_metrics.csv"))
    assert rows_before == (2 if interruption == "before_partner_1" else 4)

    resumed = execute(tiny_config(tmp_path, bank, cl_method, resume=str(run_dir)))

    assert resumed.run_dir == run_dir and len([p for p in tmp_path.iterdir() if p.is_dir()]) == 1
    assert resumed.counters == reference.counters
    assert_trees_close(resumed.ego_params, reference.ego_params)
    assert_trees_close(resumed.cl_state, reference.cl_state)
    for name in ("train_metrics.csv", "eval_metrics.csv"):
        assert_rows_match(read_rows(run_dir / name), read_rows(reference.run_dir / name))
    event_ids = [r["row_id"] for r in read_rows(run_dir / "eval_metrics.csv")]
    assert len(event_ids) == len(set(event_ids))

    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "complete" and len(status["attempts"]) == 2
    assert status["attempts"][1]["resumed_from_stage"] == 1
    assert status["attempts"][1]["discarded_rows_of_uncommitted_stages"] == (0 if rows_before == 2 else 2 + 8 + 1)  # train, eval, timing rows
    # the resumed attempt writes its own W&B run containing only partner 1, with strictly increasing steps
    steps = [s for s, _ in fake_wandb[-1].logged]
    assert steps == sorted(set(steps)) and min(steps) > 2 * 4 * 16
    assert fake_wandb[-1].finished


def test_resume_of_complete_run_changes_nothing(bank, references):
    reference = references("ft")
    skip = ("status.json", "params_seed0.pt")
    before = {p.name: p.read_bytes() for p in reference.run_dir.iterdir() if p.name not in skip}
    again = execute(tiny_config(reference.run_dir.parent, bank, "ft", resume=str(reference.run_dir)))
    assert again.run_dir == reference.run_dir
    assert_trees_close(again.ego_params, reference.ego_params)
    after = {p.name: p.read_bytes() for p in reference.run_dir.iterdir() if p.name not in skip}
    assert before == after
    exported = pickle.load(open(reference.run_dir / "params_seed0.pt", "rb"))["actor_params"]
    assert_trees_close(exported, reference.ego_params)


# ----------------------------------------------------------------------------------------------------------
# what a run records
# ----------------------------------------------------------------------------------------------------------

def test_run_directory_contents(bank, references):
    run = references("ewc")
    d = run.run_dir
    for name in ("config.json", "config.pckl", "run.json", "partner_bank.json", "status.json", "train_metrics.csv",
                 "eval_metrics.csv", "stage_timings.csv", "latest.ckpt", "params_seed0.pt"):
        assert (d / name).is_file(), name
    meta = json.loads((d / "run.json").read_text())
    assert [p["label"] for p in meta["partners"]] == ["cramped_room/pop/m0", "cramped_room/pop/m1"]
    assert all(len(p["payload_sha256"]) == 64 for p in meta["partners"])
    assert meta["code"]["available"] and len(meta["code"]["commit"]) == 40
    assert meta["architecture"]["num_heads"] == 2 and meta["architecture"]["parameter_count"] > 0
    assert meta["budget"]["planned_train_env_steps"] == run.counters["train_env_steps"] == 256
    assert meta["budget"]["importance_env_steps_per_stage"] == 8
    assert "re-created at every partner" in meta["schedule"]["optimizer"]
    assert run.counters["importance_env_steps"] == 16 and run.counters["eval_events"] == 4
    assert run.counters["eval_env_steps"] == 4 * 2 * 2 * 16

    train = read_rows(d / "train_metrics.csv")
    assert [int(r["env_steps"]) for r in train] == [64, 128, 192, 256]
    assert [r["stage"] for r in train] == ["0", "0", "1", "1"]
    assert all(r["sparse_reward"] != "" and r["shaped_reward"] != "" for r in train)
    assert float(train[0]["shaping_coef"]) == 1.0 and float(train[1]["shaping_coef"]) < 1.0
    assert float(train[0]["shaping_coef"]) == float(train[2]["shaping_coef"])  # anneal restarts at each partner

    evals = read_rows(d / "eval_metrics.csv")
    assert len(evals) == 4 * 2 * 2
    assert len({r["row_id"] for r in evals}) == len(evals)
    assert {r["event_id"] for r in evals} == {"s000-u00001", "s000-u00002", "s001-u00001", "s001-u00002"}

    timings = read_rows(d / "stage_timings.csv")
    assert [r["compile_cached"] for r in timings] == ["0", "1"]
    assert all(r["train_eval_s"] and r["eval_event_s"] and r["importance_s"] for r in timings)
    assert all(r["memory_status"] == "unavailable" and r["peak_bytes_in_use"] == "" for r in timings)


def test_wandb_steps_are_environment_interactions_and_disabled_mode_needs_no_login(bank, tmp_path, fake_wandb):
    run = execute(tiny_config(tmp_path, bank, "ft"))
    steps = [s for s, _ in fake_wandb[0].logged]
    assert steps == [64, 128, 192, 256]
    for step, data in fake_wandb[0].logged:
        assert data["env_steps"] == step and "Train/EgoValueLoss" in data
    assert all("Eval/StageID" in data for _, data in fake_wandb[0].logged)  # evaluation merged at the same step
    assert fake_wandb[0].finished and run.counters["stages_completed"] == 2


def test_allocation_never_reuses_an_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(run_outputs, "datetime", SimpleNamespace(now=lambda: datetime(2026, 1, 1)))
    first, uid = allocate_run_dir(tmp_path, "br_ft_x")
    (first / "keep").write_text("a previous run")
    real_uuid4, calls = run_outputs.uuid.uuid4, []

    def colliding_uuid4():  # same timestamp and same suffix twice, then a fresh one
        calls.append(1)
        return SimpleNamespace(hex=uid) if len(calls) <= 2 else real_uuid4()
    monkeypatch.setattr(run_outputs.uuid, "uuid4", colliding_uuid4)
    second, _ = allocate_run_dir(tmp_path, "br_ft_x")
    assert second != first and len(calls) == 3
    assert (first / "keep").read_text() == "a previous run"


def test_concurrent_allocations_are_distinct(tmp_path):
    paths, lock = [], threading.Lock()

    def work():
        p, _ = allocate_run_dir(tmp_path, "same_name")
        with lock:
            paths.append(p)
    threads = [threading.Thread(target=work) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(paths)) == 16


def test_execute_twice_with_identical_settings_creates_two_runs(bank, tmp_path, monkeypatch):
    """Cheap: stop right after the run directory exists, by failing the first stage."""
    def crash(*args, **kwargs):
        raise RuntimeError("stop")
    monkeypatch.setattr(run_br, "run_br_training", crash)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="stop"):
            execute(tiny_config(tmp_path, bank, "ft"))
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 2


# ----------------------------------------------------------------------------------------------------------
# rejected resumes and damaged checkpoints
# ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("changes, expected", [
    (dict(total_timesteps=4 * 4 * 16), "per partner"),
    (dict(num_steps=32), "num_steps"),
    (dict(num_eval_episodes=3), "num_eval_episodes"),
    (dict(eval_every=1), "eval_every"),
    (dict(cl_method="ewc"), "cl_method"),
    (dict(hidden_size=32), "hidden_size"),
    (dict(use_multihead=False), "use_multihead"),
    (dict(seed=1), "seed"),
])
def test_incompatible_resume_is_rejected_and_leaves_the_run_untouched(changes, expected, bank, references):
    reference = references("ft")
    before = {p.name: p.read_bytes() for p in reference.run_dir.iterdir()}
    with pytest.raises(ResumeError, match=expected):
        execute(tiny_config(reference.run_dir.parent, bank, resume=str(reference.run_dir),
                            **{"cl_method": "ft", **changes}))
    assert before == {p.name: p.read_bytes() for p in reference.run_dir.iterdir()}


def test_resume_rejects_a_different_layout(bank, references):
    reference = references("ft")
    with pytest.raises((ResumeError, PartnerBankError), match="coord_ring"):
        execute(tiny_config(reference.run_dir.parent, bank, "ft", resume=str(reference.run_dir),
                            layout_name="coord_ring"))
    with pytest.raises(ResumeError, match="layout_name"):
        execute(tiny_config(reference.run_dir.parent, "", "ft", resume=str(reference.run_dir),
                            layout_name="coord_ring"))


def test_resume_rejects_different_partner_order_and_bank(bank, references, tmp_path):
    reference = references("ft")
    swapped = write_bank(tmp_path / "swapped.json", [1, 0])
    with pytest.raises(ResumeError, match=r"partners\[0\]"):
        execute(tiny_config(reference.run_dir.parent, swapped, "ft", resume=str(reference.run_dir)))
    longer = write_bank(tmp_path / "longer.json", [0, 1, 2])
    with pytest.raises(ResumeError, match="partners.length"):
        execute(tiny_config(reference.run_dir.parent, longer, "ft", resume=str(reference.run_dir)))


def small_state():
    network = ActorCritic(6, "relu", 2, True, False, 8, 1, True, True)
    return MLPActorCriticPolicyCL(network, 10).init_params(jax.random.PRNGKey(0))


@pytest.mark.parametrize("damage", ["flip_byte", "truncate", "wrong_magic", "empty"])
def test_damaged_checkpoint_is_rejected(damage, tmp_path):
    params = small_state()
    path = save_checkpoint(tmp_path, params, None, {"run_uid": "x", "stages_completed": 1, "counters": {},
                                                    "fingerprint": {}})
    blob = bytearray(path.read_bytes())
    if damage == "flip_byte":
        blob[len(blob) // 2] ^= 0xFF
    elif damage == "truncate":
        blob = blob[:len(blob) // 2]
    elif damage == "wrong_magic":
        blob[0] ^= 0xFF
    else:
        blob = bytearray()
    path.write_bytes(bytes(blob))
    with pytest.raises(CheckpointError):
        read_checkpoint(path)
    with pytest.raises(CheckpointError):
        execute(tiny_config(tmp_path.parent, "", "ft", resume=str(tmp_path)))


def test_missing_checkpoint_and_wrong_structure_are_rejected(tmp_path):
    with pytest.raises(ResumeError, match="no checkpoint"):
        run_outputs.resolve_resume(tmp_path)
    params = small_state()
    path = save_checkpoint(tmp_path, params, None, {"run_uid": "x", "stages_completed": 1, "counters": {},
                                                    "fingerprint": {}})
    _, raw = read_checkpoint(path)
    bigger = ActorCritic(6, "relu", 2, True, False, 16, 1, True, True)
    other = MLPActorCriticPolicyCL(bigger, 10).init_params(jax.random.PRNGKey(0))
    with pytest.raises(CheckpointError):
        restore_state(raw["ego_params"], other, "ego parameters")
    with pytest.raises(CheckpointError):
        restore_state(raw["cl_state"], params, "continual-learning state")  # saved without CL state
    assert_trees_close(restore_state(raw["ego_params"], params, "ego parameters"), params)


def test_stale_temporary_file_does_not_hide_the_last_good_checkpoint(tmp_path):
    params = small_state()
    meta = {"run_uid": "x", "stages_completed": 1, "counters": {}, "fingerprint": {}}
    path = save_checkpoint(tmp_path, params, None, meta)
    (tmp_path / ".latest.ckpt.123.tmp").write_bytes(b"half written")
    assert read_checkpoint(path)[0]["stages_completed"] == 1
    save_checkpoint(tmp_path, params, None, dict(meta, stages_completed=2))
    assert read_checkpoint(path)[0]["stages_completed"] == 2


@pytest.mark.parametrize("method", ["mas", "ewc", "ft", "l2", "agem"])
def test_cl_state_round_trips_through_a_checkpoint(method, tmp_path):
    """MAS (and the other CL states) serialise and restore exactly, including after an importance update."""
    params = small_state()
    cfg = TrainConfig(cl_method=method, num_envs=4, num_steps=16, total_timesteps=64, regularize_critic=False)
    if method == "agem":
        cl_state = init_agem_memory(8, (10,))
    else:
        cl = dict(mas=MAS(), ewc=EWC(), ft=FT(), l2=L2())[method]
        cl_state = init_cl_state(params, False, False, cl, cfg)
        importance = jax.tree.map(lambda x: np.abs(np.asarray(x)) + 0.5, params)
        cl_state = cl.update_state(cl_state, params, importance)
    path = save_checkpoint(tmp_path, params, cl_state, {"run_uid": "x", "stages_completed": 1, "counters": {},
                                                        "fingerprint": {}})
    _, raw = read_checkpoint(path)
    restored = restore_state(raw["cl_state"], cl_state, "cl")
    assert type(restored) is type(cl_state)
    for a, b in zip(jax.tree.leaves(cl_state), jax.tree.leaves(restored)):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ----------------------------------------------------------------------------------------------------------
# recorder behaviour
# ----------------------------------------------------------------------------------------------------------

def make_events(stage, update, num_updates=4, partners=2, episodes=2):
    steps_per_update = 64
    train = build_train_record(stage, update, num_updates, steps_per_update, 3, 1.0, 5.0, 0.5, 2.0, 1.0,
                               0.1, 0.2, 0.3, 0.0, 0.4)
    train_log = build_train_log(stage, update, num_updates, 3, 1.0, 5.0, 0.2, 0.1, 0.3, 0.4,
                                env_steps=train["env_steps"])
    soups = np.ones((partners, episodes))
    ev, ev_log = build_eval_event(stage, update, 3, list(range(partners)), soups, soups * 2,
                                  np.ones((partners, episodes), bool), num_updates, steps_per_update)
    return train, train_log, ev, ev_log


def test_recorder_commits_in_update_order_whatever_the_arrival_order(tmp_path):
    logged = []
    recorder = RunRecorder(tmp_path, ["a", "b"], seed=0, wandb_log=lambda d, step: logged.append(step))
    recorder.begin_stage(0, 4, {1: "full", 3: "full", 4: "full"})
    events = []
    for u in (1, 2, 3, 4):
        train, train_log, ev, ev_log = make_events(0, u)
        events.append(("train", train, train_log))
        if u in (1, 3, 4):
            events.append(("eval", ev, ev_log))
    for index in (5, 2, 0, 3, 6, 1, 4):  # scrambled delivery
        recorder.handle(*events[index])
    summary = recorder.end_stage()
    recorder.close()
    assert logged == [64, 128, 192, 256]
    assert [int(r["update"]) for r in read_rows(tmp_path / "train_metrics.csv")] == [1, 2, 3, 4]
    assert summary["eval_events"] == 3 and summary["updates"] == 4
    ids = [r["row_id"] for r in read_rows(tmp_path / "eval_metrics.csv")]
    assert len(ids) == len(set(ids)) == 3 * 2 * 2


def test_recorder_refuses_incomplete_or_duplicate_stages(tmp_path):
    recorder = RunRecorder(tmp_path, ["a", "b"], seed=0)
    recorder.begin_stage(0, 2, {1: "full", 2: "full"})
    train, train_log, ev, ev_log = make_events(0, 1, num_updates=2)
    recorder.handle("train", train, train_log)  # evaluation of update 1 never arrives
    with pytest.raises(RunOutputError, match="incomplete"):
        recorder.end_stage()

    recorder.begin_stage(0, 2, {1: "full", 2: "full"})
    recorder.handle("train", train, train_log)
    recorder.handle("train", train, train_log)
    with pytest.raises(RunOutputError, match="duplicate"):
        recorder.end_stage()
    recorder.close()


def test_wandb_failure_keeps_local_records_and_is_reported(tmp_path):
    def broken(data, step):
        raise ConnectionError("offline")
    recorder = RunRecorder(tmp_path, ["a", "b"], seed=0, wandb_log=broken)
    recorder.begin_stage(0, 1, {1: "full"})
    train, train_log, ev, ev_log = make_events(0, 1, num_updates=1)
    recorder.handle("train", train, train_log)
    recorder.handle("eval", ev, ev_log)
    recorder.end_stage()
    recorder.close()
    assert recorder.wandb_failures == 1 and "ConnectionError" in recorder.wandb_error
    assert len(read_rows(tmp_path / "train_metrics.csv")) == 1 and len(read_rows(tmp_path / "eval_metrics.csv")) == 4


def test_truncate_csv_drops_only_uncommitted_stages(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("stage,x\n0,1\n0,2\n1,3\n2,4\n")
    assert truncate_csv(path, 1) == 1
    assert path.read_text() == "stage,x\n0,1\n0,2\n1,3\n"
    assert truncate_csv(path, 1) == 0


def test_metric_helpers():
    assert shaping_coefficient(1000, 1, 64) == 1.0
    assert shaping_coefficient(1000, 2, 64) == pytest.approx(1 - 64 / 1000)
    assert shaping_coefficient(0, 1, 64) == 0.0
    record = build_train_record(0, 1, 2, 64, 0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0)
    assert record["soups"] is None and record["ego_return"] is None  # no completed episode: blank, not zero
