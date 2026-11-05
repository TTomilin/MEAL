"""TrainConfig.stage_order: custom / returning partner sequences (CPU only)."""
import json

import pytest

from experiments.partner_adaptation import run_br
from experiments.partner_adaptation.run_br import execute, resolve_run
from experiments.partner_adaptation.run_outputs import read_checkpoint
from tests.test_run_outputs import assert_rows_match, read_rows, tiny_config, write_bank


@pytest.fixture
def fake_wandb(monkeypatch):
    runs = []

    class FakeWandb:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name")
            self.logged = []
            self.finished = False

        def log(self, data, step=None):
            self.logged.append((step, data))

        def finish(self):
            self.finished = True

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


def cfg(root, bank, **overrides):
    return tiny_config(root, bank, cl_method="ft", eval_schedule="pilot", record_video=False, **overrides)


def test_empty_stage_order_omitted_from_fingerprint(bank, tmp_path):
    assert "stage_order" not in resolve_run(cfg(tmp_path, bank)).fingerprint["config"]
    assert resolve_run(cfg(tmp_path, bank, stage_order=[0, 1])).fingerprint["config"]["stage_order"] == [0, 1]


def test_empty_matches_explicit_bank_order(bank, tmp_path, fake_wandb):
    empty = execute(cfg(tmp_path / "empty", bank))
    explicit = execute(cfg(tmp_path / "explicit", bank, stage_order=[0, 1]))
    for name in ("train_metrics.csv", "eval_metrics.csv"):
        assert_rows_match(read_rows(empty.run_dir / name), read_rows(explicit.run_dir / name))


def test_returning_partner_stage_and_partner_columns(bank, tmp_path, fake_wandb):
    run = execute(cfg(tmp_path, bank, stage_order=[0, 1, 0]))
    train = read_rows(run.run_dir / "train_metrics.csv")
    assert [int(r["stage"]) for r in train] == [0, 0, 1, 1, 2, 2]
    assert [int(r["partner_id"]) for r in train] == [0, 0, 1, 1, 0, 0]
    steps = [int(r["env_steps"]) for r in train]
    assert steps == sorted(set(steps)) and steps == sorted(steps)
    evals = read_rows(run.run_dir / "eval_metrics.csv")
    assert len({r["event_id"] for r in evals}) == len({(r["env_steps"], r["stage"], r["update"]) for r in evals})
    assert len({r["row_id"] for r in evals}) == len(evals)
    status = json.loads((run.run_dir / "status.json").read_text())
    assert status["state"] == "complete" and status["stages_completed"] == 3

    stage0 = [r for r in train if int(r["stage"]) == 0]
    stage2 = [r for r in train if int(r["stage"]) == 2]
    assert len(stage0) == len(stage2) == 2
    assert any(stage0[i][k] != stage2[i][k] for i in range(2) for k in ("soups", "actor_loss", "ego_return"))


def test_resume_after_two_of_three_matches_uninterrupted(bank, tmp_path, fake_wandb):
    reference = execute(cfg(tmp_path / "ref", bank, stage_order=[0, 1, 0]))
    config = cfg(tmp_path / "resume", bank, stage_order=[0, 1, 0])
    patch = pytest.MonkeyPatch()
    real = run_br.run_br_training

    def failing(*args, **kwargs):
        if kwargs.get("key_index") == 2:
            raise RuntimeError("simulated crash")
        return real(*args, **kwargs)

    patch.setattr(run_br, "run_br_training", failing)
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            execute(config)
    finally:
        patch.undo()
    run_dir, = [p for p in (tmp_path / "resume").iterdir() if p.is_dir()]
    assert read_checkpoint(run_dir / "latest.ckpt")[0]["stages_completed"] == 2

    resumed = execute(cfg(tmp_path / "resume", bank, stage_order=[0, 1, 0], resume=str(run_dir)))
    for name in ("train_metrics.csv", "eval_metrics.csv"):
        assert_rows_match(read_rows(resumed.run_dir / name), read_rows(reference.run_dir / name))
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "complete" and status["stages_completed"] == 3
    assert status["attempts"][1]["resumed_from_stage"] == 2


@pytest.mark.parametrize("overrides, message", [
    (dict(stage_order=[0, 2]), "out of range"),
    (dict(stage_order=[0, 0]), "never trains"),
    (dict(stage_order=[0, 1], schedule="interleaved", interleave_updates=1, reward_shaping_horizon=200),
     "interleaved"),
    (dict(stage_order=[0, 1], only_stages=[0]), "only_stages"),
])
def test_stage_order_validation_errors(overrides, message, bank, tmp_path):
    with pytest.raises(ValueError, match=message):
        resolve_run(cfg(tmp_path, bank, **overrides))
