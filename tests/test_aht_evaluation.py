import csv
import json

import jax
import numpy as np
import pytest

from experiments.partner_adaptation import run_br
from experiments.partner_adaptation.jax_aht_bridge.hf_partners import TeammateSpec, export_population
from experiments.partner_adaptation.jax_aht_bridge.generators import bank_entries
from experiments.partner_adaptation.jax_aht_bridge.policies import make_jaxaht_policy
from experiments.partner_adaptation.partner_bank import write_bank_v2
from experiments.partner_adaptation.run_outputs import ResumeError


@pytest.fixture
def bank(tmp_path):
    policy = make_jaxaht_policy("mlp", 5 * 5 * 26)
    params = [policy.init_params(jax.random.fold_in(jax.random.PRNGKey(0), n)) for n in range(3)]
    spec = TeammateSpec(set_name="meal", layout="coord_ring", name="test", actor_type="mlp", run_path="test",
                        ckpt_key="checkpoints", idx_list=[0, 1, 2], pop_size=1, test_mode=False,
                        performance_bounds={"base_return": [[0, 100]] * 3})
    gen = export_population(spec, params, tmp_path / "pop", source="test", run="test")
    entries = bank_entries(gen, "pop")
    path = tmp_path / "bank.json"
    write_bank_v2(str(path), "coord_ring", entries[:2], entries[2:], {"pop": gen})
    return path


def config(root, bank, **overrides):
    values = dict(mode="disabled", checkpoint_path=str(root), partner_bank=str(bank), layout_name="coord_ring",
                  num_heuristic_partners=0, num_envs=2, num_steps=4, total_timesteps=24, update_epochs=1,
                  num_minibatches=1, num_eval_episodes=1, eval_every=1, aht_eval_episodes=1,
                  hidden_size=8, num_layers=1, seed=0, cl_method="ft", reward_shaping_horizon=0)
    values.update(overrides)
    return run_br.TrainConfig(**values)


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def assert_equal_params(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_every_evaluation_and_crash_resume_are_checkpoint_aligned(bank, tmp_path, monkeypatch):
    every = dict(aht_heldout_eval="every_event")
    reference = run_br.execute(config(tmp_path / "reference", bank, **every))
    aht = rows(reference.run_dir / "aht_eval_metrics.csv")
    regular = rows(reference.run_dir / "eval_metrics.csv")
    assert {int(r["env_steps"]) for r in aht} == {0} | {int(r["env_steps"]) for r in regular}
    assert {int(r["event"]) for r in aht} == set(range(7))
    assert len(aht) == 7 * 3
    assert len({(r["event"], r["group"], r["index"]) for r in aht}) == len(aht)

    save = run_br.save_checkpoint

    def interrupt(directory, ego, cl, meta):
        if meta["stages_completed"] == 2:
            raise KeyboardInterrupt("after AHT evaluation, before committing stage 2")
        return save(directory, ego, cl, meta)

    monkeypatch.setattr(run_br, "save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_br.execute(config(tmp_path / "interrupted", bank, **every))
    directory, = (tmp_path / "interrupted").iterdir()
    assert len(rows(directory / "aht_eval_metrics.csv")) == len(aht)
    monkeypatch.setattr(run_br, "save_checkpoint", save)
    resumed = run_br.execute(config(tmp_path / "interrupted", bank, resume=str(directory), **every))
    assert rows(directory / "aht_eval_metrics.csv") == aht
    assert_equal_params(resumed.ego_params, reference.ego_params)
    assert resumed.counters == reference.counters

    before = (directory / "aht_eval_metrics.csv").read_bytes()
    run_br.execute(config(tmp_path / "interrupted", bank, resume=str(directory), **every))
    assert (directory / "aht_eval_metrics.csv").read_bytes() == before
    assert not (directory / run_br.AHT_FINAL_CSV).exists()


def test_heldout_test_set_is_evaluated_once_after_training(bank, tmp_path, monkeypatch):
    result = run_br.execute(config(tmp_path / "run", bank))
    aht = rows(result.run_dir / "aht_eval_metrics.csv")
    assert {r["group"] for r in aht} == {"train"} and len(aht) == 7 * 2
    final = rows(result.run_dir / run_br.AHT_FINAL_CSV)
    assert [(r["group"], r["index"], r["head"]) for r in final] == [("heldout", "0", "1")]
    assert int(final[0]["env_steps"]) == result.counters["train_env_steps"] and final[0]["norm_return"] != ""
    norm = json.loads((result.run_dir / "run.json").read_text())["normalization"]
    assert "heldout_metric" not in norm["aht"] and "subset" in norm["aht_final_heldout"]["heldout_metric"]

    logged = {}
    evaluator = run_br.run_aht_eval

    def logging_eval(*args, **kwargs):
        out = evaluator(*args, **kwargs)
        logged.update(out["metrics"])
        return out

    monkeypatch.setattr(run_br, "run_aht_eval", logging_eval)
    save = run_br.save_checkpoint

    def interrupt(directory, ego, cl, meta):
        if meta["stages_completed"] == 2:
            raise KeyboardInterrupt("before the final evaluation")
        return save(directory, ego, cl, meta)

    monkeypatch.setattr(run_br, "save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_br.execute(config(tmp_path / "broken", bank))
    directory, = (tmp_path / "broken").iterdir()
    assert not (directory / run_br.AHT_FINAL_CSV).exists() and not any(k.startswith("Test/") for k in logged)
    monkeypatch.setattr(run_br, "save_checkpoint", save)
    run_br.execute(config(tmp_path / "broken", bank, resume=str(directory)))
    assert rows(directory / run_br.AHT_FINAL_CSV) == final
    assert set(logged) >= {"Test/Heldout/Mean", "Test/Heldout/NormReturn_Mean"}
    assert not any(k.startswith("Eval/Heldout") for k in logged)
    run_br.execute(config(tmp_path / "broken", bank, resume=str(directory)))
    assert rows(directory / run_br.AHT_FINAL_CSV) == final


def test_heldout_eval_mode_is_validated_and_fingerprinted(bank, tmp_path):
    with pytest.raises(ValueError, match="aht_heldout_eval"):
        run_br.execute(config(tmp_path / "bad", bank, aht_heldout_eval="sometimes"))
    result = run_br.execute(config(tmp_path / "run", bank))
    with pytest.raises(ResumeError, match="aht_evaluation"):
        run_br.execute(config(tmp_path / "run", bank, aht_heldout_eval="every_event", resume=str(result.run_dir)))


def test_changed_heldout_is_rejected_before_run_mutation(bank, tmp_path):
    result = run_br.execute(config(tmp_path / "run", bank))
    manifest = json.loads(bank.read_text())
    manifest["heldout"] = []
    bank.write_text(json.dumps(manifest))
    before = {p.relative_to(result.run_dir): p.read_bytes() for p in result.run_dir.rglob("*") if p.is_file()}
    with pytest.raises(ResumeError, match="heldout"):
        run_br.execute(config(tmp_path / "run", bank, resume=str(result.run_dir)))
    assert {p.relative_to(result.run_dir): p.read_bytes() for p in result.run_dir.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("schedule", ["pilot", "interleaved"])
def test_pilot_and_interleaved_aht_events(bank, tmp_path, schedule, monkeypatch):
    extra = dict(eval_schedule="pilot", eval_current_every=1)
    if schedule == "interleaved":
        extra.update(schedule="interleaved", interleave_updates=1, eval_current_every=0)
    result = run_br.execute(config(tmp_path / schedule, bank, **extra))
    aht = rows(result.run_dir / "aht_eval_metrics.csv")
    regular = rows(result.run_dir / "eval_metrics.csv")
    assert {int(r["env_steps"]) for r in aht} == {int(r["env_steps"]) for r in regular}
    assert len({(r["event"], r["group"], r["index"]) for r in aht}) == len(aht)
    if schedule == "interleaved":
        save = run_br.save_checkpoint

        def interrupt(directory, ego, cl, meta):
            if meta["stages_completed"] == 4:
                raise KeyboardInterrupt("after second cycle evaluation")
            return save(directory, ego, cl, meta)

        monkeypatch.setattr(run_br, "save_checkpoint", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_br.execute(config(tmp_path / "broken", bank, **extra))
        directory, = (tmp_path / "broken").iterdir()
        monkeypatch.setattr(run_br, "save_checkpoint", save)
        resumed = run_br.execute(config(tmp_path / "broken", bank, resume=str(directory), **extra))
        assert rows(directory / "aht_eval_metrics.csv") == aht
        assert_equal_params(resumed.ego_params, result.ego_params)
        assert resumed.counters == result.counters


def test_aht_evaluation_does_not_change_training(bank, tmp_path):
    enabled = run_br.execute(config(tmp_path / "enabled", bank))
    disabled = run_br.execute(config(tmp_path / "disabled", bank, aht_eval=False))
    assert_equal_params(enabled.ego_params, disabled.ego_params)
    assert not (disabled.run_dir / "aht_eval_metrics.csv").exists()
    assert not (disabled.run_dir / run_br.AHT_FINAL_CSV).exists()