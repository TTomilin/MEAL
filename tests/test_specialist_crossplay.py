"""Specialist cross-play: exported egos against every bank partner (CPU only)."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.partner_adaptation.run_br import TrainConfig, execute
from experiments.results.specialist_crossplay import EPISODE_STEPS, main

LAYOUT = "cramped_room"
IDS = ("P01", "P02")


def write_bank(path: Path) -> Path:
    partners = [{"partner_id": i, "kind": "planner", "planner_id": pid} for i, pid in enumerate(IDS)]
    path.write_text(json.dumps({
        "format_version": 2, "name": "tiny_planners", "layout": LAYOUT, "num_partners": len(partners),
        "populations": {}, "partners": partners, "heldout": [],
    }))
    return path


def train_specialist(tmp_path: Path, bank: Path, stage: int) -> Path:
    spu = 4 * 16
    cfg = TrainConfig(
        mode="disabled", checkpoint_path=str(tmp_path / f"runs_s{stage}"), layout_name=LAYOUT,
        partner_bank=str(bank), num_heuristic_partners=0, num_envs=4, num_steps=16, total_timesteps=spu,
        update_epochs=1, num_minibatches=2, num_eval_episodes=1, hidden_size=16, num_layers=1, seed=0,
        reward_shaping_horizon=spu, aht_eval=False, record_video=False, only_stages=[stage],
        eval_schedule="pilot")
    return execute(cfg).run_dir


@pytest.fixture(scope="module")
def specialists(tmp_path_factory):
    root = tmp_path_factory.mktemp("specialist_xp")
    bank = write_bank(root / "bank.json")
    runs = [train_specialist(root, bank, i) for i in range(2)]
    return bank, runs


def test_matrix_shape_order_and_behaviour_sums(specialists, tmp_path):
    bank, runs = specialists
    out = tmp_path / "out"
    main(["--bank", str(bank), "--runs", *[str(r) for r in runs], "--episodes", "2", "--seed", "0",
          "--env-dynamics", "aht", "--out", str(out)])
    labels = [f"{LAYOUT}/{pid}" for pid in IDS]
    with open(out / "xp_matrix.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["specialist"] + labels
    assert [r[0] for r in rows[1:]] == labels
    assert len(rows) == 3 and all(len(r) == 3 for r in rows)
    with open(out / "xp_matrix_std.csv") as f:
        std_rows = list(csv.reader(f))
    assert [r[0] for r in std_rows[1:]] == labels

    data = np.load(out / "partner_behaviour.npz")
    assert list(data["labels"]) == labels
    assert data["action_counts"].shape == (2, 2, 6)
    assert data["visits"].shape[0:2] == (2, 2)
    assert data["pot_interactions"].shape[0:2] == (2, 2)
    assert int(data["episodes"]) == 2
    np.testing.assert_array_equal(data["action_counts"].sum(axis=-1), 2 * EPISODE_STEPS)


def test_missing_specialist_lists_labels(specialists, tmp_path):
    bank, runs = specialists
    with pytest.raises(SystemExit, match=r"no specialist.*cramped_room/P02"):
        main(["--bank", str(bank), "--runs", str(runs[0]), "--episodes", "2", "--out", str(tmp_path / "out")])


def test_heterogeneous_specialist_architectures_and_legacy_config_crossplay(tmp_path):
    bank = write_bank(tmp_path / "bank.json")
    spu = 4 * 16
    cfg0 = TrainConfig(
        mode="disabled", checkpoint_path=str(tmp_path / "s0"), layout_name=LAYOUT,
        partner_bank=str(bank), num_heuristic_partners=0, num_envs=4, num_steps=16, total_timesteps=spu,
        update_epochs=1, num_minibatches=2, num_eval_episodes=1, hidden_size=16, num_layers=1, seed=0,
        reward_shaping_horizon=spu, aht_eval=False, record_video=False, only_stages=[0],
        eval_schedule="pilot")
    cfg1 = TrainConfig(
        mode="disabled", checkpoint_path=str(tmp_path / "s1"), layout_name=LAYOUT,
        partner_bank=str(bank), num_heuristic_partners=0, num_envs=4, num_steps=16, total_timesteps=spu,
        update_epochs=1, num_minibatches=2, num_eval_episodes=1, hidden_size=32, num_layers=2, seed=0,
        reward_shaping_horizon=spu, aht_eval=False, record_video=False, only_stages=[1],
        eval_schedule="pilot")
    r0 = execute(cfg0).run_dir
    r1 = execute(cfg1).run_dir
    # Simulate a legacy config.json missing a defaulted architecture field.
    c0_path = r0 / "config.json"
    c0 = json.loads(c0_path.read_text())
    c0.pop("use_layer_norm")
    c0_path.write_text(json.dumps(c0))
    out = tmp_path / "out_het"
    main(["--bank", str(bank), "--runs", str(r0), str(r1), "--episodes", "1", "--seed", "0",
          "--env-dynamics", "aht", "--out", str(out)])
    with open(out / "xp_matrix.csv") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3 and all(len(r) == 3 for r in rows)
