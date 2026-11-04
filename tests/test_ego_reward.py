import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.partner_adaptation.run_br import TrainConfig, execute, resolve_run
from experiments.partner_adaptation.train_ego import ego_sparse_reward

POP_DIR = (Path(__file__).resolve().parent.parent
           / "experiments/partner_adaptation/partner_agents/BRDiv_population/cramped_room")
SPU = 4 * 16


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    path = tmp_path_factory.mktemp("bank") / "bank.json"
    path.write_text(json.dumps({
        "format_version": 1, "name": "tiny", "layout": "cramped_room", "num_partners": 1,
        "populations": {"pop": {"config": str(POP_DIR / "config.pckl"), "seed_index": 0}},
        "partners": [{"partner_id": 0, "population": "pop", "member_index": 0,
                      "checkpoint": str(POP_DIR / "params_seed0_agent0.pt")}]}))
    return str(path)


def config(root, bank, **overrides):
    settings = dict(
        mode="disabled", checkpoint_path=str(root), layout_name="cramped_room", partner_bank=bank,
        num_heuristic_partners=0, num_envs=4, num_steps=16, total_timesteps=SPU, update_epochs=1,
        num_minibatches=2, num_eval_episodes=1, hidden_size=16, num_layers=1, seed=0, use_task_id=False,
        use_multihead=False, eval_schedule="pilot", reward_shaping_horizon=SPU, record_video=False)
    settings.update(overrides)
    return TrainConfig(**settings)


def test_partner_delivery_rewards_the_ego_under_team_reward():
    reward = {"agent_0": jnp.array([0.0, 20.0, 0.0]), "agent_1": jnp.array([20.0, 0.0, 0.0])}
    np.testing.assert_array_equal(ego_sparse_reward(reward), [20.0, 20.0, 0.0])
    np.testing.assert_array_equal(ego_sparse_reward(reward, "team"), [20.0, 20.0, 0.0])
    np.testing.assert_array_equal(ego_sparse_reward(reward, "own"), [0.0, 20.0, 0.0])
    with pytest.raises(ValueError, match="ego_reward"):
        ego_sparse_reward(reward, "shared")


def test_default_is_team_and_invalid_mode_is_rejected(bank, tmp_path):
    assert TrainConfig().ego_reward == "team"
    with pytest.raises(ValueError, match="ego_reward must be"):
        resolve_run(config(tmp_path, bank, ego_reward="shared"))


def test_ego_reward_is_part_of_the_resume_fingerprint(bank, tmp_path):
    team = resolve_run(config(tmp_path, bank)).fingerprint["config"]
    own = resolve_run(config(tmp_path, bank, ego_reward="own")).fingerprint["config"]
    assert team["ego_reward"] == "team" and own["ego_reward"] == "own"


@pytest.mark.parametrize("mode", ["team", "own"])
def test_tiny_run_trains_with_either_reward(mode, bank, tmp_path):
    result = execute(config(tmp_path, bank, ego_reward=mode))
    assert json.loads((result.run_dir / "status.json").read_text())["state"] == "complete"
    assert json.loads((result.run_dir / "config.json").read_text())["ego_reward"] == mode
