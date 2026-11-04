import json
from pathlib import Path

import jax
import numpy as np
import pytest
import wandb

from experiments.partner_adaptation.partner_agents.agent_interface import ActorWithConditionalCriticPolicy
from experiments.partner_adaptation.partner_bank import assemble_bank, load_partners
from experiments.partner_adaptation.partner_generation.LBRDiv import (
    _lbrdiv_role_observations,
    run_lbrdiv,
)
from experiments.partner_adaptation.partner_generation.run import TrainConfig, generate_population
from experiments.partner_adaptation.partner_generation.population_check import check_population
from meal import make_env
from meal.env.overcooked.layouts.presets import overcooked_layouts


def test_lbrdiv_role_observations_match_deployment_seats():
    obs = {
        "agent_0": np.full((2, 2, 3), 4, dtype=np.float32),
        "agent_1": np.full((2, 2, 3), 9, dtype=np.float32),
    }

    conf_obs, br_obs = _lbrdiv_role_observations(obs, num_envs=2)

    np.testing.assert_array_equal(conf_obs, obs["agent_1"].reshape(2, -1))
    np.testing.assert_array_equal(br_obs, obs["agent_0"].reshape(2, -1))


def test_lbrdiv_rejects_single_member_population():
    config = TrainConfig(alg="lbrdiv", partner_pop_size=1)
    with pytest.raises(ValueError, match="partner_pop_size >= 2"):
        run_lbrdiv(config)


def test_algorithm_specific_ppo_defaults_respect_overrides():
    lbrdiv = TrainConfig(alg="lbrdiv")
    assert (lbrdiv.update_epochs, lbrdiv.num_minibatches, lbrdiv.clip_eps, lbrdiv.ent_coef) == (15, 4, 0.1, 0.03)
    overridden = TrainConfig(alg="lbrdiv", update_epochs=2, num_minibatches=1, clip_eps=0.2, ent_coef=0.01)
    assert (overridden.update_epochs, overridden.num_minibatches, overridden.clip_eps, overridden.ent_coef) == (
        2, 1, 0.2, 0.01
    )
    brdiv = TrainConfig(alg="brdiv")
    assert (brdiv.update_epochs, brdiv.num_minibatches, brdiv.clip_eps, brdiv.ent_coef) == (8, 16, 0.2, 0.01)


def test_tiny_lbrdiv_generation_and_bank_round_trip(tmp_path, monkeypatch):
    config = TrainConfig(
        alg="lbrdiv",
        mode="disabled",
        layout_name="coord_ring",
        seed=17,
        partner_pop_size=2,
        num_envs_xp=1,
        num_envs_sp=1,
        num_steps=16,
        total_timesteps=64,
        update_epochs=1,
        num_minibatches=1,
        num_checkpoints=1,
        num_eval_episodes=1,
        checkpoint_path=str(tmp_path / "generation"),
    )
    captured = {}
    from experiments.partner_adaptation.partner_generation import run as generation_run

    real_run_lbrdiv = generation_run.run_lbrdiv

    def capture_output(cfg):
        params, population, out = real_run_lbrdiv(cfg)
        captured["out"] = out
        return params, population, out

    monkeypatch.setattr(generation_run, "run_lbrdiv", capture_output)
    try:
        generation_dir = Path(generate_population(config))
    finally:
        wandb.finish()

    assert sorted(p.name for p in generation_dir.glob("*.pt")) == [
        "params_seed0_agent0.pt",
        "params_seed0_agent1.pt",
        "params_seed0_br0.pt",
        "params_seed0_br1.pt",
    ]
    generation = json.loads((generation_dir / "generation.json").read_text())
    assert generation["alg"] == "lbrdiv"
    assert generation["confederate_seat"] == "agent_1"
    assert [member["file"] for member in generation["br_members"]] == [
        "params_seed0_br0.pt",
        "params_seed0_br1.pt",
    ]
    report = check_population(generation_dir, "coord_ring", episodes=1, steps=16, seed=23)
    assert report.sp_xp_matrix.shape == (2, 2)
    assert report.passed is not None and not report.passed.all()

    out = captured["out"]
    assert jax.tree.leaves(out["final_params_conf"])[0].shape[:2] == (1, 2)
    assert jax.tree.leaves(out["final_params_br"])[0].shape[:2] == (1, 2)
    for name in ("lms_vertical", "lms_horizontal"):
        multipliers = np.asarray(out["metrics"][name])
        assert np.isfinite(multipliers).all()
        assert (multipliers >= 0).all()
        np.testing.assert_allclose(multipliers[..., np.arange(2), np.arange(2)], 0.5)

    bank = assemble_bank(
        "coord_ring",
        str(tmp_path / "coord_ring_lbrdiv_bank.json"),
        [str(generation_dir)],
        bundled=False,
    )
    loaded = load_partners(bank, int(np.prod(make_env(
        "overcooked", layout=overcooked_layouts["coord_ring"]
    ).observation_space().shape)))
    assert len(loaded) == 2
    assert all(isinstance(partner.policy, ActorWithConditionalCriticPolicy) for partner in loaded)
    assert [partner.record.member_index for partner in loaded] == [0, 1]


def test_population_check_degrades_gracefully_for_bundled_members():
    generation_dir = (
        Path(__file__).resolve().parents[1]
        / "experiments/partner_adaptation/partner_agents/BRDiv_population/coord_ring"
    )
    report = check_population(generation_dir, "coord_ring", episodes=1, steps=8, seed=29)

    assert report.passed is None
    assert np.isnan(report.sp_xp_matrix).all()
    assert report.solo.shape == (3,)