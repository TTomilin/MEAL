"""`--env-dynamics meal|aht`: training env for CPA (CPU only)."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.partner_adaptation.run_br import TrainConfig, execute, make_train_env, resolve_run
from experiments.partner_adaptation.run_outputs import build_fingerprint
from meal import make_env
from meal.env.overcooked.layouts.presets import overcooked_layouts
from meal.env.overcooked.overcooked_env import Overcooked
from meal.wrappers.logging import LogWrapper

STAY = 4
# Fingerprint sha256 of the tiny meal config below, computed on this branch before env_dynamics existed.
PRE_ENV_DYNAMICS_MEAL_FP_SHA256 = "f75837ffa6e35a4382a06cc3eef0b07f4d6d1f9c2f72fa1506c3ec25fdff7062"


def write_planner_bank(path: Path, layout="cramped_room", planner_id="P01"):
    path.write_text(json.dumps({
        "format_version": 2, "name": "tiny_planner", "layout": layout, "num_partners": 1,
        "populations": {}, "partners": [{"partner_id": 0, "kind": "planner", "planner_id": planner_id}],
        "heldout": [],
    }))
    return str(path)


def write_bundled_planner_bank(path: Path):
    """One bundled BRDiv member + P01 — same shape as the pre-change fingerprint capture."""
    from experiments.partner_adaptation.partner_bank import assemble_bank
    assemble_bank("cramped_room", str(path), [], bundled=True, planners=("P01",))
    return str(path)


def tiny_config(root, bank, **overrides):
    settings = dict(
        mode="disabled", checkpoint_path=str(root), layout_name="cramped_room", partner_bank=bank,
        num_heuristic_partners=0, num_envs=4, num_steps=16, total_timesteps=2 * 4 * 16, update_epochs=1,
        num_minibatches=2, num_eval_episodes=2, eval_every=2, hidden_size=16, num_layers=1,
        importance_episodes=1, importance_steps=8, cl_method="ft", reg_coef=1.0, seed=0,
        reward_shaping_horizon=1e3, aht_eval=False, record_video=False)
    settings.update(overrides)
    return TrainConfig(**settings)


def _layout_on(config, layout_name="coord_ring"):
    config.layout = {"layout": overcooked_layouts[layout_name]}
    return config


def test_default_env_dynamics_is_meal_overcooked():
    assert TrainConfig().env_dynamics == "meal"
    cfg = _layout_on(TrainConfig(layout_name="coord_ring", num_steps=16))
    env = make_train_env(cfg, "coord_ring")
    assert isinstance(env, LogWrapper)
    assert type(env._env) is Overcooked
    assert type(env._env) is type(make_env("overcooked", layout=overcooked_layouts["coord_ring"], max_steps=16))


def test_aht_resets_randomise_agent_positions_on_coord_ring():
    cfg = _layout_on(TrainConfig(layout_name="coord_ring", num_steps=400, env_dynamics="aht"))
    env = make_train_env(cfg, "coord_ring")
    keys = jax.random.split(jax.random.PRNGKey(0), 32)
    _, states = jax.vmap(env.reset)(keys)
    pairs = [tuple(map(tuple, np.asarray(states.env_state.agent_pos[i]))) for i in range(32)]
    assert len(set(pairs)) >= 2


def test_aht_autoreset_also_randomises_start_positions():
    cfg = _layout_on(TrainConfig(layout_name="coord_ring", num_steps=3, env_dynamics="aht"))
    env = make_train_env(cfg, "coord_ring")
    keys = jax.random.split(jax.random.PRNGKey(1), 32)

    def run(key):
        obs, state = env.reset(key)

        def body(carry, _):
            obs, state, key = carry
            key, step_key = jax.random.split(key)
            actions = {a: jnp.asarray(STAY, jnp.int32) for a in env.agents}
            obs, state, _, done, _ = env.step(step_key, state, actions)
            return (obs, state, key), done["__all__"]

        (obs, state, key), dones = jax.lax.scan(body, (obs, state, key), None, 3)
        return state.env_state.agent_pos, dones[-1]

    positions, done = jax.vmap(run)(keys)
    assert bool(done.all())
    pairs = [tuple(map(tuple, np.asarray(positions[i]))) for i in range(32)]
    assert len(set(pairs)) >= 2


def test_meal_fingerprint_matches_pre_env_dynamics(tmp_path):
    bank = write_bundled_planner_bank(tmp_path / "bank.json")
    cfg = tiny_config(tmp_path / "runs", bank)
    assert cfg.env_dynamics == "meal"
    resolved = resolve_run(cfg)
    assert "env_dynamics" not in resolved.fingerprint["config"]

    # Expected: same as a config dict that never had the field (pre-change resolve_run).
    config_dict = asdict(cfg)
    config_dict.pop("env_dynamics")
    config_dict.pop("schedule"), config_dict.pop("interleave_updates")
    config_dict.pop("stage_order")
    expected = build_fingerprint(config_dict, resolved.identities)
    assert resolved.fingerprint == expected
    digest = hashlib.sha256(json.dumps(resolved.fingerprint, sort_keys=True, default=str).encode()).hexdigest()
    assert digest == PRE_ENV_DYNAMICS_MEAL_FP_SHA256


def test_aht_fingerprint_differs_from_meal(tmp_path):
    bank = write_bundled_planner_bank(tmp_path / "bank.json")
    meal = resolve_run(tiny_config(tmp_path / "meal", bank))
    aht = resolve_run(tiny_config(tmp_path / "aht", bank, env_dynamics="aht"))
    assert "env_dynamics" not in meal.fingerprint["config"]
    assert aht.fingerprint["config"]["env_dynamics"] == "aht"
    assert meal.fingerprint != aht.fingerprint


def test_invalid_env_dynamics_is_rejected(tmp_path):
    bank = write_planner_bank(tmp_path / "bank.json")
    with pytest.raises(ValueError, match="env_dynamics"):
        resolve_run(tiny_config(tmp_path / "bad", bank, env_dynamics="jaxmarl"))


def test_tiny_aht_run_records_env_dynamics(tmp_path):
    bank = write_planner_bank(tmp_path / "bank.json")
    result = execute(tiny_config(tmp_path / "runs", bank, env_dynamics="aht"))
    run = json.loads((result.run_dir / "run.json").read_text())
    assert run["env_dynamics"] == "aht"
    assert run["fingerprint"]["config"]["env_dynamics"] == "aht"
    assert json.loads((result.run_dir / "status.json").read_text())["state"] == "complete"


def test_tiny_meal_run_records_env_dynamics_outside_fingerprint(tmp_path):
    bank = write_planner_bank(tmp_path / "bank.json")
    result = execute(tiny_config(tmp_path / "runs", bank))
    run = json.loads((result.run_dir / "run.json").read_text())
    assert run["env_dynamics"] == "meal"
    assert "env_dynamics" not in run["fingerprint"]["config"]


def test_seat_mismatch_warning_for_jaxaht_confederates_under_meal_only(tmp_path, caplog):
    import pickle
    from experiments.partner_adaptation import run_br
    from experiments.partner_adaptation.partner_bank import write_bank_v2

    def make_bank(kind: str, actor_type: str, trained_seat: str) -> str:
        d = tmp_path / f"pop_{kind}_{actor_type}_{trained_seat}"
        d.mkdir()
        with open(d / "m0.pt", "wb") as f:
            pickle.dump({"actor_params": {"w": np.zeros((1,), np.float32)}}, f)
        gen = {
            "status": "complete", "kind": kind, "layout_name": "cramped_room", "actor_type": actor_type,
            "partner_pop_size": 3, "num_members": 1, "activation": "tanh", "fc_hidden_dim": 64,
            "test_mode": False, "source": "test", "trained_seat": trained_seat,
            "members": [{"member_index": 0, "file": "m0.pt", "performance_bounds": {}}],
        }
        gen_path = d / "generation.json"
        gen_path.write_text(json.dumps(gen))
        bank_path = d / "bank.json"
        write_bank_v2(str(bank_path), "cramped_room",
                      [{"kind": kind, "population": "p", "member_index": 0, "checkpoint": d / "m0.pt"}],
                      [], {"p": gen_path})
        return str(bank_path)

    conf_bank = make_bank("jaxaht", "actor_with_conditional_critic", "agent_0")
    with caplog.at_level("WARNING", logger=run_br.__name__):
        resolve_run(tiny_config(tmp_path / "r1", conf_bank, env_dynamics="meal"))
    assert any("--env-dynamics aht" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING", logger=run_br.__name__):
        resolve_run(tiny_config(tmp_path / "r2", conf_bank, env_dynamics="aht"))
    assert not any("--env-dynamics aht" in r.message for r in caplog.records)

    mlp_bank = make_bank("jaxaht", "mlp", "self-play")
    caplog.clear()
    with caplog.at_level("WARNING", logger=run_br.__name__):
        resolve_run(tiny_config(tmp_path / "r3", mlp_bank, env_dynamics="meal"))
    assert not any("--env-dynamics aht" in r.message for r in caplog.records)

    brdiv_bank = write_bundled_planner_bank(tmp_path / "brdiv_bank.json")
    caplog.clear()
    with caplog.at_level("WARNING", logger=run_br.__name__):
        resolve_run(tiny_config(tmp_path / "r4", brdiv_bank, env_dynamics="meal"))
    assert not any("--env-dynamics aht" in r.message for r in caplog.records)
