"""Tests for viable event-reward shaping generation and scripts/cpa_event_partners.sh."""
import json
import subprocess
from pathlib import Path

import pytest

from experiments.partner_adaptation.jax_aht_bridge import event_shaping as es

REPO = Path(__file__).resolve().parents[1]


def test_sampled_event_shaping_is_unique_deterministic_and_viable():
    a = es.sample_event_shaping(24, setting_seed=0)
    b = es.sample_event_shaping(24, setting_seed=0)
    c = es.sample_event_shaping(24, setting_seed=1)
    assert a == b
    assert a != c
    tuples = [tuple(sorted(w.items())) for w in a]
    assert len(set(tuples)) == 24
    for w in a:
        es.validate_shaping_weights(w)
        assert w["PLACEMENT_IN_POT_REW"] >= 0.0
        assert w["SOUP_PICKUP_REWARD"] >= 0.0
        assert es.net_soup_cycle_return(w) >= es.MIN_NET_SOUP_RETURN
        assert any(v != 0.0 for v in w.values())


def test_unviable_shaping_is_rejected():
    with pytest.raises(ValueError, match="all zero"):
        es.validate_shaping_weights({k: 0.0 for k in es.EVENT_SHAPING_KEYS})
    with pytest.raises(ValueError, match="PLACEMENT_IN_POT_REW"):
        es.validate_shaping_weights({"PLACEMENT_IN_POT_REW": -3.0})
    with pytest.raises(ValueError, match="SOUP_PICKUP_REWARD"):
        es.validate_shaping_weights({"SOUP_PICKUP_REWARD": -3.0})
    with pytest.raises(ValueError, match="net soup-cycle return"):
        es.validate_shaping_weights({"ONION_PICKUP_REWARD": -4.0, "PLATE_PICKUP_REWARD": -2.0})
    with pytest.raises(ValueError, match="unknown event shaping"):
        es.validate_shaping_weights({"NOT_A_KEY": 1.0})
    with pytest.raises(ValueError, match="n_settings"):
        es.sample_event_shaping(0)
    with pytest.raises(ValueError, match="viable combinations"):
        es.sample_event_shaping(10000)
    for bad_pair in ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0)):
        with pytest.raises(ValueError, match=r"COUNTER_DROP_REWARD \+ COUNTER_PICKUP_REWARD"):
            es.validate_shaping_weights({
                "COUNTER_DROP_REWARD": bad_pair[0], "COUNTER_PICKUP_REWARD": bad_pair[1],
            })
    for good_pair in ((1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
        es.validate_shaping_weights({
            "COUNTER_DROP_REWARD": good_pair[0], "COUNTER_PICKUP_REWARD": good_pair[1],
        })
    for non_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            es.validate_shaping_weights({"PLACEMENT_IN_POT_REW": non_finite})


def test_sampled_weights_accepted_by_meal_overcooked_aht():
    try:
        from experiments.partner_adaptation.jax_aht_bridge.env_adapter import MealOvercookedAHT
    except ImportError as exc:
        pytest.skip(f"jax-aht not available: {exc}")
    w = es.sample_event_shaping(1, setting_seed=0)[0]
    env = MealOvercookedAHT("coord_ring", do_reward_shaping=True, reward_shaping_params=w)
    for key, val in w.items():
        assert list(env.reward_shaping_params[key]) == [pytest.approx(val), pytest.approx(val)]


def test_cli_outputs_valid_json_and_hydra_overrides(capsys):
    es.main(["--k", "3", "--n-settings", "12", "--seed", "0"])
    payload = json.loads(capsys.readouterr().out)
    expected = es.sample_event_shaping(12, setting_seed=0)[2]
    assert payload["weights"] == expected
    assert len(payload["overrides"]) == len(es.EVENT_SHAPING_KEYS)
    assert all(ov.startswith("task.ENV_KWARGS.reward_shaping_params.") for ov in payload["overrides"])
    with pytest.raises(SystemExit):
        es.main(["--k", "13", "--n-settings", "12", "--seed", "0"])


def test_cpa_event_partners_script_syntax_and_single_seed_default():
    script = REPO / "scripts" / "cpa_event_partners.sh"
    res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    text = script.read_text()
    assert 'SEEDS="${SEEDS:-1}"' in text
    assert "experiments.partner_adaptation.jax_aht_bridge.event_shaping" in text


def test_counter_drop_pickup_cycle_never_yields_positive_shaped_reward_in_env():
    try:
        import jax
        import jax.numpy as jnp
        from experiments.partner_adaptation.jax_aht_bridge.env_adapter import MealOvercookedAHT
        from meal.env.overcooked.common import OBJECT_TO_INDEX
    except ImportError as exc:
        pytest.skip(f"jax-aht not available: {exc}")

    # cramped_room: agent_0 at (1, 1) facing up onto wall counter (1, 0) holding an onion.
    # Step 1: interact drops onion on counter; Step 2: interact picks onion back up.
    for w in es.sample_event_shaping(24, setting_seed=0):
        env = MealOvercookedAHT("cramped_room", random_reset=False, do_reward_shaping=True,
                                reward_shaping_params=w)
        _, state = env.reset(jax.random.PRNGKey(0))
        es_raw = state.env_state
        es_raw = es_raw.replace(
            agent_pos=jnp.array([[1, 1], [3, 2]], dtype=es_raw.agent_pos.dtype),
            agent_dir_idx=jnp.array([0, 0], dtype=es_raw.agent_dir_idx.dtype),
            agent_inv=jnp.array([OBJECT_TO_INDEX["onion"], OBJECT_TO_INDEX["empty"]], dtype=es_raw.agent_inv.dtype),
        )
        state = state.replace(env_state=es_raw)
        interact = {"agent_0": jnp.int32(5), "agent_1": jnp.int32(4)}
        _, state, r1, _, info1 = env.step(jax.random.PRNGKey(1), state, interact)
        _, state, r2, _, info2 = env.step(jax.random.PRNGKey(2), state, interact)
        assert int(state.env_state.agent_inv[0]) == OBJECT_TO_INDEX["onion"]
        cycle_reward = float(info1["shaped_reward"][0] + info2["shaped_reward"][0])
        assert cycle_reward <= 0.0, f"counter farming exploit ({cycle_reward}) for weights={w}"


def test_cpa_event_partners_runs_portably_from_any_cwd_and_propagates_failure(tmp_path):
    import os
    import sys

    script = REPO / "scripts" / "cpa_event_partners.sh"
    fake_aht = tmp_path / "fake_aht"
    (fake_aht / "marl").mkdir(parents=True)
    log_file = tmp_path / "invocations.log"
    (fake_aht / "marl" / "run.py").write_text(
        "import os, sys\n"
        f"with open({str(log_file)!r}, 'a') as f:\n"
        "    f.write(os.environ.get('PYTHONPATH', '') + '|' + ' '.join(sys.argv) + '\\n')\n"
        "if os.environ.get('FAIL_ME') == '1':\n"
        "    raise SystemExit(3)\n"
    )
    env = {
        **os.environ,
        "LAYOUT": "coord_ring",
        "N_SETTINGS": "2",
        "PARALLEL": "1",
        "MEAL_CPA_ROOT": str(tmp_path / "cpa"),
        "AHT_DIR": str(fake_aht),
        "PYTHON": sys.executable,
        "PYTHONPATH": "/custom/extra",
    }
    res = subprocess.run(["/bin/bash", str(script)], cwd=str(tmp_path), env=env,
                         capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    lines = log_file.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        pypath, argv = line.split("|", 1)
        assert "/custom/extra" in pypath and str(REPO) in pypath
        assert "task.ENV_KWARGS.reward_shaping_params." in argv
    assert (tmp_path / "cpa" / "event_partners" / "coord_ring" / "event_1" / "weights.json").is_file()

    # Worker failure must propagate as non-zero exit code.
    fail_res = subprocess.run(["/bin/bash", str(script)], cwd=str(tmp_path),
                              env={**env, "FAIL_ME": "1", "MEAL_CPA_ROOT": str(tmp_path / "cpa_fail")},
                              capture_output=True, text=True, check=False)
    assert fail_res.returncode != 0
