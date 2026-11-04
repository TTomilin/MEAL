import numpy as np
import pytest

from experiments.partner_adaptation.jax_aht_bridge.generators import algorithm_config


@pytest.mark.parametrize("updates", [1, 4, 6, 7, 10])
def test_fcp_rejects_unfilled_or_overflowing_checkpoint_schedules(updates):
    with pytest.raises(ValueError, match="checkpoint schedule"):
        algorithm_config("fcp", "coord_ring", {"NUM_ENVS": 16, "ROLLOUT_LENGTH": 4,
                                              "TOTAL_TIMESTEPS": updates * 64, "NUM_CHECKPOINTS": 5})


@pytest.mark.parametrize("updates", [5, 16, 257])
def test_fcp_accepts_exactly_filled_checkpoint_schedules(updates):
    cfg = algorithm_config("fcp", "coord_ring", {"NUM_ENVS": 16, "ROLLOUT_LENGTH": 4,
                                               "TOTAL_TIMESTEPS": updates * 64, "NUM_CHECKPOINTS": 5})
    assert cfg["NUM_UPDATES"] == updates


def checkpoint_output():
    values = np.arange(1, 11, dtype=np.float32).reshape(2, 5, 1)
    return {"checkpoints": {"weights": values}, "final_params": {"weights": values[:, -1].copy()},
            "final_ckpt_idx": np.asarray([5, 5])}


@pytest.mark.parametrize("corruption", ["count", "zeros", "missing_final", "nan"])
def test_export_rejects_invalid_checkpoints(corruption):
    from experiments.partner_adaptation.jax_aht_bridge.generators import validate_fcp_checkpoints
    out = checkpoint_output()
    if corruption == "count":
        out["final_ckpt_idx"][0] = 6
    elif corruption == "zeros":
        out["checkpoints"]["weights"][0, 1] = 0
    elif corruption == "missing_final":
        out["final_params"]["weights"][0] += 1
    else:
        out["checkpoints"]["weights"][0, 1] = np.nan
    with pytest.raises(ValueError, match="fcp:"):
        validate_fcp_checkpoints(out, {"PARTNER_POP_SIZE": 2, "NUM_CHECKPOINTS": 5})


@pytest.mark.parametrize("width", [None, 32])
def test_tiny_real_fcp_exports_final_checkpoint(tmp_path, width):
    import json
    from experiments.partner_adaptation.jax_aht_bridge.generators import generate, bank_entries
    from experiments.partner_adaptation.partner_bank import write_bank_v2, load_partners
    from meal.env.overcooked.layouts.presets import overcooked_layouts
    layout = overcooked_layouts["coord_ring"]
    path = generate("fcp", "coord_ring", str(tmp_path / "fcp"), seed=0, overrides={
        "PARTNER_POP_SIZE": 1, "NUM_ENVS": 2, "NUM_MINIBATCHES": 1, "UPDATE_EPOCHS": 1,
        "ROLLOUT_LENGTH": 4, "TOTAL_TIMESTEPS": 40, "NUM_CHECKPOINTS": 5, "NUM_EVAL_EPISODES": 1,
        "ENV_KWARGS.max_steps": 4, **({"FC_HIDDEN_DIM": width} if width else {})})
    assert json.loads(path.read_text())["fc_hidden_dim"] == (width or 64)
    bank = write_bank_v2(str(tmp_path / "bank.json"), "coord_ring", bank_entries(path, "fcp"), [], {"fcp": path})
    assert len(load_partners(bank, int(layout["width"]) * int(layout["height"]) * 26)) == 5