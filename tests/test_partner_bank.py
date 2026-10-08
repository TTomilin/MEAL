"""Partner-bank manifest, loader and generation-output tests (CPU only).

Three kinds of data are used, and each test says which:
  * real bundled checkpoints  - `partner_agents/BRDiv_population/<layout>` (trained partners, read-only);
  * synthetic populations     - random-initialised parameters with the real architecture, written to tmp dirs;
  * a tiny generated run      - an actual BRDiv run with a few dozen transitions: it exercises the generation
                                output format only and its partners are untrained.
Nothing is written outside pytest's tmp dirs.
"""
import json
import pickle
import shutil
from pathlib import Path

import jax
import numpy as np
import pytest

from experiments.partner_adaptation.partner_agents.agent_interface import ActorWithConditionalCriticPolicy
from experiments.partner_adaptation.partner_bank import (
    BUNDLED_ROOT, PartnerBankError, assemble_bank, legacy_bundled_bank, load_partner_bank, load_partners,
    select_partner_bank,
)
from experiments.partner_adaptation.partner_generation.run import (
    TrainConfig, generate_population, interaction_counts, save_population_members,
)
from meal import make_env
from meal.env.overcooked.layouts.presets import overcooked_layouts

LAYOUTS = ["cramped_room", "coord_ring", "asymm_advantages", "counter_circuit"]


def obs_dim_of(layout):
    env = make_env("overcooked", layout=overcooked_layouts[layout])
    return int(np.prod(env.observation_space().shape))


def act(partner, layout):
    env = make_env("overcooked", layout=overcooked_layouts[layout])
    obs, _ = env.reset(jax.random.PRNGKey(0))
    flat = obs["agent_1"].reshape(-1)
    action, _ = partner.policy.get_action(partner.params, flat, False, np.ones(6, dtype=bool), None,
                                          jax.random.PRNGKey(1), test_mode=True)
    return int(action)


def synthetic_population(root: Path, name: str, layout: str, size: int, seed: int, declared_layout=None):
    """Random-init members with the real architecture, written like a completed generation run."""
    policy = ActorWithConditionalCriticPolicy(6, obs_dim_of(layout), size)
    keys = jax.random.split(jax.random.PRNGKey(seed), size)
    stacked = jax.tree.map(lambda *xs: np.stack(xs)[None], *[policy.init_params(k) for k in keys])
    d = root / name
    d.mkdir(parents=True)
    members = save_population_members(str(d), stacked, 1, size)
    gen = {"status": "complete", "seed": seed, "num_seeds": 1, "partner_pop_size": size, "activation": "tanh",
           "layout_name": layout if declared_layout is None else declared_layout,
           "members": members}
    (d / "generation.json").write_text(json.dumps(gen))
    return d


def write_manifest(path: Path, layout, populations, partners, num=None):
    manifest = {"format_version": 1, "name": "t", "layout": layout,
                "num_partners": len(partners) if num is None else num,
                "populations": populations, "partners": partners}
    path.write_text(json.dumps(manifest))
    return str(path)


def one_pop_manifest(tmp_path, d, layout="cramped_room", members=((0, "params_seed0_agent0.pt"),),
                     config="generation.json", seed_index=0):
    partners = [{"partner_id": i, "population": "p", "member_index": m, "checkpoint": f"{d.name}/{f}"}
                for i, (m, f) in enumerate(members)]
    return write_manifest(tmp_path / "bank.json", layout,
                          {"p": {"config": f"{d.name}/{config}", "seed_index": seed_index}}, partners)


#  real bundled data

@pytest.mark.parametrize("layout", LAYOUTS)
def test_bundled_checkpoints_load_and_act(layout):
    """Real bundled partners of all four layouts load through the bank and produce valid actions."""
    bank = legacy_bundled_bank(layout)
    assert [r.partner_id for r in bank.records] == [0, 1, 2]
    assert all(r.population_size == 3 for r in bank.records)
    partners = load_partners(bank, obs_dim_of(layout))
    assert len({id(p.policy) for p in partners}) == 1  # one shared wrapper for equal (size, activation)
    for p in partners:
        assert 0 <= act(p, layout) < 6


def test_bundled_ignores_generic_params_file_and_is_not_truncated(tmp_path):
    root = tmp_path / "BRDiv_population"
    shutil.copytree(BUNDLED_ROOT / "cramped_room", root / "cramped_room")
    shutil.copy(root / "cramped_room/params_seed0_agent2.pt", root / "cramped_room/params.pt")
    bank = legacy_bundled_bank("cramped_room", root=root)
    assert [r.checkpoint.name for r in bank.records] == [f"params_seed0_agent{i}.pt" for i in range(3)]
    assert len(legacy_bundled_bank("cramped_room", 2, root=root)) == 2
    with pytest.raises(PartnerBankError, match="only 3 bundled members"):
        legacy_bundled_bank("cramped_room", 4, root=root)


def test_legacy_requires_layout():
    with pytest.raises(PartnerBankError, match="layout-name"):
        select_partner_bank("", "", None)


#  synthetic banks

def test_bank_from_several_synthetic_populations(tmp_path):
    """Bundled members + synthetic populations of size 3, 3 and 2 -> one bank, two wrappers."""
    gens = [synthetic_population(tmp_path, f"g{s}", "cramped_room", size, s)
            for s, size in ((1001, 3), (1002, 3), (1003, 2))]
    bank = assemble_bank("cramped_room", str(tmp_path / "bank" / "bank.json"), [str(g) for g in gens])
    assert len(bank) == 3 + 3 + 3 + 2
    assert [r.partner_id for r in bank.records] == list(range(11))
    assert [r.population for r in bank.records[3:6]] == ["gseed1001"] * 3
    assert [r.generation_seed for r in bank.records[6:]] == [1002] * 3 + [1003] * 2

    reloaded = load_partner_bank(str(tmp_path / "bank" / "bank.json"), "cramped_room")  # relative to the manifest
    assert [r.payload_sha256 for r in reloaded.records] == [r.payload_sha256 for r in bank.records]

    partners = load_partners(bank, obs_dim_of("cramped_room"))
    policies = {id(p.policy) for p in partners}
    assert len(policies) == 2  # (3, tanh) shared by bundled + two size-3 runs; (2, tanh) on its own
    assert {p.policy.pop_size for p in partners} == {2, 3}
    for p in partners:
        assert 0 <= act(p, "cramped_room") < 6

    selected = select_partner_bank("cramped_room", str(tmp_path / "bank" / "bank.json"), None)
    assert len(selected) == 11
    with pytest.raises(PartnerBankError, match="declares 11"):
        select_partner_bank("cramped_room", str(tmp_path / "bank" / "bank.json"), 3)


#  rejections

def test_rejects_missing_checkpoint(tmp_path):
    d = synthetic_population(tmp_path, "g", "cramped_room", 2, 1)
    m = one_pop_manifest(tmp_path, d, members=((0, "params_seed0_agent0.pt"), (1, "params_seed0_agent1.pt")))
    (d / "params_seed0_agent1.pt").unlink()
    with pytest.raises(PartnerBankError, match="does not exist"):
        load_partner_bank(m, "cramped_room")


def test_rejects_same_resolved_file_under_two_populations(tmp_path):
    d = synthetic_population(tmp_path, "g", "cramped_room", 2, 1)
    partners = [{"partner_id": i, "population": p, "member_index": 0, "checkpoint": "g/params_seed0_agent0.pt"}
                for i, p in enumerate(["a", "b"])]
    cfg = {"config": "g/generation.json"}
    m = write_manifest(tmp_path / "bank.json", "cramped_room", {"a": cfg, "b": cfg}, partners)
    with pytest.raises(PartnerBankError, match="same file as partner 0"):
        load_partner_bank(m, "cramped_room")


def test_rejects_duplicate_payload_under_another_name(tmp_path):
    d = synthetic_population(tmp_path, "g", "cramped_room", 2, 1)
    shutil.copy(d / "params_seed0_agent0.pt", d / "params_seed0_agent1.pt")
    m = one_pop_manifest(tmp_path, d, members=((0, "params_seed0_agent0.pt"), (1, "params_seed0_agent1.pt")))
    with pytest.raises(PartnerBankError, match="payload is identical to partner 0"):
        load_partner_bank(m, "cramped_room")


def test_rejects_generic_params_file(tmp_path):
    d = synthetic_population(tmp_path, "g", "cramped_room", 2, 1)
    shutil.copy(d / "params_seed0_agent1.pt", d / "params.pt")
    m = one_pop_manifest(tmp_path, d, members=((0, "params.pt"),))
    with pytest.raises(PartnerBankError, match="never a member"):
        load_partner_bank(m, "cramped_room")


def test_rejects_bad_member_indices(tmp_path):
    d = synthetic_population(tmp_path, "g", "cramped_room", 2, 1)
    with pytest.raises(PartnerBankError, match=r"member_index must be in \[0, 2\)"):
        load_partner_bank(one_pop_manifest(tmp_path, d, members=((2, "params_seed0_agent0.pt"),)), "cramped_room")
    with pytest.raises(PartnerBankError, match="file name says seed 0 agent 1"):
        load_partner_bank(one_pop_manifest(tmp_path, d, members=((0, "params_seed0_agent1.pt"),)), "cramped_room")
    with pytest.raises(PartnerBankError, match="file name says seed 0 agent 0"):
        load_partner_bank(one_pop_manifest(tmp_path, d, seed_index=3), "cramped_room")
    m = one_pop_manifest(tmp_path, d, members=((0, "params_seed0_agent0.pt"), (0, "params_seed0_agent0.pt")))
    with pytest.raises(PartnerBankError, match="already used"):
        load_partner_bank(m, "cramped_room")


def test_rejects_wrong_count_and_partner_id(tmp_path):
    d = synthetic_population(tmp_path, "g", "cramped_room", 2, 1)
    partners = [{"partner_id": 0, "population": "p", "member_index": 0, "checkpoint": "g/params_seed0_agent0.pt"}]
    pops = {"p": {"config": "g/generation.json"}}
    with pytest.raises(PartnerBankError, match="declares 2 partners but lists 1"):
        load_partner_bank(write_manifest(tmp_path / "b1.json", "cramped_room", pops, partners, num=2), "cramped_room")
    partners[0]["partner_id"] = 5
    with pytest.raises(PartnerBankError, match="must equal its position 0"):
        load_partner_bank(write_manifest(tmp_path / "b2.json", "cramped_room", pops, partners), "cramped_room")


def test_rejects_layout_mismatch(tmp_path):
    d = synthetic_population(tmp_path, "g", "cramped_room", 2, 1)
    m = one_pop_manifest(tmp_path, d)
    with pytest.raises(PartnerBankError, match="bank is for layout 'cramped_room' but the run uses 'coord_ring'"):
        load_partner_bank(m, "coord_ring")
    d2 = synthetic_population(tmp_path, "g2", "cramped_room", 2, 2, declared_layout="coord_ring")
    with pytest.raises(PartnerBankError, match="generated for layout 'coord_ring'"):
        load_partner_bank(one_pop_manifest(tmp_path, d2), "cramped_room")


def test_rejects_checkpoint_from_another_layout_or_population_size(tmp_path):
    # trained on coord_ring observations but declared (and loaded) as cramped_room
    d = synthetic_population(tmp_path, "g", "coord_ring", 2, 1, declared_layout="")
    bank = load_partner_bank(one_pop_manifest(tmp_path, d, layout="cramped_room"), "cramped_room")
    with pytest.raises(PartnerBankError, match=r"wrong layout or population size\?.*Dense_0"):
        load_partners(bank, obs_dim_of("cramped_room"))

    # parameters of a size-2 population under a config that says size 3 -> critic input width differs
    d = synthetic_population(tmp_path, "h", "cramped_room", 2, 1)
    cfg = json.loads((d / "generation.json").read_text())
    cfg["partner_pop_size"] = 3
    (d / "generation.json").write_text(json.dumps(cfg))
    bank = load_partner_bank(one_pop_manifest(tmp_path, d), "cramped_room")
    with pytest.raises(PartnerBankError, match="partner_pop_size=3"):
        load_partners(bank, obs_dim_of("cramped_room"))


def test_assemble_rejects_incomplete_generation(tmp_path):
    d = synthetic_population(tmp_path, "g", "cramped_room", 2, 1)
    gen = json.loads((d / "generation.json").read_text())
    gen["status"] = "started"
    (d / "generation.json").write_text(json.dumps(gen))
    with pytest.raises(PartnerBankError, match="not complete"):
        assemble_bank("cramped_room", str(tmp_path / "bank.json"), [str(d)])


# generation settings and accounting

def test_interaction_counts_separate_joint_and_agent_transitions():
    cfg = TrainConfig(num_envs_xp=2, num_envs_sp=2, num_steps=8, total_timesteps=200, num_seeds=2)
    counts = interaction_counts(cfg)
    assert counts["num_updates"] == 3  # 200 // (2 agents * 8 steps * 4 envs)
    assert counts["joint_env_transitions_per_seed"] == 3 * 8 * 4
    assert counts["agent_transitions_per_seed"] == 2 * 3 * 8 * 4
    assert counts["unused_requested_agent_transitions_per_seed"] == 200 - 192
    assert counts["joint_env_transitions_total"] == 2 * 96 and counts["agent_transitions_total"] == 2 * 192


def test_reference_brdiv_defaults_are_unchanged():
    cfg = TrainConfig()
    assert (cfg.partner_pop_size, cfg.num_envs_xp, cfg.num_envs_sp, cfg.num_steps, cfg.total_timesteps,
            cfg.update_epochs, cfg.num_minibatches, cfg.lr, cfg.num_seeds) == (3, 32, 32, 400, 2.5e8, 8, 16, 1e-3, 1)
    assert interaction_counts(cfg)["num_updates"] == 4882


def test_generation_rejects_budget_below_one_update_and_unknown_layout(tmp_path):
    with pytest.raises(ValueError, match="less than one update"):
        generate_population(TrainConfig(mode="disabled", layout_name="cramped_room", total_timesteps=10,
                                        checkpoint_path=str(tmp_path)))
    with pytest.raises(ValueError, match="unknown layout 'nope'"):
        generate_population(TrainConfig(mode="disabled", layout_name="nope", checkpoint_path=str(tmp_path)))


#  generation round trip

def test_tiny_generation_to_loader_round_trip(tmp_path):
    """Runs the real BRDiv generation on a few dozen transitions (untrained partners; format check only),
    assembles a bank with the bundled members and loads it back."""
    cfg = TrainConfig(
        mode="disabled", layout_name="cramped_room", seed=1001, partner_pop_size=2, num_seeds=1,
        num_envs_xp=2, num_envs_sp=2, num_steps=8, total_timesteps=64, update_epochs=1, num_minibatches=2,
        num_checkpoints=2, num_eval_episodes=1, checkpoint_path=str(tmp_path / "ckpt"))
    out = Path(generate_population(cfg))
    assert out.name == "brdiv_cramped_room_pop2_gseed1001"
    assert sorted(p.name for p in out.glob("*.pt")) == ["params_seed0_agent0.pt", "params_seed0_agent1.pt"]
    assert not (out / "params.pt").exists()

    gen = json.loads((out / "generation.json").read_text())
    assert (gen["status"], gen["seed"], gen["layout_name"], gen["resolved_layout_name"],
            gen["partner_pop_size"], gen["num_seeds"]) == ("complete", 1001, "cramped_room", "cramped_room", 2, 1)
    assert gen["members"] == [{"seed_index": 0, "member_index": j, "file": f"params_seed0_agent{j}.pt"}
                              for j in range(2)]
    assert gen["interactions"]["joint_env_transitions_per_seed"] == 32
    assert gen["interactions"]["agent_transitions_per_seed"] == 64

    bank = assemble_bank("cramped_room", str(tmp_path / "bank.json"), [str(out)])
    assert [r.population for r in bank.records] == ["bundled_seed0"] * 3 + ["gseed1001"] * 2
    partners = load_partners(bank, obs_dim_of("cramped_room"))
    assert len({id(p.policy) for p in partners}) == 2
    for p in partners:
        assert 0 <= act(p, "cramped_room") < 6

    with pytest.raises(FileExistsError, match="refusing to mix runs"):
        generate_population(cfg)
