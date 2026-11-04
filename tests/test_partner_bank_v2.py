import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

try:
    import experiments.partner_adaptation.jax_aht_bridge  # noqa: F401
except ImportError as exc:
    pytest.skip(f"jax-aht not available: {exc}", allow_module_level=True)

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from experiments.partner_adaptation.jax_aht_bridge import hf_partners
from experiments.partner_adaptation.jax_aht_bridge.policies import JaxMarlCompatPolicy, make_jaxaht_policy
from experiments.partner_adaptation.partner_bank import (
    ImportedPartnerRecord, PartnerBankError, PartnerRecord, PlannerRecord, load_bank_partners, load_partner_bank,
    load_partners, partner_identity, write_bank_v2,
)
from meal import make_env
from meal.env.overcooked.layouts.presets import overcooked_layouts

LAYOUT = "coord_ring"


def obs_dim_of(layout=LAYOUT):
    env = make_env("overcooked", layout=overcooked_layouts[layout])
    return int(np.prod(env.observation_space().shape))


def save_run(root: Path, ckpt_key: str, actor_type: str, lead_shape, pop_size=1, seed=0) -> Path:
    policy = make_jaxaht_policy(actor_type, obs_dim_of(), pop_size=pop_size)
    n = int(np.prod(lead_shape))
    trees = [policy.init_params(k) for k in jax.random.split(jax.random.PRNGKey(seed), n)]
    stacked = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]).reshape(*lead_shape, *xs[0].shape),
                           *trees)
    run = root / "run"
    ocp.PyTreeCheckpointer().save(str((run / "saved_train_run").resolve()), {ckpt_key: stacked, "metrics": {}})
    return run


def spec(name, actor_type, idx_list, pop_size=None, set_name="eval", ckpt_key="final_params", test_mode=False,
         bounds=None):
    return hf_partners.TeammateSpec(set_name=set_name, layout=LAYOUT, name=name, actor_type=actor_type,
                                    run_path=f"overcooked-v1/{LAYOUT}/{name}/run/saved_train_run",
                                    ckpt_key=ckpt_key, idx_list=idx_list, pop_size=pop_size, test_mode=test_mode,
                                    performance_bounds=bounds or {})


def import_synthetic(tmp_path, s, lead_shape, seed=0):
    run = save_run(tmp_path / f"src_{s.set_name}_{s.name}", s.ckpt_key, s.actor_type, lead_shape,
                   pop_size=s.pop_size or 1, seed=seed)
    local = tmp_path / "hf"
    target = hf_partners.local_run_dir(s, local)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(run, target)
    return hf_partners.import_spec(s, local, tmp_path / "partners" / "jaxaht")


def entries(gen_path: Path, key: str):
    gen = json.loads(gen_path.read_text())
    return [dict(kind="jaxaht", population=key, member_index=m["member_index"], checkpoint=gen_path.parent / m["file"])
            for m in gen["members"]]


@pytest.fixture
def imported(tmp_path):
    conf = spec("comedi", "actor_with_conditional_critic", [[1, 0], [1, 2]], pop_size=4,
                ckpt_key="final_params_conf", bounds={"base_return": [[0.0, 100.0], [0.0, 120.0]]})
    ippo = spec("ippo_mlp", "mlp", [1, 2, 3], test_mode=True,
                bounds={"base_return": [[0.0, 200.0], [0.0, 210.0], [0.0, 210.0]]})
    val = spec("lbrdiv", "actor_with_conditional_critic", [[0, 0], [0, 1]], pop_size=4, set_name="val",
               ckpt_key="final_params_conf")
    gens = {"c": import_synthetic(tmp_path, conf, (2, 4), seed=1),
            "i": import_synthetic(tmp_path, ippo, (3,), seed=2),
            "v": import_synthetic(tmp_path, val, (2, 4), seed=3)}
    return tmp_path, gens


def test_import_selects_members_like_jax_aht(imported):
    _, gens = imported
    conf = json.loads(gens["c"].read_text())
    assert conf["kind"] == "jaxaht" and conf["actor_type"] == "actor_with_conditional_critic"
    assert conf["partner_pop_size"] == 4 and conf["num_members"] == 2 and conf["layout_name"] == LAYOUT
    # jnp's clamped gather maps idx 3 on 3 seeds to 2, a duplicate that is dropped once.
    assert [m["effective_index"] for m in conf["members"]] == [[1, 0], [1, 2]]
    assert [m["performance_bounds"]["base_return"] for m in conf["members"]] == [[0.0, 100.0], [0.0, 120.0]]
    ippo = json.loads(gens["i"].read_text())
    assert ippo["num_members"] == 2 and ippo["test_mode"] is True
    assert [m["effective_index"] for m in ippo["members"]] == [[1], [2]]
    assert ippo["duplicates_dropped"] == [{"requested": 3, "same_as_member": 1}]
    assert sorted(p.name for p in gens["i"].parent.glob("m*.pt")) == ["m0.pt", "m1.pt"]


def test_effective_indices_match_jnp_gather():
    x = jnp.arange(5 * 3).reshape(5, 3)
    for idx in ([0, 4, 7, -1], [[0, 1], [4, 5], [-1, -1], [9, 0]]):
        got = [int(np.asarray(x)[e]) if len(e) == 2 else int(np.asarray(x)[e][0])
               for e in hf_partners.effective_indices(idx, x.shape)]
        a = np.asarray(idx)
        want = (x[a] if a.ndim == 1 else x[a[:, 0], a[:, 1]])
        want = [int(w) if np.ndim(w) == 0 else int(w[0]) for w in np.asarray(want)]
        assert got == want


def test_v2_bank_with_heldout_loads_and_acts(imported):
    tmp_path, gens = imported
    pops = {"jaxaht_eval_comedi": gens["c"], "jaxaht_eval_ippo_mlp": gens["i"], "jaxaht_val_lbrdiv": gens["v"]}
    train = entries(gens["c"], "jaxaht_eval_comedi") + entries(gens["i"], "jaxaht_eval_ippo_mlp")
    held = entries(gens["v"], "jaxaht_val_lbrdiv")
    bank = write_bank_v2(str(tmp_path / "partners" / "bank.json"), LAYOUT, train, held, pops)
    assert bank.format_version == 2 and len(bank) == 4 and len(bank.heldout) == 2
    assert [r.partner_id for r in bank.records] == [0, 1, 2, 3]
    assert all(r.heldout for r in bank.heldout) and not any(r.heldout for r in bank.records)
    assert {r.checkpoint for r in bank.records}.isdisjoint({r.checkpoint for r in bank.heldout})
    assert bank.records[0].performance_bounds == {"base_return": [0.0, 100.0]}
    ident = partner_identity(bank.heldout[1])
    assert ident["heldout_id"] == 1 and ident["heldout"] is True and "partner_id" not in ident

    partners, heldout = load_bank_partners(bank, obs_dim_of())
    assert len(load_partners(bank, obs_dim_of())) == len(partners) == 4
    assert all(isinstance(p.policy, JaxMarlCompatPolicy) for p in partners + heldout)
    assert len({id(p.policy) for p in partners + heldout}) == 2
    env = make_env("overcooked", layout=overcooked_layouts[LAYOUT])
    obs, _ = env.reset(jax.random.PRNGKey(0))
    for p in partners + heldout:
        a, h = p.policy.get_action(p.params, obs["agent_1"].reshape(-1), False, jnp.ones(6), None,
                                   jax.random.PRNGKey(1))
        assert 0 <= int(a) < 6 and h is None


def test_imported_activation_changes_identity_with_same_weights(tmp_path):
    gen_path = import_synthetic(tmp_path, spec("ippo_mlp", "mlp", [0]), (1,), seed=0)
    bank_path = tmp_path / "bank.json"
    bank = write_bank_v2(str(bank_path), LAYOUT, entries(gen_path, "p"), [], {"p": gen_path})
    (tanh,) = load_partners(bank, obs_dim_of())
    gen = json.loads(gen_path.read_text())
    assert gen["activation"] == "tanh"
    gen["activation"] = "relu"
    gen_path.write_text(json.dumps(gen))
    (relu,) = load_partners(load_partner_bank(str(bank_path), LAYOUT), obs_dim_of())
    assert tanh.record.checkpoint == relu.record.checkpoint
    assert tanh.record.payload_sha256 == relu.record.payload_sha256
    assert jax.tree.structure(tanh.params) == jax.tree.structure(relu.params)
    for before, after in zip(jax.tree.leaves(tanh.params), jax.tree.leaves(relu.params)):
        assert before.shape == after.shape
        np.testing.assert_array_equal(before, after)
    assert partner_identity(tanh.record) != partner_identity(relu.record)
    assert partner_identity(relu.record) == dict(partner_identity(tanh.record), activation="relu")


@pytest.mark.parametrize("kind", ["jaxaht", "concord", "bc_proxy"])
@pytest.mark.parametrize("setting,value", [
    ("actor_type", "actor_with_conditional_critic"), ("population_size", 3), ("activation", "relu"),
    ("fc_hidden_dim", 128), ("test_mode", True), ("layout", "cramped_room"),
])
def test_imported_identity_records_policy_settings(kind, setting, value):
    record = ImportedPartnerRecord(
        partner_id=0, layout=LAYOUT, kind=kind, population="p", population_config=Path("generation.json"),
        checkpoint=Path("m0.pt"), member_index=0, actor_type="mlp", population_size=1, activation="tanh",
        fc_hidden_dim=64, test_mode=False, source="synthetic", performance_bounds={}, payload_sha256="hash",
        config={})
    identity = partner_identity(record)
    changed = partner_identity(replace(record, **{setting: value}))
    assert changed != identity
    assert changed[setting] == value


def test_v1_partner_identities_remain_unchanged():
    record = PartnerRecord(
        partner_id=0, layout=LAYOUT, population="p", population_config=Path("generation.json"),
        checkpoint=Path("params_seed0_agent0.pt"), member_index=0, seed_index=0, population_size=3,
        activation="tanh", generation_seed=0, payload_sha256="hash", config={})
    assert partner_identity(record) == dict(
        partner_id=0, kind="brdiv", label=f"{LAYOUT}/p/m0", checkpoint="params_seed0_agent0.pt",
        population_size=3, generation_seed=0, payload_sha256="hash")
    planner = PlannerRecord(partner_id=1, layout=LAYOUT, planner_id="P01", config={"x": 1}, config_sha256="hash")
    assert partner_identity(planner) == dict(
        partner_id=1, kind="planner", label=f"{LAYOUT}/P01", planner_id="P01", config={"x": 1},
        config_sha256="hash")


SHARED_RUNS = {
    "coord_ring": {"overcooked-v1/coord_ring/lbrdiv/2026-03-02_14-56-18",
                   "overcooked-v1/coord_ring/comedi/2026-03-01_22-32-32"},
    "asymm_advantages": {"overcooked-v1/asymm_advantages/lbrdiv/2026-03-03_17-02-53",
                         "overcooked-v1/asymm_advantages/comedi/2026-03-03_17-06-25"},
    "forced_coord": {"overcooked-v1/forced_coord/lbrdiv/2026-03-03_19-30-49",
                     "overcooked-v1/forced_coord/comedi/2026-03-04_00-28-31"},
    "cramped_room": set(), "counter_circuit": set(),
}


@pytest.mark.parametrize("layout", hf_partners.LAYOUTS)
def test_shared_runs_follow_jax_aht_configs(layout):
    from experiments.partner_adaptation.jax_aht_bridge import fetch
    assert fetch.shared_runs(layout) == SHARED_RUNS[layout]


@pytest.mark.parametrize("keep_shared", [False, True])
def test_bank_trains_on_val_and_holds_out_eval_and_bc(tmp_path, monkeypatch, keep_shared):
    from experiments.partner_adaptation.jax_aht_bridge import fetch
    imported = {}

    def fake_export(s, out_root, key):
        gen = out_root / s.set_name / s.layout / s.name / "generation.json"
        gen.parent.mkdir(parents=True, exist_ok=True)
        gen.write_text(json.dumps(dict(num_members=1, members=[dict(member_index=0, file="m0.pt")])))
        imported.setdefault(key, []).append(s)
        return gen

    def fake_import_spec(s, local_dir, out_root):
        return fake_export(s, out_root, s.set_name)

    def fake_bc(s, local_dir, out_root):
        return fake_export(s, out_root, f"{s.set_name}_bc")

    written = {}
    monkeypatch.setattr(hf_partners, "import_spec", fake_import_spec)
    monkeypatch.setattr(hf_partners, "import_bc_proxy", fake_bc)
    monkeypatch.setattr(fetch, "write_bank_v2", lambda path, layout, partners, heldout, pops, name: written.update(
        partners=partners, heldout=heldout) or type("B", (), dict(source=path, records=partners, heldout=heldout)))
    fetch.build_layout_bank(LAYOUT, tmp_path, download=False, concord=False, keep_shared_runs=keep_shared)

    val_runs = {s.run_dir for s in imported["val"]}
    want_val = {s.run_dir for s in hf_partners.load_specs("val", LAYOUT)}
    assert val_runs == (want_val if keep_shared else want_val - SHARED_RUNS[LAYOUT])
    assert {s.name for s in imported["eval"]} == {s.name for s in hf_partners.load_specs("eval", LAYOUT)}
    assert "val_bc" not in imported and len(imported["eval_bc"]) == 1
    assert all(e["population"].startswith("jaxaht_val_") for e in written["partners"])
    assert all(e["population"].startswith("jaxaht_eval_") for e in written["heldout"])
    assert [e["kind"] for e in written["heldout"]].count("bc_proxy") == 1


def test_member_cannot_be_both_training_and_heldout(imported):
    tmp_path, gens = imported
    pops = {"jaxaht_eval_comedi": gens["c"]}
    train = entries(gens["c"], "jaxaht_eval_comedi")
    with pytest.raises(PartnerBankError, match="already used"):
        write_bank_v2(str(tmp_path / "partners" / "bad.json"), LAYOUT, train, train[:1], pops)
    assert not (tmp_path / "partners" / "bad.json").exists()


def test_shape_and_layout_mismatches_are_rejected(imported):
    tmp_path, gens = imported
    gen = json.loads(gens["c"].read_text())
    gen["partner_pop_size"] = 7
    gens["c"].write_text(json.dumps(gen))
    bank = write_bank_v2(str(tmp_path / "partners" / "b.json"), LAYOUT, entries(gens["c"], "p"), [],
                         {"p": gens["c"]})
    with pytest.raises(PartnerBankError, match="does not match"):
        load_partners(bank, obs_dim_of())
    with pytest.raises(PartnerBankError, match="generated for layout"):
        write_bank_v2(str(tmp_path / "partners" / "c.json"), "cramped_room", entries(gens["c"], "p"), [],
                      {"p": gens["c"]})


def test_v1_manifest_cannot_use_v2_features(imported):
    tmp_path, gens = imported
    bank_path = tmp_path / "partners" / "v2.json"
    write_bank_v2(str(bank_path), LAYOUT, entries(gens["c"], "p"), entries(gens["v"], "v"),
                  {"p": gens["c"], "v": gens["v"]})
    data = json.loads(bank_path.read_text())
    data["format_version"] = 1
    v1 = tmp_path / "partners" / "v1.json"
    v1.write_text(json.dumps(data))
    with pytest.raises(PartnerBankError, match="heldout"):
        load_partner_bank(str(v1), LAYOUT)
    data["heldout"] = []
    v1.write_text(json.dumps(data))
    with pytest.raises(PartnerBankError, match="needs format_version 2"):
        load_partner_bank(str(v1), LAYOUT)


def test_eval_and_val_specs_parse_for_all_layouts():
    for layout in hf_partners.LAYOUTS:
        for set_name in ("eval", "val"):
            specs = hf_partners.load_specs(set_name, layout)
            assert specs and all(s.actor_type in ("mlp", "actor_with_conditional_critic") for s in specs)
            assert all(s.run_dir.startswith(f"overcooked-v1/{layout}/") for s in specs)
            assert all(s.idx_list is not None for s in specs)


def test_bc_proxy_specs_and_stateful_policy(tmp_path):
    from experiments.partner_adaptation.partner_agents.overcooked.bc_proxy_wrapper import BCProxyPolicy
    from agents.overcooked.bc_agent import BCPolicyHState
    for layout in hf_partners.LAYOUTS:
        (bc,) = hf_partners.load_bc_specs("eval", layout)
        assert bc.test_mode and bc.performance_bounds["base_return"]
        assert bc.run_path == f"overcooked-v1/{layout}/human_proxy/models"
    layout = overcooked_layouts[LAYOUT]
    policy = BCProxyPolicy(LAYOUT, layout["height"], layout["width"], 2, 1)
    d = tmp_path / "bc"
    d.mkdir()
    with open(d / "m0.pt", "wb") as f:
        import pickle
        pickle.dump({"actor_params": policy.init_params(jax.random.PRNGKey(0))}, f)
    gen = dict(status="complete", kind="bc_proxy", layout_name=LAYOUT, actor_type="bc_mlp", partner_pop_size=1,
               num_members=1, activation="relu", test_mode=True,
               members=[dict(member_index=0, file="m0.pt", performance_bounds={"base_return": [0.0, 215.0]})])
    (d / "generation.json").write_text(json.dumps(gen))
    bank = write_bank_v2(str(tmp_path / "bank.json"), LAYOUT,
                         [dict(kind="bc_proxy", population="bc", member_index=0, checkpoint=d / "m0.pt")], [],
                         {"bc": d / "generation.json"})
    (p,), _ = load_bank_partners(bank, obs_dim_of())
    assert isinstance(p.policy, BCProxyPolicy) and p.policy.needs_env_state

    env = make_env("overcooked", layout=layout)
    obs, st = env.reset(jax.random.PRNGKey(0))
    h0 = p.policy.init_hstate(1, {"agent_id": 1})
    h = h0
    for t in range(6):
        a, h = p.policy.get_action(p.params, obs["agent_1"].reshape(-1), False, None, h, jax.random.PRNGKey(t),
                                   env_state=st)
        assert 0 <= int(a) < 6
        obs, st, *_ = env.step(jax.random.PRNGKey(100 + t), st, {"agent_0": jnp.int32(4), "agent_1": a})
    assert not np.array_equal(np.asarray(h), np.asarray(h0))
    _, h_reset = p.policy.get_action(p.params, obs["agent_1"].reshape(-1), True, None, h, jax.random.PRNGKey(9),
                                     env_state=st)
    _, h_fresh = p.policy.get_action(p.params, obs["agent_1"].reshape(-1), False, None, h0, jax.random.PRNGKey(9),
                                     env_state=st)
    np.testing.assert_array_equal(np.asarray(h_reset), np.asarray(h_fresh))
    assert int(BCPolicyHState.from_numpy(h_reset).agent_id) == 1


def test_tiny_comedi_on_meal_env_exports_a_loadable_bank(tmp_path):
    from experiments.partner_adaptation.jax_aht_bridge.generators import algorithm_config
    from experiments.partner_adaptation.partner_generation.run import TrainConfig, generate_population
    cfg = algorithm_config("comedi", LAYOUT, {"LR": "3e-4"})
    assert cfg["ENV_NAME"] == "meal-overcooked" and cfg["LR"] == 3e-4 and isinstance(cfg["ENT_COEF"], float)
    config = TrainConfig(alg="comedi", layout_name=LAYOUT, partner_pop_size=2, checkpoint_path=str(tmp_path),
                         aht_num_envs=4, aht_total_timesteps=1600, mode="disabled", aht_overrides=[
                             "ENV_KWARGS.max_steps=50", "ROLLOUT_LENGTH=50", "NUM_MINIBATCHES=2", "UPDATE_EPOCHS=1",
                             "NUM_EVAL_EPISODES=2", "NUM_CHECKPOINTS=2", "NUM_ARGMAX_ROLLOUT_EPS=2"])
    out = Path(generate_population(config))
    gen = json.loads((out / "generation.json").read_text())
    assert gen["kind"] == "jaxaht" and gen["source"] == "meal-generated" and gen["num_members"] == 2
    assert gen["actor_type"] == "actor_with_conditional_critic" and gen["partner_pop_size"] == 2
    assert all(not m["performance_bounds"] for m in gen["members"])
    bank = write_bank_v2(str(tmp_path / "bank.json"), LAYOUT, entries(out / "generation.json", "comedi"), [],
                         {"comedi": out / "generation.json"})
    partners = load_partners(bank, obs_dim_of())
    env = make_env("overcooked", layout=overcooked_layouts[LAYOUT])
    obs, _ = env.reset(jax.random.PRNGKey(0))
    for p in partners:
        a, _ = p.policy.get_action(p.params, obs["agent_1"].reshape(-1), False, jnp.ones(6), None,
                                   jax.random.PRNGKey(1))
        assert 0 <= int(a) < 6


@pytest.mark.hf_data
@pytest.mark.skipif(os.environ.get("RUN_HF_DATA") != "1", reason="downloads from Hugging Face; set RUN_HF_DATA=1")
def test_real_bc_proxy_matches_upstream_and_delivers(tmp_path):
    # BC is paired with BC because P01 and the proxy can jam each other in coord_ring's corridor.
    from flax import struct
    from agents.overcooked.bc_agent import ACTION_DIM, BCModel, BCPolicy
    from agents.overcooked.bc_featurizer import BCFeaturizer
    from jaxmarl.environments.overcooked.layouts import overcooked_layouts as jaxmarl_layouts
    from experiments.partner_adaptation.jax_aht_bridge import compat

    @struct.dataclass
    class Wrapped:
        env_state: object

    (s,) = hf_partners.load_bc_specs("eval", LAYOUT)
    hf_partners.download([s], tmp_path / "hf")
    gen_path = hf_partners.import_bc_proxy(s, tmp_path / "hf", tmp_path / "partners" / "jaxaht")
    bank = write_bank_v2(str(tmp_path / "partners" / "bank.json"), LAYOUT,
                         [dict(kind="bc_proxy", population="bc", member_index=0, checkpoint=gen_path.parent / "m0.pt")],
                         [], {"bc": gen_path})
    (p,) = load_partners(bank, obs_dim_of())
    layout = overcooked_layouts[LAYOUT]
    upstream = object.__new__(BCPolicy)
    upstream.action_dim, upstream.params, upstream.unblock_if_stuck, upstream.using_log_wrapper = \
        ACTION_DIM, p.params, True, False
    upstream.network = BCModel(action_dim=ACTION_DIM, hidden_dims=(64, 64))
    upstream.featurizer = BCFeaturizer(jaxmarl_layouts[LAYOUT], num_pots=2, max_steps=400)
    env = make_env("overcooked", layout=layout, max_steps=400)

    total = 0.0
    for episode in range(4):
        obs, st = env.reset(jax.random.PRNGKey(episode))
        h = [p.policy.init_hstate(1, {"agent_id": i}) for i in (0, 1)]
        hu = upstream.init_hstate(1, {"agent_id": 1})
        for t in range(400):
            k = jax.random.split(jax.random.PRNGKey(1000 * episode + t), 3)
            acts = {}
            for i, agent in enumerate(("agent_0", "agent_1")):
                acts[agent], h[i] = p.policy.get_action(p.params, obs[agent].reshape(-1), False, None, h[i], k[i],
                                                        env_state=st)
            if episode == 0:
                jm = compat.meal_state_to_jaxmarl(st, len(layout["pot_idx"]), len(layout["goal_idx"]))
                jm = jm.replace(time=jnp.int32(t))
                au, hu = upstream.get_action(None, None, False, None, hu, k[1], env_state=Wrapped(env_state=jm),
                                             test_mode=True)
                assert int(compat.jaxmarl_action_to_meal(au)) == int(acts["agent_1"]), t
            obs, st, _, _, info = env.step_env(k[2], st, acts)
            total += float(info["soups"]["agent_0"] + info["soups"]["agent_1"])
    assert total >= 1


@pytest.mark.hf_data
@pytest.mark.skipif(os.environ.get("RUN_HF_DATA") != "1", reason="downloads from Hugging Face; set RUN_HF_DATA=1")
def test_real_jaxaht_checkpoint_restores_and_loads(tmp_path):
    s = next(x for x in hf_partners.load_specs("eval", LAYOUT) if x.name == "lbrdiv-conf")
    hf_partners.download([s], tmp_path / "hf")
    gen_path = hf_partners.import_spec(s, tmp_path / "hf", tmp_path / "partners" / "jaxaht")
    bank = write_bank_v2(str(tmp_path / "partners" / "bank.json"), LAYOUT, entries(gen_path, "p"), [],
                         {"p": gen_path})
    partners = load_partners(bank, obs_dim_of())
    assert len(partners) == 3 and all(p.record.performance_bounds.get("base_return") for p in partners)
