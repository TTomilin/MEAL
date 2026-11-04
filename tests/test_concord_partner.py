import json
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.partner_adaptation.partner_agents.overcooked.concord_cnn import (
    OVERCOOKED_AI_ACTION_TO_MEAL, ConcordActor, ConcordCNNPolicy, torch_state_dict_to_flax,
)
from experiments.partner_adaptation.partner_agents.overcooked.concord_featurizer import concord_featurize
from experiments.partner_adaptation.partner_bank import load_bank_partners, write_bank_v2
from experiments.partner_adaptation.train_br import HeuristicPolicyPopulation, make_partner_population
from meal.env.overcooked import OBJECT_TO_INDEX, Overcooked
from meal.env.overcooked.common import DIR_TO_VEC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from convert_concord_partners import NUM_MEMBERS, convert, read_state_dict  # noqa: E402

H = W = 5


def test_overcooked_ai_action_map():
    oai = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    for i, v in enumerate(oai):
        assert tuple(np.asarray(DIR_TO_VEC[OVERCOOKED_AI_ACTION_TO_MEAL[i]])) == v
    assert OVERCOOKED_AI_ACTION_TO_MEAL[4:].tolist() == [4, 5]


def test_featurizer_on_hand_built_coord_ring_state():
    env = Overcooked(layout_name="coord_ring", max_steps=400)
    _, s = env.reset(jax.random.PRNGKey(0))
    maze = s.maze_map
    maze = maze.at[0, 3, 2].set(21)
    maze = maze.at[1, 4, 2].set(12)
    maze = maze.at[1, 0, 0].set(OBJECT_TO_INDEX["plate"])
    s = s.replace(maze_map=maze, agent_inv=jnp.array([OBJECT_TO_INDEX["onion"], OBJECT_TO_INDEX["dish"]]),
                  agent_dir_idx=jnp.array([2, 0]))
    (x0, y0), (x1, y1) = np.asarray(s.agent_pos).tolist()
    f = np.asarray(concord_featurize(s, 1))
    assert f.shape == (W, H, 20)
    assert f[x1, y1, 0] == 1 and f[..., 0].sum() == 1 and f[x0, y0, 1] == 1
    assert f[x1, y1, 2 + 0] == 1 and f[..., 2:6].sum() == 1
    assert f[x0, y0, 6 + 2] == 1 and f[..., 6:10].sum() == 1
    assert f[3, 0, 10] == f[4, 1, 10] == 1 and f[..., 10].sum() == 2
    assert f[0, 2, 13] == 1 and f[0, 3, 12] == f[1, 4, 12] == 1 and f[2, 4, 14] == 1
    assert f[0, 1, 11] == 1 and f[3, 0, 11] == 0 and f[2, 4, 11] == 0
    assert f[3, 0, 15] == 2 and f[3, 0, 16] == 0
    assert f[4, 1, 15] == 3 and f[4, 1, 16] == 8
    assert f[x1, y1, 17] == 1 and f[..., 17].sum() == 1
    assert f[0, 1, 18] == 1 and f[..., 18].sum() == 1
    assert f[x0, y0, 19] == 1 and f[..., 19].sum() == 1
    g = np.asarray(concord_featurize(s, 0))
    np.testing.assert_array_equal(g[..., 0], f[..., 1])
    np.testing.assert_array_equal(g[..., 10:], f[..., 10:])


def torch_r_actor():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    # Concord R_Actor((5, 5, 20), 6): CNNBase + ACTLayer, ReLU, hidden 64
    class RActor(nn.Module):
        def __init__(self):
            super().__init__()
            base = nn.Module()
            base.cnn = nn.Sequential(
                nn.Conv2d(20, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32 * W * H, 64), nn.ReLU(),
                nn.LayerNorm(64), nn.Linear(64, 64), nn.ReLU(), nn.LayerNorm(64), nn.Linear(64, 64), nn.ReLU(),
                nn.LayerNorm(64))
            self.base = base
            act = nn.Module()
            act.action_out = nn.Module()
            act.action_out.linear = nn.Linear(64, 6)
            self.act = act

        def forward(self, obs):
            return self.act.action_out.linear(self.base.cnn(obs.permute(0, 3, 1, 2) / 255.0))

    torch.manual_seed(0)
    net = RActor()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.5, 0.5)
    return torch, net


def test_flax_replica_matches_torch_logits(tmp_path):
    torch, net = torch_r_actor()
    path = tmp_path / "actor.pt"
    torch.save(net.state_dict(), path)
    sd_torch, sd_free = read_state_dict(path), read_state_dict(path, prefer_torch=False)
    assert sd_torch.keys() == sd_free.keys()
    for k in sd_torch:
        np.testing.assert_array_equal(sd_torch[k], sd_free[k])
    x = (np.random.RandomState(0).rand(16, W, H, 20) > 0.8).astype(np.float32)
    x[..., 15:17] *= np.random.RandomState(1).randint(0, 21, size=x[..., 15:17].shape)
    with torch.no_grad():
        want = net(torch.tensor(x * 255.0)).numpy()
    got = np.asarray(ConcordActor().apply(torch_state_dict_to_flax(sd_free), jnp.asarray(x)))
    np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-4)


def synthetic_concord_population(root: Path, n=2):
    policy = ConcordCNNPolicy(H, W)
    d = root / "concord" / "coord_ring"
    d.mkdir(parents=True)
    members = []
    for i in range(n):
        with open(d / f"r{i + 1}.pt", "wb") as f:
            pickle.dump({"actor_params": policy.init_params(jax.random.PRNGKey(i))}, f)
        members.append(dict(member_index=i, seed_index=0, file=f"r{i + 1}.pt", concord_id=f"r{i + 1}",
                            performance_bounds={}))
    gen = dict(status="complete", kind="concord", layout_name="coord_ring", actor_type="concord_cnn",
               partner_pop_size=1, num_members=n, activation="relu", test_mode=False, members=members)
    (d / "generation.json").write_text(json.dumps(gen))
    return d / "generation.json"


def test_concord_bank_partner_acts_through_state_population(tmp_path):
    gen = synthetic_concord_population(tmp_path)
    entries = [dict(kind="concord", population="concord", member_index=i, checkpoint=gen.parent / f"r{i + 1}.pt")
               for i in range(2)]
    bank = write_bank_v2(str(tmp_path / "bank.json"), "coord_ring", entries, [], {"concord": gen})
    partners, heldout = load_bank_partners(bank, H * W * 26)
    assert len(partners) == 2 and not heldout and partners[0].policy is partners[1].policy
    p = partners[0]
    population, params = make_partner_population(p.policy, p.params, ego_params=None)
    assert isinstance(population, HeuristicPolicyPopulation)
    env = Overcooked(layout_name="coord_ring", max_steps=400)
    obs, s = env.reset(jax.random.PRNGKey(0))
    acts, _ = population.get_actions(params, jnp.zeros((1,), jnp.int32), obs["agent_1"].reshape(1, 1, -1),
                                     jnp.zeros((1, 1, 1), bool), jnp.ones((1, 6)), None, jax.random.PRNGKey(1),
                                     env_state=jax.tree.map(lambda x: jnp.asarray(x)[None], s))
    assert 0 <= int(acts.reshape(())) < 6
    seat0 = p.policy._seat(obs["agent_0"].reshape(-1), s)
    seat1 = p.policy._seat(obs["agent_1"].reshape(-1), s)
    assert (int(seat0), int(seat1)) == (0, 1)


@pytest.mark.hf_data
@pytest.mark.skipif(os.environ.get("RUN_HF_DATA") != "1", reason="downloads Concord's actors; set RUN_HF_DATA=1")
def test_real_concord_partners_deliver_soups(tmp_path):
    gen = json.loads(convert(tmp_path / "concord").read_text())
    assert gen["num_members"] == NUM_MEMBERS
    policy = ConcordCNNPolicy(H, W)
    params = [pickle.load(open(tmp_path / "concord" / m["file"], "rb"))["actor_params"] for m in gen["members"][:2]]
    env = Overcooked(layout_name="coord_ring", max_steps=400)

    def episode(key):
        obs, st = env.reset(key)

        def body(c, _):
            obs, st, key, soups = c
            key, k0, k1, k2 = jax.random.split(key, 4)
            a0, _ = policy.get_action(params[0], obs["agent_0"].reshape(-1), False, None, None, k0, env_state=st)
            a1, _ = policy.get_action(params[1], obs["agent_1"].reshape(-1), False, None, None, k1, env_state=st)
            obs, st, _, _, info = env.step_env(k2, st, {"agent_0": a0, "agent_1": a1})
            return (obs, st, key, soups + info["soups"]["agent_0"] + info["soups"]["agent_1"]), None

        return jax.lax.scan(body, (obs, st, key, 0.0), None, 400)[0][3]

    soups = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(0), 4))
    assert float(soups.mean()) >= 1.0
