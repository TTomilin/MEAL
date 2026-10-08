"""CPU-only training-path tests for the continual partner-adaptation (CPA) pipeline.

Everything here uses tiny shapes (a handful of envs, 16-step episodes unless stated), a real bundled
BRDiv partner from `partner_agents/BRDiv_population/cramped_room` and the existing onion planner.
These tests exercise logic and wiring on CPU; they say nothing about GPU performance or numerics.
"""
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.continual.ewc import EWC
from experiments.continual.ft import FT
from experiments.continual.mas import MAS
from experiments.model.mlp import ActorCritic
from experiments.partner_adaptation import train_br
from experiments.partner_adaptation.partner_agents.agent_interface import (
    ActorWithConditionalCriticPolicy, MLPActorCriticPolicyCL,
)
from experiments.partner_adaptation.partner_agents.overcooked.agent_policy_wrappers import (
    OvercookedOnionPolicyWrapper,
)
from experiments.partner_adaptation.run_br import TrainConfig
from experiments.partner_adaptation.train_br import (
    DummyPolicyPopulation, HeuristicPolicyPopulation, make_partner_switches, run_br_training,
)
from experiments.partner_adaptation.train_ego import make_stage_keys, train_ppo_ego_agent
from experiments.utils import init_cl_state
from meal import make_env
from meal.env.overcooked.common import OBJECT_TO_INDEX
from meal.env.overcooked.layouts.presets import overcooked_layouts
from meal.wrappers.logging import LogWrapper

POP_DIR = (Path(__file__).resolve().parent.parent
           / "experiments/partner_adaptation/partner_agents/BRDiv_population/cramped_room")
STAY = 4


@pytest.fixture(scope="module")
def cpa():
    layout = {"layout": overcooked_layouts["cramped_room"]}
    envs = {}

    def make_logged_env(episode_steps):
        if episode_steps not in envs:
            envs[episode_steps] = LogWrapper(make_env("overcooked", **layout, max_steps=episode_steps))
        return envs[episode_steps]

    env = make_logged_env(16)
    obs_dim = int(np.prod(env.observation_space().shape))
    with open(POP_DIR / "config.pckl", "rb") as f:
        pop_cfg = pickle.load(f)
    pop_params = [pickle.load(open(POP_DIR / f"params_seed0_agent{i}.pt", "rb"))["actor_params"] for i in range(3)]
    partner_policy = ActorWithConditionalCriticPolicy(6, obs_dim=obs_dim, pop_size=pop_cfg["partner_pop_size"])
    planner = OvercookedOnionPolicyWrapper(layout=layout["layout"])
    network = ActorCritic(6, "relu", 3, True, False, 16, 1, True, True)
    ego = MLPActorCriticPolicyCL(network, obs_dim)
    ego_params = ego.init_params(jax.random.PRNGKey(1))

    def make_cfg(num_steps=16, updates=3, eval_every=2, cl_method=None, **kw):
        cfg = TrainConfig(
            layout_name="cramped_room", num_envs=4, num_steps=num_steps, total_timesteps=updates * 4 * num_steps,
            update_epochs=1, num_minibatches=2, num_eval_episodes=2, eval_every=eval_every, hidden_size=16,
            num_layers=1, importance_episodes=1, importance_steps=8, cl_method=cl_method, reg_coef=1.0, seed=0,
            reward_shaping_horizon=1e3, **kw)
        cfg.layout = layout
        return cfg

    def eval_entries(ids=(0, 1)):
        """Eval list: BRDiv partners 0/1 and the onion planner as id 2."""
        fake = jax.tree.map(lambda x: x[None], ego_params)
        entries = [(DummyPolicyPopulation(partner_policy), jax.tree.map(lambda x: x[None], pop_params[i]), i)
                   for i in ids]
        entries.append((HeuristicPolicyPopulation(planner), fake, 2))
        return entries

    return SimpleNamespace(
        layout=layout, env=env, make_env=make_logged_env, pop_cfg=pop_cfg, pop_params=pop_params,
        partner_policy=partner_policy, planner=planner, ego=ego, ego_params=ego_params, make_cfg=make_cfg,
        eval_entries=eval_entries)


def _events(logs):
    return [log for log in logs if "Eval/StageID" in log]


def _train_stage0(cpa, cfg, eval_partner, logs, eval_rng_stage=0):
    keys = make_stage_keys(cfg.seed, 0)
    partner_population = DummyPolicyPopulation(cpa.partner_policy)
    partner_params = jax.tree.map(lambda x: x[None], cpa.pop_params[0])
    return train_ppo_ego_agent(
        cfg, cpa.env, keys.train, cpa.ego, cpa.ego_params, 1, partner_population, partner_params, env_id_idx=0,
        eval_partner=eval_partner, eval_rng=keys.eval, log_fn=logs.append)


def test_eval_events_only_on_schedule_and_partners_unique(cpa):
    cfg = cpa.make_cfg(updates=3, eval_every=2)
    logs = []
    out = _train_stage0(cpa, cfg, cpa.eval_entries(), logs)

    events = _events(logs)
    assert [e["train_step"] for e in events] == [0, 2]  # after updates 1 and 3, not after update 2
    assert np.asarray(out["metrics"]["evaluated"]).tolist() == [True, False, True]
    assert all(np.asarray(v).shape == (3,) for v in out["metrics"].values())  # one scalar per update, no rollout data
    for e in events:
        evaluated = sorted(int(k.removeprefix("Eval/Episodes_Partner")) for k in e if "Episodes_Partner" in k)
        assert evaluated == [0, 1, 2]  # current partner (id 0) is evaluated once, not again as a separate population
        assert e["Eval/StageID"] == 0
        assert e["Eval/EnvSteps"] == (0 * 3 + (e["train_step"] + 1)) * 4 * 16
        assert all(e[f"Eval/Episodes_Partner{i}"] == cfg.num_eval_episodes for i in range(3))
    train_logs = [log for log in logs if "Train/EgoGradNorm" in log]
    assert [log["train_step"] for log in train_logs] == [0, 1, 2]


def test_current_partner_is_evaluated_when_missing_from_eval_list(cpa):
    cfg = cpa.make_cfg(updates=1, eval_every=1)
    logs = []
    _train_stage0(cpa, cfg, [], logs)
    (event,) = _events(logs)
    assert [k for k in event if "Episodes_Partner" in k] == ["Eval/Episodes_Partner0"]


def test_eval_frequency_does_not_change_training(cpa):
    results = {}
    for eval_every in (1, 3):
        logs = []
        results[eval_every] = (_train_stage0(cpa, cpa.make_cfg(updates=3, eval_every=eval_every),
                                             cpa.eval_entries(), logs), logs)

    (every1, logs1), (every3, logs3) = results[1], results[3]
    assert len(_events(logs1)) == 3 and len(_events(logs3)) == 2
    for a, b in zip(jax.tree.leaves(every1["final_params"]), jax.tree.leaves(every3["final_params"])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for name in ("actor_loss", "value_loss", "entropy_loss", "avg_grad_norm"):
        np.testing.assert_array_equal(np.asarray(every1["metrics"][name]), np.asarray(every3["metrics"][name]))

    # Evaluation draws depend only on the update index: an update evaluated under both schedules agrees.
    first1, first3 = _events(logs1)[0], _events(logs3)[0]
    assert first1["train_step"] == first3["train_step"] == 0
    assert first1["Eval/EgoSoup_Partner1"] == first3["Eval/EgoSoup_Partner1"]


def test_importance_rollout_is_driven_by_the_acting_partner(cpa):
    """Regression: the importance rollout used to feed the partner action 0, which is `up` (stay is 4)."""
    env = cpa.make_env(16)
    population = HeuristicPolicyPopulation(cpa.planner)
    dummy = jax.tree.map(lambda x: x[None], cpa.ego_params)
    reset_switch, step_switch = make_partner_switches(env, population, dummy)
    ego_stay = {"agent_0": jnp.full((1,), STAY, jnp.int32)}

    _, state = reset_switch(jax.random.PRNGKey(0), 0)
    key = jax.random.PRNGKey(1)
    held = []
    for _ in range(12):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = step_switch(step_key, state, ego_stay, 0)
        held.append(int(state.env_state.env_state.agent_inv[1]))
    assert OBJECT_TO_INDEX["onion"] in held  # the planner partner actually fetched an onion

    # The old behaviour, a frozen partner repeating action 0 (`up`), never gets anything.
    _, old_state = env.reset(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)
    for _ in range(12):
        key, step_key = jax.random.split(key)
        act = {"agent_0": jnp.full((1,), STAY, jnp.int32), "agent_1": jnp.zeros((1,), jnp.int32)}
        _, old_state, _, _, _ = env.step(step_key, old_state, act)
        assert int(old_state.env_state.agent_inv[1]) == OBJECT_TO_INDEX["empty"]


def test_importance_rollout_resets_planner_state_on_episode_end(cpa):
    env = cpa.make_env(16)
    population = HeuristicPolicyPopulation(cpa.planner)
    dummy = jax.tree.map(lambda x: x[None], cpa.ego_params)
    reset_switch, step_switch = make_partner_switches(env, population, dummy)
    ego_stay = {"agent_0": jnp.full((1,), STAY, jnp.int32)}

    _, state = reset_switch(jax.random.PRNGKey(0), 0)
    key, goals = jax.random.PRNGKey(1), []
    for t in range(18):  # the episode ends at step 16 and the env auto-resets
        key, step_key = jax.random.split(key)
        _, state, _, done, _ = step_switch(step_key, state, ego_stay, 0)
        goals.append((bool(done["__all__"]), int(state.partner_hstate.holding[0]), bool(state.partner_done)))
    done_step = [i for i, g in enumerate(goals) if g[0]]
    assert done_step == [15]
    assert goals[15][2]  # the done flag is carried to the partner for its next call
    assert goals[15][1] != 0, "test needs the planner to hold something when the episode ends"
    assert goals[16][1] == 0  # next call resets the planner's agent state


@pytest.mark.parametrize("method", ["ft", "ewc", "mas"])
def test_cl_smoke_with_real_partner_and_planner(cpa, method, monkeypatch):
    cl = {"ft": FT(), "ewc": EWC("online", 0.9), "mas": MAS("online", 0.9)}[method]
    cfg = cpa.make_cfg(updates=2, eval_every=2, cl_method=method)
    max_soup = {"cramped_room": 5.0}
    builds = []
    original = train_br.make_partner_importance_fn
    monkeypatch.setattr(train_br, "make_partner_importance_fn",
                        lambda *a, **k: builds.append(1) or original(*a, **k))

    params = cpa.ego_params
    cl_state = init_cl_state(params, False, False, cl, cfg)
    cache, logs, cache_sizes = {}, [], []
    stages = [(cpa.partner_policy, cpa.pop_params[0]), (cpa.partner_policy, cpa.pop_params[1]),
              (cpa.planner, None)]
    for stage, (policy, partner_params) in enumerate(stages):
        params, cl_state = run_br_training(
            cfg, cpa.env, cpa.pop_cfg, cpa.ego, params, policy, partner_params, env_id_idx=stage,
            eval_partner=cpa.eval_entries(), max_soup_dict=max_soup, cl=cl, cl_state=cl_state,
            log_fn=logs.append, compiled_cache=cache)
        cache_sizes.append(len(cache))
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params))

    # Two BRDiv partners share one compiled train (and importance) function; the planner adds its own.
    assert cache_sizes[0] == cache_sizes[1] < cache_sizes[2]
    events = _events(logs)
    assert sorted({e["Eval/StageID"] for e in events}) == [0, 1, 2]
    assert len(events) == 3 * 2  # three stages, events after updates 1 and 2 (eval_every=2, last update)

    importance = [np.asarray(x) for x in jax.tree.leaves(cl_state.importance)]
    assert all(np.isfinite(x).all() for x in importance)
    if method == "ft":
        assert builds == []  # no importance rollout is built or run
        assert all((x == 0).all() for x in importance)
    else:
        assert len(builds) == 2  # one per partner type: BRDiv (stages 0-1) and planner (stage 2)
        assert any((x != 0).any() for x in importance)


def test_importance_depends_on_partner_behaviour(cpa):
    """EWC states visited with a planner partner differ from those visited with the old zero-action partner."""
    env = cpa.make_env(16)
    cfg = cpa.make_cfg(cl_method="ewc")
    ewc = EWC("online", 0.9)
    population = HeuristicPolicyPopulation(cpa.planner)
    dummy = jax.tree.map(lambda x: x[None], cpa.ego_params)
    key = jax.random.PRNGKey(3)

    with_partner = train_br.make_partner_importance_fn(ewc, env, cpa.ego.network, population, cfg)(
        cpa.ego_params, jnp.int32(0), key, dummy)

    def old_step(k, state, actions, task_idx):
        return env.step(k, state, {**actions, "agent_1": jnp.zeros_like(actions["agent_0"])})

    old = ewc.make_importance_fn(
        lambda k, t: env.reset(k), old_step, cpa.ego.network, ["agent_0"], False, cfg.importance_episodes,
        cfg.importance_steps, False, cfg.importance_stride)(cpa.ego_params, jnp.int32(0), key)
    diff = sum(float(jnp.abs(a - b).sum()) for a, b in zip(jax.tree.leaves(with_partner), jax.tree.leaves(old)))
    assert diff > 0


def test_incomplete_episodes_are_not_reported_as_returns(cpa):
    """A 16-step rollout inside a 400-step episode never completes an episode, so no episode value exists."""
    env = cpa.make_env(400)
    cfg = cpa.make_cfg(num_steps=16, updates=1, eval_every=1)
    keys = make_stage_keys(cfg.seed, 0)
    logs = []
    out = train_ppo_ego_agent(
        cfg, env, keys.train, cpa.ego, cpa.ego_params, 1, DummyPolicyPopulation(cpa.partner_policy),
        jax.tree.map(lambda x: x[None], cpa.pop_params[0]), env_id_idx=0, eval_partner=cpa.eval_entries(),
        eval_rng=keys.eval, log_fn=logs.append)

    assert int(np.asarray(out["metrics"]["n_episodes"]).sum()) == 0
    (train_log,) = [log for log in logs if "Train/EgoGradNorm" in log]
    assert train_log["Train/EpisodesCompleted"] == 0
    assert "Train/EgoSoup" not in train_log and "Train/EgoReturn" not in train_log
    (event,) = _events(logs)
    assert all(event[f"Eval/Episodes_Partner{i}"] == 0 for i in range(3))
    assert not any("Eval/EgoSoup_Partner" in k or "Eval/EgoReturn_Partner" in k for k in event)
