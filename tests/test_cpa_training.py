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
from experiments.partner_adaptation import train_br, train_ego
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
from experiments.partner_adaptation.partner_generation.run_episodes import run_episodes
from experiments.partner_adaptation.train_ego import (
    AGENT_STAT_COLUMNS, STAY_INDEX, make_stage_keys, train_ppo_ego_agent,
)
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

    def make_logged_env(episode_steps, agent_stats=False):
        if (episode_steps, agent_stats) not in envs:
            envs[(episode_steps, agent_stats)] = LogWrapper(
                make_env("overcooked", **layout, max_steps=episode_steps), agent_stats=agent_stats)
        return envs[(episode_steps, agent_stats)]

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


# ----------------------------------------------------------------------------------------------------------
# per-agent diagnostics (ego = agent_0, partner = agent_1)
# ----------------------------------------------------------------------------------------------------------

def _train_with_stats(cpa, cfg, env, partner="learned", eval_partner=None):
    """One stage; returns (out, W&B-style logs, (kind, record) pairs). Single seed, fixed keys."""
    keys = make_stage_keys(cfg.seed, 0)
    logs, records = [], []
    if partner == "learned":
        population, params = DummyPolicyPopulation(cpa.partner_policy), jax.tree.map(lambda x: x[None], cpa.pop_params[0])
    else:
        population, params = HeuristicPolicyPopulation(cpa.planner), jax.tree.map(lambda x: x[None], cpa.ego_params)
    out = train_ppo_ego_agent(
        cfg, env, keys.train, cpa.ego, cpa.ego_params, 1, population, params, env_id_idx=0,
        eval_partner=cpa.eval_entries() if eval_partner is None else eval_partner, eval_rng=keys.eval,
        log_fn=logs.append, record_fn=lambda kind, record, log: records.append((kind, record)))
    return out, logs, records


def _stat(vector, name):
    return float(np.asarray(vector)[AGENT_STAT_COLUMNS.index(name)])


def test_agent_stats_do_not_change_the_experiment(cpa, monkeypatch):
    cfg = cpa.make_cfg(updates=2, eval_every=1)
    batches = []
    real = train_ego._create_minibatches_no_time
    monkeypatch.setattr(train_ego, "_create_minibatches_no_time",
                        lambda traj_batch, *a, **k: batches.append(sorted(traj_batch.info)) or real(traj_batch, *a, **k))
    off, off_logs, off_records = _train_with_stats(cpa, cfg, cpa.make_env(16, False))
    on, on_logs, on_records = _train_with_stats(cpa, cfg, cpa.make_env(16, True))

    # Same policy, same learning: instrumentation reads values the env already computes and draws no random numbers,
    # so parameters agree to float round-off (XLA may fuse the two programs differently).
    for a, b in zip(jax.tree.leaves(off["final_params"]), jax.tree.leaves(on["final_params"])):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)
    for name in off["metrics"]:
        np.testing.assert_allclose(np.asarray(off["metrics"][name]), np.asarray(on["metrics"][name]),
                                   rtol=1e-6, atol=1e-7, err_msg=name)
    assert set(on["metrics"]) == set(off["metrics"])  # no diagnostic array is added to the scan output
    # the diagnostics never enter the PPO batch (and so neither minibatches nor replay memory)
    assert len(batches) == 2 and batches[0] == batches[1]
    assert not any("episode_actions" in k or "episode_shaped" in k for k in batches[1])
    # team scores and evaluation results agree; legacy keys keep their team semantics in both
    for name in ("Train/EgoSoup", "Train/EgoReturn"):
        assert [l[name] for l in off_logs if name in l] == [l[name] for l in on_logs if name in l]
    for a, b in zip(_events(off_logs), _events(on_logs)):
        for pid in range(3):
            assert a[f"Eval/EgoSoup_Partner{pid}"] == b[f"Eval/EgoSoup_Partner{pid}"]
            assert a[f"Eval/EgoReturn_Partner{pid}"] == b[f"Eval/EgoReturn_Partner{pid}"]
    assert all(r["agent_stats"] is None for k, r in off_records if k == "eval")
    assert not any("EgoSoupsDelivered" in k for l in off_logs for k in l)

    # identical evaluation episodes (same key) with and without the extra counters: same actions, same outcome
    results = []
    for flag in (False, True):
        results.append(run_episodes(
            jax.random.PRNGKey(7), cpa.make_env(16, flag), cpa.ego_params, cpa.ego, cpa.pop_params[0],
            cpa.partner_policy, max_episode_steps=16, num_eps=3))
    for key in ("returned_episode_returns", "returned_episode_soups", "returned_episode_lengths"):
        np.testing.assert_array_equal(np.asarray(results[0][key]), np.asarray(results[1][key]))


@pytest.mark.parametrize("partner", ["learned", "planner"])
def test_training_and_evaluation_agent_stats_are_consistent(cpa, partner):
    cfg = cpa.make_cfg(updates=2, eval_every=1)
    _, logs, records = _train_with_stats(cpa, cfg, cpa.make_env(16, True), partner)
    steps = 16

    trains = [r for k, r in records if k == "train"]
    assert len(trains) == 2 and all(r["episodes"] == 4 for r in trains)  # every env finishes its 16-step episode
    for r in trains:
        for agent in ("ego", "partner"):
            executed = sum(r[f"{agent}_act_{n}"] for n in train_ego.ACTION_NAMES)
            assert executed == pytest.approx(steps)  # each executed action counted once, none after termination
            assert r[f"{agent}_stay_fraction"] == pytest.approx(r[f"{agent}_act_stay"] / steps)
        assert r["ego_soups"] + r["partner_soups"] == pytest.approx(r["soups"])
    for log, r in zip([l for l in logs if "Train/EgoGradNorm" in l], trains):  # W&B agrees with the record
        assert log["Train/TeamSoups"] == log["Train/EgoSoup"] == r["soups"]
        assert log["Train/EgoSoupsDelivered"] == r["ego_soups"]
        assert log["Train/PartnerRawShapedReturn"] == r["partner_raw_shaped_return"]
        assert log["Train/EgoStayFraction"] == r["ego_stay_fraction"]
    if partner == "planner":  # a planning partner is not idle
        assert any(r["partner_stay_fraction"] < 1.0 for r in trains)

    evals = [r for k, r in records if k == "eval"]
    assert len(evals) == 2
    for ev, log in zip(evals, _events(logs)):
        stats = ev["agent_stats"]
        assert stats.shape == (3, cfg.num_eval_episodes, len(AGENT_STAT_COLUMNS)) and ev["completed"].all()
        counts = stats[..., [AGENT_STAT_COLUMNS.index(f"{a}_act_{n}") for a in ("ego", "partner")
                                for n in train_ego.ACTION_NAMES]].reshape(3, cfg.num_eval_episodes, 2, 6)
        np.testing.assert_array_equal(counts.sum(-1), steps)  # includes the first and the terminal action
        np.testing.assert_allclose(stats[..., 4:6], counts[..., STAY_INDEX] / steps)
        np.testing.assert_allclose(stats[..., 0] + stats[..., 1], ev["team_soups"])  # per episode
        assert (log["EvalFull/MeanTeamSoups"]
                == pytest.approx(np.mean([log[f"EvalFull/TeamSoups_Partner{i}"] for i in range(3)])))
        for i in range(3):
            assert log[f"EvalFull/EgoSoupsDelivered_Partner{i}"] == pytest.approx(stats[i, :, 0].mean())
            assert log[f"EvalFull/PartnerStayFraction_Partner{i}"] == pytest.approx(stats[i, :, 5].mean())
            assert log[f"EvalFull/TeamSoups_Partner{i}"] == log[f"Eval/EgoSoup_Partner{i}"]  # legacy = team


def test_agent_stats_with_no_completed_episode_are_absent_not_zero(cpa):
    """A 16-step rollout inside a 400-step episode completes nothing: no episode mean, no synthetic zero."""
    cfg = cpa.make_cfg(num_steps=16, updates=1, eval_every=1)
    _, logs, records = _train_with_stats(cpa, cfg, cpa.make_env(400, True))
    (train_log,) = [l for l in logs if "Train/EgoGradNorm" in l]
    assert not any(k in train_log for k in (
        "Train/TeamSoups", "Train/EgoSoupsDelivered", "Train/PartnerSoupsDelivered", "Train/EgoRawShapedReturn",
        "Train/PartnerRawShapedReturn", "Train/EgoStayFraction", "Train/PartnerStayFraction"))
    (train_record,) = [r for k, r in records if k == "train"]
    assert all(train_record.get(c) is None for c in AGENT_STAT_COLUMNS)
    (event,) = _events(logs)
    assert not any(k.startswith("EvalFull/") and k != "EvalFull/EnvSteps" for k in event)
    (eval_record,) = [r for k, r in records if k == "eval"]
    assert not eval_record["completed"].any()


def test_full_and_current_events_are_separate_and_use_their_own_names(cpa):
    cfg = cpa.make_cfg(updates=3, eval_every=1, eval_schedule="pilot", eval_current_every=1)
    _, logs, _ = _train_with_stats(cpa, cfg, cpa.make_env(16, True))
    events = _events(logs)
    assert [e["train_step"] for e in events] == [0, 1, 2]  # one event per update, never both kinds at one update
    scopes = ["current" if "EvalCurrent/EnvSteps" in e else "full" for e in events]
    assert scopes == ["current", "current", "full"]
    for event, scope in zip(events, scopes):
        other = "EvalFull" if scope == "current" else "EvalCurrent"
        assert not any(k.startswith(other) for k in event)
        assert event[f"Eval{scope.capitalize()}/EnvSteps"] == event["Eval/EnvSteps"]
    assert "EvalCurrent/TeamSoups" in events[0] and "EvalCurrent/EgoStayFraction" in events[0]
    assert "EvalCurrent/TeamSoups_Partner0" not in events[0]
    assert "EvalFull/MeanTeamSoups" in events[2] and "EvalFull/EgoStayFraction_Partner2" in events[2]
