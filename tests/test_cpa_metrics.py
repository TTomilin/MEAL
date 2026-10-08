"""Metric aggregation, evaluation-event and RNG-stream tests for the continual partner-adaptation (CPA) path."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import FrozenDict

from experiments.partner_adaptation.train_ego import (
    build_eval_event, build_train_log, group_eval_partners, make_stage_keys, should_evaluate,
    team_episode_summary,
)
from meal import make_env
from meal.env.overcooked import DELIVERY_REWARD
from meal.env.overcooked.layouts.presets import cramped_room
from meal.wrappers.logging import LogWrapper

A = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'S': 4, 'I': 5}
ONION_CYCLE = [A['L'], A['I'], A['R'], A['U'], A['I']]
COOK_AND_DELIVER = (
        ONION_CYCLE * 3 + [A['S']] * 5
        + [A['D'], A['L'], A['D'], A['I'], A['U'], A['R'], A['U'], A['I'], A['D'], A['R'], A['D'], A['I']]
)
PLATE_THEN_DELIVER_0 = (
        ONION_CYCLE * 3 + [A['S']] * 5
        + [A['D'], A['L'], A['D'], A['I'], A['U'], A['R'], A['U'], A['I'], A['D'], A['I'], A['L']]
)


def run_logged_episode(actions_0, actions_1, max_steps, n_steps=None):
    """Scripted rollout through the same LogWrapper'd env the CPA pipeline uses.

    Steps `n_steps` (default: the whole episode) and returns the final info plus the per-step raw
    rewards and soups of both agents.
    """
    env = LogWrapper(make_env("overcooked", layout=FrozenDict(cramped_room), max_steps=max_steps, cook_time=5))
    pad = max_steps - len(actions_0)
    actions_0 = list(actions_0) + [A['S']] * pad
    actions_1 = list(actions_1) + [A['S']] * (max_steps - len(actions_1))
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    rewards, soups, info = [], [], None
    for t in range(n_steps or max_steps):
        key, step_key = jax.random.split(key)
        act = {"agent_0": jnp.uint32(actions_0[t]), "agent_1": jnp.uint32(actions_1[t])}
        _, state, reward, _, info = env.step(step_key, state, act)
        rewards.append([float(reward["agent_0"]), float(reward["agent_1"])])
        soups.append([float(info["soups"]["agent_0"]), float(info["soups"]["agent_1"])])
    return info, np.array(rewards), np.array(soups)


def test_single_agent_delivery_counted_once():
    info, rewards, soups = run_logged_episode(COOK_AND_DELIVER, [], max_steps=len(COOK_AND_DELIVER))
    team_soups, team_returns, completed = team_episode_summary(
        info["returned_episode_soups"], info["returned_episode_returns"], info["returned_episode"])
    assert bool(completed)
    assert float(team_soups) == 1.0
    assert float(team_returns) == DELIVERY_REWARD
    # The env reports delivery only for the delivering agent, so there is nothing shared to double count.
    assert rewards[:, 1].sum() == 0 and soups[:, 1].sum() == 0
    assert soups.sum() == 1.0 and rewards.sum() == DELIVERY_REWARD


def test_split_roles_delivery_counted_once():
    wait = len(ONION_CYCLE) * 3 + 5 + 10
    actions_1 = [A['S']] * wait + [A['L'], A['D'], A['I'], A['R'], A['D'], A['I']]
    info, rewards, soups = run_logged_episode(PLATE_THEN_DELIVER_0, actions_1, max_steps=len(actions_1))
    team_soups, team_returns, completed = team_episode_summary(
        info["returned_episode_soups"], info["returned_episode_returns"], info["returned_episode"])
    assert bool(completed)
    assert float(team_soups) == soups.sum() == 1.0
    assert soups[:, 0].sum() == 0 and soups[:, 1].sum() == 1.0  # credited to the deliverer only
    assert float(team_returns) == rewards.sum() == DELIVERY_REWARD


def test_incomplete_episode_is_flagged_not_invented():
    info, _, soups = run_logged_episode(
        COOK_AND_DELIVER, [], max_steps=len(COOK_AND_DELIVER) + 50, n_steps=len(COOK_AND_DELIVER))
    assert soups.sum() == 1.0  # the soup was delivered ...
    team_soups, team_returns, completed = team_episode_summary(
        info["returned_episode_soups"], info["returned_episode_returns"], info["returned_episode"])
    assert not bool(completed)  # ... but the episode has not completed, so no episode value is reported
    assert float(team_soups) == 0.0 and float(team_returns) == 0.0


def test_should_evaluate_schedule():
    def events(every, total):
        return [u for u in range(1, total + 1) if bool(should_evaluate(jnp.int32(u), every, total))]

    assert events(1, 4) == [1, 2, 3, 4]
    assert events(2, 4) == [1, 3, 4]
    assert events(3, 7) == [1, 4, 7]
    assert events(10, 5) == [1, 5]


def test_eval_event_means_only_completed_episodes():
    soups = np.array([[2.0, 4.0, 9.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    rets = soups * DELIVERY_REWARD
    completed = np.array([[True, True, False], [True, True, True], [False, False, False]])
    record, log = build_eval_event(
        stage_id=1, update_steps=3, train_episodes=12, partner_ids=[0, 1, 2], team_soups=soups, team_returns=rets,
        completed=completed, num_updates=10, steps_per_update=100, max_soup=5.0)

    assert record["stage_id"] == 1 and record["update"] == 3 and record["train_episodes"] == 12
    assert record["partner_ids"] == [0, 1, 2] and record["episode_ids"] == [0, 1, 2]
    assert record["env_steps"] == (1 * 10 + 3) * 100
    assert log["Eval/EnvSteps"] == 1300 and log["Eval/StageID"] == 1 and log["train_step"] == 12

    assert log["Eval/EgoSoup_Partner0"] == 3.0  # the uncompleted 9.0 is excluded
    assert log["Eval/EgoReturn_Partner0"] == 3.0 * DELIVERY_REWARD
    assert log["Eval/EgoSoup_Partner1"] == 1.0
    assert log["Eval/EgoSoup_scaled_Partner1"] == pytest.approx(0.2)
    assert log["Eval/Episodes_Partner2"] == 0
    assert "Eval/EgoSoup_Partner2" not in log  # no completed episode: omitted, not logged as zero
    # Headline Eval/Ego* metrics describe the partner of the current stage (id == stage_id == 1)
    assert log["Eval/EgoSoup"] == 1.0


def test_train_log_omits_episode_stats_without_completed_episodes():
    kwargs = dict(stage_id=0, update_steps=2, num_updates=5, value_loss=1.0, actor_loss=2.0, entropy_loss=3.0,
                  grad_norm=4.0)
    assert "Train/EgoSoup" not in build_train_log(n_episodes=0, team_soup=0.0, ego_return=0.0, **kwargs)
    log = build_train_log(n_episodes=3, team_soup=2.0, ego_return=40.0, max_soup=4.0, **kwargs)
    assert log["Train/EgoSoup"] == 2.0 and log["Train/EgoSoup_scaled"] == 0.5 and log["train_step"] == 1


def _entry(policy, idx):
    return SimpleNamespace(policy_cls=policy), {"w": jnp.full((1, 2), float(idx))}, idx


def test_group_eval_partners_dedupes_and_adds_current_once():
    shared, planner = object(), object()
    entries = [_entry(shared, 0), _entry(shared, 1), _entry(planner, 2), _entry(shared, 1)]  # id 1 listed twice

    policies, params, ids = group_eval_partners(entries, current=_entry(shared, 1))
    flat = [int(i) for g in ids for i in g]
    assert sorted(flat) == [0, 1, 2]  # duplicate id and the (listed) current partner appear once
    assert policies == (shared, planner)
    assert ids[0].dtype == jnp.int32 and params[0]["w"].shape == (2, 2)

    # A current partner that is not in the list is added
    _, _, ids = group_eval_partners(entries[:1], current=_entry(planner, 5))
    assert sorted(int(i) for g in ids for i in g) == [0, 5]


def test_stage_keys_are_reproducible_and_independent():
    a, b = make_stage_keys(0, 0), make_stage_keys(0, 1)
    again = make_stage_keys(0, 0)
    flat = [np.asarray(jax.random.key_data(k)).tobytes() for k in (*a, *b)]
    assert len(set(flat)) == 6  # three streams per stage, none shared across stages
    assert all(np.array_equal(jax.random.key_data(x), jax.random.key_data(y)) for x, y in zip(a, again))
    assert not np.array_equal(jax.random.key_data(make_stage_keys(1, 0).train), jax.random.key_data(a.train))
