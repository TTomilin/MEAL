import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.model.mlp import ActorCritic
from experiments.partner_adaptation import run_br
from experiments.partner_adaptation.partner_agents.agent_interface import MLPActorCriticPolicyCL
from experiments.partner_adaptation.partner_agents.overcooked.agent_policy_wrappers import \
    OvercookedOnionPolicyWrapper
from experiments.partner_adaptation.run_br import TrainConfig, execute, video_name
from experiments.partner_adaptation.train_br import (
    HeuristicPolicyPopulation, make_partner_population, rollout_ego_with_partner)
from meal import make_env
from meal.env.overcooked.common import OBJECT_TO_INDEX
from meal.env.overcooked.layouts.presets import overcooked_layouts
from meal.wrappers.logging import LogWrapper

POP_DIR = (Path(__file__).resolve().parent.parent
           / "experiments/partner_adaptation/partner_agents/BRDiv_population/cramped_room")
STAY = 4


class StayPopulation(HeuristicPolicyPopulation):
    calls = 0

    def get_actions(self, pop_params, agent_indices, obs, done, avail_actions, hstate, rng, env_state, aux_obs=None):
        StayPopulation.calls += 1
        return jnp.full((agent_indices.shape[0],), STAY, jnp.int32), hstate


@pytest.fixture(scope="module")
def setup():
    layout = overcooked_layouts["cramped_room"]
    env = LogWrapper(make_env("overcooked", layout=layout, max_steps=40))
    obs_dim = int(np.prod(env.observation_space().shape))
    network = ActorCritic(6, "relu", 2, False, False, 16, 1, False, False)
    ego_params = MLPActorCriticPolicyCL(network, obs_dim).init_params(jax.random.PRNGKey(0))
    return env, network, ego_params, layout


def positions(states, agent):
    return {tuple(np.asarray(s.agent_pos[agent]).tolist()) for s in states}


def test_agent_1_is_the_partner_not_the_ego(setup):
    env, network, ego_params, layout = setup
    population = StayPopulation(policy_cls=OvercookedOnionPolicyWrapper(layout=layout))
    _, partner_params = make_partner_population(population.policy_cls, None, ego_params)
    StayPopulation.calls = 0
    states = rollout_ego_with_partner(jax.random.PRNGKey(3), env, network, ego_params, 0, population,
                                      partner_params, max_steps=30)
    assert len(states) == 31
    assert StayPopulation.calls >= 1
    assert len(positions(states, 1)) == 1
    assert len(positions(states, 0)) > 1


def test_planner_partner_acts_in_the_video(setup):
    env, network, ego_params, layout = setup
    population, partner_params = make_partner_population(OvercookedOnionPolicyWrapper(layout=layout), None,
                                                         ego_params)
    cache = {}
    states = rollout_ego_with_partner(jax.random.PRNGKey(0), env, network, ego_params, 0, population,
                                      partner_params, max_steps=30, cache=cache)
    held = [int(s.agent_inv[1]) for s in states]
    assert OBJECT_TO_INDEX["onion"] in held
    assert len(cache) == 1
    rollout_ego_with_partner(jax.random.PRNGKey(1), env, network, ego_params, 0, population, partner_params,
                             max_steps=30, cache=cache)
    assert len(cache) == 1


def test_video_name_is_filesystem_and_wandb_safe():
    assert video_name(3, "coord_ring/P10") == "stage_03_coord_ring_P10"
    assert video_name(12, "cramped_room/pop/m0 (x)") == "stage_12_cramped_room_pop_m0_x"


class FakeRun:
    def __init__(self, **kwargs):
        self.name, self.logged = kwargs.get("name"), []

    def log(self, data, step=None):
        self.logged.append((step, data))

    def finish(self):
        pass


def test_run_records_one_partner_video_per_stage(tmp_path, monkeypatch):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({
        "format_version": 1, "name": "tiny", "layout": "cramped_room", "num_partners": 2,
        "populations": {"pop": {"config": str(POP_DIR / "config.pckl"), "seed_index": 0}},
        "partners": [{"partner_id": 0, "population": "pop", "member_index": 0,
                      "checkpoint": str(POP_DIR / "params_seed0_agent0.pt")},
                     {"partner_id": 1, "kind": "planner", "planner_id": "P01"}]}))
    runs = []
    monkeypatch.setattr(run_br.wandb, "init", lambda **kw: runs.append(FakeRun(**kw)) or runs[-1])
    rollouts = []
    real = run_br.rollout_ego_with_partner

    def spy(rng, env, ego_network, ego_params, env_idx, population, partner_params, max_steps, **kw):
        rollouts.append((env_idx, type(population).__name__, max_steps))
        return real(rng, env, ego_network, ego_params, env_idx, population, partner_params, max_steps, **kw)
    monkeypatch.setattr(run_br, "rollout_ego_with_partner", spy)

    config = TrainConfig(
        mode="offline", checkpoint_path=str(tmp_path / "runs"), layout_name="cramped_room", partner_bank=str(bank),
        num_heuristic_partners=0, num_envs=4, num_steps=16, total_timesteps=4 * 16, update_epochs=1,
        num_minibatches=2, num_eval_episodes=1, hidden_size=16, num_layers=1, cl_method="ft", seed=0,
        reward_shaping_horizon=32, use_task_id=False, use_multihead=False, eval_schedule="pilot",
        record_video=True, gif_len=16)
    result = execute(config)

    assert rollouts == [(0, "DummyPolicyPopulation", 16), (1, "HeuristicPolicyPopulation", 16)]
    videos = sorted(p.name for p in (result.run_dir / "videos").iterdir())
    assert videos == ["stage_00_cramped_room_pop_m0.mp4", "stage_01_cramped_room_P01.mp4"]
    logged = {k: step for step, d in runs[0].logged for k in d if k.startswith("Video/")}
    assert set(logged) == {"Video/stage_00_cramped_room_pop_m0", "Video/stage_01_cramped_room_P01"}
    assert logged["Video/stage_00_cramped_room_pop_m0"] == 4 * 16
    assert logged["Video/stage_01_cramped_room_P01"] == 2 * 4 * 16
    assert json.loads((result.run_dir / "status.json").read_text())["state"] == "complete"
