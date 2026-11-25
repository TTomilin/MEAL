import numpy as np
import pytest

try:
    import experiments.partner_adaptation.jax_aht_bridge  # noqa: F401
except ImportError as exc:  # JaxAHTUnavailable is an ImportError
    pytest.skip(f"jax-aht not available: {exc}", allow_module_level=True)

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from envs.base_env import BaseEnv, WrappedEnvState
from envs.overcooked.augmented_layouts import augmented_layouts
from envs.overcooked.overcooked_v1 import OvercookedV1
from experiments.partner_adaptation.jax_aht_bridge import compat, make_env, MEAL_OVERCOOKED
from experiments.partner_adaptation.jax_aht_bridge.env_adapter import MealOvercookedAHT
from meal.env.overcooked import Overcooked, overcooked_layouts

LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring", "counter_circuit", "forced_coord"]
# jax-aht teammate_generation/configs/task/overcooked-v1/*.yaml
TASK_SHAPING = {"PLACEMENT_IN_POT_REW": 0.5, "PLATE_PICKUP_REWARD": 0.1, "SOUP_PICKUP_REWARD": 1.0,
                "ONION_PICKUP_REWARD": 0.1, "COUNTER_PICKUP_REWARD": 0.0, "COUNTER_DROP_REWARD": 0.0}
BATCH, STEPS = 64, 40


def test_action_and_direction_tables_are_consistent_permutations():
    for table in (compat.JAXMARL_ACTION_TO_MEAL, compat.MEAL_ACTION_TO_JAXMARL,
                  compat.OVERCOOKED_AI_ACTION_TO_MEAL, compat.MEAL_ACTION_TO_OVERCOOKED_AI):
        assert sorted(table.tolist()) == list(range(6))
        assert table[4] == 4 and table[5] == 5
    a = jnp.arange(6)
    np.testing.assert_array_equal(compat.meal_action_to_jaxmarl(compat.jaxmarl_action_to_meal(a)), a)
    np.testing.assert_array_equal(compat.MEAL_ACTION_TO_OVERCOOKED_AI[compat.OVERCOOKED_AI_ACTION_TO_MEAL],
                                  np.arange(6))
    from jaxmarl.environments.overcooked.common import DIR_TO_VEC as JM_DIR
    from meal.env.overcooked.common import DIR_TO_VEC as MEAL_DIR
    for j in range(4):
        np.testing.assert_array_equal(np.asarray(JM_DIR[j]), np.asarray(MEAL_DIR[compat.JAXMARL_ACTION_TO_MEAL[j]]))
        np.testing.assert_array_equal(compat.OVERCOOKED_AI_DIR_TO_VEC[j],
                                      np.asarray(MEAL_DIR[compat.OVERCOOKED_AI_ACTION_TO_MEAL[j]]))


@pytest.mark.parametrize("name", LAYOUTS)
def test_layouts_match(name):
    meal, jm = overcooked_layouts[name], augmented_layouts[name]
    assert (int(meal["height"]), int(meal["width"])) == (int(jm["height"]), int(jm["width"]))
    for key in ["wall_idx", "agent_idx", "goal_idx", "plate_pile_idx", "onion_pile_idx", "pot_idx"]:
        assert set(np.asarray(meal[key]).tolist()) == set(np.asarray(jm[key]).tolist()), key


def test_following_is_allowed_only_in_aht_dynamics():
    from experiments.partner_adaptation.jax_aht_bridge.heldout import AhtEvalOvercooked
    current = jnp.asarray([[1, 1], [2, 1]], jnp.uint32)
    proposed = jnp.asarray([[2, 1], [3, 1]], jnp.uint32)
    adapter = MealOvercookedAHT("cramped_room")
    evaluation = AhtEvalOvercooked(overcooked_layouts["cramped_room"])
    for env in (adapter.env, evaluation):
        np.testing.assert_array_equal(env._resolve_collisions(current, proposed), proposed)
        np.testing.assert_array_equal(env._resolve_collisions(current, current[::-1]), current)
        np.testing.assert_array_equal(env._resolve_collisions(current, jnp.asarray([[2, 1], [2, 1]])), current)
    legacy = Overcooked(layout_name="cramped_room")
    np.testing.assert_array_equal(legacy._resolve_collisions(current, proposed), [[1, 1], [3, 1]])


@pytest.mark.parametrize("name", LAYOUTS)
def test_one_step_parity_with_overcooked_v1(name):
    layout = overcooked_layouts[name]
    n_pots, n_goals = len(layout["pot_idx"]), len(layout["goal_idx"])
    sampler = Overcooked(layout_name=name, max_steps=400, random_reset=True)
    adapter = MealOvercookedAHT(name, random_reset=False, do_reward_shaping=True, reward_shaping_params=TASK_SHAPING)
    jm = OvercookedV1(layout=FrozenDict(augmented_layouts[name]), max_steps=400, do_reward_shaping=True,
                      reward_shaping_params=FrozenDict({k: [v, v] for k, v in TASK_SHAPING.items()}))
    to_jm = jax.vmap(lambda s: compat.meal_state_to_jaxmarl(s, n_pots, n_goals))
    adapter_step = jax.jit(jax.vmap(adapter.step))
    jm_step = jax.jit(jax.vmap(jm.step_env))
    meal_obs = jax.jit(jax.vmap(sampler.get_obs))
    jm_obs = jax.jit(jax.vmap(jm.get_obs))

    key = jax.random.PRNGKey(0)
    key, k = jax.random.split(key)
    _, state = jax.vmap(sampler.reset)(jax.random.split(k, BATCH))
    js = to_jm(state)
    zeros = jnp.zeros((BATCH, 2))
    compared = deliveries = shaped_events = 0
    for _ in range(STEPS):
        mo, jo = meal_obs(state), jm_obs(js)
        for agent in ("agent_0", "agent_1"):
            np.testing.assert_array_equal(np.asarray(compat.meal_obs_to_jaxmarl(mo[agent])), np.asarray(jo[agent]))

        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (BATCH, 2), 0, 6)
        steps = jax.random.split(k_step, BATCH)
        wrapped = WrappedEnvState(state, zeros, zeros, jnp.zeros((BATCH,), jnp.int32))
        a_obs, a_state, a_rew, _, a_info = adapter_step(steps, wrapped, {"agent_0": acts[:, 0], "agent_1": acts[:, 1]})
        j_obs, j_state, j_rew, _, j_info = jm_step(steps, js, {"agent_0": compat.meal_action_to_jaxmarl(acts[:, 0]),
                                                              "agent_1": compat.meal_action_to_jaxmarl(acts[:, 1])})
        conv = to_jm(a_state.env_state)
        for field in ("agent_pos", "agent_dir_idx", "agent_inv", "maze_map"):
            np.testing.assert_array_equal(np.asarray(getattr(conv, field)),
                                          np.asarray(getattr(j_state, field)), err_msg=field)
        np.testing.assert_allclose(np.asarray(a_info["base_reward"]), np.asarray(j_info["base_reward"]))
        np.testing.assert_allclose(np.asarray(a_info["shaped_reward"]), np.asarray(j_info["shaped_reward"]),
                                   atol=1e-6)
        for agent in ("agent_0", "agent_1"):
            np.testing.assert_allclose(np.asarray(a_rew[agent]), np.asarray(j_rew[agent]), atol=1e-6)
            np.testing.assert_array_equal(np.asarray(a_obs[agent]),
                                          np.asarray(j_obs[agent]).reshape(BATCH, -1))
        compared += BATCH
        deliveries += int((np.asarray(j_info["base_reward"])[:, 0] > 0).sum())
        shaped_events += int((np.asarray(j_info["shaped_reward"]) > 0).sum())
        state = a_state.env_state
        js = j_state
    assert compared == BATCH * STEPS
    assert shaped_events > 0
    if name in ("cramped_room", "coord_ring", "counter_circuit"):
        assert deliveries > 0, "random object states should produce at least one delivery"


def test_adapter_contract_and_episode_bookkeeping():
    env = make_env(MEAL_OVERCOOKED, {"layout": "coord_ring", "max_steps": 5})
    assert isinstance(env, BaseEnv)
    obs, state = env.reset(jax.random.PRNGKey(0))
    h, w = env.height, env.width
    assert obs["agent_0"].shape == (h * w * 26,) == env.observation_space("agent_0").shape
    assert env.action_space("agent_0").n == 6
    np.testing.assert_array_equal(env.get_avail_actions(state)["agent_1"], np.ones(6))

    _, reset_target = env.reset(jax.random.PRNGKey(42))
    key = jax.random.PRNGKey(1)
    for t in range(5):
        key, k = jax.random.split(key)
        obs, state, rew, done, info = env.step(k, state, {"agent_0": jnp.int32(4), "agent_1": jnp.int32(4)},
                                               reset_state=reset_target)
        assert int(env.get_step_count(state)) == (t + 1) % 5
        assert info["base_return"].shape == (2,)
    assert bool(done["__all__"])
    for field in ("agent_pos", "agent_dir_idx", "maze_map"):
        np.testing.assert_array_equal(getattr(state.env_state, field), getattr(reset_target.env_state, field))
    np.testing.assert_array_equal(state.base_return_so_far, np.zeros(2))


@pytest.mark.parametrize("name", ["asymm_advantages", "forced_coord"])
def test_random_reset_puts_agents_in_different_components(name):
    env = MealOvercookedAHT(name, random_reset=True)
    assert env.num_components >= 2
    comp = np.asarray(env.free_space_map)
    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 32))
    pos = np.asarray(states.env_state.agent_pos)
    idx = pos[..., 1] * env.width + pos[..., 0]
    assert (comp[idx[:, 0]] != comp[idx[:, 1]]).all()
    assert (comp[idx] > 0).all()
    assert len({tuple(p.ravel()) for p in pos}) > 1


def test_random_obj_state_randomises_pots_and_inventory():
    env = MealOvercookedAHT("cramped_room", random_reset=False, random_obj_state=True)
    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 64))
    s = states.env_state
    pots = np.asarray(s.maze_map)[:, np.asarray(s.pot_pos[0, :1, 1]), np.asarray(s.pot_pos[0, :1, 0]), 2]
    assert len(np.unique(pots)) > 1 and pots.max() <= 23
    assert len(np.unique(np.asarray(s.agent_inv))) > 1


def test_step_info_leaves_match_jaxaht_num_agents_contract():
    env = MealOvercookedAHT("coord_ring", random_reset=False, do_reward_shaping=True)
    _, state = env.reset(jax.random.PRNGKey(0))
    _, _, _, _, info = env.step(jax.random.PRNGKey(1), state, {"agent_0": jnp.int32(4), "agent_1": jnp.int32(4)})
    assert "events" not in info
    for key, val in info.items():
        assert val.shape == (env.num_agents,), f"info[{key!r}] has shape {val.shape}, expected ({env.num_agents},)"


def test_custom_cook_time_uses_dynamic_pot_empty_status():
    env = MealOvercookedAHT("cramped_room", random_reset=False, random_obj_state=True,
                            do_reward_shaping=True, cook_time=10)
    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 64))
    s = states.env_state
    pots = np.asarray(s.maze_map)[:, np.asarray(s.pot_pos[0, :1, 1]), np.asarray(s.pot_pos[0, :1, 0]), 2]
    assert int(s.pot_empty_status[0]) == 13
    assert len(np.unique(pots)) > 1 and pots.max() <= 13


def test_custom_and_random_cook_time_step_shaping_under_jit():
    from meal.env.overcooked.common import OBJECT_TO_INDEX

    env = MealOvercookedAHT("cramped_room", random_reset=False, do_reward_shaping=True,
                            reward_shaping_params={"PLACEMENT_IN_POT_REW": 3.0}, cook_time=10)
    _, state = env.reset(jax.random.PRNGKey(0))
    raw = state.env_state
    # Agent 0 at (2, 1) facing up onto pot (2, 0), holding onion; pot status = 13 (> pot_full_status=10).
    raw = raw.replace(
        agent_pos=jnp.array([[2, 1], [3, 2]], dtype=raw.agent_pos.dtype),
        agent_dir_idx=jnp.array([0, 0], dtype=raw.agent_dir_idx.dtype),
        agent_inv=jnp.array([OBJECT_TO_INDEX["onion"], OBJECT_TO_INDEX["empty"]], dtype=raw.agent_inv.dtype),
    )
    _, nxt, _, _, info = env.step(jax.random.PRNGKey(1), state.replace(env_state=raw),
                                  {"agent_0": jnp.int32(5), "agent_1": jnp.int32(4)})
    assert float(info["shaped_reward"][0]) == pytest.approx(3.0)
    assert int(nxt.env_state.maze_map[0, 2, 2]) == 12

    rand_env = MealOvercookedAHT("cramped_room", random_reset=True, random_obj_state=True,
                                 do_reward_shaping=True, random_cook_time=True, random_pot_size=True)
    keys = jax.random.split(jax.random.PRNGKey(2), 16)
    _, states = jax.vmap(rand_env.reset)(keys)
    assert len(np.unique(np.asarray(states.env_state.pot_empty_status))) > 1
    _, _, _, _, infos = jax.vmap(rand_env.step)(
        jax.random.split(jax.random.PRNGKey(3), 16), states,
        {"agent_0": jnp.full((16,), 5, jnp.int32), "agent_1": jnp.full((16,), 4, jnp.int32)},
    )
    assert infos["shaped_reward"].shape == (16, 2)
