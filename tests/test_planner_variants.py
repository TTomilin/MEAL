"""Planner partners P01-P12 (CPU only).

State-based tests build an Overcooked state, move agents / set pots and counters directly, and check the planner's
decision (and the action it takes) for the variant under test. Rollout tests use the real env with the existing
competent planners as counterparts. Nothing here measures strength or diversity of the partners; see
`partner_quality.py` for the diagnostic.
"""
import json
import pickle
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.model.mlp import ActorCritic
from experiments.partner_adaptation.partner_agents.agent_interface import (
    ActorWithConditionalCriticPolicy, MLPActorCriticPolicyCL,
)
from experiments.partner_adaptation.partner_agents.overcooked.agent_policy_wrappers import (
    OvercookedPlannerPolicyWrapper,
)
from experiments.partner_adaptation.partner_agents.overcooked.planner_variants import (
    PLANNER_SPECS, STALL_LIMIT, Case, PlannerAgent, layout_collapses, planner_params,
)
from experiments.partner_adaptation.partner_bank import (
    PartnerBankError, assemble_bank, load_partner_bank, load_partners,
)
from experiments.partner_adaptation.partner_generation.run import save_population_members
from experiments.partner_adaptation.partner_quality import QualityConfig, diagnose, format_report, run_layout
from experiments.partner_adaptation.run_br import TrainConfig
from experiments.partner_adaptation.train_br import HeuristicPolicyPopulation, run_br_training
from experiments.utils import init_cl_state
from experiments.continual.ewc import EWC
from meal import make_env
from meal.env.overcooked.common import OBJECT_TO_INDEX
from meal.env.overcooked.layouts.presets import overcooked_layouts
from meal.wrappers.logging import LogWrapper

STAY, INTERACT = 4, 5
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
ONION, PLATE, DISH = (OBJECT_TO_INDEX[k] for k in ("onion", "plate", "dish"))
EMPTY = OBJECT_TO_INDEX["empty"]
IDS = list(PLANNER_SPECS)

# coord_ring (y, x):  ###P#   pots (0,3) (1,4); onion piles (3,0) (4,1); plate pile (2,0); goal (4,2)
#                     #.A.P   agents start at (2,1) [agent_1, the planner] and (1,2)
#                     BA#.#   free counters reachable from the floor, row-major: (0,1) (0,2) (1,0) (2,2) (2,4) (3,4) (4,3)
#                     O...#
#                     #OX##
RING_COUNTERS = [(0, 1), (0, 2), (1, 0), (2, 2), (2, 4), (3, 4), (4, 3)]
RING_POTS = [(0, 3), (1, 4)]


@lru_cache(maxsize=None)
def world(layout="coord_ring"):
    env = make_env("overcooked", layout=overcooked_layouts[layout], max_steps=100)
    agent = PlannerAgent(overcooked_layouts[layout])
    _, state = env.reset(jax.random.PRNGKey(0))
    return env, agent, jax.jit(agent.act), state


def arrange(state, pos1=None, pos0=None, inv1=None, inv0=None, dir1=None, pots=None, items=None):
    """Return `state` with agent_1 / agent_0 moved ((y, x) positions), inventories, pot statuses {(y,x): s}
    and items on counters {(y,x): object}."""
    pos, inv, dirs, maze = state.agent_pos, state.agent_inv, state.agent_dir_idx, state.maze_map
    for who, p in ((1, pos1), (0, pos0)):
        if p is not None:
            pos = pos.at[who].set(jnp.asarray([p[1], p[0]], pos.dtype))
    for who, v in ((1, inv1), (0, inv0)):
        if v is not None:
            inv = inv.at[who].set(jnp.asarray(v, inv.dtype))
    if dir1 is not None:
        dirs = dirs.at[1].set(jnp.asarray(dir1, dirs.dtype))
    for (y, x), s in (pots or {}).items():
        maze = maze.at[y, x, 2].set(s)
    for (y, x), o in (items or {}).items():
        maze = maze.at[y, x, 0].set(o)
    return state.replace(agent_pos=pos, agent_inv=inv, agent_dir_idx=dirs, maze_map=maze)


def decide(layout, state, planner, pstate=None, rng=0, params=None):
    """(action, new planner state, info) of planner `planner` (id) for the agent_1 seat."""
    env, agent, act, _ = world(layout)
    params = planner_params(planner) if params is None else params
    pstate = agent.init_agent_state(1) if pstate is None else pstate
    obs = env.get_obs(state)["agent_1"]
    action, new, info = act(params, obs, state, pstate, jax.random.PRNGKey(rng))
    return int(action), new, {k: np.asarray(v) for k, v in info.items()}


def case_of(layout, state, planner, **kw):
    return int(decide(layout, state, planner, **kw)[2]["case"])


@pytest.fixture(scope="module")
def ring():
    return world("coord_ring")[3]


#  registry

def test_registry_has_twelve_stable_distinct_planners():
    assert IDS == [f"P{i:02d}" for i in range(1, 13)]
    assert len({s.config_sha256 for s in PLANNER_SPECS.values()}) == 12
    assert len({s.name for s in PLANNER_SPECS.values()}) == 12
    p = {k: v.params for k, v in PLANNER_SPECS.items()}
    assert (p["P01"]["p_onion_on_counter"], p["P02"]["p_onion_on_counter"]) == (0.0, 0.8)
    assert (p["P03"]["prefetch_steps"], p["P04"]["prefetch_steps"]) == (0, 10)
    assert (p["P07"]["yield_wait"], p["P08"]["yield_wait"]) == (0, 2)
    assert (p["P11"]["history_window"], p["P12"]["history_window"]) == (8, 32)
    # a pair differs from the fixed generalist only through its own option
    for a, b, key in (("P01", "P02", "p_onion_on_counter"), ("P03", "P04", "prefetch_steps"),
                      ("P05", "P06", "priority"), ("P07", "P08", "yield_wait"), ("P09", "P10", "handoff_rule")):
        assert {k for k in p[a] if p[a][k] != p[b][k]} == {key}


#  variants change their decision

def test_supply_variants_route_onions_differently(ring):
    state = arrange(ring, inv1=ONION)
    env, agent, act, _ = world()
    obs = env.get_obs(state)["agent_1"]
    keys = jax.random.split(jax.random.PRNGKey(0), 64)

    def cases(pid):
        fn = jax.vmap(lambda k: act(planner_params(pid), obs, state, agent.init_agent_state(1), k)[2]["case"])
        return np.asarray(fn(keys))

    assert (cases("P01") == Case.ONION_TO_POT).all()
    counter_share = (cases("P02") == Case.ONION_TO_COUNTER).mean()
    assert 0.6 <= counter_share <= 0.95  # configured 0.8


def test_onion_destination_is_drawn_once_per_held_onion(ring):
    state = arrange(ring, inv1=ONION)
    cases = {k: case_of("coord_ring", state, "P02", rng=k) for k in range(40)}
    to_counter = next(k for k, c in cases.items() if c == Case.ONION_TO_COUNTER)
    to_pot = next(k for k, c in cases.items() if c == Case.ONION_TO_POT)
    _, held, _ = decide("coord_ring", state, "P02", rng=to_counter)
    for rng in (to_pot, to_pot + 1, to_counter):  # later draws must not change the committed choice
        assert case_of("coord_ring", state, "P02", pstate=held, rng=rng) == Case.ONION_TO_COUNTER
    _, committed_pot, _ = decide("coord_ring", state, "P02", rng=to_pot)
    assert case_of("coord_ring", state, "P02", pstate=committed_pot, rng=to_counter) == Case.ONION_TO_POT


@pytest.mark.parametrize("remaining, p03, p04", [
    (0, Case.PICK_PLATE, Case.PICK_PLATE),  # ready: both fetch a plate
    (8, Case.NONE, Case.PICK_PLATE),  # cooking, within P04's 10 steps
    (12, Case.NONE, Case.NONE),  # cooking, beyond it
])
def test_plate_variants_differ_on_prefetch(ring, remaining, p03, p04):
    state = arrange(ring, pots={p: remaining for p in RING_POTS})  # no pot to fill either
    assert case_of("coord_ring", state, "P03") == p03
    assert case_of("coord_ring", state, "P04") == p04


def test_priority_variants_split_only_when_both_feasible(ring):
    both = arrange(ring, pots={(0, 3): 0})  # one pot ready, the other empty: fill and serve both possible
    assert case_of("coord_ring", both, "P05") == Case.PICK_ONION
    assert case_of("coord_ring", both, "P06") == Case.PICK_PLATE
    serve_only = arrange(ring, pots={(0, 3): 0, (1, 4): 12})
    fill_only = arrange(ring)
    for pid in ("P05", "P06"):
        assert case_of("coord_ring", serve_only, pid) == Case.PICK_PLATE
        assert case_of("coord_ring", fill_only, pid) == Case.PICK_ONION


def test_yield_variants_replan_or_wait_when_blocked(ring):
    """The env keeps agent_1 in place (something blocks it); P07 retries at once, P08 waits two steps first."""
    state = arrange(ring, inv1=ONION)  # heads for a pot, so it tries to move
    sequences = {}
    for pid in ("P07", "P08"):
        pstate, seq = None, []
        for _ in range(5):
            action, pstate, _ = decide("coord_ring", state, pid, pstate=pstate, params=planner_params(pid).replace(
                p_onion_on_counter=jnp.float32(0.0)))
            seq.append(action)
        sequences[pid] = seq
    assert sequences["P07"][0] < 4 and all(a < 4 for a in sequences["P07"])
    assert sequences["P08"][0] < 4 and sequences["P08"][1:3] == [STAY, STAY] and sequences["P08"][3] < 4


def test_yield_wait_ends_early_when_the_other_agent_moves(ring):
    state = arrange(ring, inv1=ONION)
    params = planner_params("P08").replace(p_onion_on_counter=jnp.float32(0.0))
    _, pstate, _ = decide("coord_ring", state, "P08", params=params)
    action, pstate, info = decide("coord_ring", state, "P08", pstate=pstate, params=params)
    assert action == STAY and info["waiting"]
    moved = arrange(state, pos0=(3, 1))
    action, _, info = decide("coord_ring", moved, "P08", pstate=pstate, params=params)
    assert action < 4 and not info["waiting"]


def test_handoff_variants_pick_lowest_and_highest_free_shared_counter(ring):
    holding = arrange(ring, inv1=ONION)

    def target(pid, state):
        params = planner_params(pid).replace(p_onion_on_counter=jnp.float32(1.0))
        _, _, info = decide("coord_ring", state, pid, params=params)
        assert info["case"] == Case.ONION_TO_COUNTER
        return tuple(int(v) for v in info["target"])

    assert target("P09", holding) == RING_COUNTERS[0] == (0, 1)  # border corner (0,0) is not a counter
    assert target("P10", holding) == RING_COUNTERS[-1] == (4, 3)
    full = arrange(holding, items={RING_COUNTERS[0]: ONION, RING_COUNTERS[-1]: PLATE})  # occupied counters are skipped
    assert target("P09", full) == RING_COUNTERS[1]
    assert target("P10", full) == RING_COUNTERS[-2]


def _feed_other_inventory(state, planner, inventories):
    pstate = None
    for inv in inventories:
        _, pstate, info = decide("coord_ring", arrange(state, inv0=inv), planner, pstate=pstate)
    return pstate, info


def test_complement_variants_use_their_own_window(ring):
    """The other agent filled for 12 steps, then did nothing for the last 8: P11 (window 8) sees nothing and falls
    back to FILL; P12 (window 32) sees a fill-heavy partner and takes the less covered subtask, SERVE."""
    both = arrange(ring, pots={(0, 3): 0})
    history = [EMPTY, ONION] * 6 + [EMPTY] * 9  # its last release is 9 steps before the decision
    cases = {}
    for pid in ("P11", "P12"):
        pstate, info = _feed_other_inventory(both, pid, history)
        cases[pid] = int(info["case"])
        assert not info["recovered"]
    assert cases == {"P11": Case.PICK_ONION, "P12": Case.PICK_PLATE}


def test_other_agent_events_and_tie_breaking(ring):
    both = arrange(ring, pots={(0, 3): 0})
    # first sight of the other agent holding an onion is not an event; ties go to FILL
    _, info = _feed_other_inventory(both, "P12", [ONION])
    assert (info["covered_fill"], info["covered_serve"], info["case"]) == (0, 0, Case.PICK_ONION)
    # releasing the onion = FILL, picking up a plate, plating and delivering = SERVE
    _, info = _feed_other_inventory(both, "P12", [ONION, EMPTY, PLATE, DISH, EMPTY])
    assert (info["covered_fill"], info["covered_serve"]) == (1, 3)
    assert info["case"] == Case.PICK_ONION and info["complement_decided"]  # serve is covered more: fill


#  unavailable, unreachable, full and carried

@pytest.mark.parametrize("pid", IDS)
def test_nothing_to_do_means_stay(ring, pid):
    cooking = arrange(ring, pots={p: 15 for p in RING_POTS})
    action, _, info = decide("coord_ring", cooking, pid)
    assert (action, info["case"]) == (STAY, Case.NONE)


def test_onion_with_no_pot_and_full_counters_stays_then_uses_a_freed_counter(ring):
    full = arrange(ring, inv1=ONION, pots={p: 15 for p in RING_POTS}, items={c: ONION for c in RING_COUNTERS})
    assert decide("coord_ring", full, "P01")[::2][0] == STAY
    freed = arrange(full, items={(2, 4): 2})  # wall = empty counter
    _, _, info = decide("coord_ring", freed, "P01")  # P01 never prefers counters, but has no pot to use
    assert info["case"] == Case.ONION_TO_COUNTER and tuple(info["target"]) == (2, 4)


def test_plate_in_hand_serves_waits_or_stashes(ring):
    at_pots = arrange(ring, inv1=PLATE, pos1=(1, 3), dir1=RIGHT)  # faces pot (1,4)
    ready = arrange(at_pots, pots={(1, 4): 0, (0, 3): 12})
    action, _, info = decide("coord_ring", ready, "P01")
    assert (info["case"], action) == (Case.PLATE_TO_POT, INTERACT)
    cooking = arrange(at_pots, pots={(1, 4): 9, (0, 3): 12})
    pstate = None
    for _ in range(30):  # waiting beside a cooking pot is not a deadlock
        action, pstate, info = decide("coord_ring", cooking, "P01", pstate=pstate)
        assert (info["case"], action, bool(info["recovered"])) == (Case.PLATE_WAIT, STAY, False)
    idle = arrange(at_pots)  # no soup anywhere: put the plate on a counter
    assert case_of("coord_ring", idle, "P01") == Case.PLATE_STASH


def test_dish_is_delivered(ring):
    at_goal = arrange(ring, inv1=DISH, pos1=(3, 2), dir1=DOWN)  # faces the goal (4,2)
    action, _, info = decide("coord_ring", at_goal, "P01")
    assert (info["case"], action) == (Case.DELIVER, INTERACT)
    far = arrange(ring, inv1=DISH)
    action, _, info = decide("coord_ring", far, "P01")
    assert info["case"] == Case.DELIVER and action in (UP, DOWN, RIGHT, LEFT)


def test_objects_the_agent_dropped_are_not_picked_up_again(ring):
    """An onion left on a counter is meant for the other agent."""
    near_counter = arrange(ring, inv1=ONION, pos1=(1, 2), dir1=UP)  # faces counter (0,2)
    params = planner_params("P02").replace(p_onion_on_counter=jnp.float32(1.0))
    action, pstate, info = decide("coord_ring", near_counter, "P02", params=params)
    assert info["case"] == Case.ONION_TO_COUNTER and tuple(info["target"]) == (0, 2) and action == INTERACT
    dropped = arrange(ring, items={(0, 2): ONION}, pos1=(1, 2), dir1=UP)  # the env's result of that interact
    _, _, info = decide("coord_ring", dropped, "P02", pstate=pstate, params=params)
    assert info["case"] == Case.PICK_ONION and tuple(info["target"]) in [(3, 0), (4, 1)]  # piles, not (0,2)
    fresh_agent = decide("coord_ring", dropped, "P02", params=params)[2]  # someone else's onion is fair game
    assert tuple(fresh_agent["target"]) == (0, 2)


def test_targets_stay_in_the_agents_own_floor_component():
    """asymm_advantages: two rooms. Nothing in the other room (or a handoff across the wall) is ever a target."""
    env, agent, act, state = world("asymm_advantages")
    left = arrange(state, pos1=(3, 1), pos0=(3, 7))
    _, _, info = decide("asymm_advantages", left, "P01")
    assert info["case"] == Case.PICK_ONION and int(info["target"][1]) < 4  # an onion pile of the left room
    holding = arrange(left, inv1=ONION)
    forced = lambda pid: planner_params(pid).replace(p_onion_on_counter=jnp.float32(1.0))
    cases = {pid: decide("asymm_advantages", holding, pid, params=forced(pid))[2] for pid in ("P09", "P10")}
    assert all(i["case"] == Case.ONION_TO_POT for i in cases.values())  # no shared counter: use a pot
    assert (cases["P09"]["target"] == cases["P10"]["target"]).all()


#  legality, reset, deadlock, integration

def test_deadlock_recovery_steps_aside_and_waiting_for_a_soup_does_not_trigger_it(ring):
    """Frozen state while trying to reach a pot (the env keeps rejecting every move): one legal random step."""
    blocked = arrange(ring, inv1=ONION, pos1=(1, 2), pos0=(1, 3))  # the other agent sits on both pots' approach tile
    params = planner_params("P07").replace(p_onion_on_counter=jnp.float32(0.0))
    pstate, recovered_at = None, []
    for t in range(2 * STALL_LIMIT + 4):
        action, pstate, info = decide("coord_ring", blocked, "P07", pstate=pstate, rng=t, params=params)
        if info["recovered"]:
            recovered_at.append(t)
            assert action in (UP, DOWN, RIGHT, LEFT)  # a step to a free floor tile of its own
            y, x = 1, 2
            dy, dx = {UP: (-1, 0), DOWN: (1, 0), RIGHT: (0, 1), LEFT: (0, -1)}[action]
            assert (y + dy, x + dx) != (1, 3)
            assert overcooked_layouts["coord_ring"]["wall_idx"].tolist().count((y + dy) * 5 + x + dx) == 0
    assert recovered_at and recovered_at[0] <= STALL_LIMIT + 1
    assert len(recovered_at) <= 2


def test_done_clears_planner_memory_before_acting(ring):
    wrapper = OvercookedPlannerPolicyWrapper(overcooked_layouts["coord_ring"])
    env = world()[0]
    held = arrange(ring, inv1=ONION, pos0=(3, 3))
    hstate = wrapper.init_hstate(1, {"agent_id": 1})
    for t in range(5):
        now = arrange(held, inv0=PLATE if t % 2 else EMPTY)
        _, hstate = wrapper.get_action(planner_params("P12"), env.get_obs(now)["agent_1"], jnp.asarray(False), None,
                                       hstate, jax.random.PRNGKey(t), now)
    assert int(hstate.case_run) > 0 and bool((hstate.hist >= 0).any())
    obs = env.get_obs(ring)["agent_1"]
    fresh = wrapper.init_hstate(1, {"agent_id": 1})
    a_done, h_done = wrapper.get_action(planner_params("P12"), obs, jnp.asarray(True), None, hstate,
                                        jax.random.PRNGKey(9), ring)
    a_new, h_new = wrapper.get_action(planner_params("P12"), obs, jnp.asarray(False), None, fresh,
                                      jax.random.PRNGKey(9), ring)
    assert int(a_done) == int(a_new)
    for a, b in zip(jax.tree.leaves(h_done), jax.tree.leaves(h_new)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("layout", ["coord_ring", "cramped_room", "asymm_advantages", "counter_circuit"])
def test_every_planner_runs_with_existing_counterparts_on_all_layouts(layout):
    """Construction and a 60-step rollout of all twelve planners against the onion specialist; no BRDiv partners."""
    cfg = QualityConfig(layouts=[layout], counterparts=["onion"], episodes=2, steps=60)
    result = run_layout(layout, cfg)
    assert list(result["rows"]) == IDS
    for pid in IDS:
        actions = result["runs"][pid, "onion"]["action"]
        assert actions.shape == (2, 60) and ((0 <= actions) & (actions <= 5)).all()
    report = format_report(result, diagnose(result, cfg), cfg)
    assert all(pid in report for pid in IDS)


def test_quality_diagnostic_flags_identical_never_active_and_collapsed_variants():
    cfg = QualityConfig(layouts=["cramped_room"], planners=["P05", "P06", "P11", "P12", "P03", "P04"],
                        counterparts=["independent"], episodes=2, steps=60)
    result = run_layout("cramped_room", cfg)
    flags = diagnose(result, cfg)
    assert set(layout_collapses(overcooked_layouts["cramped_room"])) == {"P05/P06", "P11/P12"}
    assert {f.split(":")[0] for f in flags["collapsed"]} == {"P05/P06", "P11/P12"}
    assert not any(f.startswith(("P05", "P06", "P11", "P12")) for f in flags["never_active"])  # explained by collapse
    assert any({"P05", "P06"} <= set(group.split(" = ")) for group in flags["identical"])
    # a variant whose option never mattered is reported when the layout does not explain it
    fake = dict(result, collapses={}, rows={**result["rows"], "P04": dict(
        result["rows"]["P04"], activations={**result["rows"]["P04"]["activations"], "prefetch": 0})})
    assert any(f.startswith("P04") for f in diagnose(fake, cfg)["never_active"])


def test_layouts_report_the_distinctions_they_cannot_express():
    assert set(layout_collapses(overcooked_layouts["asymm_advantages"])) == {"P01/P02", "P07/P08", "P09/P10"}
    assert layout_collapses(overcooked_layouts["coord_ring"]) == {}
    assert layout_collapses(overcooked_layouts["counter_circuit"]) == {}


def _tiny_training(layout_name, planners, cl):
    layout = {"layout": overcooked_layouts[layout_name]}
    env = LogWrapper(make_env("overcooked", **layout, max_steps=16))
    obs_dim = int(np.prod(env.observation_space().shape))
    ego = MLPActorCriticPolicyCL(ActorCritic(6, "relu", len(planners), True, False, 16, 1, True, True), obs_dim)
    params = ego.init_params(jax.random.PRNGKey(1))
    cfg = TrainConfig(layout_name=layout_name, num_envs=4, num_steps=16, total_timesteps=2 * 4 * 16, update_epochs=1,
                      num_minibatches=2, num_eval_episodes=2, eval_every=2, hidden_size=16, num_layers=1,
                      importance_episodes=1, importance_steps=8, cl_method="ewc", reg_coef=1.0, seed=0,
                      reward_shaping_horizon=1e3)
    cfg.layout = layout
    policy = OvercookedPlannerPolicyWrapper(overcooked_layouts[layout_name])
    evals = [(HeuristicPolicyPopulation(policy), jax.tree.map(lambda x: x[None], planner_params(pid)), i)
             for i, pid in enumerate(planners)]
    cl_state = init_cl_state(params, False, False, cl, cfg)
    logs, cache = [], {}
    for stage, pid in enumerate(planners):
        params, cl_state = run_br_training(
            cfg, env, {}, ego, params, policy, planner_params(pid), env_id_idx=stage, eval_partner=evals,
            max_soup_dict={layout_name: 5.0}, cl=cl, cl_state=cl_state, log_fn=logs.append, compiled_cache=cache)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params))
    return logs, cache


@pytest.mark.parametrize("layout_name", ["coord_ring", "cramped_room"])
def test_planners_train_a_continual_ego_with_one_compiled_graph(layout_name):
    """Three planner stages with EWC: planners are real partners (their params are their configuration)."""
    logs, cache = _tiny_training(layout_name, ["P02", "P08", "P11"], EWC("online", 0.9))
    events = [log for log in logs if "Eval/StageID" in log]
    assert sorted({e["Eval/StageID"] for e in events}) == [0, 1, 2]
    assert all(f"Eval/Episodes_Partner{i}" in e for e in events for i in range(3))
    assert len(cache) == 2  # one train function and one importance function shared by all three planners


#  the bank

def _synthetic_brdiv(root: Path, name, layout, size, seed):
    env = make_env("overcooked", layout=overcooked_layouts[layout])
    policy = ActorWithConditionalCriticPolicy(6, int(np.prod(env.observation_space().shape)), size)
    keys = jax.random.split(jax.random.PRNGKey(seed), size)
    stacked = jax.tree.map(lambda *xs: np.stack(xs)[None], *[policy.init_params(k) for k in keys])
    d = root / name
    d.mkdir(parents=True)
    members = save_population_members(str(d), stacked, 1, size)
    (d / "generation.json").write_text(json.dumps({
        "status": "complete", "seed": seed, "num_seeds": 1, "partner_pop_size": size, "activation": "tanh",
        "layout_name": layout, "members": members}))
    return d


def test_expanded_bank_has_24_unique_resolved_partners(tmp_path):
    gens = [str(_synthetic_brdiv(tmp_path, f"g{s}", "coord_ring", 3, s)) for s in (1001, 1002, 1003, 1004)]
    out = tmp_path / "bank" / "bank.json"
    bank = assemble_bank("coord_ring", str(out), gens, bundled=False, planners=("all",))
    assert len(bank) == 24 and [r.partner_id for r in bank.records] == list(range(24))
    assert [r.kind for r in bank.records] == ["brdiv"] * 12 + ["planner"] * 12
    assert len({r.label for r in bank.records}) == 24
    assert [r.planner_id for r in bank.records[12:]] == IDS
    assert len({r.config_sha256 for r in bank.records[12:]}) == 12

    partners = load_partners(load_partner_bank(str(out), "coord_ring"), obs_dim=5 * 5 * 26)  # resolved before training
    assert len({id(p.policy) for p in partners[12:]}) == 1  # one wrapper, so one compiled graph
    assert partners[12].policy.is_planner and not getattr(partners[0].policy, "is_planner", False)
    assert [round(float(p.params.p_onion_on_counter), 3) for p in partners[12:14]] == [0.0, 0.8]


def _planner_manifest(tmp_path, partners, layout="coord_ring"):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"format_version": 1, "name": "t", "layout": layout, "num_partners": len(partners),
                                "populations": {}, "partners": partners}))
    return str(path)


def test_bank_rejects_bad_planner_records(tmp_path):
    planner = lambda i, pid, **kw: {"partner_id": i, "kind": "planner", "planner_id": pid, **kw}
    with pytest.raises(PartnerBankError, match="already used by partner 0"):
        load_partner_bank(_planner_manifest(tmp_path, [planner(0, "P01"), planner(1, "P01")]), "coord_ring")
    with pytest.raises(PartnerBankError, match="unknown planner id"):
        load_partner_bank(_planner_manifest(tmp_path, [planner(0, "P13")]), "coord_ring")
    with pytest.raises(PartnerBankError, match="must equal its position"):
        load_partner_bank(_planner_manifest(tmp_path, [planner(1, "P01")]), "coord_ring")
    with pytest.raises(PartnerBankError, match="definition changed"):
        load_partner_bank(_planner_manifest(tmp_path, [planner(0, "P01", config_sha256="0" * 64)]), "coord_ring")
    with pytest.raises(PartnerBankError, match="unknown kind"):
        load_partner_bank(_planner_manifest(tmp_path, [{"partner_id": 0, "kind": "magic"}]), "coord_ring")
    ok = load_partner_bank(_planner_manifest(tmp_path, [planner(0, "P01", config_sha256=PLANNER_SPECS["P01"].config_sha256)]),
                           "coord_ring")
    assert len(ok) == 1
