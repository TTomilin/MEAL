"""Diagnostic rollouts for the planner partners P01-P12 (not a benchmark).

Each planner plays agent_1 (its training seat) for one episode per seed against each existing competent
counterpart in the agent_0 seat: the independent planner (no counters), the onion specialist and the plate
specialist. All planners see the same environment resets and the same per-step random keys ("matched
conditions"), so paired variants differ only through their own decisions.

Per planner the report prints soups per episode (team / by the planner), idle share and action mix of the planner,
counter handoffs (items one agent put on a counter and the other took) and how often the variant's distinguishing
option mattered ("activations", defined per variant by `PlannerSpec.activation`). It then lists

  * layout collapses known from the static geometry (a distinction the layout cannot express),
  * variants that never activated (unless already explained by a collapse),
  * planners whose sampled behaviour (actions and positions in every episode) is identical to another's.

It is a smoke diagnostic. Few episodes against three fixed counterparts do not show behavioural diversity, and a
planner that scores badly with these counterparts is not thereby proven unsolvable or unhelpful.

    python -m experiments.partner_adaptation.partner_quality --layouts coord_ring cramped_room
"""
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from experiments.partner_adaptation.partner_agents.overcooked.agent_policy_wrappers import (
    OvercookedIndependentPolicyWrapper, OvercookedOnionPolicyWrapper, OvercookedPlannerPolicyWrapper,
    OvercookedPlatePolicyWrapper)
from experiments.partner_adaptation.partner_agents.overcooked.planner_variants import (
    PLANNER_SPECS, layout_collapses, planner_params)
from meal import make_env
from meal.env.overcooked.common import OBJECT_TO_INDEX
from meal.env.overcooked.layouts.presets import overcooked_layouts
from meal.wrappers.logging import LogWrapper

ACTION_NAMES = ("up", "down", "right", "left", "stay", "interact")
ITEMS = [OBJECT_TO_INDEX[k] for k in ("onion", "plate", "dish")]
PAIRS = (("P01", "P02"), ("P03", "P04"), ("P05", "P06"), ("P07", "P08"), ("P09", "P10"), ("P11", "P12"))
DIRECTIONS = jnp.array([[0, -1], [0, 1], [1, 0], [-1, 0]])  # (dx, dy) of up, down, right, left
FLAGS = ("pot_route", "counter_route", "plate_ready", "prefetch", "both", "blocked", "waiting", "multi_handoff",
         "complement_decided")


@dataclass
class QualityConfig:
    layouts: List[str] = field(default_factory=lambda: ["coord_ring", "cramped_room"])
    planners: List[str] = field(default_factory=lambda: list(PLANNER_SPECS))
    counterparts: List[str] = field(default_factory=lambda: ["independent", "onion", "plate"])
    episodes: int = 8
    steps: int = 400  # the training episode length
    seed: int = 0


def _counterparts(layout):
    return {
        "independent": OvercookedIndependentPolicyWrapper(layout, p_onion_on_counter=0.0, p_plate_on_counter=0.0),
        "onion": OvercookedOnionPolicyWrapper(layout),
        "plate": OvercookedPlatePolicyWrapper(layout),
    }


def _make_rollout(env, planner_policy, counterpart, steps):
    wall = OBJECT_TO_INDEX["wall"]
    is_item = lambda o: jnp.isin(o, jnp.array(ITEMS))
    avail = jnp.ones(6, jnp.float32)
    not_done = jnp.asarray(False)
    H, W = env.height, env.width

    def episode(params, key):
        reset_key, act_key, step_key = jax.random.split(key, 3)
        obs, state = env.reset(reset_key)
        h0 = counterpart.init_hstate(1, {"agent_id": 0})
        h1 = planner_policy.init_hstate(1, {"agent_id": 1})

        def step(carry, t):
            obs, state, h0, h1, owner = carry
            k0, k1 = jax.random.split(jax.random.fold_in(act_key, t))
            a0, h0 = counterpart.get_action(None, obs["agent_0"], not_done, avail, h0, k0, env_state=state)
            a1, h1, info = planner_policy.policy.act(params, obs["agent_1"], state, h1, k1)
            acts = {"agent_0": jnp.asarray([a0], jnp.uint32), "agent_1": jnp.asarray([a1], jnp.uint32)}
            obs2, state2, _, done, einfo = env.step(jax.random.fold_in(step_key, t), state, acts)

            raw, raw2 = state.env_state, state2.env_state
            before, after = raw.maze_map[..., 0].astype(jnp.int32), raw2.maze_map[..., 0].astype(jnp.int32)
            put_tiles = (before == wall) & is_item(after)
            take_tiles = is_item(before) & (after == wall)
            pos = raw.agent_pos.astype(jnp.int32)
            faced = pos + DIRECTIONS[raw.agent_dir_idx.astype(jnp.int32)]
            fy, fx = jnp.clip(faced[:, 1], 0, H - 1), jnp.clip(faced[:, 0], 0, W - 1)
            interacted = (jnp.asarray([a0, a1]) == 5) & ~done["__all__"]
            puts = interacted & put_tiles[fy, fx]
            takes = interacted & take_tiles[fy, fx]
            handoff = takes & (owner[fy, fx] >= 0) & (owner[fy, fx] != jnp.arange(2))
            owner = owner.at[fy, fx].set(jnp.where(puts, jnp.arange(2), jnp.where(takes, -1, owner[fy, fx])))
            out = dict(
                action=jnp.asarray(a1, jnp.int32), pos=pos[1], soups=jnp.stack(
                    [einfo["soups"]["agent_0"], einfo["soups"]["agent_1"]]),
                puts=puts, handoff=handoff, **{k: info[k] for k in FLAGS})
            return (obs2, state2, h0, h1, owner), out

        owner0 = -jnp.ones((H, W), jnp.int32)
        _, out = jax.lax.scan(step, (obs, state, h0, h1, owner0), jnp.arange(steps))
        return out

    return jax.jit(jax.vmap(episode, in_axes=(None, 0)))


def run_layout(layout_name: str, cfg: QualityConfig) -> Dict:
    layout = overcooked_layouts[layout_name]
    env = LogWrapper(make_env("overcooked", layout=layout, max_steps=cfg.steps))
    planner_policy = OvercookedPlannerPolicyWrapper(layout)
    keys = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.episodes)
    counterparts = {k: v for k, v in _counterparts(layout).items() if k in cfg.counterparts}
    if not counterparts:
        raise ValueError(f"no known counterpart in {cfg.counterparts}; choose from independent, onion, plate")
    rollouts = {name: _make_rollout(env, planner_policy, cp, cfg.steps) for name, cp in counterparts.items()}

    runs = {}  # (planner, counterpart) -> numpy outputs
    for pid in cfg.planners:
        params = planner_params(pid)
        for cname in counterparts:
            runs[pid, cname] = jax.tree.map(np.asarray, rollouts[cname](params, keys))

    rows = {}
    for pid in cfg.planners:
        outs = [runs[pid, c] for c in counterparts]
        actions = np.concatenate([o["action"].ravel() for o in outs])
        digest = hashlib.sha256()
        for o in outs:
            digest.update(np.ascontiguousarray(o["action"]).tobytes())
            digest.update(np.ascontiguousarray(o["pos"]).tobytes())
        rows[pid] = dict(
            soups_team=float(np.mean([o["soups"].sum(axis=(1, 2)).mean() for o in outs])),
            soups_planner=float(np.mean([o["soups"][..., 1].sum(axis=1).mean() for o in outs])),
            soups_by_counterpart={c: float(runs[pid, c]["soups"].sum(axis=(1, 2)).mean()) for c in counterparts},
            idle=float((actions == 4).mean()),
            mix={n: float((actions == i).mean()) for i, n in enumerate(ACTION_NAMES)},
            puts=float(np.mean([o["puts"][..., 1].sum(axis=1).mean() for o in outs])),
            handoff_p2c=float(np.mean([o["handoff"][..., 0].sum(axis=1).mean() for o in outs])),
            handoff_c2p=float(np.mean([o["handoff"][..., 1].sum(axis=1).mean() for o in outs])),
            activations={k: int(sum(o[k].sum() for o in outs)) for k in FLAGS},
            digest=digest.hexdigest())
    return dict(layout=layout_name, rows=rows, runs=runs, counterparts=list(counterparts),
                collapses=layout_collapses(layout))


def diagnose(result: Dict, cfg: QualityConfig) -> Dict[str, List[str]]:
    rows, collapses = result["rows"], result["collapses"]
    flags = {"collapsed": [], "never_active": [], "identical": [], "no_soup": []}
    for pair, why in collapses.items():
        a, b = pair.split("/")
        if a in rows and b in rows:
            flags["collapsed"].append(f"{pair}: {why}")
    collapsed_ids = {i for pair in collapses for i in pair.split("/")}
    for pid, r in rows.items():
        spec = PLANNER_SPECS[pid]
        if r["activations"][spec.activation] == 0 and pid not in collapsed_ids:
            flags["never_active"].append(f"{pid} {spec.name}: '{spec.activation}' never occurred")
        if r["soups_team"] == 0:
            flags["no_soup"].append(f"{pid} {spec.name}: no soup with any of the counterparts used here")
    by_digest = {}
    for pid, r in rows.items():
        by_digest.setdefault(r["digest"], []).append(pid)
    flags["identical"] = [" = ".join(ids) for ids in by_digest.values() if len(ids) > 1]
    return flags


def format_report(result: Dict, flags: Dict[str, List[str]], cfg: QualityConfig) -> str:
    rows, runs = result["rows"], result["runs"]
    cps = result["counterparts"]
    lines = [f"== {result['layout']}: {cfg.episodes} episodes x {cfg.steps} steps, counterparts {', '.join(cps)}, "
             f"seed {cfg.seed}",
             "id   name              soups(team/self) " + " ".join(f"{c[:5]:>5}" for c in cps) +
             "  idle  u/d/r/l/s/i %                 puts  hand p>c c>p  activation"]
    for pid, r in rows.items():
        spec = PLANNER_SPECS[pid]
        mix = "/".join(f"{100 * r['mix'][n]:.0f}" for n in ACTION_NAMES)
        lines.append(
            f"{pid}  {spec.name:<16} {r['soups_team']:6.2f}/{r['soups_planner']:<6.2f}  "
            + " ".join(f"{r['soups_by_counterpart'][c]:5.2f}" for c in cps)
            + f"  {r['idle']:.2f}  {mix:<24} {r['puts']:6.2f}  {r['handoff_p2c']:5.2f} {r['handoff_c2p']:4.2f}"
              f"  {spec.activation}={r['activations'][spec.activation]}")
    lines.append("paired variants (matched episodes):  soups(a) soups(b)  mean first differing step")
    for a, b in PAIRS:
        if a in rows and b in rows:
            first = []
            for c in cps:
                diff = runs[a, c]["action"] != runs[b, c]["action"]
                first += [int(np.argmax(d)) if d.any() else cfg.steps for d in diff]
            same = "identical" if rows[a]["digest"] == rows[b]["digest"] else f"{np.mean(first):.0f}"
            lines.append(f"  {a}/{b}  {rows[a]['soups_team']:6.2f} {rows[b]['soups_team']:6.2f}    {same}"
                         f"  (steps={cfg.steps} means never differed)")
    for title, key in (("layout collapses", "collapsed"), ("never activated", "never_active"),
                       ("identical sampled behaviour", "identical"), ("no soup", "no_soup")):
        for item in flags[key]:
            lines.append(f"FLAG {title}: {item}")
    if not any(flags.values()):
        lines.append("no flags")
    return "\n".join(lines)


def main():
    import tyro
    cfg = tyro.cli(QualityConfig)
    for layout in cfg.layouts:
        result = run_layout(layout, cfg)
        print(format_report(result, diagnose(result, cfg), cfg))


if __name__ == "__main__":
    main()
