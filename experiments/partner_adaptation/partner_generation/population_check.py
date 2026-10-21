"""Matched self-play and cross-play sanity check for generated partner populations."""
import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from experiments.partner_adaptation.partner_agents.agent_interface import ActorWithConditionalCriticPolicy
from meal import make_env
from meal.env.overcooked.layouts.presets import overcooked_layouts
from meal.wrappers.logging import LogWrapper

MEMBER_RE = re.compile(r"^params_seed(\d+)_agent(\d+)\.pt$")
BR_RE = re.compile(r"^params_seed(\d+)_br(\d+)\.pt$")
ACTION_NAMES = ("up", "down", "right", "left", "stay", "interact")


@dataclass
class PopulationReport:
    sp_xp_matrix: np.ndarray
    solo: np.ndarray
    shaped_share: np.ndarray
    action_mix: np.ndarray
    passed: Optional[np.ndarray]


def _load_config(generation_dir: Path):
    generation_path = generation_dir / "generation.json"
    if generation_path.is_file():
        config = json.loads(generation_path.read_text())
        if config.get("status") != "complete":
            raise ValueError(f"{generation_path}: generation is not complete")
        return config
    config_path = generation_dir / "config.pckl"
    if not config_path.is_file():
        raise FileNotFoundError(f"No generation.json or config.pckl in {generation_dir}")
    with config_path.open("rb") as f:
        return pickle.load(f)


def _load_params(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "actor_params" not in payload:
        raise ValueError(f"{path}: expected a pickle containing actor_params")
    return payload["actor_params"]


def _make_evaluator(env, policy, steps, greedy=False):
    # Bank partners are deployed with sampled actions (train_ego: get_actions(test_mode=False)), so the gate samples
    # too by default; greedy=True evaluates the policy modes instead.
    test_mode = bool(greedy)
    def episode(conf_params, other_params, keys):
        def run_one_episode(rng):
            rng, reset_rng = jax.random.split(rng)
            obs, state = env.reset(reset_rng)
            h0 = policy.init_hstate(1, {"agent_id": 0})
            h1 = policy.init_hstate(1, {"agent_id": 1})

            def step(carry, t):
                obs, state, done, h0, h1 = carry

                def active(_):
                    step_rng = jax.random.fold_in(rng, t)
                    key0, key1, env_key = jax.random.split(step_rng, 3)
                    avail = env._env.get_avail_actions(state.env_state)
                    a0, h0_next = policy.get_action(
                        params=other_params,
                        obs=obs["agent_0"].reshape(1, -1),
                        done=jnp.asarray([done]),
                        avail_actions=avail["agent_0"].astype(jnp.float32),
                        hstate=h0,
                        rng=key0,
                        env_state=state,
                        test_mode=test_mode,
                    )
                    a1, h1_next = policy.get_action(
                        params=conf_params,
                        obs=obs["agent_1"].reshape(1, -1),
                        done=jnp.asarray([done]),
                        avail_actions=avail["agent_1"].astype(jnp.float32),
                        hstate=h1,
                        rng=key1,
                        env_state=state,
                        test_mode=test_mode,
                    )
                    a0, a1 = a0.squeeze(), a1.squeeze()
                    obs_next, state_next, _, done_dict, info = env.step(
                        env_key,
                        state,
                        {"agent_0": a0.astype(jnp.uint32), "agent_1": a1.astype(jnp.uint32)},
                    )
                    soup = info["soups"]["agent_0"] + info["soups"]["agent_1"]
                    conf_shaped = info["shaped_reward"]["agent_1"]
                    team_shaped = conf_shaped + info["shaped_reward"]["agent_0"]
                    action_count = jax.nn.one_hot(a1, len(ACTION_NAMES), dtype=jnp.float32)
                    return (obs_next, state_next, done_dict["__all__"], h0_next, h1_next), (
                        soup, conf_shaped, team_shaped, action_count
                    )

                def inactive(_):
                    zeros = jnp.zeros((), dtype=jnp.float32)
                    return carry, (zeros, zeros, zeros, jnp.zeros((len(ACTION_NAMES),), dtype=jnp.float32))

                return jax.lax.cond(done, inactive, active, operand=None)

            initial_carry = (obs, state, jnp.asarray(False), h0, h1)
            _, (soups, conf_shaped, team_shaped, action_counts) = jax.lax.scan(
                step, initial_carry, jnp.arange(steps)
            )
            total_shaped = team_shaped.sum()
            shaped_share = jnp.where(
                jnp.abs(total_shaped) > 1e-8,
                conf_shaped.sum() / total_shaped,
                0.0,
            )
            return soups.sum(), shaped_share, action_counts.sum(axis=0)

        return jax.vmap(run_one_episode)(keys)

    return jax.jit(episode)


def _member_stats(evaluator, conf_params, other_params, episode_keys):
    soups, shaped_share, action_counts = evaluator(conf_params, other_params, episode_keys)
    soups = np.asarray(soups)
    shaped_share = np.asarray(shaped_share)
    action_counts = np.asarray(action_counts).sum(axis=0)
    action_total = action_counts.sum()
    action_mix = action_counts / action_total if action_total > 0 else np.zeros(len(ACTION_NAMES))
    return float(soups.mean()), float(shaped_share.mean()), action_mix


def check_population(
    gen_dir: Path,
    layout: str,
    episodes: int = 5,
    steps: int = 400,
    seed: int = 0,
    greedy: bool = False,
) -> PopulationReport:
    """Evaluate every confederate against each saved BR and an untrained MLP baseline."""
    gen_dir = Path(gen_dir)
    if layout not in overcooked_layouts:
        raise ValueError(f"unknown layout {layout!r}; available: {sorted(overcooked_layouts)}")
    if episodes < 1 or steps < 1:
        raise ValueError("episodes and steps must both be positive")
    config = _load_config(gen_dir)
    pop_size = int(config.get("partner_pop_size", 0))
    if pop_size < 1:
        raise ValueError(f"{gen_dir}: config has no valid partner_pop_size")
    declared_layout = config.get("layout_name") or config.get("resolved_layout_name")
    if declared_layout and declared_layout != layout:
        raise ValueError(f"{gen_dir}: generated for {declared_layout!r}, not requested layout {layout!r}")

    members = {}
    br_members = {}
    for path in gen_dir.iterdir():
        member_match = MEMBER_RE.match(path.name)
        br_match = BR_RE.match(path.name)
        if member_match:
            members[(int(member_match.group(1)), int(member_match.group(2)))] = path
        if br_match:
            br_members[(int(br_match.group(1)), int(br_match.group(2)))] = path
    expected_keys = {(0, j) for j in range(pop_size)}
    if not expected_keys.issubset(members):
        raise ValueError(f"{gen_dir}: missing confederate checkpoints for members {sorted(expected_keys - members.keys())}")

    has_br = bool(br_members)
    if not has_br and config.get("alg") == "lbrdiv":
        raise FileNotFoundError(f"{gen_dir}: LBRDiv generation is missing its saved BR checkpoints")
    if has_br and not expected_keys.issubset(br_members):
        raise ValueError(f"{gen_dir}: missing BR checkpoints for members {sorted(expected_keys - br_members.keys())}")

    env = LogWrapper(make_env("overcooked", layout=overcooked_layouts[layout], max_steps=steps))
    obs_dim = int(np.prod(env.observation_space().shape))
    policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=obs_dim,
        pop_size=pop_size,
        activation=config.get("activation", "tanh"),
    )
    baseline_params = policy.init_params(jax.random.PRNGKey(seed + 314159))
    episode_keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
    evaluator = _make_evaluator(env, policy, steps, greedy)
    conf_params = [_load_params(members[(0, j)]) for j in range(pop_size)]
    baseline_scores = np.zeros(pop_size, dtype=np.float64)
    shaped_share = np.zeros(pop_size, dtype=np.float64)
    action_mix = np.zeros((pop_size, len(ACTION_NAMES)), dtype=np.float64)

    if has_br:
        sp_xp_matrix = np.zeros((pop_size, pop_size), dtype=np.float64)
        for conf_id in range(pop_size):
            for br_id in range(pop_size):
                br_params = _load_params(br_members[(0, br_id)])
                score, share, mix = _member_stats(evaluator, conf_params[conf_id], br_params, episode_keys)
                sp_xp_matrix[conf_id, br_id] = score
                if conf_id == br_id:
                    shaped_share[conf_id] = share
                    action_mix[conf_id] = mix
            baseline_scores[conf_id], _, _ = _member_stats(
                evaluator, conf_params[conf_id], baseline_params, episode_keys
            )
        passed = np.zeros(pop_size, dtype=bool)
        for i in range(pop_size):
            xp_scores = np.delete(sp_xp_matrix[i], i)
            passed[i] = (
                sp_xp_matrix[i, i] > 0
                and sp_xp_matrix[i, i] > baseline_scores[i]
                and sp_xp_matrix[i, i] >= 1.5 * xp_scores.max()
            )
    else:
        sp_xp_matrix = np.full((pop_size, pop_size), np.nan, dtype=np.float64)
        for conf_id in range(pop_size):
            baseline_scores[conf_id], shaped_share[conf_id], action_mix[conf_id] = _member_stats(
                evaluator, conf_params[conf_id], baseline_params, episode_keys
            )
        passed = None

    print(f"Population check: {gen_dir} ({pop_size} members, {episodes} episodes x {steps} steps, "
          f"{'greedy' if greedy else 'sampled'} actions)")
    if has_br:
        print("Soup matrix (confederate rows, BR columns):")
        print("       " + " ".join(f"BR{j:>6}" for j in range(pop_size)))
        for i, row in enumerate(sp_xp_matrix):
            print(f"conf{i}: " + " ".join(f"{value:7.2f}" for value in row))
    else:
        print("No BR checkpoints found; reporting confederate vs untrained-ego baseline only (diagnostic mode).")
    for i in range(pop_size):
        mix = ", ".join(f"{name}={value:.3f}" for name, value in zip(ACTION_NAMES, action_mix[i]))
        if has_br:
            verdict = "PASS" if passed[i] else "FAIL"
            print(
                f"member {i}: SP={sp_xp_matrix[i, i]:.2f}, max_XP="
                f"{np.delete(sp_xp_matrix[i], i).max():.2f}, baseline={baseline_scores[i]:.2f}, "
                f"shaped_share={shaped_share[i]:.3f}, actions=({mix}) [{verdict}]"
            )
        else:
            print(f"member {i}: baseline_soups={baseline_scores[i]:.2f}, actions=({mix})")

    return PopulationReport(sp_xp_matrix, baseline_scores, shaped_share, action_mix, passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--layout", default="coord_ring")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true",
                        help="evaluate policy modes instead of sampled actions (deployment samples)")
    args = parser.parse_args()
    report = check_population(args.generation_dir, args.layout, args.episodes, args.steps, args.seed, args.greedy)
    if report.passed is not None and not report.passed.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()