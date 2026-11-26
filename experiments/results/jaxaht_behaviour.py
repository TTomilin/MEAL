"""Behaviour Vendi of the jax-aht held-out teammates, each playing with its own trained best response.

Uses the teammates, best responses and evaluation seeds behind
third_party/jax-aht/evaluation/xp_matrices/<layout>.csv, so the diagonal of that csv checks the rollouts.
Run it with jax-aht's own environment, not MEAL's (see experiments/results/README.MD).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from omegaconf import OmegaConf

from experiments.partner_adaptation.diversity import behaviour_kernel, response_kernel, vendi_score

JAX_AHT = Path(__file__).resolve().parents[2] / "third_party" / "jax-aht"
sys.path.insert(0, str(JAX_AHT))
from agents.overcooked.bc_agent import BCPolicy  # noqa: E402
from common.run_episodes import run_episodes  # noqa: E402
from common.save_load_utils import load_train_run  # noqa: E402
from envs import make_env  # noqa: E402
from envs.base_env import WrappedEnvState  # noqa: E402
from envs.log_wrapper import LogWrapper  # noqa: E402
from evaluation.heldout_core import load_heldout_set  # noqa: E402

NUM_ACTIONS = 6


def load_bc_params(bc):
    # BCPolicy restores without restore args, which fails off the GPU the checkpoint was saved on;
    # load_train_run falls back to a NumPy restore.
    return load_train_run(str(bc.model_base_dir / str(bc.run_id)))["params"]


BCPolicy._load_params = load_bc_params


@dataclass
class CountedState(WrappedEnvState):
    counts: jnp.ndarray


class CountTeammate:
    """Counts agent_0's steps per (tile, action); agent_0 is the teammate's seat in jax-aht's XP code."""

    def __init__(self, env):
        self._env = env
        self.width = env.env.width
        self.cells = env.env.height * env.env.width

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, key):
        obs, s = self._env.reset(key)
        counts = jnp.zeros(self.cells * NUM_ACTIONS)
        return obs, CountedState(s.env_state, s.base_return_so_far, s.avail_actions, s.step, counts)

    def step(self, key, state, actions, reset_state=None):
        x, y = state.env_state.agent_pos[0].astype(jnp.int32)
        counts = state.counts.at[(y * self.width + x) * NUM_ACTIONS + actions["agent_0"]].add(1)
        obs, s, reward, done, info = self._env.step(key, state, actions, reset_state)
        s = CountedState(s.env_state, s.base_return_so_far, s.avail_actions, s.step, counts)
        return obs, s, reward, done, {**info, "counts": counts}


def read_xp(path, teammates):
    """XP[i, j] is the base return of teammate j with the best response trained for teammate i."""
    rows = [r for r in csv.DictReader(open(path))
            if r["metric_name"] == "base_return" and r["aggregate_stat"] == "mean" and r["seating"] == "new"]
    br_of = {r["row_agent"]: r["col_agent"] for r in rows if r["is_diag"] == "True"}
    score = {(r["col_agent"], r["row_agent"]): float(r["mean"]) for r in rows}
    return np.array([[score[br_of[i], j] for j in teammates] for i in teammates])


def behaviour_vendi(counts, n_boot, seed=0):
    """Vendi of the summed episodes, with a 95% interval from resampling each teammate's episodes."""
    n, episodes, _ = counts.shape
    rng = np.random.default_rng(seed)
    rows = np.arange(n)[:, None]
    boot = [vendi_score(behaviour_kernel(counts[rows, rng.integers(0, episodes, (n, episodes))].sum(1)))
            for _ in range(n_boot)]
    return [vendi_score(behaviour_kernel(counts.sum(1))), *np.percentile(boot, [2.5, 97.5])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", required=True,
                    help="cramped_room, asymm_advantages, coord_ring, counter_circuit or forced_coord")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default=Path(os.environ.get("MEAL_CPA_ROOT", "cpa_root")) / "behaviour")
    args = ap.parse_args(argv)

    configs = JAX_AHT / "evaluation" / "configs"
    cfg = OmegaConf.merge(
        {"task": OmegaConf.load(configs / "task" / "overcooked-v1" / f"{args.layout}.yaml")},
        OmegaConf.load(configs / "global_heldout_settings.yaml"),
        OmegaConf.load(configs / "global_heldout_br.yaml"))
    task, settings = cfg.task.TASK_NAME, cfg.global_heldout_settings
    env = LogWrapper(CountTeammate(make_env(cfg.task.ENV_NAME, cfg.task.ENV_KWARGS)))

    # same key chain as jax-aht's evaluation/generate_xp_matrix.py, so these are the XP diagonal episodes
    _, teammate_key, br_key, eval_key = jax.random.split(jax.random.PRNGKey(settings.EVAL_SEED), 4)
    teammates = load_heldout_set(cfg.heldout_set[task], env, task, cfg.task.ENV_KWARGS, teammate_key)
    brs = load_heldout_set(cfg.best_response_set[task], env, task, cfg.task.ENV_KWARGS, br_key)
    names, n = list(teammates), len(teammates)
    row_keys = jax.random.split(jax.random.split(eval_key)[1], n)

    counts, own = [], []
    for i, (name, br) in enumerate(zip(names, brs)):
        policy, params, test_mode, _ = teammates[name]
        br_policy, br_params, br_test_mode, _ = brs[br]
        key = jax.random.split(jax.random.split(row_keys[i])[1], n)[i]
        info = run_episodes(key, env, agent_0_param=params, agent_0_policy=policy,
                            agent_1_param=br_params, agent_1_policy=br_policy,
                            max_episode_steps=settings.MAX_EPISODE_STEPS, num_eps=settings.NUM_EVAL_EPISODES,
                            agent_0_test_mode=test_mode, agent_1_test_mode=br_test_mode)
        counts.append(np.asarray(info["counts"]))
        own.append(float(np.mean(info["base_return"])))
        print(f"{name:24s} + {br:30s} own score {own[-1]:6.1f}", flush=True)

    xp = read_xp(JAX_AHT / "evaluation" / "xp_matrices" / f"{args.layout}.csv", names)
    keep = np.flatnonzero(np.diag(xp) > 0)
    tile_action = np.stack(counts)[keep]
    tiles = tile_action.reshape(*tile_action.shape[:2], -1, NUM_ACTIONS).sum(-1)
    result = dict(
        layout=args.layout,
        teammates=[names[i] for i in keep],
        response_vendi=vendi_score(response_kernel(xp[np.ix_(keep, keep)])),
        behaviour_vendi_tiles=behaviour_vendi(tiles, args.n_boot),
        behaviour_vendi_tile_action=behaviour_vendi(tile_action, args.n_boot),
        own_score=[own[i] for i in keep],
        xp_diagonal=np.diag(xp)[keep].tolist(),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.layout}.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez(out / f"{args.layout}_counts.npz", teammates=np.asarray(names), counts=np.stack(counts))

    print(f"\n{args.layout}: {len(keep)} teammates, response Vendi {result['response_vendi']:.2f}")
    for kind in ("tiles", "tile_action"):
        v, lo, hi = result[f"behaviour_vendi_{kind}"]
        print(f"  behaviour Vendi ({kind}) {v:.2f} [{lo:.2f}, {hi:.2f}]")
    gap = np.abs(np.array(result["own_score"]) - result["xp_diagonal"]).max()
    print(f"  largest |own score - XP diagonal| {gap:.1f}")


if __name__ == "__main__":
    main()
