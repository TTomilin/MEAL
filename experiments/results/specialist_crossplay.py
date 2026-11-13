"""Cross-play every exported specialist ego against every training partner of a bank (no training)."""
import argparse
import csv
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from experiments.partner_adaptation.jax_aht_bridge.heldout import AHT_EVAL_STEPS, make_aht_eval_env
from experiments.partner_adaptation.partner_bank import load_bank_partners, load_partner_bank
from experiments.partner_adaptation.run_br import build_ego_network
from experiments.partner_adaptation.run_outputs import code_revision
from experiments.partner_adaptation.train_br import PartnerRolloutState, make_partner_population
from experiments.results.cpa_eval import resolve_run_dir
from meal import make_env
from meal.env.overcooked.common import DIR_TO_VEC
from meal.env.overcooked.layouts.presets import overcooked_layouts
from meal.env.overcooked.overcooked_env import Actions
from meal.wrappers.logging import LogWrapper

EPISODE_STEPS = AHT_EVAL_STEPS
NUM_ACTIONS = 6
ENV_DYNAMICS = ("meal", "aht")


def load_specialist(path: Path):
    path = resolve_run_dir(path)
    config = json.loads((path / "config.json").read_text())
    stages = config.get("only_stages")
    if not (isinstance(stages, list) and len(stages) == 1):
        raise SystemExit(f"{path}: expected only_stages=[i], got {stages!r}")
    head = int(stages[0])
    identities = json.loads((path / "partner_bank.json").read_text())
    if head < 0 or head >= len(identities):
        raise SystemExit(f"{path}: only_stages=[{head}] out of range for partner_bank.json")
    label = identities[head]["label"]
    seed = int(config["seed"])
    params_path = path / f"params_seed{seed}.pt"
    if not params_path.is_file():
        raise SystemExit(f"{path}: missing {params_path.name}")
    payload = pickle.loads(params_path.read_bytes())
    if not isinstance(payload, dict) or "actor_params" not in payload:
        raise SystemExit(f"{params_path}: expected a pickle of {{'actor_params': ...}}")
    return SimpleNamespace(path=path, config=config, head=head, label=label, params=payload["actor_params"])


def make_eval_env(layout, layout_name: str, env_dynamics: str):
    if env_dynamics == "aht":
        return make_aht_eval_env(layout, layout_name)
    return LogWrapper(make_env("overcooked", layout=layout, layout_name=layout_name, max_steps=EPISODE_STEPS))


def make_behaviour_eval_fn(env, ego_network, partner_population, num_episodes: int, use_cnn: bool,
                           height: int, width: int, num_pots: int):
    ego, partner = env.agents[0], env.agents[1]
    n_cells = height * width

    def episode(rng, ego_params, env_idx, partner_params):
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key)
        state = PartnerRolloutState(
            env_state=env_state,
            partner_obs=obs[partner],
            partner_done=jnp.zeros((), dtype=bool),
            partner_hstate=partner_population.init_hstate(1),
        )

        def step(carry, _):
            obs, state, rng, soups, action_counts, visits, pot_hits = carry
            rng, act_key, step_key, partner_key = jax.random.split(rng, 4)
            ego_obs = obs[ego][None] if use_cnn else obs[ego].reshape(1, -1)
            pi, _, _ = ego_network.apply(ego_params, ego_obs, env_idx=env_idx)
            ego_act = pi.sample(seed=act_key)

            raw = env.unwrap_env_state(state.env_state)
            avail = env.get_avail_actions(state.env_state.env_state)[partner].astype(jnp.float32)
            act_partner, partner_hstate = partner_population.get_actions(
                partner_params,
                jnp.zeros((1,), dtype=jnp.int32),
                state.partner_obs.reshape(1, 1, -1),
                state.partner_done.reshape(1, 1, 1),
                avail[None],
                state.partner_hstate,
                partner_key,
                env_state=jax.tree.map(lambda x: x[None], state.env_state),
                aux_obs=None,
            )
            partner_act = act_partner.reshape(()).astype(jnp.int32)
            pos = raw.agent_pos[1].astype(jnp.int32)
            faced = pos + DIR_TO_VEC[raw.agent_dir_idx[1].astype(jnp.int32)]
            pot_pos = raw.pot_pos[:num_pots].astype(jnp.int32)
            pot_mask = raw.pot_mask[:num_pots]
            pot_hit = ((partner_act == int(Actions.interact)) & pot_mask
                       & (faced[0] == pot_pos[:, 0]) & (faced[1] == pot_pos[:, 1]))
            cell = pos[1] * width + pos[0]

            joint = {ego: ego_act, partner: act_partner.reshape(1).astype(ego_act.dtype)}
            obs, env_state, _, done, info = env.step(step_key, state.env_state, joint)
            next_state = PartnerRolloutState(
                env_state=env_state,
                partner_obs=obs[partner],
                partner_done=done["__all__"],
                partner_hstate=partner_hstate,
            )
            soups = soups + sum(jnp.asarray(info["soups"][a]).reshape(()) for a in env.agents)
            action_counts = action_counts.at[partner_act].add(1)
            visits = visits.at[cell].add(1)
            pot_hits = pot_hits + pot_hit.astype(jnp.float32)
            return (obs, next_state, rng, soups, action_counts, visits, pot_hits), None

        init = (obs, state, rng, jnp.float32(0), jnp.zeros(NUM_ACTIONS, jnp.float32),
                jnp.zeros(n_cells, jnp.float32), jnp.zeros(num_pots, jnp.float32))
        (_, _, _, soups, action_counts, visits, pot_hits), _ = jax.lax.scan(
            step, init, None, EPISODE_STEPS)
        return soups, action_counts, visits, pot_hits

    def run(rng, ego_params, env_idx, partner_params):
        keys = jax.random.split(rng, num_episodes)
        return jax.vmap(episode, in_axes=(0, None, None, None))(keys, ego_params, env_idx, partner_params)

    return jax.jit(run)


def write_matrix_csv(path: Path, corner: str, labels, matrix):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([corner] + list(labels))
        for i, label in enumerate(labels):
            w.writerow([label] + [f"{matrix[i, j]:.6g}" for j in range(len(labels))])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="specialist run directories (shell-expanded globs)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-dynamics", choices=ENV_DYNAMICS, default="aht")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")

    manifest = json.loads(Path(args.bank).read_text())
    layout_name = manifest["layout"]
    bank = load_partner_bank(args.bank, layout_name)
    labels = [r.label for r in bank.records]
    kinds = [r.kind for r in bank.records]
    P = len(bank)
    layout = overcooked_layouts[layout_name]
    num_pots = int(np.asarray(layout["pot_idx"]).shape[0])
    height, width = int(layout["height"]), int(layout["width"])

    by_label = {}
    for d in args.runs:
        spec = load_specialist(Path(d))
        if spec.label in by_label:
            raise SystemExit(
                f"two specialists for label {spec.label!r}: {by_label[spec.label].path} and {spec.path}")
        if spec.label not in labels:
            raise SystemExit(f"{spec.path}: trained on {spec.label!r}, which is not in the bank")
        by_label[spec.label] = spec
    missing = [lab for lab in labels if lab not in by_label]
    if missing:
        raise SystemExit(f"bank partners with no specialist: {missing}")

    env = make_eval_env(layout, layout_name, args.env_dynamics)
    obs_dim = int(np.prod(env.observation_space().shape))
    partners, _ = load_bank_partners(bank, obs_dim=obs_dim)
    populations = [make_partner_population(p.policy, p.params, None) for p in partners]

    means = np.zeros((P, P), np.float64)
    stds = np.zeros((P, P), np.float64)
    action_counts = np.zeros((P, P, NUM_ACTIONS), np.float64)
    visits = np.zeros((P, P, height * width), np.float64)
    pot_interactions = np.zeros((P, P, num_pots), np.float64)

    groups = {}
    for j, (population, params) in enumerate(populations):
        key = (type(population), population.policy_cls, getattr(population, "test_mode", None))
        groups.setdefault(key, []).append((j, population, params))

    for i, label in enumerate(labels):
        spec = by_label[label]
        cfg = SimpleNamespace(**spec.config)
        ego_network = build_ego_network(cfg, NUM_ACTIONS)
        ego_params = spec.params
        head = jnp.asarray(spec.head, jnp.int32)
        local_fns = {}
        for key, items in groups.items():
            population = items[0][1]
            if key not in local_fns:
                local_fns[key] = make_behaviour_eval_fn(
                    env, ego_network, population, args.episodes, bool(cfg.use_cnn),
                    height, width, num_pots)
            fn = local_fns[key]
            indices = jnp.asarray([j for j, _, _ in items], jnp.int32)
            stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *(params for _, _, params in items))
            base = jax.random.fold_in(jax.random.PRNGKey(args.seed), i)
            keys = jax.vmap(lambda n: jax.random.fold_in(base, n))(indices)
            soups, acts, vis, pots = jax.vmap(fn, in_axes=(0, None, None, 0))(
                keys, ego_params, head, stacked)
            soups, acts, vis, pots = map(np.asarray, (soups, acts, vis, pots))
            for n, j in enumerate(int(x) for x in indices):
                means[i, j] = soups[n].mean()
                stds[i, j] = soups[n].std(ddof=0)
                action_counts[i, j] = acts[n].sum(axis=0)
                visits[i, j] = vis[n].sum(axis=0)
                pot_interactions[i, j] = pots[n].sum(axis=0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(out / "xp_matrix.csv", "specialist", labels, means)
    write_matrix_csv(out / "xp_matrix_std.csv", "specialist", labels, stds)
    np.savez(out / "partner_behaviour.npz", labels=np.asarray(labels), kinds=np.asarray(kinds),
             action_counts=action_counts, visits=visits, pot_interactions=pot_interactions,
             episodes=np.asarray(args.episodes))
    (out / "run_info.json").write_text(json.dumps({
        "bank": str(Path(args.bank).resolve()),
        "runs": [str(by_label[lab].path) for lab in labels],
        "episodes": args.episodes,
        "seed": args.seed,
        "env_dynamics": args.env_dynamics,
        "git": code_revision(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
