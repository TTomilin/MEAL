from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import numpy as np
import yaml
from omegaconf import OmegaConf

from experiments.partner_adaptation.jax_aht_bridge import JAX_AHT_ROOT, MEAL_OVERCOOKED
from experiments.partner_adaptation.jax_aht_bridge.hf_partners import TeammateSpec, export_population

ALGORITHMS = ("comedi", "fcp")
CONFIG_ROOT = JAX_AHT_ROOT / "teammate_generation" / "configs"


class NullLogger:

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _yaml(path: Path) -> Dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=False) or {}
    data.pop("defaults", None)
    return data


def _parse_value(text: str):
    try:
        return OmegaConf.to_container(OmegaConf.create({"v": yaml.safe_load(f"v: {text}")["v"]}))["v"] \
            if not _is_float(text) else float(text)
    except yaml.YAMLError:
        return text


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return not text.strip().lstrip("+-").isdigit()


def algorithm_config(alg: str, layout: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if alg not in ALGORITHMS:
        raise ValueError(f"alg must be one of {ALGORITHMS}, got {alg!r}")
    base = _yaml(CONFIG_ROOT / "base_config_teammate.yaml")["algorithm"]
    task = _yaml(CONFIG_ROOT / "task" / "overcooked-v1" / f"{layout}.yaml")
    cfg = OmegaConf.merge(
        OmegaConf.create(_yaml(CONFIG_ROOT / "algorithm" / alg / "_base_.yaml")),
        OmegaConf.create(_yaml(CONFIG_ROOT / "algorithm" / alg / "overcooked-v1" / f"{layout}.yaml")),
        OmegaConf.create({k: v for k, v in base.items() if not isinstance(v, str) or "${" not in v}),
        OmegaConf.create(dict(ENV_NAME=MEAL_OVERCOOKED, ENV_KWARGS=task["ENV_KWARGS"],
                              ROLLOUT_LENGTH=task["ROLLOUT_LENGTH"], NUM_SEEDS=1)),
    )
    for key, value in (overrides or {}).items():
        OmegaConf.update(cfg, key, _parse_value(value) if isinstance(value, str) else value, merge=True)
    out = OmegaConf.to_container(cfg, resolve=True)
    out.pop("ego_train_algorithm", None)
    for key in ("TOTAL_TIMESTEPS", "TOTAL_TIMESTEPS_PER_ITERATION"):
        if key in out:
            out[key] = float(out[key])
    num_envs, num_minibatches = out.get("NUM_ENVS"), out.get("NUM_MINIBATCHES")
    if num_envs and num_minibatches and int(num_envs) % int(num_minibatches):
        raise ValueError(f"{alg}: NUM_ENVS={num_envs} must be a multiple of NUM_MINIBATCHES={num_minibatches} "
                         f"(jax-aht splits the environments into minibatches)")
    if alg == "fcp":
        validate_fcp_schedule(out)
    else:
        selection = out["PARTNER_POP_SIZE"] * out["NUM_ARGMAX_ROLLOUT_EPS"] * out["ROLLOUT_LENGTH"] // 2
        out["NUM_UPDATES"] = int(out["TOTAL_TIMESTEPS_PER_ITERATION"] //
                                 (selection + 4 * out["ROLLOUT_LENGTH"] * out["NUM_ENVS"]))
        if out["NUM_UPDATES"] < 1:
            raise ValueError("comedi: budget must cover at least one update per iteration")
    return out


def validate_fcp_schedule(config):
    n = int(config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"])
    count = int(config["NUM_CHECKPOINTS"])
    interval = n // max(1, count - 1)
    stores = (n - 1) // interval + 1 + int((n - 1) % interval != 0) if interval > 0 else 0
    if count < 1 or n < 1 or interval < 1 or stores != count:
        raise ValueError(f"fcp: unsafe checkpoint schedule: {n} updates, {count} slots, interval={interval}, "
                         f"writes={stores}; choose a budget giving exactly NUM_CHECKPOINTS writes "
                         "(e.g. 5 or 16 updates for 5 checkpoints)")
    config["NUM_UPDATES"] = n


def validate_fcp_checkpoints(out, config):
    count, pop = int(config["NUM_CHECKPOINTS"]), int(config["PARTNER_POP_SIZE"])
    counts = np.asarray(jax.device_get(out["final_ckpt_idx"]))
    if counts.shape != (pop,) or not np.all(counts == count):
        raise ValueError(f"fcp: expected {count} filled checkpoints per member, got {counts}")
    checkpoints, final = jax.device_get((out["checkpoints"], out["final_params"]))
    nonzero = np.zeros((pop, count), dtype=bool)
    for leaf, last in zip(jax.tree.leaves(checkpoints), jax.tree.leaves(final)):
        if leaf.shape[:2] != (pop, count) or not np.isfinite(leaf).all():
            raise ValueError("fcp: invalid checkpoint shape or non-finite parameters")
        nonzero |= np.any(leaf.reshape(pop, count, -1) != 0, axis=-1)
        if not np.array_equal(leaf[:, -1], last):
            raise ValueError("fcp: final checkpoint does not contain the final trained policy")
    if not nonzero.all():
        raise ValueError("fcp: unfilled all-zero checkpoint members cannot be exported")


def make_generation_env(config: Dict[str, Any]):
    from envs import make_env
    from envs.log_wrapper import LogWrapper
    return LogWrapper(make_env(config["ENV_NAME"], config["ENV_KWARGS"]))


def train(alg: str, config: Dict[str, Any], seed: int):
    if alg == "fcp":
        validate_fcp_schedule(config)
    env = make_generation_env(config)
    rng = jax.random.PRNGKey(seed)
    if alg == "comedi":
        from teammate_generation.CoMeDi import train_comedi_partners
        out = jax.jit(lambda r: train_comedi_partners(r, NullLogger(), env, config))(rng)
        members = out["final_params_conf"]
        actor_type, pop_size = "actor_with_conditional_critic", int(config["PARTNER_POP_SIZE"])
    else:
        from teammate_generation.fcp import train_fcp_partners
        out = train_fcp_partners(rng, env, config, NullLogger())
        validate_fcp_checkpoints(out, config)
        ckpts = out["checkpoints"]
        members = jax.tree.map(lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), ckpts)
        actor_type, pop_size = "mlp", 1
    members = jax.device_get(members)
    n = jax.tree.leaves(members)[0].shape[0]
    return [jax.tree.map(lambda x: np.asarray(x[k]), members) for k in range(n)], actor_type, pop_size, out


def generate(alg: str, layout: str, out_dir: str, seed: int = 0, overrides: Optional[Dict[str, Any]] = None) -> Path:
    config = algorithm_config(alg, layout, overrides)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; choose another output directory (also for interrupted runs)")
    print(f"{alg}: {config['NUM_ENVS']} environments, {config['NUM_UPDATES']} PPO updates"
          + (" per iteration" if alg == "comedi" else ""), flush=True)
    start = time.time()
    members, actor_type, pop_size, _ = train(alg, config, seed)
    spec = TeammateSpec(set_name="meal", layout=layout, name=alg, actor_type=actor_type, run_path=str(out_dir),
                        ckpt_key="final_params_conf" if alg == "comedi" else "checkpoints",
                        idx_list=list(range(len(members))), pop_size=pop_size, test_mode=False)
    path = export_population(spec, members, out_dir, source="meal-generated", run=str(out_dir),
                             activation=str(config.get("ACTIVATION", "tanh")),
                             fc_hidden_dim=int(config.get("FC_HIDDEN_DIM", 64)),
                             extra=dict(algorithm=alg, seed=seed, algorithm_config=config,
                                        train_seconds=round(time.time() - start, 1)))
    print(f"wrote {len(members)} {alg} members to {out_dir}")
    return path


def bank_entries(gen_path: Path, population: str):
    gen = json.loads(Path(gen_path).read_text())
    return [dict(kind="jaxaht", population=population, member_index=m["member_index"],
                 checkpoint=Path(gen_path).parent / m["file"]) for m in gen["members"]]
