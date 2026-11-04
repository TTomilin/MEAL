from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from experiments.partner_adaptation.jax_aht_bridge import JAX_AHT_ROOT
from experiments.partner_adaptation.jax_aht_bridge.policies import ACTOR_TYPES

TASK_PREFIX = "overcooked-v1"
LAYOUTS = ("cramped_room", "asymm_advantages", "coord_ring", "counter_circuit", "forced_coord")
SETS = {
    "eval": dict(config="evaluation/configs/global_heldout_settings.yaml", repo_id="jaxaht/eval-teammates",
                 path_prefix="eval_teammates/"),
    "val": dict(config="evaluation/configs/global_validation_settings.yaml", repo_id="jaxaht/val-teammates",
                path_prefix="val_teammates/"),
}
DEFAULT_ACTIVATION, DEFAULT_FC_HIDDEN = "tanh", 64


@dataclass
class TeammateSpec:
    set_name: str
    layout: str
    name: str
    actor_type: str
    run_path: str
    ckpt_key: str = "checkpoints"
    idx_list: Any = None
    pop_size: Optional[int] = None
    test_mode: bool = False
    custom_loader: Optional[Dict[str, Any]] = None
    performance_bounds: Dict[str, Any] = field(default_factory=dict)

    @property
    def repo_id(self) -> str:
        return SETS[self.set_name]["repo_id"]

    @property
    def run_dir(self) -> str:
        return self.run_path.rstrip("/").rsplit("/", 1)[0] if self.run_path.rstrip("/").endswith(
            "saved_train_run") else self.run_path.rstrip("/")

    @property
    def population_key(self) -> str:
        return f"jaxaht_{self.set_name}_{self.name}"


def load_specs(set_name: str, layout: str, root: Path = JAX_AHT_ROOT, kinds=ACTOR_TYPES) -> List[TeammateSpec]:
    cfg = yaml.safe_load((Path(root) / SETS[set_name]["config"]).read_text())
    entries = cfg["heldout_set"].get(f"{TASK_PREFIX}/{layout}")
    if entries is None:
        raise KeyError(f"{SETS[set_name]['config']} has no {TASK_PREFIX}/{layout} entry")
    prefix = SETS[set_name]["path_prefix"]
    specs = []
    for name, entry in entries.items():
        if entry.get("actor_type") not in kinds:
            continue
        path = str(entry["path"])
        if not path.startswith(prefix):
            raise ValueError(f"{set_name}/{layout}/{name}: path {path!r} does not start with {prefix!r}")
        specs.append(TeammateSpec(
            set_name=set_name, layout=layout, name=name, actor_type=entry["actor_type"],
            run_path=path[len(prefix):].rstrip("/"), ckpt_key=entry.get("ckpt_key", "checkpoints"),
            idx_list=entry.get("idx_list"), pop_size=entry.get("POP_SIZE"),
            test_mode=bool(entry.get("test_mode", False)), custom_loader=entry.get("custom_loader"),
            performance_bounds=dict(entry.get("performance_bounds") or {})))
    return specs


BC_ACTOR_TYPE = "bc_proxy"


def load_bc_specs(set_name: str, layout: str, root: Path = JAX_AHT_ROOT) -> List[TeammateSpec]:
    cfg = yaml.safe_load((Path(root) / SETS[set_name]["config"]).read_text())
    prefix = SETS[set_name]["path_prefix"]
    out = []
    for name, entry in (cfg["heldout_set"].get(f"{TASK_PREFIX}/{layout}") or {}).items():
        if entry.get("actor_type") != BC_ACTOR_TYPE:
            continue
        out.append(TeammateSpec(set_name=set_name, layout=layout, name=name, actor_type=BC_ACTOR_TYPE,
                                run_path=str(entry["path"])[len(prefix):].rstrip("/"), ckpt_key="params",
                                idx_list=[int(entry.get("run_id", 0))], pop_size=1,
                                test_mode=bool(entry.get("test_mode", True)),
                                performance_bounds=dict(entry.get("performance_bounds") or {})))
    return out


def download(specs: List[TeammateSpec], local_dir: Path, token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download
    by_repo: Dict[str, List[str]] = {}
    for s in specs:
        if s.actor_type == BC_ACTOR_TYPE:
            by_repo.setdefault(s.repo_id, []).append(f"{s.run_path}/{s.idx_list[0]}/**")
            continue
        by_repo.setdefault(s.repo_id, []).extend([f"{s.run_dir}/saved_train_run/**", f"{s.run_dir}/.hydra/*"])
    for repo_id, patterns in by_repo.items():
        snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=sorted(set(patterns)),
                          local_dir=str(Path(local_dir) / repo_id.split("/")[-1]),
                          token=token or os.environ.get("HF_TOKEN") or None)


def local_run_dir(spec: TeammateSpec, local_dir: Path) -> Path:
    return Path(local_dir) / spec.repo_id.split("/")[-1] / spec.run_dir


def restore_checkpoint(path: Path) -> Dict[str, Any]:
    import jax
    import orbax.checkpoint as ocp
    path = Path(path).resolve()
    ckptr = ocp.PyTreeCheckpointer()
    meta = ckptr.metadata(str(path))
    tree = getattr(meta, "item_metadata", None)
    tree = getattr(meta, "tree", None) if tree is None else getattr(tree, "tree", tree)
    if tree is None:
        tree = meta
    args = jax.tree.map(lambda leaf: ocp.ArrayRestoreArgs(restore_type=np.ndarray) if hasattr(leaf, "shape")
                        else ocp.RestoreArgs(), tree)
    return ckptr.restore(str(path), restore_args=args)


def checkpoint_params(restored: Dict[str, Any], spec: TeammateSpec):
    import jax
    loader = (spec.custom_loader or {}).get("name")
    if loader not in (None, "fcp", "partial_load"):
        raise ValueError(f"{spec.name}: custom_loader {loader!r} is not supported for partner import")
    params = restored[spec.ckpt_key]
    if loader == "fcp":
        params = jax.tree.map(lambda x: x.reshape(x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:]), params)
    return params


def _jax_index(i: int, n: int) -> int:
    i = i + n if i < 0 else i
    return min(max(i, 0), n - 1)


def effective_indices(idx_list, shape) -> List[tuple]:
    if idx_list is None:
        raise ValueError("idx_list is required (jax-aht selects explicit members for evaluation)")
    idxs = np.asarray(idx_list, dtype=np.int64)
    if idxs.ndim == 1:
        return [(_jax_index(int(i), shape[0]),) for i in idxs]
    if idxs.ndim == 2 and idxs.shape[1] == 2:
        return [(_jax_index(int(r), shape[0]), _jax_index(int(c), shape[1])) for r, c in idxs]
    raise ValueError(f"invalid idx_list shape {idxs.shape}")


def select_members(params, idx_list) -> List[Any]:
    import jax
    shape = jax.tree.leaves(params)[0].shape
    return [jax.tree.map(lambda x: np.asarray(x)[idx], params) for idx in effective_indices(idx_list, shape)]


def _member_bounds(bounds: Dict[str, Any], k: int, n: int) -> Dict[str, List[float]]:
    out = {}
    for metric, value in bounds.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            out[metric] = arr.tolist()
        elif arr.ndim == 2 and arr.shape[0] == n:
            out[metric] = arr[k].tolist()
        else:
            raise ValueError(f"performance_bounds[{metric!r}] has shape {arr.shape}, expected (2,) or ({n}, 2)")
    return out


def _hydra_settings(run_dir: Path) -> Dict[str, Any]:
    found = {}
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if cfg_path.is_file():
        def walk(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k in ("ACTIVATION", "FC_HIDDEN_DIM") and not isinstance(v, (dict, list)):
                        found.setdefault(k, v)
                    walk(v)
        walk(((yaml.safe_load(cfg_path.read_text()) or {}).get("algorithm")) or {})
    return dict(activation=str(found.get("ACTIVATION", DEFAULT_ACTIVATION)),
                fc_hidden_dim=int(found.get("FC_HIDDEN_DIM", DEFAULT_FC_HIDDEN)))


def export_population(spec: TeammateSpec, members: List[Any], out_dir: Path, *, source: str, run: str,
                      activation: str = DEFAULT_ACTIVATION, fc_hidden_dim: int = DEFAULT_FC_HIDDEN,
                      extra: Optional[Dict[str, Any]] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("m*.pt"):
        stale.unlink()
    n = len(members)
    records, dropped = [], []
    idx_list = spec.idx_list if spec.idx_list is not None else list(range(n))
    effective = (extra or {}).get("effective_indices") or [None] * n
    seen = {}
    for k, params in enumerate(members):
        if effective[k] is not None and tuple(effective[k]) in seen:
            # jax-aht's clamped gather can select the same member twice
            dropped.append(dict(requested=idx_list[k], same_as_member=seen[tuple(effective[k])]))
            continue
        member_index = len(records)
        if effective[k] is not None:
            seen[tuple(effective[k])] = member_index
        fname = f"m{member_index}.pt"
        with open(out_dir / fname, "wb") as f:
            pickle.dump({"actor_params": params}, f)
        records.append(dict(member_index=member_index, seed_index=0, file=fname, source_index=idx_list[k],
                            effective_index=list(effective[k]) if effective[k] is not None else None,
                            performance_bounds=_member_bounds(spec.performance_bounds, k, n)))
    gen = dict(status="complete", kind="jaxaht", source=source, run=run, layout_name=spec.layout,
               name=spec.name, set=spec.set_name, actor_type=spec.actor_type,
               partner_pop_size=int(spec.pop_size or 1), num_members=len(records), activation=activation,
               fc_hidden_dim=fc_hidden_dim, test_mode=spec.test_mode, ckpt_key=spec.ckpt_key,
               idx_list=spec.idx_list, obs_convention="jaxmarl",
               trained_seat="agent_0" if spec.actor_type == "actor_with_conditional_critic" else "self-play",
               members=records, duplicates_dropped=dropped,
               **{k: v for k, v in (extra or {}).items() if k != "effective_indices"})
    (out_dir / "generation.json").write_text(json.dumps(gen, indent=2) + "\n")
    return out_dir / "generation.json"


def import_bc_proxy(spec: TeammateSpec, local_dir: Path, out_root: Path) -> Path:
    run_id = spec.idx_list[0]
    restored = restore_checkpoint(Path(local_dir) / spec.repo_id.split("/")[-1] / spec.run_path / str(run_id))
    path = export_population(spec, [restored["params"]], Path(out_root) / spec.set_name / spec.layout / spec.name,
                             source=spec.repo_id, run=f"{spec.run_path}/{run_id}", activation="relu",
                             extra=dict(run_id=run_id, bc_hidden_dims=[64, 64]))
    gen = json.loads(path.read_text())
    gen.update(kind="bc_proxy", actor_type="bc_mlp", trained_seat="both", obs_convention="jaxaht_bc_featurizer")
    path.write_text(json.dumps(gen, indent=2) + "\n")
    return path


def import_spec(spec: TeammateSpec, local_dir: Path, out_root: Path) -> Path:
    run_dir = local_run_dir(spec, local_dir)
    import jax
    restored = restore_checkpoint(run_dir / "saved_train_run")
    params = checkpoint_params(restored, spec)
    effective = effective_indices(spec.idx_list, jax.tree.leaves(params)[0].shape)
    members = select_members(params, spec.idx_list)
    return export_population(spec, members, Path(out_root) / spec.set_name / spec.layout / spec.name,
                             source=spec.repo_id, run=spec.run_path, extra=dict(effective_indices=effective),
                             **_hydra_settings(run_dir))
