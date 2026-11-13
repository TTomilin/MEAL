"""Explicit bank of CPA partners: learned BRDiv members and configurable planners.

A bank is a JSON manifest that lists every partner the ego agent is trained against, in training
order. Each BRDiv *population* is one trained run (e.g. three members); a bank may combine members of many
populations, so the size of a bank says nothing about the architecture of a member's policy.

Manifest (``format_version`` 1). **Relative paths resolve against the directory that contains the manifest.**

    {
      "format_version": 1,
      "name": "coord_ring_bank",
      "layout": "coord_ring",
      "num_partners": 6,
      "populations": {
        "bundled_seed0": {"config": "../partner_agents/BRDiv_population/coord_ring/config.pckl", "seed_index": 0},
        "gseed1001":     {"config": "gen/coord_ring_gseed1001/generation.json", "seed_index": 0}
      },
      "partners": [
        {"partner_id": 0, "population": "bundled_seed0", "member_index": 0, "checkpoint": "<path>/params_seed0_agent0.pt"},
        ...
      ]
    }

A population ``config`` is the BRDiv config of the run that trained it (``config.pckl`` or a ``generation.json``
written by ``partner_generation/run.py``). It supplies ``partner_pop_size`` (the critic's teammate-id width),
``activation`` and, when present, ``layout_name`` and ``seed``. Checkpoints are pickles of
``{"actor_params": ...}``.

Rules enforced when a bank is loaded (``PartnerBankError`` with the offending partner on any violation):
exactly ``num_partners`` records; ``partner_id`` equal to the record position (it is the ego head / eval id);
``member_index`` inside its population; each (population, member) once; each resolved file once; each
checkpoint payload once; the generic ``params.pt`` is never a member; ``params_seed{i}_agent{j}.pt`` names must
agree with the declared ``seed_index``/``member_index``; population layout must match the bank layout.

Planner partners (``partner_agents/overcooked/planner_variants.py``) sit in the same ``partners`` list. They need no
population or checkpoint; their configuration is the registry entry of the id, resolved when the bank is loaded:

        {"partner_id": 12, "kind": "planner", "planner_id": "P01", "config_sha256": "<optional, checked if present>"}

A planner id may appear once, and two planners must not resolve to the same configuration. A recorded
``config_sha256`` that no longer matches the registry is an error (the registry changed under a saved bank).

``load_partners`` additionally checks every checkpoint against the policy its population config declares.
Populations with equal ``(partner_pop_size, activation)`` share one policy object (and so one compiled graph and
one vmapped evaluation group); otherwise each gets its own policy. Parameter trees are never stacked across
different policies.

Without a manifest, ``select_partner_bank`` reproduces the legacy behaviour: the checkpoints bundled in
``partner_agents/BRDiv_population/<layout>``.

Format version 2 adds imported partner kinds (``IMPORTED_KINDS``) and an optional ``heldout`` list keyed by
``heldout_id``. Held-out partners are only evaluated, and the uniqueness rules span both lists:

        {"partner_id": 0, "kind": "jaxaht", "population": "jaxaht_eval_ippo_mlp", "member_index": 1,
         "checkpoint": "jaxaht/eval/coord_ring/ippo_mlp/m1.pt"}

Assemble a manifest from generation outputs (no files are moved or copied):

    python -m experiments.partner_adaptation.partner_bank --layout coord_ring --out bank.json \
        --generation-dirs checkpoints/brdiv_coord_ring_pop3_gseed1001 ...
"""
import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

BANK_FORMAT_VERSION = 1
BANK_FORMAT_VERSION_V2 = 2
SUPPORTED_FORMAT_VERSIONS = (BANK_FORMAT_VERSION, BANK_FORMAT_VERSION_V2)
IMPORTED_KINDS = ("jaxaht", "concord", "bc_proxy")
BUNDLED_ROOT = Path(__file__).resolve().parent / "partner_agents" / "BRDiv_population"
LEGACY_DEFAULT_PARTNERS = 3
MEMBER_FILE = re.compile(r"^params_seed(\d+)_agent(\d+)\.pt$")
GENERIC_CHECKPOINT = "params.pt"


class PartnerBankError(ValueError):
    pass


@dataclass(frozen=True)
class PartnerRecord:
    partner_id: int
    layout: str
    population: str
    population_config: Path
    checkpoint: Path
    member_index: int
    seed_index: int
    population_size: int
    activation: str
    generation_seed: Optional[int]
    payload_sha256: str
    config: Dict[str, Any] = field(repr=False)

    @property
    def label(self) -> str:
        return f"{self.layout}/{self.population}/m{self.member_index}"


    kind = "brdiv"


@dataclass(frozen=True)
class PlannerRecord:
    partner_id: int
    layout: str
    planner_id: str
    config: Dict[str, Any] = field(repr=False)
    config_sha256: str

    kind = "planner"

    @property
    def label(self) -> str:
        return f"{self.layout}/{self.planner_id}"


@dataclass(frozen=True)
class ImportedPartnerRecord:
    partner_id: int
    layout: str
    kind: str
    population: str
    population_config: Path
    checkpoint: Path
    member_index: int
    actor_type: str
    population_size: int
    activation: str
    fc_hidden_dim: int
    test_mode: bool
    source: str
    performance_bounds: Dict[str, Any]
    payload_sha256: str
    config: Dict[str, Any] = field(repr=False)
    heldout: bool = False

    @property
    def label(self) -> str:
        return f"{self.layout}/{self.population}/m{self.member_index}"

    @property
    def generation_seed(self):
        return self.config.get("seed")


AnyRecord = Union[PartnerRecord, PlannerRecord, ImportedPartnerRecord]


@dataclass(frozen=True)
class PartnerBank:
    name: str
    layout: str
    source: Optional[Path]
    records: Tuple[AnyRecord, ...]
    heldout: Tuple[AnyRecord, ...] = ()
    format_version: int = BANK_FORMAT_VERSION

    def __len__(self):
        return len(self.records)


@dataclass(frozen=True)
class LoadedPartner:
    record: AnyRecord
    policy: Any
    params: Any


def partner_identity(record: AnyRecord) -> Dict[str, Any]:
    """JSON-serialisable identity of one bank partner: what a saved run needs to recognise the same partner."""
    base = dict(partner_id=record.partner_id, kind=record.kind, label=record.label)
    if record.kind == "brdiv":
        return dict(base, checkpoint=record.checkpoint.name, population_size=record.population_size,
                    generation_seed=record.generation_seed, payload_sha256=record.payload_sha256)
    if record.kind in IMPORTED_KINDS:
        ident = dict(base, checkpoint=record.checkpoint.name, population=record.population,
                     member_index=record.member_index, actor_type=record.actor_type, source=record.source,
                     layout=record.layout, population_size=record.population_size, activation=record.activation,
                     fc_hidden_dim=record.fc_hidden_dim,
                     run=record.config.get("run"), test_mode=record.test_mode,
                     performance_bounds=record.performance_bounds, payload_sha256=record.payload_sha256)
        if record.heldout:
            ident = dict(ident, heldout_id=ident.pop("partner_id"), heldout=True)
        return ident
    return dict(base, planner_id=record.planner_id, config=record.config, config_sha256=record.config_sha256)


def _read_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise PartnerBankError(f"population config not found: {path}")
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    with open(path, "rb") as f:  # pickled by BRDiv generation; trusted like every other checkpoint here
        return pickle.load(f)


def _read_actor_params(path: Path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "actor_params" not in payload:
        raise PartnerBankError(f"{path}: expected a pickle of {{'actor_params': ...}}")
    return payload["actor_params"]


def _payload_sha256(params) -> str:
    import jax
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        a = np.asarray(leaf)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _planner_record(entry, position, layout, seen_id, seen_cfg) -> PlannerRecord:
    from experiments.partner_adaptation.partner_agents.overcooked.planner_variants import PLANNER_SPECS
    partner_id, planner_id = entry["partner_id"], entry.get("planner_id")
    where = f"partner {partner_id} (planner '{planner_id}')"
    if partner_id != position:
        raise PartnerBankError(
            f"{where}: partner_id must equal its position {position} (it is the ego head and eval id)")
    if planner_id not in PLANNER_SPECS:
        raise PartnerBankError(f"{where}: unknown planner id; known: {sorted(PLANNER_SPECS)}")
    spec = PLANNER_SPECS[planner_id]
    if entry.get("config_sha256") not in (None, spec.config_sha256):
        raise PartnerBankError(f"{where}: recorded configuration hash differs from the registry's; the planner "
                               f"definition changed after this bank was written")
    if planner_id in seen_id:
        raise PartnerBankError(f"{where}: planner already used by partner {seen_id[planner_id]}")
    if spec.config_sha256 in seen_cfg:
        raise PartnerBankError(f"{where}: resolved configuration is identical to partner {seen_cfg[spec.config_sha256]}")
    seen_id[planner_id] = seen_cfg[spec.config_sha256] = partner_id
    return PlannerRecord(partner_id=partner_id, layout=layout, planner_id=planner_id,
                         config=spec.config, config_sha256=spec.config_sha256)


def _imported_record(entry, position, layout, configs, seen, heldout: bool) -> ImportedPartnerRecord:
    seen_member, seen_file, seen_hash = seen
    id_key = "heldout_id" if heldout else "partner_id"
    kind, pop, member, ckpt = entry["kind"], entry["population"], entry["member_index"], entry["checkpoint"]
    pid = entry.get(id_key)
    where = f"{'held-out' if heldout else 'partner'} {pid} ({kind} population '{pop}', member {member}, {ckpt})"
    if pid != position:
        raise PartnerBankError(f"{where}: {id_key} must equal its position {position}")
    if pop not in configs:
        raise PartnerBankError(f"{where}: unknown population '{pop}'; declared: {sorted(configs)}")
    cfg = configs[pop][0]
    if cfg.get("kind") != kind:
        raise PartnerBankError(f"{where}: population config declares kind {cfg.get('kind')!r}")
    members = {m["member_index"]: m for m in cfg.get("members", [])}
    n = int(cfg.get("num_members", len(members)))
    if not isinstance(member, int) or not 0 <= member < n or member not in members:
        raise PartnerBankError(f"{where}: member_index must be one of the {n} exported members")
    if (pop, member) in seen_member:
        raise PartnerBankError(f"{where}: member already used by {seen_member[(pop, member)]}")
    if not ckpt.is_file():
        raise PartnerBankError(f"{where}: checkpoint file does not exist")
    real = ckpt.resolve()
    if real in seen_file:
        raise PartnerBankError(f"{where}: same file as {seen_file[real]}")
    digest = _payload_sha256(_read_actor_params(ckpt))
    if digest in seen_hash:
        raise PartnerBankError(f"{where}: checkpoint payload is identical to {seen_hash[digest]}")
    seen_member[(pop, member)] = seen_file[real] = seen_hash[digest] = where.split(" (")[0]
    return ImportedPartnerRecord(
        partner_id=pid, layout=layout, kind=kind, population=pop, population_config=configs[pop][3],
        checkpoint=real, member_index=member, actor_type=str(cfg.get("actor_type", "")),
        population_size=int(cfg.get("partner_pop_size", 1)), activation=cfg.get("activation", "tanh"),
        fc_hidden_dim=int(cfg.get("fc_hidden_dim", 64)), test_mode=bool(cfg.get("test_mode", False)),
        source=str(cfg.get("source", "")), performance_bounds=dict(members[member].get("performance_bounds") or {}),
        payload_sha256=digest, config=cfg, heldout=heldout)


def _build_bank(name, layout, source, populations, entries, declared, heldout_entries=(),
                format_version=BANK_FORMAT_VERSION) -> PartnerBank:
    if len(entries) != declared:
        raise PartnerBankError(
            f"bank '{name}' declares {declared} partners but lists {len(entries)}; "
            f"explicit banks need exactly the declared number of members")

    configs = {}
    for key, (cfg_path, seed_index) in populations.items():
        cfg = _read_config(cfg_path)
        size = cfg.get("partner_pop_size")
        if not isinstance(size, int) or size < 1:
            raise PartnerBankError(f"population '{key}': config {cfg_path} has no valid partner_pop_size")
        cfg_layout = cfg.get("layout_name") or ""
        if cfg_layout and cfg_layout != layout:
            raise PartnerBankError(
                f"population '{key}' was generated for layout '{cfg_layout}' but the bank layout is '{layout}'")
        configs[key] = (cfg, size, seed_index, cfg_path)

    records, seen_member, seen_file, seen_hash, seen_planner, seen_planner_cfg = [], {}, {}, {}, {}, {}
    for position, entry in enumerate(entries):
        if isinstance(entry, dict) and entry.get("kind") in IMPORTED_KINDS:
            records.append(_imported_record(entry, position, layout, configs, (seen_member, seen_file, seen_hash),
                                            heldout=False))
            continue
        if isinstance(entry, dict):
            records.append(_planner_record(entry, position, layout, seen_planner, seen_planner_cfg))
            continue
        partner_id, pop, member, ckpt = entry
        where = f"partner {partner_id} (population '{pop}', member {member}, {ckpt})"
        if partner_id != position:
            raise PartnerBankError(
                f"{where}: partner_id must equal its position {position} (it is the ego head and eval id)")
        if pop not in configs:
            raise PartnerBankError(f"{where}: unknown population '{pop}'; declared: {sorted(configs)}")
        cfg, size, seed_index, _ = configs[pop]
        if cfg.get("kind", "brdiv") in IMPORTED_KINDS:
            raise PartnerBankError(f"{where}: population '{pop}' holds {cfg['kind']!r} members; "
                                   f"declare the partner with \"kind\": \"{cfg['kind']}\"")
        if not isinstance(member, int) or not 0 <= member < size:
            raise PartnerBankError(f"{where}: member_index must be in [0, {size}) for this population")
        if (pop, member) in seen_member:
            raise PartnerBankError(f"{where}: member already used by partner {seen_member[(pop, member)]}")
        if ckpt.name == GENERIC_CHECKPOINT:
            raise PartnerBankError(
                f"{where}: '{GENERIC_CHECKPOINT}' is a copy of whichever member was saved last, never a member")
        if not ckpt.is_file():
            raise PartnerBankError(f"{where}: checkpoint file does not exist")
        named = MEMBER_FILE.match(ckpt.name)
        if named and (int(named.group(1)), int(named.group(2))) != (seed_index, member):
            raise PartnerBankError(
                f"{where}: file name says seed {named.group(1)} agent {named.group(2)} but the record declares "
                f"seed_index {seed_index}, member_index {member}")
        real = ckpt.resolve()
        if real in seen_file:
            raise PartnerBankError(f"{where}: same file as partner {seen_file[real]}")
        digest = _payload_sha256(_read_actor_params(ckpt))
        if digest in seen_hash:
            raise PartnerBankError(f"{where}: checkpoint payload is identical to partner {seen_hash[digest]}")
        seen_member[(pop, member)] = seen_file[real] = seen_hash[digest] = partner_id
        records.append(PartnerRecord(
            partner_id=partner_id, layout=layout, population=pop, population_config=populations[pop][0],
            checkpoint=real, member_index=member, seed_index=seed_index, population_size=size,
            activation=cfg.get("activation", "tanh"), generation_seed=cfg.get("seed"),
            payload_sha256=digest, config=cfg))
    heldout = []
    for position, entry in enumerate(heldout_entries):
        if entry.get("kind") not in IMPORTED_KINDS:
            raise PartnerBankError(f"held-out {entry.get('heldout_id')}: kind {entry.get('kind')!r} cannot be "
                                   f"held out (supported: {IMPORTED_KINDS})")
        heldout.append(_imported_record(entry, position, layout, configs, (seen_member, seen_file, seen_hash),
                                        heldout=True))
    return PartnerBank(name=name, layout=layout, source=source, records=tuple(records), heldout=tuple(heldout),
                       format_version=format_version)


def load_partner_bank(manifest: str, layout: str) -> PartnerBank:
    path = Path(manifest).resolve()
    if not path.is_file():
        raise PartnerBankError(f"partner bank manifest not found: {path}")
    with open(path) as f:
        data = json.load(f)
    version = data.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise PartnerBankError(f"{path}: unsupported format_version {version!r}")
    if version == BANK_FORMAT_VERSION and data.get("heldout"):
        raise PartnerBankError(f"{path}: a 'heldout' list needs format_version {BANK_FORMAT_VERSION_V2}")
    for key in ("layout", "num_partners", "populations", "partners"):
        if key not in data:
            raise PartnerBankError(f"{path}: missing '{key}'")
    if layout and data["layout"] != layout:
        raise PartnerBankError(f"{path}: bank is for layout '{data['layout']}' but the run uses '{layout}'")

    base = path.parent
    populations = {k: (base / v["config"], int(v.get("seed_index", 0))) for k, v in data["populations"].items()}
    entries = []
    for p in data["partners"]:
        kind = p.get("kind", "brdiv")
        if kind == "planner":
            entries.append({k: p.get(k) for k in ("partner_id", "planner_id", "config_sha256")})
        elif kind == "brdiv":
            entries.append((p["partner_id"], p["population"], p["member_index"], base / p["checkpoint"]))
        elif kind in IMPORTED_KINDS and version >= BANK_FORMAT_VERSION_V2:
            entries.append(_imported_entry(p, base, "partner_id"))
        elif kind in IMPORTED_KINDS:
            raise PartnerBankError(f"{path}: partner {p.get('partner_id')} of kind {kind!r} needs format_version "
                                   f"{BANK_FORMAT_VERSION_V2}")
        else:
            raise PartnerBankError(f"{path}: partner {p.get('partner_id')} has unknown kind {kind!r}")
    heldout = [_imported_entry(p, base, "heldout_id") for p in data.get("heldout", [])]
    return _build_bank(data.get("name", path.stem), data["layout"], path, populations, entries,
                       data["num_partners"], heldout_entries=heldout, format_version=version)


def _imported_entry(p: Dict[str, Any], base: Path, id_key: str) -> Dict[str, Any]:
    missing = [k for k in (id_key, "kind", "population", "member_index", "checkpoint") if k not in p]
    if missing:
        raise PartnerBankError(f"entry {p} is missing {missing}")
    return {id_key: p[id_key], "kind": p["kind"], "population": p["population"],
            "member_index": p["member_index"], "checkpoint": base / p["checkpoint"]}


def bundled_members(layout: str, root: Path = BUNDLED_ROOT):
    """Populations and (population, member, file) triples of the checkpoints bundled for ``layout``,
    ordered by (seed index, member index). Only ``params_seed{i}_agent{j}.pt`` files count."""
    pop_dir = Path(root) / layout
    if not (pop_dir / "config.pckl").is_file():
        raise PartnerBankError(f"no bundled BRDiv population for layout '{layout}' in {pop_dir}")
    found = sorted((int(m.group(1)), int(m.group(2)), p) for p in pop_dir.iterdir()
                   if (m := MEMBER_FILE.match(p.name)))
    if not found:
        raise PartnerBankError(f"{pop_dir} has no params_seed*_agent*.pt member checkpoints")
    populations = {f"bundled_seed{s}": (pop_dir / "config.pckl", s) for s in sorted({s for s, _, _ in found})}
    return populations, [(f"bundled_seed{s}", a, p) for s, a, p in found]


def legacy_bundled_bank(layout: str, num_requested: Optional[int] = None, root: Path = BUNDLED_ROOT) -> PartnerBank:
    """The default (no manifest) bank: the first ``num_requested`` (default 3) bundled members."""
    if not layout:
        raise PartnerBankError("population partners require --layout-name (bundled populations are per layout)")
    populations, members = bundled_members(layout, root)
    n = LEGACY_DEFAULT_PARTNERS if num_requested is None else num_requested
    if n > len(members):
        raise PartnerBankError(
            f"num_population_partners={n} but only {len(members)} bundled members exist for '{layout}'")
    entries = [(i, pop, a, p) for i, (pop, a, p) in enumerate(members[:n])]
    return _build_bank(f"bundled_{layout}", layout, Path(root) / layout, populations, entries, n)


def select_partner_bank(layout: str, bank_path: str = "", num_requested: Optional[int] = None) -> PartnerBank:
    if not bank_path:
        return legacy_bundled_bank(layout, num_requested)
    bank = load_partner_bank(bank_path, layout)
    if num_requested is not None and num_requested != len(bank):
        raise PartnerBankError(
            f"num_population_partners={num_requested} but the bank declares {len(bank)}; "
            f"omit the flag or pass {len(bank)}")
    return bank


def _flat_shapes(tree) -> Dict[str, Tuple[int, ...]]:
    import jax
    return {jax.tree_util.keystr(k): tuple(v.shape) for k, v in jax.tree_util.tree_flatten_with_path(tree)[0]}


def _imported_policy(rec: ImportedPartnerRecord, obs_dim: int, action_dim: int):
    from meal.env.overcooked.layouts.presets import overcooked_layouts
    if rec.kind == "concord":
        from experiments.partner_adaptation.partner_agents.overcooked.concord_cnn import ConcordCNNPolicy
        layout = overcooked_layouts[rec.layout]
        return ConcordCNNPolicy(height=layout["height"], width=layout["width"], test_mode=rec.test_mode)
    try:
        if rec.kind == "bc_proxy":
            from experiments.partner_adaptation.partner_agents.overcooked.bc_proxy_wrapper import BCProxyPolicy
            layout = overcooked_layouts[rec.layout]
            return BCProxyPolicy(rec.layout, layout["height"], layout["width"], num_pots=len(layout["pot_idx"]),
                                 num_goals=len(layout["goal_idx"]), test_mode=rec.test_mode)
        from experiments.partner_adaptation.jax_aht_bridge.policies import JaxMarlCompatPolicy, make_jaxaht_policy
    except ImportError as exc:
        raise PartnerBankError(f"partner {rec.partner_id} ({rec.label}) is a {rec.kind!r} partner: {exc}") from exc
    layout = overcooked_layouts[rec.layout]
    inner = make_jaxaht_policy(rec.actor_type, obs_dim, action_dim, pop_size=rec.population_size,
                               activation=rec.activation, fc_hidden_dim=rec.fc_hidden_dim)
    return JaxMarlCompatPolicy(inner, height=layout["height"], width=layout["width"], test_mode=rec.test_mode)


def _load_records(records, obs_dim: int, action_dim: int, cache: Dict[Any, Any]) -> List[LoadedPartner]:
    import jax
    from experiments.partner_adaptation.partner_agents.agent_interface import ActorWithConditionalCriticPolicy

    loaded = []
    for rec in records:
        if rec.kind == "planner":
            from meal.env.overcooked.layouts.presets import overcooked_layouts
            from experiments.partner_adaptation.partner_agents.overcooked.agent_policy_wrappers import (
                OvercookedPlannerPolicyWrapper)
            from experiments.partner_adaptation.partner_agents.overcooked.planner_variants import planner_params
            if rec.layout not in overcooked_layouts:
                raise PartnerBankError(f"partner {rec.partner_id} ({rec.label}): unknown layout '{rec.layout}'")
            if "planner" not in cache:
                cache["planner"] = OvercookedPlannerPolicyWrapper(overcooked_layouts[rec.layout])
            loaded.append(LoadedPartner(rec, cache["planner"], planner_params(rec.planner_id)))
            continue
        if rec.kind in IMPORTED_KINDS:
            key = (rec.kind, rec.actor_type, rec.population_size, rec.activation, rec.fc_hidden_dim, rec.test_mode,
                   rec.layout)
            if key not in cache:
                cache[key] = _imported_policy(rec, obs_dim, action_dim)
            what = f"the {rec.kind} {rec.actor_type} policy"
            hint = f"actor_type={rec.actor_type}, partner_pop_size={rec.population_size}"
        else:
            key = (rec.population_size, rec.activation)
            if key not in cache:
                cache[key] = ActorWithConditionalCriticPolicy(
                    action_dim, obs_dim=obs_dim, pop_size=rec.population_size, activation=rec.activation)
            what = "the BRDiv actor-critic"
            hint = f"partner_pop_size={rec.population_size}"
        shapes_key = ("shapes",) + key if isinstance(key, tuple) else ("shapes", key)
        if shapes_key not in cache:
            cache[shapes_key] = _flat_shapes(jax.eval_shape(cache[key].init_params, jax.random.PRNGKey(0)))
        expected = cache[shapes_key]
        params = _read_actor_params(rec.checkpoint)
        got = _flat_shapes(params)
        if got != expected:
            bad = [f"{k}: checkpoint {got.get(k)} vs expected {expected.get(k)}"
                   for k in sorted(set(got) | set(expected)) if got.get(k) != expected.get(k)]
            raise PartnerBankError(
                f"partner {rec.partner_id} ({rec.label}, {rec.checkpoint}) does not match {what} "
                f"declared for it (obs_dim={obs_dim}, {hint}, "
                f"activation={rec.activation}); wrong layout or population size? "
                f"Mismatches: {'; '.join(bad[:3])}")
        loaded.append(LoadedPartner(rec, cache[key], params))
    return loaded


def load_bank_partners(bank: PartnerBank, obs_dim: int, action_dim: int = 6
                       ) -> Tuple[List[LoadedPartner], List[LoadedPartner]]:
    cache: Dict[Any, Any] = {}
    obs_dim = int(obs_dim)
    return (_load_records(bank.records, obs_dim, action_dim, cache),
            _load_records(bank.heldout, obs_dim, action_dim, cache))


def load_partners(bank: PartnerBank, obs_dim: int, action_dim: int = 6) -> List[LoadedPartner]:
    """Build one policy per distinct (partner_pop_size, activation) and load every member's parameters,
    checking each against the architecture its own population config declares. All planners of a bank share
    one policy object (their configuration is the params), so they share one compiled graph."""
    return _load_records(bank.records, int(obs_dim), action_dim, {})


def _relpath(target: Path, base: Path) -> str:
    return os.path.relpath(Path(target).resolve(), base.resolve())


def assemble_bank(layout: str, out: str, generation_dirs: List[str], name: str = "", bundled: bool = True,
                  bundled_root: Path = BUNDLED_ROOT, planners: Tuple[str, ...] = ()) -> PartnerBank:
    """Write a manifest listing the bundled members (optional), every member recorded in each
    ``generation.json`` and then the requested planners (ids, or ``("all",)`` for P01-P12 only), and load it back so an
    invalid bank is never left behind silently. Order in the manifest is training order."""
    from experiments.partner_adaptation.partner_agents.overcooked.planner_variants import (
        CLASSIC_PLANNER_IDS, PLANNER_SPECS)
    planners = CLASSIC_PLANNER_IDS if tuple(planners) == ("all",) else tuple(planners)
    out_path = Path(out).resolve()
    base = out_path.parent
    populations, members = {}, []
    if bundled:
        pops, mem = bundled_members(layout, bundled_root)
        populations.update({k: (c, s) for k, (c, s) in pops.items()})
        members += mem
    for d in generation_dirs:
        gen_path = Path(d) / "generation.json"
        if not gen_path.is_file():
            raise PartnerBankError(f"{gen_path} not found; was generation completed?")
        gen = json.loads(gen_path.read_text())
        if gen.get("status") != "complete" or not gen.get("members"):
            raise PartnerBankError(f"{gen_path}: generation not complete (status={gen.get('status')!r})")
        for m in gen["members"]:
            key = f"gseed{gen['seed']}" + (f"_i{m['seed_index']}" if gen["num_seeds"] > 1 else "")
            populations[key] = (gen_path, m["seed_index"])
            members.append((key, m["member_index"], Path(d) / m["file"]))

    manifest = {
        "format_version": BANK_FORMAT_VERSION,
        "name": name or f"{layout}_bank",
        "layout": layout,
        "num_partners": len(members) + len(planners),
        "populations": {k: {"config": _relpath(c, base), "seed_index": s} for k, (c, s) in populations.items()},
        "partners": [{"partner_id": i, "population": pop, "member_index": a, "checkpoint": _relpath(p, base)}
                     for i, (pop, a, p) in enumerate(members)]
                    + [{"partner_id": len(members) + j, "kind": "planner", "planner_id": pid,
                        "config_sha256": PLANNER_SPECS[pid].config_sha256 if pid in PLANNER_SPECS else None}
                       for j, pid in enumerate(planners)],
    }
    base.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return load_partner_bank(str(out_path), layout)


def write_bank_v2(out: str, layout: str, partners: List[Dict[str, Any]], heldout: List[Dict[str, Any]] = (),
                  populations: Optional[Dict[str, Path]] = None, name: str = "") -> PartnerBank:
    out_path = Path(out).resolve()
    base = out_path.parent
    tmp = out_path.with_name(f".{out_path.name}.writing.json")

    def entry(e, id_key, i):
        d = {id_key: i, "kind": e["kind"]}
        if e["kind"] == "planner":
            return dict(d, planner_id=e["planner_id"], config_sha256=e.get("config_sha256"))
        return dict(d, population=e["population"], member_index=e["member_index"],
                    checkpoint=_relpath(e["checkpoint"], base), **{k: v for k, v in e.items() if k in ("note",)})

    manifest = {
        "format_version": BANK_FORMAT_VERSION_V2,
        "name": name or f"aht_{layout}",
        "layout": layout,
        "num_partners": len(partners),
        "populations": {k: {"config": _relpath(c, base), "seed_index": 0} for k, c in (populations or {}).items()},
        "partners": [entry(e, "partner_id", i) for i, e in enumerate(partners)],
        "heldout": [entry(e, "heldout_id", i) for i, e in enumerate(heldout)],
    }
    base.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        load_partner_bank(str(tmp), layout)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out_path)
    return load_partner_bank(str(out_path), layout)


def main():
    import tyro

    @dataclass
    class AssembleConfig:
        layout: str
        out: str
        generation_dirs: List[str] = field(default_factory=list)
        name: str = ""
        bundled: bool = True  # include the bundled members of the layout first
        planners: List[str] = field(default_factory=list)  # planner ids (P01 ...) appended last, or "all"

    cfg = tyro.cli(AssembleConfig)
    bank = assemble_bank(cfg.layout, cfg.out, cfg.generation_dirs, cfg.name, cfg.bundled, planners=tuple(cfg.planners))
    print(f"wrote {cfg.out}: {len(bank)} partners for {bank.layout}")
    for r in bank.records:
        print(f"  {r.partner_id:3d}  {r.label}  {r.checkpoint.name if r.kind == 'brdiv' else r.config['name']}")


if __name__ == "__main__":
    main()
