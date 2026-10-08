"""Explicit bank of learned CPA partners.

A bank is a JSON manifest that lists every learned partner the ego agent is trained against, in training
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

``load_partners`` additionally checks every checkpoint against the policy its population config declares.
Populations with equal ``(partner_pop_size, activation)`` share one policy object (and so one compiled graph and
one vmapped evaluation group); otherwise each gets its own policy. Parameter trees are never stacked across
different policies.

Without a manifest, ``select_partner_bank`` reproduces the legacy behaviour: the checkpoints bundled in
``partner_agents/BRDiv_population/<layout>``.

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BANK_FORMAT_VERSION = 1
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


@dataclass(frozen=True)
class PartnerBank:
    name: str
    layout: str
    source: Optional[Path]
    records: Tuple[PartnerRecord, ...]

    def __len__(self):
        return len(self.records)


@dataclass(frozen=True)
class LoadedPartner:
    record: PartnerRecord
    policy: Any
    params: Any


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


def _build_bank(name, layout, source, populations, entries, declared) -> PartnerBank:
    """populations: key -> (config path, seed_index). entries: (partner_id, population, member_index, checkpoint)."""
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
        configs[key] = (cfg, size, seed_index)

    records, seen_member, seen_file, seen_hash = [], {}, {}, {}
    for position, (partner_id, pop, member, ckpt) in enumerate(entries):
        where = f"partner {partner_id} (population '{pop}', member {member}, {ckpt})"
        if partner_id != position:
            raise PartnerBankError(
                f"{where}: partner_id must equal its position {position} (it is the ego head and eval id)")
        if pop not in configs:
            raise PartnerBankError(f"{where}: unknown population '{pop}'; declared: {sorted(configs)}")
        cfg, size, seed_index = configs[pop]
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
    return PartnerBank(name=name, layout=layout, source=source, records=tuple(records))


def load_partner_bank(manifest: str, layout: str) -> PartnerBank:
    path = Path(manifest).resolve()
    if not path.is_file():
        raise PartnerBankError(f"partner bank manifest not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if data.get("format_version") != BANK_FORMAT_VERSION:
        raise PartnerBankError(f"{path}: unsupported format_version {data.get('format_version')!r}")
    for key in ("layout", "num_partners", "populations", "partners"):
        if key not in data:
            raise PartnerBankError(f"{path}: missing '{key}'")
    if layout and data["layout"] != layout:
        raise PartnerBankError(f"{path}: bank is for layout '{data['layout']}' but the run uses '{layout}'")

    base = path.parent
    populations = {k: (base / v["config"], int(v.get("seed_index", 0))) for k, v in data["populations"].items()}
    entries = [(p["partner_id"], p["population"], p["member_index"], base / p["checkpoint"])
               for p in data["partners"]]
    return _build_bank(data.get("name", path.stem), data["layout"], path, populations, entries,
                       data["num_partners"])


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


def load_partners(bank: PartnerBank, obs_dim: int, action_dim: int = 6) -> List[LoadedPartner]:
    """Build one policy per distinct (partner_pop_size, activation) and load every member's parameters,
    checking each against the architecture its own population config declares."""
    import jax
    from experiments.partner_adaptation.partner_agents.agent_interface import ActorWithConditionalCriticPolicy

    obs_dim = int(obs_dim)
    policies, expected, loaded = {}, {}, []
    for rec in bank.records:
        key = (rec.population_size, rec.activation)
        if key not in policies:
            policies[key] = ActorWithConditionalCriticPolicy(
                action_dim, obs_dim=obs_dim, pop_size=rec.population_size, activation=rec.activation)
            expected[key] = _flat_shapes(jax.eval_shape(policies[key].init_params, jax.random.PRNGKey(0)))
        params = _read_actor_params(rec.checkpoint)
        got = _flat_shapes(params)
        if got != expected[key]:
            bad = [f"{k}: checkpoint {got.get(k)} vs expected {expected[key].get(k)}"
                   for k in sorted(set(got) | set(expected[key])) if got.get(k) != expected[key].get(k)]
            raise PartnerBankError(
                f"partner {rec.partner_id} ({rec.label}, {rec.checkpoint}) does not match the BRDiv actor-critic "
                f"declared for it (obs_dim={obs_dim}, partner_pop_size={rec.population_size}, "
                f"activation={rec.activation}); wrong layout or population size? "
                f"Mismatches: {'; '.join(bad[:3])}")
        loaded.append(LoadedPartner(rec, policies[key], params))
    return loaded


def _relpath(target: Path, base: Path) -> str:
    return os.path.relpath(Path(target).resolve(), base.resolve())


def assemble_bank(layout: str, out: str, generation_dirs: List[str], name: str = "", bundled: bool = True,
                  bundled_root: Path = BUNDLED_ROOT) -> PartnerBank:
    """Write a manifest listing the bundled members (optional) followed by every member recorded in each
    ``generation.json``, then load it back so an invalid bank is never left behind silently."""
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
        "num_partners": len(members),
        "populations": {k: {"config": _relpath(c, base), "seed_index": s} for k, (c, s) in populations.items()},
        "partners": [{"partner_id": i, "population": pop, "member_index": a, "checkpoint": _relpath(p, base)}
                     for i, (pop, a, p) in enumerate(members)],
    }
    base.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
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

    cfg = tyro.cli(AssembleConfig)
    bank = assemble_bank(cfg.layout, cfg.out, cfg.generation_dirs, cfg.name, cfg.bundled)
    print(f"wrote {cfg.out}: {len(bank)} partners for {bank.layout}")
    for r in bank.records:
        print(f"  {r.partner_id:3d}  {r.label}  {r.checkpoint.name}")


if __name__ == "__main__":
    main()
