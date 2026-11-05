from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

from experiments.partner_adaptation.jax_aht_bridge import hf_partners
from experiments.partner_adaptation.partner_bank import PartnerBank, write_bank_v2

KIND = "jaxaht"


def default_root() -> Path:
    return Path(os.environ.get("MEAL_CPA_ROOT", "cpa_root")) / "partners"


def shared_runs(layout: str) -> set:
    runs = {s: {spec.run_dir for spec in hf_partners.load_specs(s, layout)} for s in hf_partners.SETS}
    return set.intersection(*runs.values())


def import_set(set_name: str, layout: str, root: Path, download: bool, exclude_runs=(),
               bc_proxy: bool = True) -> Tuple[List[Dict], Dict[str, Path]]:
    import json
    specs = []
    for spec in hf_partners.load_specs(set_name, layout):
        if spec.run_dir in exclude_runs:
            print(f"  [{set_name}] {spec.name}: skipped (run {spec.run_dir} also supplies the other set)")
            continue
        specs.append(spec)
    hf_dir = root / "jaxaht" / "hf"
    if download:
        hf_partners.download(specs, hf_dir)
    entries, populations = [], {}
    for spec in specs:
        gen_path = hf_partners.import_spec(spec, hf_dir, root / "jaxaht")
        gen = json.loads(gen_path.read_text())
        populations[spec.population_key] = gen_path
        entries += [dict(kind=KIND, population=spec.population_key, member_index=m["member_index"],
                         checkpoint=gen_path.parent / m["file"]) for m in gen["members"]]
        print(f"  [{set_name}] {spec.name}: {gen['num_members']} members ({spec.actor_type}, run {spec.run_path})")
    for spec in (hf_partners.load_bc_specs(set_name, layout) if bc_proxy else []):
        if download:
            hf_partners.download([spec], hf_dir)
        gen_path = hf_partners.import_bc_proxy(spec, hf_dir, root / "jaxaht")
        key = f"{spec.population_key}"
        populations[key] = gen_path
        entries.append(dict(kind="bc_proxy", population=key, member_index=0, checkpoint=gen_path.parent / "m0.pt"))
        print(f"  [{set_name}] {spec.name}: BC human proxy run {spec.idx_list[0]} ({spec.run_path})")
    return entries, populations


CONCORD_LAYOUT = "coord_ring"


def concord_members(root: Path) -> Tuple[List[Dict], Dict[str, Path]]:
    import json
    import sys
    gen_path = root / "concord" / CONCORD_LAYOUT / "generation.json"
    if not gen_path.is_file():
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
        from convert_concord_partners import convert
        convert(gen_path.parent)
    gen = json.loads(gen_path.read_text())
    key = "concord_coord_ring"
    print(f"  [concord] {gen['num_members']} MEP/HSP members ({gen['source']})")
    return ([dict(kind="concord", population=key, member_index=m["member_index"],
                  checkpoint=gen_path.parent / m["file"]) for m in gen["members"]], {key: gen_path})


def brdiv_members(dirs) -> Tuple[List[Dict], Dict[str, Path]]:
    import json
    entries, populations = [], {}
    for d in dirs:
        d = Path(d)
        gen_path = d / "generation.json"
        gen = json.loads(gen_path.read_text())
        if gen.get("status") != "complete":
            raise ValueError(f"{gen_path}")
        key = f"brdiv_gseed{gen['seed']}"
        populations[key] = gen_path
        entries += [dict(kind="brdiv", population=key, member_index=m["member_index"],
                         checkpoint=d / m["file"]) for m in gen["members"]]
    return entries, populations


def build_layout_bank(layout: str, root: Path, download: bool = True, extra_partners=None,
                      extra_populations=None, concord: bool = True, keep_shared_runs: bool = False,
                      planners=()) -> PartnerBank:
    print(f"{layout}:")
    excluded = set() if keep_shared_runs else shared_runs(layout)
    partners, populations = import_set("val", layout, root, download, exclude_runs=excluded)
    heldout, test_pops = import_set("eval", layout, root, download)
    populations.update(test_pops)
    if concord and layout == CONCORD_LAYOUT:
        c_entries, c_pops = concord_members(root)
        partners += c_entries
        populations.update(c_pops)
    partners += list(extra_partners or [])
    partners += [dict(kind="planner", planner_id=p) for p in planners]
    for p in planners:
        print(f"  [planner] {p}")
    populations.update(extra_populations or {})
    bank = write_bank_v2(str(root / f"bank_{layout}.json"), layout, partners, heldout, populations,
                         name=f"aht_{layout}")
    print(f"  wrote {bank.source}: {len(bank.records)} training partners, {len(bank.heldout)} held-out")
    return bank


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", nargs="+", default=list(hf_partners.LAYOUTS), choices=hf_partners.LAYOUTS)
    parser.add_argument("--root", type=Path, default=None, help="output root (default $MEAL_CPA_ROOT/partners)")
    parser.add_argument("--skip-download", action="store_true", help="reuse <root>/jaxaht/hf")
    parser.add_argument("--no-concord", action="store_true", help="do not add Concord's coord_ring teammates")
    parser.add_argument("--keep-shared-runs", action="store_true",
                        help="also train on val members whose run supplies held-out (eval) members")
    parser.add_argument("--planners", nargs="*", default=[], help="planner ids appended to training, e.g. P01")
    parser.add_argument("--brdiv-dirs", nargs="*", type=Path, default=[])
    args = parser.parse_args(argv)
    root = (args.root or default_root()).resolve()
    extra, extra_pops = brdiv_members(args.brdiv_dirs)
    for layout in args.layouts:
        build_layout_bank(layout, root, download=not args.skip_download, concord=not args.no_concord,
                          keep_shared_runs=args.keep_shared_runs, planners=args.planners,
                          extra_partners=extra, extra_populations=extra_pops)


if __name__ == "__main__":
    main()
