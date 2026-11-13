"""Turn single-partner best-response runs into a cross-play matrix and select a bank."""
import argparse, csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.partner_adaptation.partner_bank import load_partner_bank
from experiments.results.cpa_eval import read_csv, resolve_run_dir

LEARNABILITY_FRACTION = 0.8


def xp_row(path, P):
    path = resolve_run_dir(path)
    stages = json.loads((path / "config.json").read_text()).get("only_stages")
    if not (isinstance(stages, list) and len(stages) == 1):
        raise SystemExit(f"{path}: expected only_stages=[i], got {stages!r}")
    events = {}
    for r in read_csv(path / "eval_metrics.csv"):
        if r["scope"] != "full" or not int(float(r["completed"])): continue
        events.setdefault(int(r["env_steps"]), {}).setdefault(int(r["eval_partner_id"]), []).append(float(r["soups"]))
    if not events: raise SystemExit(f"{path}: no full evaluation")
    row = np.full(P, np.nan)
    for j, vals in events[max(events)].items(): row[j] = float(np.mean(vals))
    return int(stages[0]), row


def br_run_dir(path):
    path = Path(path)
    if (path / "config.json").is_file():
        return path
    return resolve_run_dir(path)


def partner_index(path):
    path = br_run_dir(path)
    stages = json.loads((path / "config.json").read_text()).get("only_stages")
    if not (isinstance(stages, list) and len(stages) == 1):
        raise SystemExit(f"{path}: expected only_stages=[i], got {stages!r}")
    return int(stages[0]), path


def read_xp_matrix(path, labels):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows or len(rows[0]) < 2:
        raise SystemExit(f"{path}: expected a labelled square matrix")
    header = rows[0][1:]
    if len(header) != len(labels) or set(header) != set(labels):
        raise SystemExit(f"{path}: matrix labels do not match the bank")
    col = {lab: j for j, lab in enumerate(header)}
    by_row = {}
    for row in rows[1:]:
        if len(row) != len(header) + 1:
            raise SystemExit(f"{path}: row {row[0]!r} has the wrong width")
        by_row[row[0]] = row[1:]
    missing = [lab for lab in labels if lab not in by_row]
    if missing:
        raise SystemExit(f"{path}: missing rows for {missing}")
    P = len(labels)
    XP = np.empty((P, P), float)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            XP[i, j] = float(by_row[li][col[lj]])
    return XP


def kind_match(kind, partner_kind):
    if kind == "learned":
        return partner_kind != "planner"
    return partner_kind == kind


def components(kept, sim, tau):
    rem, groups = set(kept), []
    for seed in sorted(kept):
        if seed not in rem: continue
        stack, comp = [seed], []
        rem.remove(seed)
        while stack:
            a = stack.pop(); comp.append(a)
            for b in [b for b in rem if sim[a, b] >= tau]: rem.remove(b); stack.append(b)
        groups.append(sorted(comp))
    return groups


def distinctness_rows(kept, XP, labels, kinds, threshold):
    rows, non_distinct = [], []
    for j in kept:
        own = float(XP[j, j])
        others = [i for i in kept if i != j]
        if not others:
            rows.append(dict(label=labels[j], kind=kinds[j], own_score=own,
                             closest_specialist="", ratio="", distinct="true"))
            continue
        if own == 0:
            closest = max(others, key=lambda i: XP[i, j])
            rows.append(dict(label=labels[j], kind=kinds[j], own_score=own,
                             closest_specialist=labels[closest], ratio="", distinct="false"))
            non_distinct.append((labels[j], labels[closest], "own score is 0"))
            continue
        closest = max(others, key=lambda i: XP[i, j] / own)
        ratio = float(XP[closest, j] / own)
        distinct = ratio < threshold
        rows.append(dict(label=labels[j], kind=kinds[j], own_score=own,
                         closest_specialist=labels[closest],
                         ratio=f"{ratio:.6g}", distinct="true" if distinct else "false"))
        if not distinct:
            non_distinct.append((labels[j], labels[closest], f"ratio={ratio:.6g}"))
    return rows, non_distinct


def learnability_row(path, label, own_score):
    path = Path(path)
    out = dict(label=label, own_score=own_score, final_train_soups="", steps_to_80pct="",
               last_quarter_gain="")
    tm = path / "train_metrics.csv"
    if not tm.is_file():
        print(f"{path}: no train_metrics.csv, learnability fields empty")
        return out
    usable = [r for r in read_csv(tm) if r.get("soups") not in ("", None)]
    if not usable:
        return out
    final = float(usable[-1]["soups"])
    out["final_train_soups"] = final
    target = LEARNABILITY_FRACTION * final
    for r in usable:
        if float(r["soups"]) >= target:
            out["steps_to_80pct"] = int(r["env_steps"])
            break
    q = len(usable) // 4
    if q == 0:
        return out
    last = [float(r["soups"]) for r in usable[-q:]]
    third = [float(r["soups"]) for r in usable[-2 * q:-q]]
    out["last_quarter_gain"] = float(np.mean(last) - np.mean(third))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--matrix", default=None, help="xp_matrix.csv from specialist_crossplay (labels match the bank)")
    ap.add_argument("--keep", action="append", default=[], metavar="KIND=N")
    ap.add_argument("--keep-by", choices=("similarity", "own_score"), default="similarity")
    ap.add_argument("--min-own-score", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--distinct-threshold", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-bank", required=True)
    args = ap.parse_args(argv)
    bank_path, out_bank_path = Path(args.bank), Path(args.out_bank)
    if out_bank_path.resolve().parent != bank_path.resolve().parent:
        raise SystemExit(
            f"--out-bank must be in the same directory as --bank (partner paths are relative to the manifest); "
            f"got {out_bank_path.resolve().parent} vs {bank_path.resolve().parent}")
    manifest = json.loads(bank_path.read_text())
    bank = load_partner_bank(args.bank, manifest["layout"])
    labels, kinds, P = [r.label for r in bank.records], [r.kind for r in bank.records], len(bank)
    run_of = {}
    if args.matrix:
        XP = read_xp_matrix(args.matrix, labels)
        for d in args.runs:
            i, path = partner_index(d)
            if i in run_of: raise SystemExit(f"partner {i} has two runs")
            run_of[i] = path
    else:
        by = {}
        for d in args.runs:
            i, row = xp_row(d, P)
            if i in by: raise SystemExit(f"partner {i} has two runs")
            by[i] = row
            run_of[i] = br_run_dir(d)
        for i in range(P):
            if i not in by: raise SystemExit(f"partner {i} has no run")
        XP = np.stack([by[i] for i in range(P)])
    for i in range(P):
        if i not in run_of: raise SystemExit(f"partner {i} has no run")
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = XP / np.where(np.diag(XP) == 0, np.nan, np.diag(XP))
    sim = np.minimum(rel, rel.T); np.fill_diagonal(sim, np.nan)
    for j in range(P):
        if np.nanargmax(XP[:, j]) != j: print(f"warning: {labels[j]}: its best response is weak")
    kept = list(range(P))
    for item in args.keep:
        kind, _, n = item.partition("="); n = int(n)
        if args.keep_by == "own_score":
            cands = [i for i in kept if kind_match(kind, kinds[i])]
            if len(cands) > n:
                ranked = sorted(cands, key=lambda i: (-XP[i, i], i))
                drop = set(ranked[n:])
                kept = [i for i in kept if i not in drop]
            continue
        while sum(kind_match(kind, kinds[i]) for i in kept) > n:
            low = [i for i in kept if kind_match(kind, kinds[i]) and XP[i, i] < args.min_own_score]
            if not low: break
            kept.remove(min(low, key=lambda i: XP[i, i]))
        while sum(kind_match(kind, kinds[i]) for i in kept) > n:
            cands = [i for i in kept if kind_match(kind, kinds[i])]
            def max_sim(i):
                others = [j for j in kept if j != i]
                vals = sim[i, others]
                if not others or np.all(np.isnan(vals)): return np.inf
                return float(np.nanmax(vals))
            kept.remove(max(cands, key=lambda i: (max_sim(i), -XP[i, i])))
    if args.keep_by != "own_score":
        for i in kept:
            if XP[i, i] < args.min_own_score:
                print(f"warning: {labels[i]}: own score below min-own-score ({XP[i, i]:.3f})")
    dropped = [i for i in range(P) if i not in set(kept)]
    comps = components(kept, sim, args.tau)
    groups = [[labels[i] for i in g] for g in comps]
    dist_rows, non_distinct = distinctness_rows(kept, XP, labels, kinds, args.distinct_threshold)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "xp_matrix.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["best_response"] + labels)
        for i in range(P): w.writerow([labels[i]] + [f"{XP[i, j]:.3f}" for j in range(P)])
    (out / "selection.json").write_text(json.dumps(
        dict(tau=args.tau, kept=kept, dropped=dropped, groups=groups, distinct=len(comps)), indent=2) + "\n")
    with open(out / "distinctness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "kind", "own_score", "closest_specialist", "ratio", "distinct"])
        w.writeheader()
        for row in dist_rows:
            w.writerow(row)
    (out / "reference_scores.json").write_text(json.dumps(
        {labels[j]: float(XP[j, j]) for j in kept}, indent=2) + "\n")
    with open(out / "learnability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "own_score", "final_train_soups", "steps_to_80pct",
                                          "last_quarter_gain"])
        w.writeheader()
        for i in range(P):
            w.writerow(learnability_row(run_of[i], labels[i], float(XP[i, i])))
    out_bank_path.write_text(json.dumps(
        {**manifest, "num_partners": len(kept),
         "partners": [{**manifest["partners"][i], "partner_id": n} for n, i in enumerate(kept)]}, indent=2) + "\n")
    load_partner_bank(args.out_bank, manifest["layout"])
    print(dict(Counter(kinds[i] for i in kept)), f"distinct={len(comps)}")
    for g in groups:
        if len(g) > 1: print(g)
    by_kind = Counter(r["kind"] for r in dist_rows if r["distinct"] == "true")
    print(f"distinct partners by kind: {dict(by_kind)}")
    for label, closest, why in non_distinct:
        print(f"not distinct: {label} (closest {closest}; {why})")


if __name__ == "__main__":
    main()
