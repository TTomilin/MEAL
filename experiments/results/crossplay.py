"""Turn single-partner best-response runs into a cross-play matrix and select a bank."""
import argparse, csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.partner_adaptation.partner_bank import load_partner_bank
from experiments.results.cpa_eval import read_csv, resolve_run_dir
from experiments.partner_adaptation.diversity import (
    DEFAULT_MAX_OFF_DIAG, DEFAULT_MIN_VENDI_GAIN, diagnose_specialists, diversity_summary,
    greedy_vendi_curve,
)

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


def read_partner_events(path, labels):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "partner" not in rows[0]:
        raise SystemExit(f"{path}: expected a partner_events.csv with a partner column")
    by_label = {}
    for row in rows:
        label = row["partner"]
        by_label[label] = sum(float(row[k]) for k in row if k != "partner" and row[k] != "")
    missing = [lab for lab in labels if lab not in by_label]
    if missing:
        raise SystemExit(f"{path}: missing rows for {missing}")
    return np.array([by_label[lab] for lab in labels], float)


def largest_all_different(ratio_matrix, candidates, threshold):
    """Largest subset of candidates that are pairwise different (ratio < threshold both ways)."""
    cand = sorted(candidates)
    nbrs = {
        i: {j for j in cand if j != i
            and ratio_matrix[i, j] < threshold and ratio_matrix[j, i] < threshold}
        for i in cand
    }
    best = []

    def search(r, p, x):
        nonlocal best
        if not p and not x:
            chosen = sorted(r)
            if len(chosen) > len(best) or (len(chosen) == len(best) and chosen < best):
                best = chosen
            return
        u = next(iter(p | x))
        for v in list(p - nbrs[u]):
            search(r | {v}, p & nbrs[v], x & nbrs[v])
            p.remove(v)
            x.add(v)

    search(set(), set(cand), set())
    return best


def read_behaviour_visits(path, labels):
    with np.load(path) as data:
        for req in ("labels", "visits"):
            if req not in data:
                raise SystemExit(f"{path}: missing required array {req!r}")
        beh_labels = [str(x) for x in data["labels"]]
        visits = np.asarray(data["visits"], dtype=np.float64)
        episodes = int(data["episodes"]) if "episodes" in data else None
    n = len(beh_labels)
    if visits.ndim != 3 or visits.shape[:2] != (n, n):
        raise SystemExit(f"{path}: visits shape {visits.shape} does not match ({n}, {n}, n_cells)")
    index = {lab: i for i, lab in enumerate(beh_labels)}
    missing = [lab for lab in labels if lab not in index]
    if missing:
        raise SystemExit(f"{path}: missing partners {missing}")
    idx = [index[lab] for lab in labels]
    return visits[np.ix_(idx, idx)], episodes


def _parse_keep_specs(items):
    parsed = []
    for item in items:
        kind, sep, raw = item.partition("=")
        if sep != "=" or not kind:
            raise SystemExit(f"--keep expects KIND=N, got {item!r}")
        try:
            n = int(raw)
        except ValueError as exc:
            raise SystemExit(f"--keep expects integer N in KIND=N, got {item!r}") from exc
        if n < 0:
            raise SystemExit(f"--keep count must be >= 0, got {item!r}")
        parsed.append((kind, n))
    return parsed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--matrix", default=None, help="xp_matrix.csv from specialist_crossplay (labels match the bank)")
    ap.add_argument("--xp-std", default=None, help="xp_matrix_std.csv from specialist_crossplay (for Vendi CI)")
    ap.add_argument("--behaviour", default=None, help="partner_behaviour.npz from specialist_crossplay (for behaviour Vendi)")
    ap.add_argument("--episodes", type=int, default=None, help="eval episodes per XP cell for Vendi bootstrap CI")
    ap.add_argument("--events", default=None, help="partner_events.csv from specialist_crossplay")
    ap.add_argument("--reference", default=None, metavar="LABEL",
                    help="idle planner; its own score is the ego-alone score")
    ap.add_argument("--helpful-ratio", type=float, default=0.8)
    ap.add_argument("--min-events", type=float, default=6)
    ap.add_argument("--different", type=float, default=0.7)
    ap.add_argument("--keep", action="append", default=[], metavar="KIND=N")
    ap.add_argument("--keep-by", choices=("similarity", "own_score", "vendi"), default="similarity")
    ap.add_argument("--max-partners", type=int, default=None, help="optional hard cap on partners under --keep-by vendi")
    ap.add_argument("--min-vendi-gain", type=float, default=DEFAULT_MIN_VENDI_GAIN)
    ap.add_argument("--max-off-diag", type=float, default=DEFAULT_MAX_OFF_DIAG)
    ap.add_argument("--min-own-score", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--distinct-threshold", type=float, default=0.7)
    ap.add_argument("--strict-specialists", action="store_true",
                    help="abort if any bank partner has ZERO_OWN_SCORE, WEAK_SPECIALIST, or STILL_LEARNING")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-bank", required=True)
    args = ap.parse_args(argv)
    if args.events and not args.reference:
        raise SystemExit("--reference is required with --events")
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
    XP_std = read_xp_matrix(args.xp_std, labels) if args.xp_std else None
    visits, beh_episodes = read_behaviour_visits(args.behaviour, labels) if args.behaviour else (None, None)
    ci_episodes = args.episodes if args.episodes is not None else (beh_episodes or 20)
    ref_i = ref_score = event_sums = works = reasons = None
    if args.reference is not None:
        if args.reference not in labels:
            raise SystemExit(f"--reference {args.reference!r} is not in the bank")
        ref_i = labels.index(args.reference)
        ref_score = float(XP[ref_i, ref_i])

    learn_rows = [learnability_row(run_of[i], labels[i], float(XP[i, i])) for i in range(P)]
    health = diagnose_specialists(XP, labels, learn_rows, min_own_score=args.min_own_score)
    if args.strict_specialists:
        bad = [f"{p['label']} ({','.join(p['flags'])})"
               for p in health["partners"] if p["defect"] and p["index"] != ref_i]
        if bad:
            raise SystemExit(
                f"unhealthy specialist(s) detected ({len(bad)}/{P}): {', '.join(bad)}; "
                f"retrain weak/unconverged specialists before selecting a bank")
    if args.events:
        event_sums = read_partner_events(args.events, labels)
        floor = args.helpful_ratio * ref_score
        works, reasons = [], []
        for i in range(P):
            if event_sums[i] < args.min_events:
                works.append(False); reasons.append("no_job")
            elif XP[i, i] < floor:
                works.append(False); reasons.append("below_ego_alone")
            else:
                works.append(True); reasons.append("")

    with np.errstate(invalid="ignore", divide="ignore"):
        rel = XP / np.where(np.diag(XP) == 0, np.nan, np.diag(XP))
    sim = np.minimum(rel, rel.T); np.fill_diagonal(sim, np.nan)
    for j in range(P):
        if np.isfinite(XP[:, j]).any() and np.nanargmax(XP[:, j]) != j:
            print(f"warning: {labels[j]}: its best response is weak")
    vendi_candidates = [
        i for i in range(P)
        if XP[i, i] >= args.min_own_score and XP[i, i] > 0
        and (ref_i is None or i != ref_i)
        and (works is None or works[i])
    ]
    keep_specs = _parse_keep_specs(args.keep)
    kind_caps = dict(keep_specs)
    vendi_curve = greedy_vendi_curve(
        XP, vendi_candidates, max_k=args.max_partners, min_gain=args.min_vendi_gain,
        max_off_diag=args.max_off_diag, kinds=kinds,
        kind_caps=kind_caps if args.keep_by == "vendi" and kind_caps else None,
    )
    if args.keep_by == "vendi":
        kept = sorted(vendi_curve["recommended_indices"])
    else:
        kept = list(range(P))
        for kind, n in keep_specs:
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
    if not kept:
        raise SystemExit(
            f"no partners remained after selection (keep_by={args.keep_by!r}, min_own_score={args.min_own_score})")
    dropped = [i for i in range(P) if i not in set(kept)]
    comps = components(kept, sim, args.tau)
    groups = [[labels[i] for i in g] for g in comps]
    dist_rows, non_distinct = distinctness_rows(kept, XP, labels, kinds, args.distinct_threshold)
    div_full = diversity_summary(XP, labels, kinds, xp_std=XP_std, episodes=ci_episodes, visits=visits)
    div_kept = diversity_summary(XP, labels, kinds, indices=kept, xp_std=XP_std, episodes=ci_episodes, visits=visits)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "xp_matrix.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["best_response"] + labels)
        for i in range(P): w.writerow([labels[i]] + [f"{XP[i, j]:.3f}" for j in range(P)])
    (out / "selection.json").write_text(json.dumps(
        dict(tau=args.tau, kept=kept, dropped=dropped, groups=groups, distinct=len(comps)), indent=2) + "\n")
    (out / "diversity.json").write_text(json.dumps(
        dict(
            keep_by=args.keep_by,
            full=div_full,
            kept=div_kept,
            vendi_curve={**vendi_curve, "recommended_labels": [labels[i] for i in vendi_curve["recommended_indices"]]},
            specialist_health=health,
        ), indent=2) + "\n")
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
        for row in learn_rows:
            w.writerow(row)
    out_bank_path.write_text(json.dumps(
        {**manifest, "num_partners": len(kept),
         "partners": [{**manifest["partners"][i], "partner_id": n} for n, i in enumerate(kept)]}, indent=2) + "\n")
    load_partner_bank(args.out_bank, manifest["layout"])
    print(dict(Counter(kinds[i] for i in kept)), f"distinct={len(comps)}",
          f"vendi_kept={div_kept['vendi_resp']:.2f}/{div_kept['n_valid']}",
          f"vendi_full={div_full['vendi_resp']:.2f}/{div_full['n_valid']}",
          f"recommended_k={vendi_curve['recommended_k']}",
          f"specialist_defects={health['n_defects']}")
    for g in groups:
        if len(g) > 1: print(g)
    by_kind = Counter(r["kind"] for r in dist_rows if r["distinct"] == "true")
    print(f"distinct partners by kind: {dict(by_kind)}")
    for label, closest, why in non_distinct:
        print(f"not distinct: {label} (closest {closest}; {why})")

    if args.events:
        candidates = [i for i in range(P) if works[i] and i != ref_i]
        best = XP.max(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = XP / np.where(best == 0, np.nan, best)
        chosen = set(largest_all_different(ratio, candidates, args.different))
        with open(out / "working_set.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label", "kind", "own_score", "events_per_episode",
                                              "works", "reason", "in_set"])
            w.writeheader()
            for i in range(P):
                w.writerow(dict(
                    label=labels[i], kind=kinds[i], own_score=float(XP[i, i]),
                    events_per_episode=float(event_sums[i]),
                    works="true" if works[i] else "false",
                    reason=reasons[i],
                    in_set="true" if i in chosen else "false"))
        print(f"reference_score={ref_score} working_set_size={len(chosen)}")


if __name__ == "__main__":
    main()
