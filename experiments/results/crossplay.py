"""Turn single-partner best-response runs into a cross-play matrix and select a bank."""
import argparse, csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.partner_adaptation.partner_bank import load_partner_bank
from experiments.results.cpa_eval import read_csv, resolve_run_dir


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--keep", action="append", default=[], metavar="KIND=N")
    ap.add_argument("--min-own-score", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-bank", required=True)
    args = ap.parse_args(argv)
    manifest = json.loads(Path(args.bank).read_text())
    bank = load_partner_bank(args.bank, manifest["layout"])
    labels, kinds, P = [r.label for r in bank.records], [r.kind for r in bank.records], len(bank)
    by = {}
    for d in args.runs:
        i, row = xp_row(d, P)
        if i in by: raise SystemExit(f"partner {i} has two runs")
        by[i] = row
    for i in range(P):
        if i not in by: raise SystemExit(f"partner {i} has no run")
    XP = np.stack([by[i] for i in range(P)])
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = XP / np.where(np.diag(XP) == 0, np.nan, np.diag(XP))
    sim = np.minimum(rel, rel.T); np.fill_diagonal(sim, np.nan)
    for j in range(P):
        if np.nanargmax(XP[:, j]) != j: print(f"warning: {labels[j]}: its best response is weak")
    kept = list(range(P))
    for item in args.keep:
        kind, _, n = item.partition("="); n = int(n)
        while sum(kinds[i] == kind for i in kept) > n:
            low = [i for i in kept if kinds[i] == kind and XP[i, i] < args.min_own_score]
            if not low: break
            kept.remove(min(low, key=lambda i: XP[i, i]))
        while sum(kinds[i] == kind for i in kept) > n:
            cands = [i for i in kept if kinds[i] == kind]
            def max_sim(i):
                others = [j for j in kept if j != i]
                vals = sim[i, others]
                if not others or np.all(np.isnan(vals)): return np.inf
                return float(np.nanmax(vals))
            kept.remove(max(cands, key=lambda i: (max_sim(i), -XP[i, i])))
    for i in kept:
        if XP[i, i] < args.min_own_score:
            print(f"warning: {labels[i]}: own score below min-own-score ({XP[i, i]:.3f})")
    dropped = [i for i in range(P) if i not in set(kept)]
    comps = components(kept, sim, args.tau)
    groups = [[labels[i] for i in g] for g in comps]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "xp_matrix.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["best_response"] + labels)
        for i in range(P): w.writerow([labels[i]] + [f"{XP[i, j]:.3f}" for j in range(P)])
    (out / "selection.json").write_text(json.dumps(
        dict(tau=args.tau, kept=kept, dropped=dropped, groups=groups, distinct=len(comps)), indent=2) + "\n")
    Path(args.out_bank).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_bank).write_text(json.dumps(
        {**manifest, "num_partners": len(kept),
         "partners": [{**manifest["partners"][i], "partner_id": n} for n, i in enumerate(kept)]}, indent=2) + "\n")
    load_partner_bank(args.out_bank, manifest["layout"])
    print(dict(Counter(kinds[i] for i in kept)), f"distinct={len(comps)}")
    for g in groups:
        if len(g) > 1: print(g)


if __name__ == "__main__":
    main()
