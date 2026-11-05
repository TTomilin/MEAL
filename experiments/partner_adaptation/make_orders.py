"""Write training partner orders from a bank manifest."""
import argparse, json
from pathlib import Path
import numpy as np

R_SEEDS = {"r1": 1, "r2": 2, "r3": 3}
REVISIT_EVERY = 3


def revisit_order(r1):
    out = []
    for i, p in enumerate(r1):
        out.append(int(p))
        if (i + 1) % REVISIT_EVERY == 0:
            out.append(out[-REVISIT_EVERY])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    N = json.loads(Path(args.bank).read_text())["num_partners"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    orders = {"bank": list(range(N))}
    for name, seed in R_SEEDS.items():
        orders[name] = np.random.default_rng(seed).permutation(N).tolist()
    orders["revisit"] = revisit_order(orders["r1"])
    for name, seq in orders.items():
        (out / f"{name}.txt").write_text(" ".join(map(str, seq)) + "\n")
    meta = {**orders, "seeds": dict(R_SEEDS), "revisit_from": "r1", "revisit_every": REVISIT_EVERY}
    (out / "orders.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
