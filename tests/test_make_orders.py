"""Training order generation from a bank manifest."""
import json
from pathlib import Path

from experiments.partner_adaptation.make_orders import main

N = 24


def write_bank(path: Path) -> Path:
    path.write_text(json.dumps({"num_partners": N}))
    return path


def read_order(path: Path) -> list[int]:
    return list(map(int, path.read_text().split()))


def test_orders_n24(tmp_path):
    bank = write_bank(tmp_path / "bank.json")
    out = tmp_path / "orders"
    main(["--bank", str(bank), "--out", str(out)])
    bank_ord = read_order(out / "bank.txt")
    r1, r2, r3 = (read_order(out / f"r{i}.txt") for i in (1, 2, 3))
    revisit = read_order(out / "revisit.txt")
    assert bank_ord == list(range(N))
    for r in (r1, r2, r3):
        assert sorted(r) == list(range(N))
    assert r1 != r2 != r3 and r1 != r3
    assert len(revisit) == 32
    assert set(revisit) == set(range(N))
    assert revisit[3] == revisit[0] and revisit[7] == revisit[4] and revisit[11] == revisit[8]
    meta = json.loads((out / "orders.json").read_text())
    assert meta["seeds"] == {"r1": 1, "r2": 2, "r3": 3}
    assert meta["revisit_from"] == "r1" and meta["revisit_every"] == 3
    assert meta["r1"] == r1 and meta["revisit"] == revisit


def test_deterministic(tmp_path):
    bank = write_bank(tmp_path / "bank.json")
    out1, out2 = tmp_path / "a", tmp_path / "b"
    main(["--bank", str(bank), "--out", str(out1)])
    main(["--bank", str(bank), "--out", str(out2)])
    for name in ("bank", "r1", "r2", "r3", "revisit", "orders"):
        ext = ".json" if name == "orders" else ".txt"
        assert (out1 / f"{name}{ext}").read_text() == (out2 / f"{name}{ext}").read_text()
