"""Cross-play matrix + partner selection from single-partner BR runs (CPU only)."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.partner_adaptation.partner_bank import load_partner_bank
from experiments.results.crossplay import main

LAYOUT = "coord_ring"
IDS = ("P01", "P02", "P03", "P04")
# Diagonals; P01/P02 cross at rel 0.9, everything else at rel 0.3.
DIAG = np.array([10.0, 8.0, 12.0, 14.0])
REL = np.array([
    [1.0, 0.9, 0.3, 0.3],
    [0.9, 1.0, 0.3, 0.3],
    [0.3, 0.3, 1.0, 0.3],
    [0.3, 0.3, 0.3, 1.0],
])
XP = REL * DIAG[np.newaxis, :]


def write_bank(path: Path) -> Path:
    partners = [{"partner_id": i, "kind": "planner", "planner_id": pid} for i, pid in enumerate(IDS)]
    path.write_text(json.dumps({
        "format_version": 2, "name": "xp_planners", "layout": LAYOUT, "num_partners": len(partners),
        "populations": {}, "partners": partners, "heldout": [],
    }))
    return path


def write_run(root: Path, i: int, row: np.ndarray) -> Path:
    d = root / f"br_{i}"
    d.mkdir(parents=True)
    bank = [{"partner_id": j, "kind": "planner", "label": f"{LAYOUT}/{pid}", "planner_id": pid}
            for j, pid in enumerate(IDS)]
    (d / "config.json").write_text(json.dumps({"only_stages": [i]}))
    (d / "partner_bank.json").write_text(json.dumps(bank))
    cols = ["event_id", "row_id", "scope", "env_steps", "stage", "update", "train_episodes",
            "eval_partner_id", "eval_partner", "eval_episode", "episode_key", "soups", "ego_return",
            "completed", "attempt"]
    with open(d / "eval_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        for j, soups in enumerate(row):
            w.writerow({"event_id": "ev0", "row_id": f"ev0-p{j}-e0", "scope": "full", "env_steps": 1000,
                        "eval_partner_id": j, "eval_partner": bank[j]["label"], "eval_episode": 0,
                        "soups": float(soups), "completed": 1, "attempt": 0})
    return d


@pytest.fixture
def setup(tmp_path):
    bank = write_bank(tmp_path / "bank.json")
    runs = [write_run(tmp_path / "runs", i, XP[i]) for i in range(4)]
    out, out_bank = tmp_path / "out", tmp_path / "final_bank.json"
    return bank, runs, out, out_bank


def test_keep_drops_weaker_duplicate_and_writes_bank(setup):
    bank, runs, out, out_bank = setup
    main(["--bank", str(bank), "--runs", *[str(r) for r in runs], "--keep", "planner=3",
          "--out", str(out), "--out-bank", str(out_bank)])
    sel = json.loads((out / "selection.json").read_text())
    assert sel["distinct"] == 3 and len(sel["kept"]) == 3
    assert set(sel["dropped"]) <= {0, 1} and len(sel["dropped"]) == 1
    # Lower diagonal of the P01/P02 pair is P02 (id 1).
    assert sel["dropped"] == [1]
    loaded = load_partner_bank(str(out_bank), LAYOUT)
    assert len(loaded) == 3 and [r.partner_id for r in loaded.records] == [0, 1, 2]
    assert [r.planner_id for r in loaded.records] == ["P01", "P03", "P04"]
    with open(out / "xp_matrix.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "best_response" and len(rows) == 5


def test_missing_run_raises(setup):
    bank, runs, out, out_bank = setup
    with pytest.raises(SystemExit, match="partner 3 has no run"):
        main(["--bank", str(bank), "--runs", *[str(r) for r in runs[:3]],
              "--out", str(out), "--out-bank", str(out_bank)])
