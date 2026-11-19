"""Cross-play matrix + partner selection from single-partner BR runs (CPU only)."""
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from experiments.partner_adaptation.partner_bank import load_partner_bank, write_bank_v2
from experiments.results.crossplay import LEARNABILITY_FRACTION, largest_all_different, main

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

BASELINE_XP = (
    "best_response,coord_ring/P01,coord_ring/P02,coord_ring/P03,coord_ring/P04\n"
    "coord_ring/P01,10.000,7.200,3.600,4.200\n"
    "coord_ring/P02,9.000,8.000,3.600,4.200\n"
    "coord_ring/P03,3.000,2.400,12.000,4.200\n"
    "coord_ring/P04,3.000,2.400,3.600,14.000\n"
)
BASELINE_SELECTION = (
    "{\n"
    '  "tau": 0.8,\n'
    '  "kept": [\n'
    "    0,\n"
    "    2,\n"
    "    3\n"
    "  ],\n"
    '  "dropped": [\n'
    "    1\n"
    "  ],\n"
    '  "groups": [\n'
    "    [\n"
    '      "coord_ring/P01"\n'
    "    ],\n"
    "    [\n"
    '      "coord_ring/P03"\n'
    "    ],\n"
    "    [\n"
    '      "coord_ring/P04"\n'
    "    ]\n"
    "  ],\n"
    '  "distinct": 3\n'
    "}\n"
)
BASELINE_OUT_BANK = (
    "{\n"
    '  "format_version": 2,\n'
    '  "name": "xp_planners",\n'
    '  "layout": "coord_ring",\n'
    '  "num_partners": 3,\n'
    '  "populations": {},\n'
    '  "partners": [\n'
    "    {\n"
    '      "partner_id": 0,\n'
    '      "kind": "planner",\n'
    '      "planner_id": "P01"\n'
    "    },\n"
    "    {\n"
    '      "partner_id": 1,\n'
    '      "kind": "planner",\n'
    '      "planner_id": "P03"\n'
    "    },\n"
    "    {\n"
    '      "partner_id": 2,\n'
    '      "kind": "planner",\n'
    '      "planner_id": "P04"\n'
    "    }\n"
    "  ],\n"
    '  "heldout": []\n'
    "}\n"
)


def write_bank(path: Path) -> Path:
    partners = [{"partner_id": i, "kind": "planner", "planner_id": pid} for i, pid in enumerate(IDS)]
    path.write_text(json.dumps({
        "format_version": 2, "name": "xp_planners", "layout": LAYOUT, "num_partners": len(partners),
        "populations": {}, "partners": partners, "heldout": [],
    }))
    return path


def write_run(root: Path, i: int, row: np.ndarray, train_soups=None) -> Path:
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
    if train_soups is not None:
        write_train_metrics(d, train_soups)
    return d


def write_train_metrics(run: Path, soups):
    cols = ["env_steps", "stage", "partner_id", "partner", "update", "episodes", "soups", "ego_return"]
    with open(run / "train_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        for u, s in enumerate(soups):
            row = {"env_steps": (u + 1) * 100, "stage": 0, "partner_id": 0, "partner": "",
                   "update": u, "episodes": 1, "soups": s, "ego_return": 0}
            w.writerow(row)


def fake_population(root: Path, kind: str, name: str, n: int) -> Path:
    d = root / name
    d.mkdir(parents=True)
    members = []
    for i in range(n):
        ckpt = d / f"m{i}.pt"
        with open(ckpt, "wb") as f:
            pickle.dump({"actor_params": {"w": np.array([float(i + (0 if kind == "jaxaht" else 10))],
                                                        dtype=np.float32)}}, f)
        members.append({"member_index": i, "file": f"m{i}.pt", "performance_bounds": {}})
    gen = {"status": "complete", "kind": kind, "layout_name": LAYOUT, "actor_type": "mlp",
           "partner_pop_size": 1, "num_members": n, "activation": "tanh", "test_mode": False,
           "members": members, "source": "test"}
    path = d / "generation.json"
    path.write_text(json.dumps(gen))
    return path


def write_mixed_bank(root: Path) -> tuple[Path, list[str], list[str]]:
    """3 learned (2 jaxaht + 1 concord) and 2 planners."""
    ja = fake_population(root / "pops", "jaxaht", "ja", 2)
    co = fake_population(root / "pops", "concord", "co", 1)
    partners = [
        dict(kind="jaxaht", population="ja", member_index=0, checkpoint=ja.parent / "m0.pt"),
        dict(kind="jaxaht", population="ja", member_index=1, checkpoint=ja.parent / "m1.pt"),
        dict(kind="concord", population="co", member_index=0, checkpoint=co.parent / "m0.pt"),
        dict(kind="planner", planner_id="P01"),
        dict(kind="planner", planner_id="P02"),
    ]
    bank_path = root / "bank.json"
    write_bank_v2(str(bank_path), LAYOUT, partners, [], {"ja": ja, "co": co}, name="mixed")
    labels = [f"{LAYOUT}/ja/m0", f"{LAYOUT}/ja/m1", f"{LAYOUT}/co/m0",
              f"{LAYOUT}/P01", f"{LAYOUT}/P02"]
    kinds = ["jaxaht", "jaxaht", "concord", "planner", "planner"]
    return bank_path, labels, kinds


def write_matrix_csv(path: Path, labels, xp, order=None):
    order = list(labels) if order is None else list(order)
    idx = {lab: i for i, lab in enumerate(labels)}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["specialist"] + order)
        for lab in order:
            i = idx[lab]
            w.writerow([lab] + [f"{xp[i, idx[c]]:.6g}" for c in order])


def write_matrix_run(root: Path, i: int, label: str, train_soups=None) -> Path:
    d = root / f"br_{i}"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"only_stages": [i]}))
    if train_soups is not None:
        write_train_metrics(d, train_soups)
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


def test_keep_drops_zero_own_score_before_duplicate(tmp_path):
    # Same duplicate pair as the default fixture, but P04 has own score 0.
    diag = np.array([10.0, 8.0, 12.0, 0.0])
    xp = REL * diag[np.newaxis, :]
    bank = write_bank(tmp_path / "bank.json")
    runs = [write_run(tmp_path / "runs", i, xp[i]) for i in range(4)]
    out, out_bank = tmp_path / "out", tmp_path / "final_bank.json"
    main(["--bank", str(bank), "--runs", *[str(r) for r in runs], "--keep", "planner=3",
          "--out", str(out), "--out-bank", str(out_bank)])
    sel = json.loads((out / "selection.json").read_text())
    assert sel["dropped"] == [3]
    assert set(sel["kept"]) == {0, 1, 2}


def test_missing_run_raises(setup):
    bank, runs, out, out_bank = setup
    with pytest.raises(SystemExit, match="partner 3 has no run"):
        main(["--bank", str(bank), "--runs", *[str(r) for r in runs[:3]],
              "--out", str(out), "--out-bank", str(out_bank)])


def test_default_path_byte_identical_legacy_outputs(setup):
    bank, runs, out, out_bank = setup
    main(["--bank", str(bank), "--runs", *[str(r) for r in runs], "--keep", "planner=3",
          "--out", str(out), "--out-bank", str(out_bank)])
    assert (out / "xp_matrix.csv").read_text() == BASELINE_XP
    assert (out / "selection.json").read_text() == BASELINE_SELECTION
    assert out_bank.read_text() == BASELINE_OUT_BANK
    assert (out / "distinctness.csv").is_file()
    assert (out / "reference_scores.json").is_file()
    assert (out / "learnability.csv").is_file()


def test_own_score_keep_learned_distinctness_learnability_and_out_bank_dir(tmp_path, capsys):
    bank, labels, kinds = write_mixed_bank(tmp_path)
    # Diagonals 10, 9, 8, 0, 6. Off-diag ratios vs kept: col0 has 0.75, col1 has 0.65.
    xp = np.array([
        [10.0, 5.85, 1.0, 0.0, 1.0],
        [7.5, 9.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 8.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 6.0],
    ], dtype=float)
    matrix = tmp_path / "xp_matrix.csv"
    # Shuffle column order so matching is by label, not position.
    write_matrix_csv(matrix, labels, xp, order=[labels[2], labels[4], labels[0], labels[1], labels[3]])

    soups = [float(s) for s in range(1, 13)]  # 12 updates, q = 3
    runs = [write_matrix_run(tmp_path / "runs", i, labels[i],
                             train_soups=soups if i == 0 else None) for i in range(5)]

    out = tmp_path / "out"
    out_bank = tmp_path / "other" / "final_bank.json"
    with pytest.raises(SystemExit, match="same directory"):
        main(["--bank", str(bank), "--runs", *[str(r) for r in runs], "--matrix", str(matrix),
              "--keep", "learned=2", "--keep-by", "own_score",
              "--out", str(out), "--out-bank", str(out_bank)])

    out_bank = tmp_path / "kept_bank.json"
    main(["--bank", str(bank), "--runs", *[str(r) for r in runs], "--matrix", str(matrix),
          "--keep", "learned=2", "--keep-by", "own_score",
          "--out", str(out), "--out-bank", str(out_bank)])
    printed = capsys.readouterr().out

    sel = json.loads((out / "selection.json").read_text())
    assert sel["dropped"] == [2]
    assert sel["kept"] == [0, 1, 3, 4]

    with open(out / "distinctness.csv") as f:
        rows = {r["label"]: r for r in csv.DictReader(f)}
    assert rows[labels[0]]["distinct"] == "false"
    assert float(rows[labels[0]]["ratio"]) == pytest.approx(0.75)
    assert rows[labels[0]]["closest_specialist"] == labels[1]
    assert rows[labels[1]]["distinct"] == "true"
    assert float(rows[labels[1]]["ratio"]) == pytest.approx(0.65)
    assert rows[labels[3]]["distinct"] == "false"
    assert rows[labels[3]]["ratio"] == ""
    assert rows[labels[3]]["own_score"] == "0" or float(rows[labels[3]]["own_score"]) == 0.0
    assert "0" in printed and ("diag" in printed.lower() or "zero" in printed.lower()
                               or "own score" in printed.lower() or labels[3] in printed)
    assert labels[0] in printed and labels[1] in printed

    ref = json.loads((out / "reference_scores.json").read_text())
    assert ref == {labels[0]: 10.0, labels[1]: 9.0, labels[3]: 0.0, labels[4]: 6.0}

    with open(out / "learnability.csv") as f:
        learn = {r["label"]: r for r in csv.DictReader(f)}
    assert set(learn) == set(labels)
    assert float(learn[labels[0]]["final_train_soups"]) == 12.0
    assert float(learn[labels[0]]["steps_to_80pct"]) == 1000.0  # soups 10 at env_steps 1000; 0.8*12=9.6
    assert float(learn[labels[0]]["last_quarter_gain"]) == pytest.approx(3.0)  # mean(10,11,12)-mean(7,8,9)
    assert LEARNABILITY_FRACTION == 0.8
    assert learn[labels[1]]["final_train_soups"] == ""
    assert f"{runs[1]}" in printed or runs[1].name in printed

    loaded = load_partner_bank(str(out_bank), LAYOUT)
    assert [r.kind for r in loaded.records] == ["jaxaht", "jaxaht", "planner", "planner"]


def test_largest_all_different_exact_max_clique():
    # Five partners; pairwise different iff both directed ratios are below 0.7.
    # Edges: 0-1, 0-2, 0-3, 1-2, 1-3, 2-3 form a K4 on {0,1,2,3}; 4 only meets 0.
    # Largest all-different set is therefore {0,1,2,3}.
    thr = 0.7
    ratio = np.ones((5, 5))
    np.fill_diagonal(ratio, 0.0)
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 4)]:
        ratio[i, j] = ratio[j, i] = 0.5
    assert largest_all_different(ratio, range(5), thr) == [0, 1, 2, 3]
    assert largest_all_different(ratio, [0, 4], thr) == [0, 4]
    assert largest_all_different(ratio, [1, 4], thr) == [1]
