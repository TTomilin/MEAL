"""Partner-diversity plots from specialist cross-play + bank selection (CPU only)."""
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from experiments.partner_adaptation.partner_bank import write_bank_v2
from experiments.results.plotting import partner_diversity as pdiversity

LAYOUT = "coord_ring"
N_ACTIONS = 6
HEIGHT, WIDTH, N_POTS = 5, 5, 2


def fake_population(root: Path, kind: str, name: str, n: int) -> Path:
    d = root / name
    d.mkdir(parents=True)
    members = []
    for i in range(n):
        ckpt = d / f"m{i}.pt"
        with open(ckpt, "wb") as f:
            pickle.dump({"actor_params": {"w": np.array(
                [float(i + (0 if kind == "jaxaht" else 10))], dtype=np.float32)}}, f)
        members.append({"member_index": i, "file": f"m{i}.pt", "performance_bounds": {}})
    gen = {"status": "complete", "kind": kind, "layout_name": LAYOUT, "actor_type": "mlp",
           "partner_pop_size": 1, "num_members": n, "activation": "tanh", "test_mode": False,
           "members": members, "source": "test"}
    path = d / "generation.json"
    path.write_text(json.dumps(gen))
    return path


def write_bank(root: Path):
    """4 learned + 2 planners (bank order = learned then planner)."""
    ja = fake_population(root / "pops", "jaxaht", "ja", 3)
    co = fake_population(root / "pops", "concord", "co", 1)
    partners = [
        dict(kind="jaxaht", population="ja", member_index=0, checkpoint=ja.parent / "m0.pt"),
        dict(kind="jaxaht", population="ja", member_index=1, checkpoint=ja.parent / "m1.pt"),
        dict(kind="jaxaht", population="ja", member_index=2, checkpoint=ja.parent / "m2.pt"),
        dict(kind="concord", population="co", member_index=0, checkpoint=co.parent / "m0.pt"),
        dict(kind="planner", planner_id="P01"),
        dict(kind="planner", planner_id="P02"),
    ]
    bank_path = root / "bank.json"
    write_bank_v2(str(bank_path), LAYOUT, partners, [], {"ja": ja, "co": co}, name="main")
    labels = [f"{LAYOUT}/ja/m0", f"{LAYOUT}/ja/m1", f"{LAYOUT}/ja/m2",
              f"{LAYOUT}/co/m0", f"{LAYOUT}/P01", f"{LAYOUT}/P02"]
    kinds = ["jaxaht", "jaxaht", "jaxaht", "concord", "planner", "planner"]
    return bank_path, labels, kinds


def write_matrix(path: Path, labels, xp):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["specialist"] + list(labels))
        for i, lab in enumerate(labels):
            w.writerow([lab] + [f"{xp[i, j]:.6g}" for j in range(len(labels))])


def write_behaviour(path: Path, labels, kinds, rng):
    p = len(labels)
    n_cells = HEIGHT * WIDTH
    action_counts = rng.integers(1, 20, size=(p, p, N_ACTIONS)).astype(np.float64)
    visits = rng.integers(0, 10, size=(p, p, n_cells)).astype(np.float64)
    visits[..., 0] += 1
    pot_interactions = rng.integers(0, 5, size=(p, p, N_POTS)).astype(np.float64)
    np.savez(path, labels=np.asarray(labels), kinds=np.asarray(kinds),
             action_counts=action_counts, visits=visits, pot_interactions=pot_interactions,
             episodes=np.asarray(2))


def write_distinctness(path: Path, labels, kinds):
    # Partner 0 fails via specialist 1; everyone else distinct.
    rows = []
    for j, (lab, kind) in enumerate(zip(labels, kinds)):
        if j == 0:
            rows.append(dict(label=lab, kind=kind, own_score="10", closest_specialist=labels[1],
                             ratio="0.85", distinct="false"))
        else:
            rows.append(dict(label=lab, kind=kind, own_score="10", closest_specialist=labels[0],
                             ratio="0.4", distinct="true"))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "kind", "own_score", "closest_specialist", "ratio", "distinct"])
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    bank, labels, kinds = write_bank(tmp_path)
    p = len(labels)
    diag = np.array([10.0, 9.0, 8.0, 7.0, 12.0, 11.0])
    xp = np.full((p, p), 2.0)
    np.fill_diagonal(xp, diag)
    xp[1, 0] = 8.5  # closest for partner 0
    matrix = tmp_path / "xp_matrix.csv"
    write_matrix(matrix, labels, xp)
    behaviour = tmp_path / "partner_behaviour.npz"
    write_behaviour(behaviour, labels, kinds, np.random.default_rng(0))
    distinctness = tmp_path / "distinctness.csv"
    write_distinctness(distinctness, labels, kinds)
    return dict(bank=bank, labels=labels, kinds=kinds, matrix=matrix, behaviour=behaviour,
                distinctness=distinctness, xp=xp)


def test_relative_xp_diagonal_is_one(inputs):
    rel = pdiversity.relative_xp(inputs["xp"])
    np.testing.assert_allclose(np.diag(rel), 1.0)


def test_perplexity_below_partner_count(inputs):
    n = len(inputs["labels"])
    perp = pdiversity.tsne_perplexity(n)
    assert perp < n
    assert perp == min(pdiversity.PERPLEXITY_MAX, n - 1)


def test_figures_and_tsne_csv_written(inputs, tmp_path):
    out = tmp_path / "figs"
    pdiversity.main([
        "--matrix", str(inputs["matrix"]),
        "--behaviour", str(inputs["behaviour"]),
        "--bank", str(inputs["bank"]),
        "--distinctness", str(inputs["distinctness"]),
        "--layout_name", LAYOUT,
        "--output_dir", str(out),
        "--formats", "png",
        "--no_show",
        "--seed", "0",
    ])
    assert (out / f"{LAYOUT}_crossplay_heatmap.png").is_file()
    assert (out / f"{LAYOUT}_behaviour_tsne.png").is_file()
    csv_path = out / f"{LAYOUT}_behaviour_tsne.csv"
    assert csv_path.is_file()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["label"] for r in rows] == inputs["labels"]
    assert all("x" in r and "y" in r for r in rows)
