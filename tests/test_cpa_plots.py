"""CPU-only tests of the CSV mode of the CPA plotting scripts (experiments/results/plotting).

Run directories are tiny synthetic ones written to tmp_path (same file formats as run_br.py), so selection,
validation and aggregation are exercised on known numbers: irregular env_steps, current/full separation, duplicates,
incomplete events, condition metadata. The six real completed runs are only used for a cross-check when they exist
locally (skipped otherwise). Nothing here trains, contacts W&B or writes below a run directory.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.results.plotting import cumulative_eval_partners as cumulative
from experiments.results.plotting import eval_per_task_partners as per_partner
from experiments.results.plotting.utils import cpa_csv
from experiments.results.plotting.utils.cpa_csv import CpaCsvError

REPO = Path(__file__).resolve().parent.parent
STAGE = 1000
P, E = 3, 2
STEPS = [0, 1000, 2500, 3000]          # irregular on purpose: the measured values must be used as recorded
HEADER = ["event_id", "row_id", "scope", "env_steps", "stage", "update", "train_episodes", "eval_partner_id",
          "eval_partner", "eval_episode", "episode_key", "soups", "ego_return", "completed", "attempt"]


def soups(t, p, e):
    return 10 * t + p + 0.5 * e         # mean over the E=2 episodes: 10 t + p + 0.25


def make_run(root, name, layout="cramped_room", cl_method="ft", importance="online", task_id=True, multihead=True,
             partners=P, state="complete", stages=None, seed=0, mutate=None, extra_columns=False):
    d = Path(root) / name
    d.mkdir(parents=True)
    bank = [({"kind": "brdiv", "label": f"{layout}/bundled_seed0/m{i}"} if i < 2 else
             {"kind": "planner", "label": f"{layout}/P01", "planner_id": "P01"}) | {"partner_id": i}
            for i in range(partners)]
    config = {"layout_name": layout, "layout_name_resolved": layout, "cl_method": cl_method,
              "importance_mode": importance, "seed": seed, "use_task_id": task_id, "use_multihead": multihead,
              "num_heads": 1 if not multihead else partners}
    run = {"budget": {"env_steps_per_stage": STAGE, "num_steps": 400, "eval_episodes_per_partner": E,
                      "planned_full_evaluations": len(STEPS)}, "partners": [b["label"] for b in bank]}
    status = {"state": state, "stages_completed": partners if stages is None else stages}
    for fname, obj in (("config.json", config), ("run.json", run), ("status.json", status),
                       ("partner_bank.json", bank)):
        (d / fname).write_text(json.dumps(obj))
    rows = []
    for t, steps in enumerate(STEPS):
        for p in range(partners):
            for e in range(E):
                rows.append({"event_id": f"ev{t}", "row_id": f"ev{t}-p{p}-e{e}", "scope": "full", "env_steps": steps,
                             "eval_partner_id": p, "eval_episode": e, "soups": soups(t, p, e), "completed": 1,
                             "attempt": 0})
        # frequent current-only events at other x values with large values that must never enter the means
        rows.append({"event_id": f"cur{t}", "row_id": f"cur{t}-p0-e0", "scope": "current", "env_steps": steps + 100,
                     "eval_partner_id": 0, "eval_episode": 0, "soups": 999, "completed": 1, "attempt": 0})
    if mutate:
        rows = mutate(rows)
    cols = HEADER + (["ego_soups", "partner_soups", "stay_fraction"] if extra_columns else [])
    with open(d / "eval_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    return d


def events_of(path):
    return cpa_csv.load_full_bank_events(cpa_csv.load_run(path))


# ----------------------------------------------------------------------------------------------- aggregation

def test_full_events_use_measured_steps_and_equal_weight_partner_mean(tmp_path):
    ev = events_of(make_run(tmp_path, "a"))
    assert ev.env_steps.tolist() == STEPS and ev.env_steps.dtype == np.int64
    expected = np.array([[np.mean([soups(t, p, e) for e in range(E)]) for p in range(P)] for t in range(len(STEPS))])
    np.testing.assert_allclose(ev.partner_means, expected)
    np.testing.assert_allclose(ev.mean, expected.mean(axis=1))
    assert 999 not in ev.partner_means          # current-scope rows are not mixed into the full-bank average


def test_rows_in_any_order_are_sorted_by_env_steps(tmp_path):
    ev = events_of(make_run(tmp_path, "a", mutate=lambda rows: rows[::-1]))
    assert ev.env_steps.tolist() == STEPS and ev.event_ids == ["ev0", "ev1", "ev2", "ev3"]


def test_diagnostic_columns_are_not_needed_or_used(tmp_path):
    plain, extra = events_of(make_run(tmp_path, "a")), events_of(make_run(tmp_path, "b", extra_columns=True))
    np.testing.assert_array_equal(plain.partner_means, extra.partner_means)


def test_identical_duplicate_rows_count_once(tmp_path):
    ev = events_of(make_run(tmp_path, "a", mutate=lambda rows: rows + [dict(rows[0]), dict(rows[3])]))
    np.testing.assert_allclose(ev.partner_means[0, 0], 0.25)


def test_conflicting_duplicate_rows_are_an_error(tmp_path):
    def conflict(rows):
        return rows + [dict(rows[0], soups=77, attempt=1)]
    with pytest.raises(CpaCsvError, match="twice with different measurements"):
        events_of(make_run(tmp_path, "a", mutate=conflict))


def test_same_episode_under_two_row_ids_is_an_error(tmp_path):
    with pytest.raises(CpaCsvError, match="same episode"):
        events_of(make_run(tmp_path, "a", mutate=lambda rows: rows + [dict(rows[0], row_id="other")]))


def test_incomplete_episode_makes_the_event_an_error_not_a_smaller_average(tmp_path):
    def drop_completion(rows):
        rows[1]["completed"] = 0
        return rows
    with pytest.raises(CpaCsvError, match="completed episodes"):
        events_of(make_run(tmp_path, "a", mutate=drop_completion))


def test_missing_partner_and_unequal_episode_counts_are_errors(tmp_path):
    with pytest.raises(CpaCsvError, match="partners"):
        events_of(make_run(tmp_path, "a", mutate=lambda rows: [r for r in rows if r["eval_partner_id"] != 2]))
    with pytest.raises(CpaCsvError, match="completed episodes"):
        events_of(make_run(tmp_path, "b", mutate=lambda rows: [
            r for r in rows if not (r["event_id"] == "ev2" and r["eval_partner_id"] == 1 and r["eval_episode"] == 1)]))


def test_event_count_must_match_the_plan_and_event_steps_must_be_unique(tmp_path):
    with pytest.raises(CpaCsvError, match="planned"):
        events_of(make_run(tmp_path, "a", mutate=lambda rows: [r for r in rows if r["event_id"] != "ev3"]))
    with pytest.raises(CpaCsvError, match="same env_steps"):
        events_of(make_run(tmp_path, "b", mutate=lambda rows: [
            dict(r, env_steps=1000) if r["event_id"] == "ev2" else r for r in rows]))


def test_a_run_without_full_rows_is_an_error(tmp_path):
    with pytest.raises(CpaCsvError, match="no full-bank"):
        events_of(make_run(tmp_path, "a", mutate=lambda rows: [r for r in rows if r["scope"] != "full"]))


# ----------------------------------------------------------------------------------------------- selection

def select(root, method="FT", layout="cramped_room", identity="available", arch="multihead", partners=P, **kw):
    return cpa_csv.select_run(root, method, layout, 0, identity, arch, partners, **kw)


def test_selection_uses_saved_metadata_not_folder_names(tmp_path):
    make_run(tmp_path, "ft_online_ewc_hidden_singlehead", cl_method="ft")          # misleading name
    ewc = make_run(tmp_path, "x1", cl_method="ewc", importance="online")
    mas = make_run(tmp_path, "x2", cl_method="mas", importance="online")
    make_run(tmp_path, "x3", cl_method="ewc", importance="offline")                # not Online_EWC
    assert select(tmp_path, "FT").path.name == "ft_online_ewc_hidden_singlehead"
    assert select(tmp_path, "Online_EWC").path == ewc
    assert select(tmp_path, "Online_MAS").path == mas


def test_selection_by_layout_identity_head_and_partner_count(tmp_path):
    want = make_run(tmp_path, "w", layout="coord_ring", task_id=False, multihead=False, partners=P)
    make_run(tmp_path, "o1", layout="cramped_room", task_id=False, multihead=False)
    make_run(tmp_path, "o2", layout="coord_ring", task_id=True, multihead=False)
    make_run(tmp_path, "o3", layout="coord_ring", task_id=False, multihead=True)
    make_run(tmp_path, "o4", layout="coord_ring", task_id=False, multihead=False, partners=P + 1)
    # num_heads in the metadata is 1 for the single-head run and P for the multi-head run; only the flag counts
    assert select(tmp_path, layout="coord_ring", identity="hidden", arch="singlehead").path == want


def test_missing_condition_error_lists_what_exists(tmp_path):
    make_run(tmp_path, "a", task_id=True)
    with pytest.raises(CpaCsvError, match="no run matches.*identity hidden") as e:
        select(tmp_path, identity="hidden")
    assert "use_task_id=True" in str(e.value)
    with pytest.raises(CpaCsvError, match="no run matches"):
        select(tmp_path, "Online_MAS")                                              # missing method is an error


def test_incomplete_runs_are_not_selected(tmp_path):
    make_run(tmp_path, "running", state="running", stages=1)
    with pytest.raises(CpaCsvError, match="not complete"):
        select(tmp_path)
    done = make_run(tmp_path, "done")
    assert select(tmp_path).path == done


def test_several_completed_matches_require_run_dirs(tmp_path):
    a, b = make_run(tmp_path, "a"), make_run(tmp_path, "b")
    with pytest.raises(CpaCsvError, match="2 completed runs.*--run_dirs") as e:
        select(tmp_path)
    assert str(a) in str(e.value) and str(b) in str(e.value)
    assert select(tmp_path, run_dirs=[str(b)]).path == b


def test_run_without_boolean_architecture_flags_is_unusable(tmp_path):
    d = make_run(tmp_path, "a")
    config = json.loads((d / "config.json").read_text())
    del config["use_multihead"]
    (d / "config.json").write_text(json.dumps(config))
    with pytest.raises(CpaCsvError, match="use_multihead"):
        select(tmp_path, run_dirs=[str(d)])
    with pytest.raises(CpaCsvError, match="(?s)no run matches.*unreadable"):
        select(tmp_path)


def test_input_files_are_not_modified(tmp_path):
    d = make_run(tmp_path, "a")
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    select(tmp_path)
    events_of(d)
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


# ----------------------------------------------------------------------------------------------- helpers

def test_boundaries_come_from_the_budget_and_labels_distinguish_position_from_planner_id(tmp_path):
    run = cpa_csv.load_run(make_run(tmp_path, "a"))
    assert cpa_csv.stage_boundaries(run) == [0, 1000, 2000, 3000]
    assert cpa_csv.partner_labels(run) == ["1 · BRDiv m0", "2 · BRDiv m1", "3 · planner P01"]
    assert cpa_csv.partner_groups(run) == [("brdiv", 0, 1), ("planner", 2, 2)]


def test_smoothing_is_off_by_default_and_changes_only_the_drawn_line():
    y = np.array([0.0, 4.0, 0.0, 4.0])
    np.testing.assert_array_equal(cpa_csv.smooth_line(y, 0), y)
    assert not np.allclose(cpa_csv.smooth_line(y, 1.0), y)


# ----------------------------------------------------------------------------------------------- scripts

def run_script(module, monkeypatch, argv, capture=None):
    """Run `module.plot()` with argv; with `capture` the figures that were saved are collected there."""
    saved = []
    monkeypatch.setattr(module, "save_plot", lambda fig, out_dir, stem, formats: (
        saved.append((fig, Path(out_dir), stem, formats)), [fig.savefig(Path(out_dir) / f"{stem}.{f}") for f in formats]))
    monkeypatch.setattr(sys, "argv", ["prog"] + argv)
    module.plot()
    return saved


def csv_argv(root, out, layout_flag, methods=("FT",), extra=()):
    return ["--run_root", str(root), layout_flag, "cramped_room", "--methods", *methods, "--identity", "available",
            "--arch", "multihead", "--num_partners", str(P), "--output_dir", str(out), "--no_show", "--formats", "png", *extra]


def test_average_figure_plots_the_all_partner_mean_of_the_measured_events(tmp_path, monkeypatch):
    d = make_run(tmp_path / "runs", "a")
    (fig, out_dir, stem, formats), = run_script(cumulative, monkeypatch, csv_argv(tmp_path / "runs", tmp_path / "o",
                                                                                   "--layout_names"))
    ev = events_of(d)
    line, markers = fig.axes[0].lines[0], fig.axes[0].lines[1]
    np.testing.assert_array_equal(markers.get_xdata(), STEPS)
    np.testing.assert_allclose(markers.get_ydata(), ev.mean)
    np.testing.assert_allclose(line.get_ydata(), ev.mean)                       # no smoothing by default
    assert (out_dir / f"{stem}.png").exists() and "cramped_room" in stem and "seed0" in stem
    title = fig.get_suptitle()
    assert all(s in title for s in ("cramped_room", "FT", "available", "multi-head", "seed 0", f"{P} partners"))
    plt.close(fig)


def test_per_partner_curves_feed_the_average_and_mark_training_intervals(tmp_path, monkeypatch):
    d = make_run(tmp_path / "runs", "a")
    (fig, out_dir, stem, formats), = run_script(per_partner, monkeypatch, csv_argv(tmp_path / "runs", tmp_path / "o",
                                                                                    "--layout_name"))
    ev, ax = events_of(d), fig.axes[0]
    markers = [l for l in ax.lines if l.get_linestyle() == "None"]
    assert len(markers) == P
    for p, m in enumerate(markers):
        np.testing.assert_array_equal(m.get_xdata(), STEPS)
        np.testing.assert_allclose(m.get_ydata(), ev.partner_means[:, p])
    np.testing.assert_allclose(np.mean([m.get_ydata() for m in markers], axis=0), ev.mean)
    thick = [l for l in ax.lines if l.get_linewidth() > 3]                      # one thick segment per partner
    assert len(thick) == P
    for p, seg in enumerate(thick):
        lo, hi = p * STAGE, (p + 1) * STAGE
        assert all(lo <= x <= hi for x in seg.get_xdata())
    assert len(ax.get_legend().get_texts()) == P
    plt.close(fig)


def test_one_per_partner_figure_per_method_and_pdf_on_request(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    make_run(root, "a")
    make_run(root, "b", cl_method="ewc")
    saved = run_script(per_partner, monkeypatch, csv_argv(root, tmp_path / "o", "--layout_name", ("FT", "Online_EWC"))[:-2]
                       + ["--formats", "png", "pdf"])
    assert "_FT_" in saved[0][2] and "_Online_EWC_" in saved[1][2] and len(saved) == 2
    assert all(s[3] == ["png", "pdf"] for s in saved)
    assert sorted(p.suffix for p in (tmp_path / "o").iterdir()) == [".pdf", ".pdf", ".png", ".png"]


def test_csv_mode_rejects_ambiguous_requests(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    make_run(root, "a")
    base = ["--run_root", str(root), "--no_show", "--output_dir", str(tmp_path / "o")]
    for argv, msg in [
        (base + ["--layout_names", "cramped_room", "coord_ring"], "exactly one layout"),
        (base, "exactly one layout"),
        (base + ["--layout_names", "cramped_room", "--seeds", "0", "1"], "one seed"),
        (base + ["--layout_names", "cramped_room", "--compare_by", "arch"], "methods only"),
        (base + ["--layout_names", "cramped_room", "--methods", "EWC"], "supports the methods"),
    ]:
        with pytest.raises(CpaCsvError, match=msg):
            run_script(cumulative, monkeypatch, argv)
    assert not (tmp_path / "o").exists() or not list((tmp_path / "o").iterdir())


# ----------------------------------------------------------------------------------------------- legacy mode

def test_legacy_defaults_and_flags_are_unchanged(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--methods", "FT"])
    args = cumulative._parse_args()
    assert (args.run_root, args.layout_names, args.num_partners, args.seeds, args.sigma, args.no_show) == (
        None, cumulative.DEFAULT_LAYOUTS, 8, [1, 2, 3, 4, 5], 1.5, False)
    monkeypatch.setattr(sys, "argv", ["prog", "--methods", "FT"])
    args = per_partner.parse_args()
    assert (args.layout_name, args.num_partners, args.seeds, args.sigma, args.algo) == (
        "cramped_room", 8, [1, 2, 3, 4, 5], 1.5, "ippo")


@pytest.fixture
def legacy_data(tmp_path):
    for seed in (1, 2):
        d = tmp_path / "data/ppo/FT/cramped_room/multihead/partners_2" / f"seed_{seed}"
        d.mkdir(parents=True)
        for i in range(2):
            (d / f"eval_partner_{i}_soup.json").write_text(json.dumps([float(seed + i + k) for k in range(6)]))
    return tmp_path / "data"


def test_legacy_json_mode_still_plots_both_scripts(legacy_data, tmp_path, monkeypatch):
    common = ["--data_root", str(legacy_data), "--methods", "FT", "--num_partners", "2", "--seeds", "1", "2",
              "--no_show", "--output_dir", str(tmp_path / "o"), "--formats", "png"]
    (fig, out_dir, stem, _), = run_script(cumulative, monkeypatch,
                                           common + ["--layout_names", "cramped_room", "--sigma", "1e-6"])
    # the legacy average: per-seed mean over the partners, then mean over the seeds
    expected = np.mean([[np.mean([s + i + k for i in range(2)]) for k in range(6)] for s in (1, 2)], axis=0)
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), expected)
    assert stem == "ppo_cumulative_methods_cramped_room_multihead_partners2"
    assert (out_dir / f"{stem}.png").exists()
    plt.close(fig)
    (fig, out_dir, stem, _), = run_script(per_partner, monkeypatch,
                                           common + ["--layout_name", "cramped_room", "--algo", "ppo"])
    assert stem == "partner_per_task_score_FT" and (out_dir / f"{stem}.png").exists()
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- environment

def test_csv_path_imports_neither_jax_nor_wandb():
    code = ("import sys; import experiments.results.plotting.cumulative_eval_partners, "
            "experiments.results.plotting.eval_per_task_partners, experiments.results.plotting.utils.cpa_csv; "
            "bad = [m for m in ('jax', 'flax', 'wandb') if m in sys.modules]; assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


REAL_ROOT = REPO / "cpa_results" / "runs"
REAL_FINAL = {("coord_ring", "FT"): 4.516667, ("coord_ring", "Online_EWC"): 7.716667,
              ("coord_ring", "Online_MAS"): 7.708333, ("cramped_room", "FT"): 4.491667,
              ("cramped_room", "Online_EWC"): 8.716667, ("cramped_room", "Online_MAS"): 9.141667}


@pytest.mark.skipif(not REAL_ROOT.is_dir(), reason="the completed local runs are not available")
@pytest.mark.parametrize("layout,method", sorted(REAL_FINAL))
def test_real_runs_reproduce_the_final_average_and_the_recorded_x_axis(layout, method):
    run = cpa_csv.select_run(REAL_ROOT, method, layout, 0, "available", "multihead", 24)
    ev = cpa_csv.load_full_bank_events(run)
    assert len(ev.env_steps) == 25 and ev.env_steps[0] == 0 and ev.env_steps[-1] == 2_398_617_600
    assert ev.env_steps.tolist() == cpa_csv.stage_boundaries(run)
    assert round(float(ev.mean[-1]), 6) == REAL_FINAL[(layout, method)]
