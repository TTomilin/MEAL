"""CPU-only tests of the CPA pilot launcher (experiments/partner_adaptation/pilot.py and pilot_gpu.py).

Launcher parsing, dry-run plans, job reuse and command-failure handling are tested without starting a real
generation or training job; subprocess launching is replaced by a recorder. One tiny real run (4 partners, 3 updates,
16-step episodes, W&B disabled) provides run directories for the reuse and resume tests. Runtime projections are
checked on synthetic profiles: they test the arithmetic, not any real device. Nothing here says anything about GPU
behaviour.
"""
import json
import re
from pathlib import Path

import jax
import pytest

from experiments.partner_adaptation import pilot, pilot_gpu
from experiments.partner_adaptation.partner_bank import assemble_bank
from experiments.partner_adaptation.partner_generation.run import (
    MEMBER_FILE_FMT, write_generation_record)
from experiments.partner_adaptation.run_br import TrainConfig, execute
from experiments.partner_adaptation.train_ego import eval_plan

PLAN = ["--steps-per-partner", "4915200"]       # six updates of 2048 envs x 400 steps


def run_main(argv, capsys):
    code = pilot.main(argv)
    return code, capsys.readouterr()


@pytest.fixture
def launched(monkeypatch):
    """Record the commands the launcher would start; every command exits with `launched.code`."""
    calls = []

    class Recorder:
        code = 0

    def fake(command, log_path):
        calls.append(command)
        return Recorder.code

    monkeypatch.setattr(pilot, "run_logged", fake)
    Recorder.calls = calls
    return Recorder


# ---------------------------------------------------------------------------------------------------------
# parsing and dry-run plans
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", pilot.LAYOUTS)
def test_plan_for_every_layout(layout, tmp_path, capsys):
    code, out = run_main(["plan", "--root", str(tmp_path), "--layouts", layout, *PLAN], capsys)
    assert code == 0, out.err
    text = out.out
    assert f"{layout}_bank24.json" in text
    assert f"{layout}/ft__id-mh/seed0" in text and f"{layout}/online_ewc__id-mh/seed0" in text
    assert "partners 24 (12 learned + 12 planners)" in text
    assert text.count("MISSING") == 3                                   # three generation seeds
    assert "(6 updates of 819,200)" in text
    assert "reward shaping: weight 1 -> 0 linearly over 25,000,000 joint steps" in text


def test_default_selection_is_pilot_ft_ewc_seed0_identity_available(tmp_path):
    sel = pilot.Plan(root=str(tmp_path), steps_per_partner=4915200)
    jobs = pilot.make_jobs(sel, tmp_path)
    assert [j.name for j in jobs] == [
        "coord_ring/ft__id-mh/seed0", "coord_ring/online_ewc__id-mh/seed0",
        "cramped_room/ft__id-mh/seed0", "cramped_room/online_ewc__id-mh/seed0"]
    for job in jobs:
        assert job.overrides["use_task_id"] and job.overrides["use_multihead"]
        assert job.overrides["seed"] == 0 and job.overrides["eval_schedule"] == "pilot"
    ewc = jobs[1].overrides
    assert (ewc["cl_method"], ewc["importance_mode"]) == ("ewc", "online")
    assert sel.eval_episodes == 5 and sel.eval_current_every == 5 and sel.num_steps == 400


def test_mas_only_maps_to_online_mas_and_selects_no_other_work(tmp_path, capsys, monkeypatch, launched):
    monkeypatch.setattr(pilot, "require_banks", lambda jobs: None)
    code, out = run_main(["train", "--root", str(tmp_path), "--methods", "online_mas", *PLAN], capsys)
    assert code == 0, out.err
    assert "online_mas__id-mh/seed0" in out.out and "ft__" not in out.out and "online_ewc__" not in out.out
    assert "cl_method=mas" in out.out
    assert "BRDiv generation" not in out.out                          # MAS later never repeats generation
    mas = pilot.make_jobs(pilot.Plan(root=str(tmp_path), methods=("online_mas",), steps_per_partner=4915200),
                          tmp_path)[0].overrides
    assert (mas["cl_method"], mas["importance_mode"]) == ("mas", "online")
    assert launched.calls == []                                        # dry run starts nothing


def test_a_bank_that_does_not_hold_24_partners_is_rejected(tmp_path, capsys):
    path = pilot.bank_path(tmp_path, "cramped_room", 24)
    assemble_bank("cramped_room", str(path), [], bundled=True, planners=("P01",))       # 4 partners at the 24 path
    code, out = run_main(["plan", "--root", str(tmp_path), "--layouts", "cramped_room", "--methods", "ft", *PLAN],
                         capsys)
    assert code == 2 and "num_population_partners=24 but the bank declares 4" in out.err


def test_one_seed_by_default_and_several_only_when_listed(tmp_path):
    one = pilot.make_jobs(pilot.Plan(root=str(tmp_path), layouts=("coord_ring",), steps_per_partner=4915200), tmp_path)
    assert sorted({j.seed for j in one}) == [0] and len(one) == 2
    many = pilot.make_jobs(pilot.Plan(root=str(tmp_path), layouts=("coord_ring",), seeds=(0, 1),
                                      steps_per_partner=4915200), tmp_path)
    assert len(many) == 4 and sorted({j.seed for j in many}) == [0, 1]
    with pytest.raises(pilot.PilotError):
        pilot.make_jobs(pilot.Plan(root=str(tmp_path), seeds=(0, 0)), tmp_path)


def test_hidden_identity_disables_input_and_routing_independently(tmp_path):
    def flags(**kw):
        job = pilot.make_jobs(pilot.Plan(root=str(tmp_path), layouts=("coord_ring",), methods=("ft",), **kw),
                              tmp_path)[0]
        return job.overrides["use_task_id"], job.overrides["use_multihead"], job.tag, job.notes

    assert flags()[:3] == (True, True, "id-mh")
    assert flags(identity="hidden")[:3] == (False, False, "noid-1h")
    assert flags(use_task_id=False)[:3] == (False, True, "noid-mh")
    assert flags(use_multihead=False)[:3] == (True, False, "id-1h")
    assert "NOT hidden" in flags(use_task_id=False)[3][0]
    # distinct tags keep identity conditions in separate output directories
    assert len({flags()[2], flags(identity="hidden")[2], flags(use_task_id=False)[2]}) == 3


def test_run_command_carries_the_resolved_flags(tmp_path):
    job = pilot.make_jobs(pilot.Plan(root=str(tmp_path), layouts=("coord_ring",), methods=("online_ewc",),
                                     identity="hidden", steps_per_partner=4915200), tmp_path)[0]
    args = pilot.cli_args(TrainConfig, job.overrides)
    assert "--no-use-task-id" in args and "--no-use-multihead" in args
    assert args[args.index("--cl-method") + 1] == "ewc" and args[args.index("--importance-mode") + 1] == "online"
    assert args[args.index("--mode") + 1] == "online"                 # W&B online unless told otherwise
    assert args[args.index("--total-timesteps") + 1] == "4915200.0"
    TrainConfig_fields = {"--" + f.replace("_", "-") for f in TrainConfig.__dataclass_fields__}
    assert {a for a in args if a.startswith("--")} <= TrainConfig_fields | {"--no-use-task-id", "--no-use-multihead",
                                                                           "--no-anneal-lr"}


# ---------------------------------------------------------------------------------------------------------
# budgets and failures
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["train"],                                                          # no budget
    ["train", "--steps-per-partner", "0"],                              # zero updates
    ["train", "--steps-per-partner", "1000"],                           # below one update
    ["train", "--steps-per-partner", "5000000"],                        # not a whole number of updates
    ["plan", "--steps-per-partner", "4915200", "--num-minibatches", "7"],   # invalid minibatches
    ["plan", "--steps-per-partner", "4915200", "--num-steps", "100"],   # horizon is part of the experiment
    ["plan", "--layouts", "no_such_layout"],
    ["plan", "--methods", "ewc_offline"],
    ["train", "--steps-per-partner", "4915200"],                        # banks not assembled
])
def test_invalid_requests_return_nonzero(argv, tmp_path, capsys, launched):
    code, out = run_main([argv[0], "--root", str(tmp_path), *argv[1:]], capsys)
    assert code == 2 and "error" in out.err
    assert launched.calls == []


def test_unknown_action_or_option_is_a_parse_error(tmp_path):
    with pytest.raises(SystemExit) as e:
        pilot.main(["frobnicate"])
    assert e.value.code != 0
    with pytest.raises(SystemExit):
        pilot.main(["plan", "--no-such-option"])


def test_root_is_required(monkeypatch, capsys):
    monkeypatch.delenv("MEAL_CPA_ROOT", raising=False)
    code, out = run_main(["plan"], capsys)
    assert code == 2 and "--root" in out.err


def test_explicit_protocol_change_is_labelled(tmp_path, capsys):
    code, out = run_main(["plan", "--root", str(tmp_path), "--layouts", "coord_ring", "--steps-per-partner", "1024",
                          "--num-envs", "16", "--num-steps", "64", "--allow-protocol-change",
                          "--eval-episodes", "2"], capsys)
    assert code == 0, out.err
    assert "PROTOCOL CHANGE: episode horizon 64" in out.out and "PROTOCOL CHANGE: 2 evaluation episodes" in out.out


def test_unmeasured_budget_and_target_are_not_claimed(tmp_path, capsys):
    code, out = run_main(["plan", "--root", str(tmp_path), "--layouts", "coord_ring"], capsys)
    assert code == 0
    assert "budget UNSET" in out.out and "runtime projection: NONE" in out.out and "not verified" in out.out


# ---------------------------------------------------------------------------------------------------------
# generation reuse
# ---------------------------------------------------------------------------------------------------------

def make_population(root, layout="coord_ring", seed=1001, status="complete", total=2 * 8 * 4 * 2, members=True,
                    gen_mode="disabled", **extra):
    cfg = pilot.generation_config(layout, seed, root, 3, total, gen_mode, **extra)
    d = pilot.population_dir(cfg)
    d.mkdir(parents=True)
    files = [MEMBER_FILE_FMT.format(i=0, j=j) for j in range(3)]
    for f in files if members else []:
        (d / f).write_bytes(b"x")
    write_generation_record(str(d), cfg, layout, "preset", status,
                            [{"seed_index": 0, "member_index": j, "file": f} for j, f in enumerate(files)])
    return cfg, d


def test_population_state_detects_valid_conflicting_and_interrupted(tmp_path):
    cfg, d = make_population(tmp_path)
    assert pilot.population_state(cfg)[0] == "valid"
    other = pilot.generation_config("coord_ring", 1001, tmp_path, 3, 2 * 8 * 4 * 4, "disabled")
    state, detail = pilot.population_state(other)
    assert state == "conflict" and "total_timesteps" in detail
    assert pilot.population_state(pilot.generation_config("coord_ring", 1002, tmp_path, 3, None, "disabled"))[0] \
        == "missing"
    cfg2, _ = make_population(tmp_path, seed=1002, status="started")
    assert pilot.population_state(cfg2)[0] == "invalid"
    cfg3, _ = make_population(tmp_path, seed=1003, members=False)
    assert pilot.population_state(cfg3)[0] == "invalid"


def test_generate_never_regenerates_a_valid_population(tmp_path, capsys, launched):
    gen = ["--root", str(tmp_path), "--layouts", "coord_ring", "--gen-seeds", "1001", "--gen-total-timesteps",
           "64", "--gen-mode", "disabled"]
    make_population(tmp_path, total=64)
    code, out = run_main(["generate", *gen, "--run"], capsys)
    assert code == 0 and "valid population exists" in out.out and launched.calls == []
    # the same seed requested with another budget conflicts: nothing is started and nothing is touched
    code, out = run_main(["generate", *gen[:-4], "--gen-total-timesteps", "128", "--gen-mode", "disabled", "--run"],
                         capsys)
    assert code == 1 and "conflict" in out.out and launched.calls == []


def test_generate_runs_only_missing_populations_and_checks_the_result(tmp_path, capsys, launched):
    gen = ["--root", str(tmp_path), "--layouts", "coord_ring", "--gen-seeds", "1001", "1002", "--gen-mode", "disabled"]
    code, out = run_main(["generate", *gen], capsys)
    assert code == 0 and launched.calls == [] and "[dry-run]" in out.out    # preview only
    code, out = run_main(["generate", *gen, "--run"], capsys)
    # the recorder creates nothing, so completion is not accepted just because the command exited zero
    assert code == 1 and len(launched.calls) == 1 and "population missing" in out.out
    launched.calls.clear()
    launched.code = 3
    code, out = run_main(["generate", *gen, "--run", "--keep-going"], capsys)
    assert code == 1 and len(launched.calls) == 2
    assert any("partner_generation.run" in part for part in launched.calls[0])


def test_bank_fails_when_populations_are_missing(tmp_path, capsys):
    code, out = run_main(["bank", "--root", str(tmp_path), "--layouts", "coord_ring"], capsys)
    assert code == 1 and "run `generate` first" in out.out


# ---------------------------------------------------------------------------------------------------------
# reuse of completed runs, resume (one tiny real run)
# ---------------------------------------------------------------------------------------------------------

TINY = dict(mode="disabled", layout_name="cramped_room", num_heuristic_partners=0, num_envs=4, num_steps=16,
            update_epochs=1, num_minibatches=2, num_eval_episodes=2, eval_schedule="pilot", eval_current_every=2,
            hidden_size=16, num_layers=1, importance_episodes=1, importance_steps=8, reg_coef=1.0, seed=0,
            reward_shaping_horizon=1e3, cl_method="ft")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("pilot")
    bank = root / "bank.json"
    assemble_bank("cramped_room", str(bank), [], bundled=True, planners=("P01",))
    runs = root / "runs"
    overrides = dict(TINY, partner_bank=str(bank), checkpoint_path=str(runs))
    job = pilot.Job("cramped_room", "ft", 0, "id-mh", runs, overrides)
    sel = pilot.Plan(root=str(root), steps_per_partner=192, num_envs=4, num_steps=16)
    result = execute(pilot.build_config(job, 192))
    return dict(root=root, job=job, sel=sel, run_dir=result.run_dir, result=result, bank=bank)


def describe(tiny, job=None):
    return pilot.describe_jobs(tiny["sel"], tiny["root"], [job or tiny["job"]], out=lambda *_: None)


def test_completed_run_is_recognised_and_not_repeated(tiny, launched):
    info = describe(tiny)
    item = info[tiny["job"].name]
    assert item["status"] == "complete" and item["run_dir"] == tiny["run_dir"]
    assert pilot.run_jobs(tiny["sel"], tiny["root"], info, False, False, run=True) == 0
    assert launched.calls == []


def test_a_different_configuration_is_a_new_job_not_a_reuse(tiny, launched):
    other = pilot.Job("cramped_room", "ft", 1, "id-mh", tiny["job"].root, dict(tiny["job"].overrides, seed=1))
    other_info = describe(tiny, other)
    assert other_info[other.name]["status"] == "new"
    changed = pilot.Job("cramped_room", "ft", 0, "id-mh", tiny["job"].root,
                        dict(tiny["job"].overrides, num_eval_episodes=3))
    assert describe(tiny, changed)[changed.name]["status"] == "new"
    assert pilot.run_jobs(tiny["sel"], tiny["root"], other_info, False, False, run=True) == 0
    assert len(launched.calls) == 1 and "experiments.partner_adaptation.run_br" in launched.calls[0]


def test_directory_existence_is_not_completion(tiny, tmp_path, launched):
    # an incomplete run of the same configuration is found by its fingerprint, not trusted as complete
    import shutil
    copy_root = tmp_path / "runs"
    shutil.copytree(tiny["run_dir"], copy_root / tiny["run_dir"].name)
    status_path = copy_root / tiny["run_dir"].name / "status.json"
    status = json.loads(status_path.read_text())
    status.update(state="running", stages_completed=2)
    status_path.write_text(json.dumps(status))
    job = pilot.Job("cramped_room", "ft", 0, "id-mh", copy_root, tiny["job"].overrides)
    info = describe(tiny, job)
    assert info[job.name]["status"] == "resumable"
    assert pilot.run_jobs(tiny["sel"], tiny["root"], info, False, False, run=True) == 1       # blocked, not repeated
    assert launched.calls == []
    assert pilot.run_jobs(tiny["sel"], tiny["root"], info, False, True, run=True) == 0       # explicit --restart
    assert len(launched.calls) == 1
    # a directory with a matching name but no run.json/status.json is ignored altogether
    (tmp_path / "other" / "fake_run").mkdir(parents=True)
    job2 = pilot.Job("cramped_room", "ft", 0, "id-mh", tmp_path / "other", tiny["job"].overrides)
    assert describe(tiny, job2)[job2.name]["status"] == "new"


def test_failed_job_returns_nonzero_and_stops_unless_keep_going(tiny, launched):
    jobs = [pilot.Job("cramped_room", "ft", s, "id-mh", tiny["job"].root, dict(tiny["job"].overrides, seed=s))
            for s in (1, 2)]
    info = pilot.describe_jobs(tiny["sel"], tiny["root"], jobs, out=lambda *_: None)
    launched.code = 5
    assert pilot.run_jobs(tiny["sel"], tiny["root"], info, False, False, run=True) == 1
    assert len(launched.calls) == 1
    launched.calls.clear()
    assert pilot.run_jobs(tiny["sel"], tiny["root"], info, True, False, run=True) == 1
    assert len(launched.calls) == 2


def test_resume_reads_the_run_configuration_and_uses_the_resume_flag(tiny, capsys, launched):
    code, out = run_main(["resume", str(tiny["run_dir"])], capsys)
    assert code == 0, out.err
    assert "4/4 partners done" in out.out and "[dry-run]" in out.out and launched.calls == []
    code, out = run_main(["resume", str(tiny["run_dir"]), "--run", "--wandb-mode", "disabled"], capsys)
    assert code == 0 and len(launched.calls) == 1
    command = launched.calls[0]
    assert command[command.index("--resume") + 1] == str(tiny["run_dir"])
    assert command[command.index("--mode") + 1] == "disabled"
    launched.code = 4
    assert run_main(["resume", str(tiny["run_dir"]), "--run"], capsys)[0] == 1


def test_resume_of_a_missing_run_fails(tmp_path, capsys):
    code, out = run_main(["resume", str(tmp_path / "nope")], capsys)
    assert code == 2


def test_tiny_run_used_the_disabled_wandb_path_and_pilot_schedule(tiny):
    evals = pilot_gpu.read_csv(tiny["run_dir"] / "eval_metrics.csv")
    pilot_gpu.verify_run(tiny["result"], 4, pilot.build_config(tiny["job"], 192))
    assert {r["scope"] for r in evals} == {"full", "current"}
    assert tiny["result"].counters["eval_full_events"] == 5            # initialization + after each of 4 stages


# ---------------------------------------------------------------------------------------------------------
# evaluation schedule and projection arithmetic
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("updates,every", [(1, 5), (5, 5), (6, 5), (10, 5), (10, 3), (7, 0), (4, 1)])
def test_current_event_formula_matches_the_training_schedule(updates, every):
    cfg = TrainConfig(**dict(TINY, total_timesteps=updates * 64, num_heuristic_partners=0,
                             partner_bank="", eval_current_every=every))
    plan = eval_plan(cfg)
    assert list(plan.values()).count("current") == pilot_gpu.current_events_per_stage(updates, every)
    assert plan[updates] == "full" and list(plan.values()).count("full") == 1


def profile(expanded=True, planner=True):
    path = lambda c, u, i, ic, ec: dict(stage_ids=[0, 1], compile_s=c, first_importance_s=i + 6, update_s=u,
                                        importance_s=i, eval_current_s=ec)
    method = lambda imp: {"learned": path(30, 4, imp, 0, 0.5), "planner": path(40, 6, imp, 0, 0.5) if planner else None,
                          "eval_full_s": 2.0, "first_full_eval_s_including_compile": 12.0}
    return {"kind": "ego", "device": {"device_kind": "TestGPU", "cuda": True, "memory_limit_bytes": None,
                                      "device_count": 1},
            "bank": {"expanded": expanded, "projection": "measured on the expanded bank" if expanded else
                     "PROVISIONAL: evaluation scales linearly from the profiled bank"},
            "layout": "coord_ring",
            "settings": {"num_envs": 2048, "num_steps": 400, "update_epochs": 8, "num_minibatches": 16,
                         "joint_steps_per_update": 819200, "updates_per_stage": 6, "eval_partners": 24,
                         "eval_episodes": 5},
            "methods": {"ft": method(0), "online_ewc": method(2.0)}}


BRDIV = {"kind": "brdiv", "device": {"device_kind": "TestGPU", "cuda": True, "memory_limit_bytes": None,
                                     "device_count": 1},
         "settings": {"num_envs_xp": 32, "num_envs_sp": 32, "num_steps": 400, "partner_pop_size": 3,
                      "agent_transitions_per_update": 51200},
         "compile_s": 60.0, "update_s": 1.0, "fixed_s": 10.0}


def test_ego_job_projection_is_the_documented_arithmetic():
    r = pilot_gpu.ego_job_seconds(profile(), "ft", 24, 12, 12, 10, 5, 5)
    c = r["components"]
    assert c["compile"] == pytest.approx(30 + 40 + (12 - 2))
    assert c["train"] == pytest.approx(12 * 10 * 4 + 12 * 10 * 6)
    assert c["importance"] == 0
    assert c["evaluation_full"] == pytest.approx(25 * 2.0)
    assert c["evaluation_current"] == pytest.approx(1 * (12 * 0.5 + 12 * 0.5))
    assert r["total"] == pytest.approx(sum(c.values()))
    ewc = pilot_gpu.ego_job_seconds(profile(), "online_ewc", 24, 12, 12, 10, 5, 5)["components"]
    assert ewc["importance"] == pytest.approx(24 * 2.0) and ewc["compile"] == pytest.approx(80 + 2 * 6)
    # equal budgets: the training component does not depend on the method
    assert ewc["train"] == c["train"]


def test_unmeasured_paths_and_methods_are_reported_not_guessed():
    r = pilot_gpu.ego_job_seconds(profile(planner=False), "ft", 24, 12, 12, 10, 5, 5)
    assert "total" not in r and "planner path" in r["missing"][0]
    assert "not profiled" in pilot_gpu.ego_job_seconds(profile(), "online_mas", 24, 12, 12, 10, 5, 5)["missing"][0]


def write_profiles(tmp_path, **kw):
    ego, brdiv = tmp_path / "ego.json", tmp_path / "brdiv.json"
    ego.write_text(json.dumps(profile(**kw)))
    brdiv.write_text(json.dumps(BRDIV))
    return str(ego), str(brdiv)


def test_projection_reports_device_batch_units_and_provisional_label(tmp_path, capsys):
    ego, brdiv = write_profiles(tmp_path, expanded=False)
    sel = pilot.Plan(root=str(tmp_path), profile=ego, brdiv_profile=brdiv, steps_per_partner=4915200)
    info = {"j": {"numbers": dict(partners=24, learned=12, planners=12, updates_per_partner=6)}}
    assert pilot_gpu.report_projection(sel, tmp_path, info) == 0
    text = capsys.readouterr().out
    assert "TestGPU" in text and "2048 envs x 400 steps" in text and "819,200 joint steps/update" in text
    assert "PROVISIONAL" in text and "51,200 agent transitions/update" in text
    assert "not measured" in text


def test_a_profile_from_other_batch_settings_is_not_applied(tmp_path, capsys):
    ego, brdiv = write_profiles(tmp_path)
    sel = pilot.Plan(root=str(tmp_path), profile=ego, brdiv_profile=brdiv, num_envs=1024, steps_per_partner=2457600)
    assert pilot_gpu.report_projection(sel, tmp_path, {"j": {"numbers": dict(
        partners=24, learned=12, planners=12, updates_per_partner=6)}}) == 0
    text = capsys.readouterr().out
    assert "ego profile is NOT USED" in text and "num_envs: profiled 2048, selected 1024" in text
    assert "projected total" not in text and "SUGGESTION" not in text


def test_suggested_budget_is_the_largest_that_fits_and_is_labelled(tmp_path, capsys):
    ego, brdiv = write_profiles(tmp_path)
    sel = pilot.Plan(root=str(tmp_path), profile=ego, brdiv_profile=brdiv, layouts=("coord_ring", "cramped_room"))
    assert pilot_gpu.report_projection(sel, tmp_path, None) == 0
    text = capsys.readouterr().out
    assert "SUGGESTION ONLY" in text and "you must choose --steps-per-partner yourself" in text
    updates = int(re.search(r"<= 13 h: (\d+) updates", text).group(1))
    counts = pilot.expected_partners(3, sel.gen_seeds)
    ego_profile, brdiv_profile = json.loads(Path(ego).read_text()), json.loads(Path(brdiv).read_text())

    def total(u):
        parts = pilot_gpu.projection_totals(sel, ego_profile, brdiv_profile, u, counts)["parts"]
        return sum(parts.values()) + sum(pilot_gpu.generation_seconds(sel, brdiv_profile, tmp_path).values())

    assert total(updates) <= 13 * 3600 < total(updates + 1)


def test_projection_counts_only_populations_still_to_generate(tmp_path):
    _, brdiv = write_profiles(tmp_path)
    sel = pilot.Plan(root=str(tmp_path), layouts=("coord_ring",), gen_seeds=(1001, 1002))
    full = sum(pilot_gpu.generation_seconds(sel, json.loads(Path(brdiv).read_text()), tmp_path).values())
    pop_cfg = pilot.generation_config("coord_ring", 1001, tmp_path, 3, None, "online")
    make_population(tmp_path, total=int(pop_cfg.total_timesteps), gen_mode="online")
    half = sum(pilot_gpu.generation_seconds(sel, json.loads(Path(brdiv).read_text()), tmp_path).values())
    assert half == pytest.approx(full / 2)


def test_over_budget_train_is_refused_with_per_layout_commands_unless_accepted(tmp_path, capsys):
    ego, brdiv = write_profiles(tmp_path)
    info = {"j": {"numbers": dict(partners=24, learned=12, planners=12, updates_per_partner=2000)}}
    sel = pilot.Train(root=str(tmp_path), profile=ego, brdiv_profile=brdiv, steps_per_partner=2000 * 819200.0)
    assert pilot_gpu.report_projection(sel, tmp_path, info, gate=True) == 1
    text = capsys.readouterr().out
    assert "refusing to start" in text and "bottlenecks" in text
    assert "--layouts coord_ring" in text and "--layouts cramped_room" in text      # neither layout is dropped
    sel.accept_over_budget = True
    assert pilot_gpu.report_projection(sel, tmp_path, info, gate=True) == 0


# ---------------------------------------------------------------------------------------------------------
# device checks
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(jax.default_backend() != "cpu", reason="checks the refusal to fall back to CPU")
def test_cpu_is_refused_without_the_explicit_rehearsal_flag(tmp_path, capsys):
    with pytest.raises(pilot.PilotError, match="no CUDA device"):
        pilot_gpu.device_report(allow_cpu=False)
    assert pilot_gpu.device_report(allow_cpu=True)["cuda"] is False
    code, out = run_main(["preflight", "--root", str(tmp_path)], capsys)
    assert code == 1 and "[FAIL] device" in out.out and "[stop]" in out.out
    assert run_main(["profile", "--root", str(tmp_path)], capsys)[0] == 2
