"""Smoke, GPU preflight, profiling and runtime projection for the CPA pilot launcher (see pilot.py).

Nothing here starts a full job: smoke and preflight train a few updates of a handful of partners, profiling trains
`updates` updates of four stages (two learned, two planner) per method and two short BRDiv runs. Every measurement
comes from `block_until_ready`-synchronized work; compilation is timed apart from steady-state execution. A
projection is arithmetic on those measurements and is labelled with what was and was not measured.
"""
import csv
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.partner_adaptation.pilot import (
    HORIZON, METHODS, TARGET_GPU_HOURS, PilotError, check_layouts, check_methods, expected_partners,
    generation_config, hours, human, population_dir, population_state, quiet, resolve_root, shaping_note)
from experiments.partner_adaptation.run_outputs import atomic_write, write_json


# ---------------------------------------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------------------------------------

def device_report(allow_cpu: bool) -> Dict[str, Any]:
    """Verify that JAX runs on a CUDA device (no silent CPU fallback) and that work really lands on it."""
    import jax
    import jax.numpy as jnp
    device = jax.devices()[0]
    version = str(getattr(getattr(device, "client", None), "platform_version", ""))
    is_cuda = device.platform in ("gpu", "cuda") and "rocm" not in version.lower()
    if not is_cuda and not allow_cpu:
        raise PilotError(
            f"no CUDA device: JAX's default backend is {device.platform!r} ({device.device_kind}). Refusing to "
            f"continue on CPU. Check `nvidia-smi`, that a CUDA build of jax is installed and that JAX_PLATFORMS / "
            f"JAX_PLATFORM_NAME do not force cpu. (--allow-cpu is a rehearsal and not a GPU verification.)")
    x = jax.device_put(jnp.ones((512, 512), jnp.float32), device)
    y = jax.block_until_ready(x @ x)
    if device not in y.devices():
        raise PilotError(f"a test computation did not run on {device}")
    memory = None
    try:
        stats = device.memory_stats()
        memory = stats.get("bytes_limit") if stats else None
    except Exception:  # noqa: BLE001
        pass
    return {"platform": device.platform, "device_kind": device.device_kind, "cuda": is_cuda,
            "platform_version": version, "device_count": len(jax.devices()), "memory_limit_bytes": memory,
            "jax_platform_env": {k: os.environ[k] for k in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME") if k in os.environ}}


def describe_device(report: Dict[str, Any]) -> str:
    kind = "CUDA" if report["cuda"] else "NOT A GPU (cpu rehearsal)"
    memory = f", {report['memory_limit_bytes'] / 2 ** 30:.1f} GiB" if report["memory_limit_bytes"] else ""
    return f"{report['device_kind']} [{kind}{memory}] x{report['device_count']}"


# ---------------------------------------------------------------------------------------------------------
# smoke (CPU, no W&B)
# ---------------------------------------------------------------------------------------------------------

def tiny_ego_config(root: Path, layout: str, bank: Path, method: str, **overrides):
    from experiments.partner_adaptation.run_br import TrainConfig
    settings = dict(
        mode="disabled", checkpoint_path=str(root / "runs" / layout / method), layout_name=layout,
        partner_bank=str(bank), num_heuristic_partners=0, num_envs=4, num_steps=16, total_timesteps=3 * 4 * 16,
        update_epochs=1, num_minibatches=2, num_eval_episodes=2, eval_schedule="pilot", eval_current_every=2,
        hidden_size=16, num_layers=1, importance_episodes=1, importance_steps=8, reg_coef=1.0, seed=0,
        reward_shaping_horizon=1e3, **METHODS[method])
    settings.update(overrides)
    with quiet():
        return TrainConfig(**settings)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def verify_run(result, partners: int, config):
    """Checks shared by smoke and preflight: completion, schedule, unique events, finite losses, records."""
    import math
    from experiments.partner_adaptation.run_outputs import read_checkpoint
    d = result.run_dir
    status = json.loads((d / "status.json").read_text())
    assert status["state"] == "complete" and status["stages_completed"] == partners, status
    train = read_csv(d / "train_metrics.csv")
    assert len(train) == partners * int(config.num_updates), len(train)
    assert all(math.isfinite(float(r[c])) for r in train for c in ("actor_loss", "value_loss", "entropy")), \
        "non-finite loss"
    evals = read_csv(d / "eval_metrics.csv")
    row_ids = [r["row_id"] for r in evals]
    assert len(row_ids) == len(set(row_ids)), "duplicate evaluation rows"
    full = {r["event_id"] for r in evals if r["scope"] == "full"}
    assert len(full) == partners + 1 and "s000-u00000" in full, sorted(full)
    for event in full:
        assert {int(r["eval_partner_id"]) for r in evals if r["event_id"] == event} == set(range(partners))
    meta, _ = read_checkpoint(d / "latest.ckpt")
    assert meta["stages_completed"] == partners
    assert (d / "params_seed0.pt").is_file()
    return evals


def smoke(opts) -> int:
    """Tiny generation -> bank -> FT / Online EWC / Online MAS ego runs -> no-op resume, all on the local device."""
    import tempfile
    from experiments.partner_adaptation.partner_bank import assemble_bank
    from experiments.partner_adaptation.partner_generation.run import generate_population
    from experiments.partner_adaptation.run_br import execute
    check_layouts(opts.layouts)
    check_methods(opts.methods)
    os.environ["WANDB_MODE"] = "disabled"
    root = Path(opts.root).expanduser().resolve() if opts.root else Path(tempfile.mkdtemp(prefix="cpa_smoke_"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"smoke root: {root}  (local, W&B disabled; tiny budgets: this checks wiring, not learning)")
    for layout in opts.layouts:
        dirs = []
        if opts.generation:
            gen = generation_config(layout, 1001, root, 3, 2 * 8 * 4 * 2, "disabled", num_envs_xp=2, num_envs_sp=2,
                                    num_steps=8, update_epochs=1, num_minibatches=2, num_checkpoints=2,
                                    num_eval_episodes=1)
            if population_state(gen)[0] != "valid":
                with quiet():
                    generate_population(gen)
            assert population_state(gen)[0] == "valid", population_state(gen)
            dirs.append(str(population_dir(gen)))
            print(f"[ok] {layout}: tiny BRDiv population {population_dir(gen)}")
        bank = root / "banks" / f"{layout}_smoke.json"
        assembled = assemble_bank(layout, str(bank), dirs, bundled=True, planners=("P01",))
        n = len(assembled)
        print(f"[ok] {layout}: smoke bank with {n} partners ({n - 1} learned + 1 planner)")
        for method in opts.methods:
            config = tiny_ego_config(root, layout, bank, method)
            start = time.perf_counter()
            with quiet():
                result = execute(config)
            verify_run(result, n, config)
            with quiet():
                again = execute(tiny_ego_config(root, layout, bank, method, resume=str(result.run_dir)))
            assert again.counters == result.counters
            print(f"[ok] {layout} {method}: {n} partners, {result.counters['train_env_steps']} joint steps, "
                  f"{result.counters['eval_events']} evaluation events, resume no-op, "
                  f"{time.perf_counter() - start:.0f} s -> {result.run_dir}")
    print("smoke checks passed (CPU-tested paths only; nothing here says anything about GPU behaviour)")
    return 0


# ---------------------------------------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------------------------------------

class Skipped(Exception):
    pass


def preflight(opts) -> int:
    from experiments.partner_adaptation.partner_bank import assemble_bank, load_partners, select_partner_bank
    from experiments.partner_adaptation.run_br import execute
    check_layouts([opts.layout])
    root = resolve_root(opts.root, create=True)
    work = root / "preflight" / datetime.now().strftime("%Y%m%d-%H%M%S")
    work.mkdir(parents=True, exist_ok=True)
    results, state = [], {}

    def check(name, fn):
        try:
            detail = fn()
            results.append(("PASS", name))
        except Skipped as e:
            detail = str(e)
            results.append(("SKIP", name))
        except Exception as e:  # noqa: BLE001 - every failure is reported, none aborts the other checks
            detail = f"{type(e).__name__}: {e}"
            results.append(("FAIL", name))
        print(f"[{results[-1][0]}] {name}: {detail}")

    def device():
        state["device"] = device_report(opts.allow_cpu)
        return describe_device(state["device"])

    def writes():
        probe = work / "probe.bin"
        atomic_write(probe, os.urandom(1 << 20))
        assert probe.stat().st_size == 1 << 20
        probe.unlink()
        free = shutil.disk_usage(work).free / 2 ** 30
        return f"{work} writable (atomic write and fsync ok), {free:.1f} GiB free; this root must live on storage " \
               f"that survives the pod"

    def bundled():
        import jax
        import numpy as np
        from meal import make_env
        from meal.env.overcooked.layouts.presets import overcooked_layouts
        bank = select_partner_bank(opts.layout, "", None)
        env = make_env("overcooked", layout=overcooked_layouts[opts.layout])
        obs_dim = int(np.prod(env.observation_space().shape))
        partners = load_partners(bank, obs_dim)
        obs, _ = env.reset(jax.random.PRNGKey(0))
        for p in partners:
            action, _ = p.policy.get_action(p.params, obs["agent_1"].reshape(-1), False, np.ones(6, dtype=bool), None,
                                            jax.random.PRNGKey(1), test_mode=True)
            assert 0 <= int(jax.block_until_ready(action)) < 6
        return f"{len(partners)} bundled {opts.layout} policies loaded and acted on {jax.default_backend()}"

    def training():
        bank_file = work / "bank.json"
        bank = assemble_bank(opts.layout, str(bank_file), [], bundled=True, planners=("P01",))
        n = len(bank)
        config = _preflight_config(work, opts, bank_file)
        result = execute(config)
        evals = verify_run(result, n, config)
        timings = read_csv(result.run_dir / "stage_timings.csv")
        backend = timings[0]["backend"]
        if not opts.allow_cpu and backend not in ("gpu", "cuda"):
            raise PilotError(f"training ran on backend {backend!r}")
        kinds = {r["eval_partner"] for r in evals}
        return (f"{n} stages ({n - 1} learned + planner P01), EWC step and evaluation on {backend}, "
                f"{len(evals)} evaluation rows over {len(kinds)} partners, checkpoint/CSV/export written and read "
                f"back, compile {sum(float(r['compile_s'] or 0) for r in timings):.1f} s")

    def wandb_auth():
        if opts.wandb_mode != "online":
            raise Skipped(f"mode {opts.wandb_mode!r}: online authentication NOT verified")
        import wandb
        if not wandb.login(key=os.environ.get("WANDB_API_KEY"), verify=True):
            raise PilotError("wandb.login returned False; set WANDB_API_KEY or run `wandb login`")
        return "authenticated (credential taken from the environment / stored login, not printed)"

    for name, fn in (("device", device), ("output root", writes), ("bundled policies", bundled),
                     ("tiny training + evaluation", training), ("W&B authentication", wandb_auth)):
        check(name, fn)
        if name == "device" and results[-1][0] == "FAIL":
            print("[stop] the remaining checks are meaningless without the target device")
            break
    failed = [n for s, n in results if s == "FAIL"]
    print("\npreflight " + ("FAILED: " + ", ".join(failed) if failed else
                            "passed" + (" (CPU rehearsal: NOT a GPU verification)" if opts.allow_cpu else "")))
    return 1 if failed else 0


def _preflight_config(work: Path, opts, bank_file: Path):
    from experiments.partner_adaptation.run_br import TrainConfig
    with quiet():
        return TrainConfig(
            mode="disabled", checkpoint_path=str(work / "runs"), layout_name=opts.layout, partner_bank=str(bank_file),
            num_heuristic_partners=0, num_envs=opts.num_envs, num_steps=HORIZON, total_timesteps=opts.num_envs * HORIZON,
            update_epochs=1, num_minibatches=16, num_eval_episodes=2, eval_schedule="pilot", eval_current_every=1,
            importance_episodes=1, importance_steps=50, seed=0, **METHODS["online_ewc"])


# ---------------------------------------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------------------------------------

def _f(row, key) -> Optional[float]:
    value = row.get(key)
    return float(value) if value not in (None, "") else None


def path_summary(rows: Dict[int, Dict[str, str]], ids: List[int], updates: int) -> Optional[Dict[str, Any]]:
    """Measurements of one compilation signature: the first stage of the path compiles, the second one is the
    steady-state sample (compile_cached must be set for it)."""
    if not ids:
        return None
    first = rows[ids[0]]
    out: Dict[str, Any] = {"stage_ids": ids, "compile_s": _f(first, "compile_s"),
                           "first_importance_s": _f(first, "importance_s")}
    if len(ids) >= 2:
        second = rows[ids[1]]
        scan = _f(second, "train_eval_s") or 0.0
        in_scan_eval = ((_f(second, "eval_events") or 0) * (_f(second, "eval_event_s") or 0)
                        + (_f(second, "eval_current_events") or 0) * (_f(second, "eval_current_event_s") or 0))
        out.update(
            compile_reused_by_second_stage=second["compile_cached"] == "1",
            update_s=max(scan - in_scan_eval, 0.0) / updates, importance_s=_f(second, "importance_s"),
            eval_current_s=_f(second, "eval_current_event_s"), peak_bytes_in_use=_f(second, "peak_bytes_in_use"))
    return out


def profile_ego(opts) -> int:
    from experiments.partner_adaptation.partner_bank import assemble_bank, load_partner_bank
    from experiments.partner_adaptation.run_br import TrainConfig, execute
    check_layouts([opts.layout])
    check_methods(opts.methods)
    if not 2 <= opts.updates <= 50:
        raise PilotError("--updates must be between 2 and 50: profiling never trains a full sequence, and a "
                         "current-partner evaluation event needs a non-final update")
    current_every = max(1, min(opts.eval_current_every, opts.updates - 1))
    if opts.num_steps != HORIZON and not opts.allow_protocol_change:
        raise PilotError(f"episode horizon {HORIZON} is part of the experiment; {opts.num_steps} is a protocol change "
                         f"(--allow-protocol-change)")
    root = resolve_root(opts.root, create=True)
    device = device_report(opts.allow_cpu)
    out_dir = root / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    total, learned_n, planners_n = expected_partners(opts.pop_size, opts.gen_seeds)
    if opts.bank:
        bank_file = Path(opts.bank)
    else:
        bank_file = out_dir / f"_profile_bank_{opts.layout}.json"
        assemble_bank(opts.layout, str(bank_file), [], bundled=True, planners=("P01", "P02"))
    bank = load_partner_bank(str(bank_file), opts.layout)
    kinds = [r.kind for r in bank.records]
    learned_ids = [i for i, k in enumerate(kinds) if k == "brdiv"][:2]
    planner_ids = [i for i, k in enumerate(kinds) if k == "planner"][:2]
    expanded = len(bank) == total and kinds.count("planner") == planners_n
    only = learned_ids + planner_ids
    print(f"profiling on {describe_device(device)}: bank {bank_file} ({len(bank)} partners, "
          f"{'expanded bank' if expanded else 'NOT the expanded bank: projections are PROVISIONAL'}), stages {only}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    methods: Dict[str, Any] = {}
    for method in opts.methods:
        with quiet():
            config = TrainConfig(
                mode="disabled", checkpoint_path=str(out_dir / "runs" / f"{opts.layout}_{method}_{stamp}"),
                layout_name=opts.layout, partner_bank=str(bank_file), num_heuristic_partners=0,
                num_envs=opts.num_envs, num_steps=opts.num_steps, update_epochs=opts.update_epochs,
                num_minibatches=opts.num_minibatches, total_timesteps=opts.updates * opts.num_envs * opts.num_steps,
                num_eval_episodes=opts.eval_episodes, eval_schedule="pilot",
                eval_current_every=current_every, only_stages=only, save_checkpoints=False, seed=0,
                **METHODS[method])
        result = execute(config)
        rows = {int(r["stage"]): r for r in read_csv(result.run_dir / "stage_timings.csv")}
        full = [_f(r, "eval_event_s") for r in rows.values() if _f(r, "eval_event_s") is not None]
        methods[method] = {
            "learned": path_summary(rows, learned_ids, opts.updates),
            "planner": path_summary(rows, planner_ids, opts.updates),
            "eval_full_s": sum(full) / len(full) if full else None,
            "first_full_eval_s_including_compile": _f(rows[only[0]], "init_eval_s"),
            "run_dir": str(result.run_dir)}
        print(f"  {method}: " + json.dumps({k: v for k, v in methods[method].items() if k != "run_dir"}, default=str))
    profile = {
        "kind": "ego", "created": stamp, "device": device, "layout": opts.layout,
        "bank": {"path": str(bank_file), "partners": len(bank), "learned": kinds.count("brdiv"),
                 "planners": kinds.count("planner"), "expanded": expanded,
                 "projection": "measured on the expanded bank" if expanded else
                 "PROVISIONAL: evaluation scales linearly from the profiled bank; paths absent from it are unmeasured"},
        "settings": {"num_envs": opts.num_envs, "num_steps": opts.num_steps, "update_epochs": opts.update_epochs,
                     "num_minibatches": opts.num_minibatches, "updates_per_stage": opts.updates,
                     "eval_episodes": opts.eval_episodes, "eval_current_every_profiled": current_every,
                     "eval_partners": len(bank), "joint_steps_per_update": opts.num_envs * opts.num_steps,
                     "unit": "joint environment steps (both agents act once per step)"},
        "methods": methods}
    path = out_dir / f"ego_{opts.layout}_{stamp}.json"
    write_json(path, profile)
    print(f"wrote {path}\nuse it with `plan --profile {path}`")
    return 0


def profile_brdiv(opts) -> int:
    import wandb
    from experiments.partner_adaptation.partner_generation.BRDiv import run_brdiv
    from experiments.partner_adaptation.partner_generation.run import TrainConfig as GenConfig, resolve_layout
    check_layouts([opts.layout])
    a, b = sorted(opts.updates)
    if not 1 <= a < b <= 20:
        raise PilotError("--updates needs two different counts between 1 and 20")
    root = resolve_root(opts.root, create=True)
    device = device_report(opts.allow_cpu)
    out_dir = root / "profiles"
    wandb.init(mode="disabled")
    runs = []
    for n in (a, b):
        with quiet():
            cfg = GenConfig(
                mode="disabled", layout_name=opts.layout, seed=0, num_seeds=1, partner_pop_size=opts.partner_pop_size,
                num_envs_xp=opts.num_envs_xp, num_envs_sp=opts.num_envs_sp, num_steps=opts.num_steps,
                num_checkpoints=2, total_timesteps=n * 2 * opts.num_steps * (opts.num_envs_xp + opts.num_envs_sp),
                checkpoint_path=str(out_dir / "brdiv_runs"))
        cfg.layout = resolve_layout(cfg)[0]
        cfg.save_dir = str(out_dir / "brdiv_runs" / f"{opts.layout}_u{n}")
        os.makedirs(cfg.save_dir, exist_ok=True)
        timings: Dict[str, float] = {}
        run_brdiv(cfg, timings)
        runs.append(dict(updates=n, **timings))
        print(f"  {n} updates: compile {timings['compile_s']:.2f} s, run {timings['run_s']:.2f} s")
    update_s = (runs[1]["run_s"] - runs[0]["run_s"]) / (b - a)
    profile = {
        "kind": "brdiv", "created": datetime.now().strftime("%Y%m%d-%H%M%S"), "device": device, "layout": opts.layout,
        "settings": {"num_envs_xp": opts.num_envs_xp, "num_envs_sp": opts.num_envs_sp, "num_steps": opts.num_steps,
                     "partner_pop_size": opts.partner_pop_size,
                     "joint_steps_per_update": opts.num_steps * (opts.num_envs_xp + opts.num_envs_sp),
                     "agent_transitions_per_update": 2 * opts.num_steps * (opts.num_envs_xp + opts.num_envs_sp),
                     "unit": "agent transitions = 2 x joint environment steps"},
        "compile_s": max(r["compile_s"] for r in runs), "update_s": update_s,
        "fixed_s": runs[0]["run_s"] - a * update_s, "runs": runs,
        "note": "per-update time is the difference of two synchronized runs; the checkpoint evaluations of a full "
                "run (5) are only represented by the fixed part"}
    path = out_dir / f"brdiv_{opts.layout}_{profile['created']}.json"
    write_json(path, profile)
    print(f"wrote {path}\nuse it with `plan --brdiv-profile {path}`")
    return 0


# ---------------------------------------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------------------------------------

def current_events_per_stage(updates: int, every: int) -> int:
    """Current-partner events in one stage under the pilot schedule (the stage end is a full event)."""
    if every <= 0:
        return 0
    return updates // every - (1 if updates % every == 0 else 0)


def ego_job_seconds(profile: Dict[str, Any], method: str, partners: int, learned: int, planners: int, updates: int,
                    current_every: int, eval_episodes: int) -> Dict[str, Any]:
    """Seconds of one ego sequence from a profile; components that were not measured are listed in `missing`."""
    m = profile["methods"].get(method)
    if m is None:
        return {"missing": [f"method {method} was not profiled"]}
    missing, c = [], {}
    paths = {"learned": learned, "planner": planners}
    importance = method != "ft"
    for kind, count in paths.items():
        p = m.get(kind)
        if count and (p is None or p.get("update_s") is None):
            missing.append(f"{kind} path (needs two {kind} stages in the profiled bank)")
    if missing:
        return {"missing": missing}
    c["compile"] = sum(m[k]["compile_s"] or 0.0 for k, n in paths.items() if n)
    if importance:
        c["compile"] += sum(max((m[k]["first_importance_s"] or 0) - (m[k]["importance_s"] or 0), 0.0)
                            for k, n in paths.items() if n)
    first_eval = m.get("first_full_eval_s_including_compile")
    c["compile"] += max((first_eval or 0.0) - (m.get("eval_full_s") or 0.0), 0.0)
    c["train"] = sum(n * updates * m[k]["update_s"] for k, n in paths.items())
    c["importance"] = sum(n * (m[k]["importance_s"] or 0.0) for k, n in paths.items()) if importance else 0.0
    scale = (partners / profile["settings"]["eval_partners"]) * (eval_episodes / profile["settings"]["eval_episodes"])
    c["evaluation_full"] = (partners + 1) * (m["eval_full_s"] or 0.0) * scale
    cur = current_events_per_stage(updates, current_every)
    cscale = eval_episodes / profile["settings"]["eval_episodes"]
    c["evaluation_current"] = cur * sum(n * (m[k]["eval_current_s"] or 0.0) * cscale for k, n in paths.items())
    return {"components": c, "total": sum(c.values()), "missing": []}


def brdiv_job_seconds(profile: Dict[str, Any], updates: int) -> Dict[str, Any]:
    c = {"compile": profile["compile_s"], "train": updates * profile["update_s"], "fixed": profile["fixed_s"]}
    return {"components": c, "total": sum(c.values()), "missing": []}


def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    return json.loads(Path(path).read_text()) if path else None


def profile_mismatches(sel, ego, brdiv) -> Dict[str, List[str]]:
    """Reasons a profile cannot be applied to the selected jobs: per-update time is only valid for the settings it
    was measured with."""
    out: Dict[str, List[str]] = {"ego": [], "brdiv": []}
    if ego is not None:
        s = ego["settings"]
        for key in ("num_envs", "num_steps", "update_epochs", "num_minibatches"):
            if s.get(key) != getattr(sel, key):
                out["ego"].append(f"{key}: profiled {s.get(key)}, selected {getattr(sel, key)}")
    if brdiv is not None:
        s = brdiv["settings"]
        cfg = generation_config(sel.layouts[0], sel.gen_seeds[0], Path("."), sel.pop_size, None, "disabled")
        for key, want in (("num_envs_xp", cfg.num_envs_xp), ("num_envs_sp", cfg.num_envs_sp),
                          ("num_steps", cfg.num_steps), ("partner_pop_size", sel.pop_size)):
            if s.get(key) != want:
                out["brdiv"].append(f"{key}: profiled {s.get(key)}, generation uses {want}")
    return out


def projection_totals(sel, ego, brdiv, updates: int, partner_counts) -> Dict[str, Any]:
    """Totals for the whole pilot at a given per-partner update count (generation jobs + all ego jobs)."""
    partners, learned, planners = partner_counts
    jobs = len(sel.layouts) * len(sel.methods) * len(sel.seeds)
    parts, missing = {}, []
    if ego is not None:
        for method in sel.methods:
            r = ego_job_seconds(ego, method, partners, learned, planners, updates, sel.eval_current_every,
                                sel.eval_episodes)
            missing += r["missing"]
            if not r["missing"]:
                for k, v in r["components"].items():
                    parts[f"ego {k}"] = parts.get(f"ego {k}", 0.0) + v * len(sel.layouts) * len(sel.seeds)
    return {"parts": parts, "missing": missing, "ego_jobs": jobs}


def generation_seconds(sel, brdiv) -> Optional[Dict[str, float]]:
    from experiments.partner_adaptation.partner_generation.run import interaction_counts
    if brdiv is None:
        return None
    cfg = generation_config(sel.layouts[0], sel.gen_seeds[0], Path("."), sel.pop_size, sel.gen_total_timesteps,
                            "disabled")
    updates = interaction_counts(cfg)["num_updates"]
    n = len(sel.layouts) * len(sel.gen_seeds)
    return {f"generation {k}": v * n for k, v in brdiv_job_seconds(brdiv, updates)["components"].items()}


def report_projection(sel, root: Path, jobs_info, gate: bool = False) -> int:
    """Print the runtime projection (if profiles are given) and, for `plan`, the suggested budget. With `gate`,
    returns 1 when the ego jobs are projected above the target and the user did not accept that."""
    ego, brdiv = _load(getattr(sel, "profile", None)), _load(getattr(sel, "brdiv_profile", None))
    if ego is None and brdiv is None:
        print("runtime projection: NONE. No measured profile was given, so no runtime is claimed "
              "(run `profile` and `profile-brdiv`; the 8-13 GPU-hour target is not verified).")
        return 0
    mismatches = profile_mismatches(sel, ego, brdiv)
    for name, reasons in mismatches.items():
        if reasons:
            print(f"runtime projection: the {name} profile is NOT USED because it was measured with other settings "
                  f"({'; '.join(reasons)}); profile again with the settings you plan to run.")
    ego = None if mismatches["ego"] else ego
    brdiv = None if mismatches["brdiv"] else brdiv
    if ego is None and brdiv is None:
        return 0
    if ego is not None and ego["layout"] not in sel.layouts:
        print(f"  note: the ego profile is from {ego['layout']}; it is applied unchanged to {list(sel.layouts)} "
              f"(layout-to-layout speed differences are not measured)")
    total, learned, planners = expected_partners(sel.pop_size, sel.gen_seeds)
    counts = (total, learned, planners)
    if jobs_info:
        first = next(iter(jobs_info.values()))["numbers"]
        counts = (first["partners"], first["learned"], first["planners"])
    lo, hi = TARGET_GPU_HOURS
    print("\nruntime projection (arithmetic on measured profiles; process start-up, bank loading and W&B upload "
          "time are not measured):")
    if ego is not None:
        s = ego["settings"]
        print(f"  ego profile: {describe_device(ego['device'])}, batch {s['num_envs']} envs x {s['num_steps']} steps "
              f"({human(s['joint_steps_per_update'])} joint steps/update), {s['updates_per_stage']} updates/stage, "
              f"{s['eval_partners']} evaluation partners x {s['eval_episodes']} episodes; {ego['bank']['projection']}")
    if brdiv is not None:
        s = brdiv["settings"]
        print(f"  brdiv profile: {describe_device(brdiv['device'])}, {s['num_envs_xp']}+{s['num_envs_sp']} envs x "
              f"{s['num_steps']} steps ({human(s['agent_transitions_per_update'])} agent transitions/update)")
    gen = generation_seconds(sel, brdiv)
    status = 0

    def table(updates):
        t = projection_totals(sel, ego, brdiv, updates, counts)
        parts = dict(t["parts"])
        if gen:
            parts.update(gen)
        return parts, t["missing"]

    updates = None
    if jobs_info:
        updates = next(iter(jobs_info.values()))["numbers"]["updates_per_partner"]
        parts, missing = table(updates)
        _print_parts(parts, missing, ego is None or brdiv is None)
        ego_total = sum(v for k, v in parts.items() if k.startswith("ego"))
        grand = sum(parts.values())
        if not missing and ego is not None and brdiv is not None:
            verdict = ("within" if lo * 3600 <= grand <= hi * 3600 else
                       "ABOVE" if grand > hi * 3600 else "below")
            print(f"  projected total {hours(grand)} GPU time ({verdict} the {lo:g}-{hi:g} h target band; the "
                  f"projection is an estimate, not a promise)")
        if gate and ego_total > hi * 3600 and not missing:
            _print_split(sel, parts, hi)
            if not sel.accept_over_budget:
                print("refusing to start: the selected ego jobs alone are projected above the target "
                      f"({hours(ego_total)} > {hi:g} h). Lower --steps-per-partner, run one layout at a time (above), "
                      "or pass --accept-over-budget.")
                status = 1
        elif ego is not None and brdiv is not None and grand > hi * 3600:
            _print_split(sel, parts, hi)
    if (getattr(sel, "suggest_budget", False) or jobs_info is None) and ego is not None:
        _print_suggestion(sel, ego, brdiv, counts, table)
    return status


def _print_parts(parts: Dict[str, float], missing: List[str], incomplete_profiles: bool):
    total = sum(parts.values()) or 1.0
    for name, value in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<26}{hours(value):>10}  {value / total:6.1%}")
    for m in missing:
        print(f"    UNMEASURED: {m}")
    if incomplete_profiles:
        print("    (a profile for the other half of the pilot was not given; this total is partial)")


def _print_split(sel, parts, hi):
    ego_total = sum(v for k, v in parts.items() if k.startswith("ego"))
    big = sorted(parts.items(), key=lambda kv: -kv[1])[:3]
    print(f"  over the {hi:g} h target. Largest bottlenecks: " + ", ".join(f"{k} {hours(v)}" for k, v in big))
    if len(sel.layouts) > 1:
        print("  run the layouts as separate jobs, in this order (none is dropped):")
        for layout in sel.layouts:
            print(f"    scripts/cpa_pilot.sh train --root {sel.root or '$MEAL_CPA_ROOT'} --layouts {layout} "
                  f"--methods {' '.join(sel.methods)} --seeds {' '.join(map(str, sel.seeds))} --steps-per-partner "
                  f"{sel.steps_per_partner:g} --run   # projected {hours(ego_total / len(sel.layouts))} each")


def _print_suggestion(sel, ego, brdiv, counts, table):
    lo, hi = TARGET_GPU_HOURS
    unit = ego["settings"]["joint_steps_per_update"]

    def total(u):
        parts, missing = table(u)
        return sum(parts.values()), missing

    base, missing = total(1)
    print("\nsuggested budget (SUGGESTION ONLY - derived from the profile above, not a promise of the "
          f"{lo:g}-{hi:g} h target; you must choose --steps-per-partner yourself):")
    if missing:
        print("  unavailable, unmeasured: " + "; ".join(missing))
        return
    if brdiv is None:
        print("  note: no BRDiv profile, so generation time is NOT in these totals")
    if base > hi * 3600:
        print(f"  even one update per partner is projected at {hours(base)} (> {hi:g} h): the fixed costs "
              f"(generation, compilation, full-bank evaluation) already exceed the target")
        return
    def largest(limit):
        """Largest update count whose projected total is <= limit (total is nondecreasing in the budget)."""
        good, bad = 1, 2
        while total(bad)[0] <= limit and bad < 1 << 24:
            good, bad = bad, bad * 2
        while bad - good > 1:
            mid = (good + bad) // 2
            good, bad = (mid, bad) if total(mid)[0] <= limit else (good, mid)
        return good

    hi_u = largest(hi * 3600)
    lo_u = largest(lo * 3600) if base <= lo * 3600 else 1
    print(f"  largest per-partner budget projected at <= {hi:g} h: {hi_u} updates = {human(hi_u * unit)} joint steps "
          f"({human(2 * hi_u * unit)} agent transitions) per partner, projected {hours(total(hi_u)[0])}")
    if total(hi_u)[0] < lo * 3600:
        print(f"  that is still below {lo:g} h of projected work")
    elif lo_u < hi_u:
        print(f"  budgets from about {lo_u} to {hi_u} updates per partner project to {lo:g}-{hi:g} h; shorter budgets "
              f"also change what the fixed shaping/LR schedules mean:")
    print("  " + shaping_note(sel.reward_shaping_horizon, hi_u, unit))
