"""Launcher for the CPA pilot: one entry point, explicit actions, dry-run by default.

    python -m experiments.partner_adaptation.pilot <action> [options]      (or scripts/cpa_pilot.sh <action> ...)

    smoke           local CPU smoke checks, no W&B (tiny generation, bank, FT / Online EWC / Online MAS)
    preflight       GPU preflight: real CUDA device, bundled policies, tiny step + evaluation, writes, W&B login
    profile         profile the ego path (compile vs steady state, learned + planner, evaluation, importance)
    profile-brdiv   profile BRDiv generation
    plan            dry run of everything: partners, jobs, interaction units, budgets, paths, projection
    generate        generate the missing BRDiv populations (existing valid ones are reused, never regenerated)
    bank            assemble / check the partner banks
    train           train ego sequences, one job at a time (completed jobs are reused)
    resume          resume an interrupted run from its directory

`generate`, `train` and `resume` only print their plan unless `--run` is given (or RUN=1, as in scripts/_common.sh).
Everything lives under a user-selected persistent output root (`--root` or $MEAL_CPA_ROOT):

    <root>/populations/brdiv_<layout>_pop<size>_gseed<seed>/     generation outputs (never modified once complete)
    <root>/banks/<layout>_bank<N>.json                           partner banks
    <root>/runs/<layout>/<method>__<identity>/seed<seed>/<run>/  one directory per run_br invocation
    <root>/profiles/*.json                                       profile results
    <root>/logs/*.log                                            one log per launched job

The scientific settings are the ones of `run_br`/`partner_generation.run` unless an option says otherwise; every
deviation from the reference protocol is listed in the plan under "PROTOCOL CHANGE".
"""
import contextlib
import dataclasses
import io
import json
import os
import shlex
import subprocess
import sys
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import tyro
from typing_extensions import Annotated, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]

LAYOUTS = ("coord_ring", "cramped_room", "asymm_advantages", "counter_circuit")
PILOT_LAYOUTS = ("coord_ring", "cramped_room")
METHODS = {
    "ft": dict(cl_method="ft"),
    "online_ewc": dict(cl_method="ewc", importance_mode="online"),
    "online_mas": dict(cl_method="mas", importance_mode="online"),
}
PILOT_METHODS = ("ft", "online_ewc")
PILOT_GEN_SEEDS = (1001, 1002, 1003)
NUM_PLANNERS = 12
HORIZON = 400
REFERENCE_EVAL_EPISODES = 5
REFERENCE_CURRENT_EVERY = 5
TARGET_GPU_HOURS = (8.0, 13.0)
IMPORTANCE_METHODS = ("ewc", "mas", "l2")

# Generation settings that decide what a population is. Logging-only settings are ignored when an existing
# population is compared with a request.
GEN_SEMANTIC = (
    "env_name", "layout_name", "layout_difficulty", "layout_idx", "seed", "num_seeds", "partner_pop_size", "activation",
    "lr", "anneal_lr", "num_envs_xp", "num_envs_sp", "num_steps", "total_timesteps", "update_epochs",
    "num_minibatches", "gamma", "gae_lambda", "clip_eps", "ent_coef", "vf_coef", "max_grad_norm", "rew_shaping_horizon",
    "xp_loss_weights", "num_agents")


class PilotError(RuntimeError):
    """A request that cannot be honoured; reported without a traceback and exits nonzero."""


# ---------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------

def resolve_root(root: str, create: bool = False) -> Path:
    root = root or os.environ.get("MEAL_CPA_ROOT", "")
    if not root:
        raise PilotError("choose a persistent output root with --root (or $MEAL_CPA_ROOT); results written elsewhere "
                         "are lost with the machine")
    path = Path(root).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def check_layouts(layouts):
    bad = [l for l in layouts if l not in LAYOUTS]
    if bad or not layouts or len(set(layouts)) != len(layouts):
        raise PilotError(f"layouts must be distinct names from {list(LAYOUTS)}, got {list(layouts)}")


def check_methods(methods):
    bad = [m for m in methods if m not in METHODS]
    if bad or not methods or len(set(methods)) != len(methods):
        raise PilotError(f"methods must be distinct names from {sorted(METHODS)}, got {list(methods)}")


def expected_partners(pop_size: int, gen_seeds) -> Tuple[int, int, int]:
    """(total, learned, planners) of a bank: bundled population + one population per generation seed + planners."""
    learned = pop_size * (1 + len(gen_seeds))
    return learned + NUM_PLANNERS, learned, NUM_PLANNERS


def cli_args(cls, overrides: Dict[str, Any]) -> List[str]:
    """Tyro command-line arguments that make `cls` take the values in `overrides` (None / empty are omitted)."""
    hints = typing.get_type_hints(cls)
    out = []
    for name, value in overrides.items():
        flag = "--" + name.replace("_", "-")
        if value is None or (isinstance(value, (list, tuple)) and not value):
            continue
        if isinstance(value, bool):
            out.append(flag if value else "--no-" + name.replace("_", "-"))
        elif isinstance(value, (list, tuple)):
            out += [flag, *map(str, value)]
        else:
            hint = hints.get(name)
            out += [flag, str(int(value)) if hint is int else repr(float(value)) if hint is float else str(value)]
    return out


def python_module_command(module: str, args: List[str]) -> List[str]:
    return [sys.executable, "-m", module, *args]


def run_logged(command: List[str], log_path: Path) -> int:
    """Run `command` from the repository root, echoing its output into `log_path`; returns its exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


@contextlib.contextmanager
def quiet():
    """Silence the progress prints of config construction while planning."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def human(n) -> str:
    return f"{int(n):,}" if float(n) == int(n) else f"{n:,.3g}"


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f} h"


# ---------------------------------------------------------------------------------------------------------
# populations
# ---------------------------------------------------------------------------------------------------------

def generation_config(layout: str, seed: int, root: Path, pop_size: int, total_timesteps: Optional[float],
                      mode: str, **extra):
    from experiments.partner_adaptation.partner_generation.run import TrainConfig as GenConfig
    kwargs = dict(layout_name=layout, seed=seed, num_seeds=1, partner_pop_size=pop_size,
                  checkpoint_path=str(root / "populations"), mode=mode, **extra)
    if total_timesteps is not None:
        kwargs["total_timesteps"] = int(total_timesteps)
    with quiet():
        return GenConfig(**kwargs)


def population_dir(cfg) -> Path:
    from experiments.partner_adaptation.partner_generation.run import resolve_save_dir
    return Path(resolve_save_dir(cfg, ""))


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return float(a) == float(b)
    return a == b


def population_state(cfg) -> Tuple[str, str]:
    """("missing"|"valid"|"invalid"|"conflict", detail) for the population `cfg` would generate.

    Valid means: generation.json says complete, lists members, every member file exists, and the settings that
    define the population equal the requested ones. Nothing is read from the checkpoints themselves here (the
    bank assembly validates those).
    """
    d = population_dir(cfg)
    if not d.exists():
        return "missing", str(d)
    gen_path = d / "generation.json"
    if not gen_path.is_file():
        return "invalid", f"{d} exists without generation.json"
    try:
        gen = json.loads(gen_path.read_text())
    except ValueError as e:
        return "invalid", f"{gen_path} is unreadable ({e})"
    if gen.get("status") != "complete":
        return "invalid", f"{d}: generation status is {gen.get('status')!r} (interrupted); it is left untouched"
    members = gen.get("members") or []
    missing = [m["file"] for m in members if not (d / m["file"]).is_file()]
    if not members or missing or len(members) != cfg.num_seeds * cfg.partner_pop_size:
        return "invalid", f"{d}: expected {cfg.num_seeds * cfg.partner_pop_size} member files, missing {missing}"
    requested = asdict(cfg)
    diffs = [f"{k}: existing {gen.get(k)!r}, requested {requested[k]!r}" for k in GEN_SEMANTIC
             if k in requested and not _same(gen.get(k), requested[k])]
    if diffs:
        return "conflict", f"{d} was generated with different settings: " + "; ".join(diffs)
    return "valid", str(d)


# ---------------------------------------------------------------------------------------------------------
# banks
# ---------------------------------------------------------------------------------------------------------

def bank_path(root: Path, layout: str, total: int) -> Path:
    return root / "banks" / f"{layout}_bank{total}.json"


def assemble_checked_bank(root: Path, layout: str, pop_size: int, gen_seeds, mode: str,
                          total_timesteps: Optional[float]):
    """Assemble the bank in a temporary file next to its final place; returns (final path, temp path, bank)."""
    from experiments.partner_adaptation.partner_bank import assemble_bank
    total, learned, planners = expected_partners(pop_size, gen_seeds)
    dirs = []
    for seed in gen_seeds:
        state, detail = population_state(generation_config(layout, seed, root, pop_size, total_timesteps, mode))
        if state != "valid":
            raise PilotError(f"{layout}: population for generation seed {seed} is {state} ({detail}); "
                             f"run `generate` first")
        dirs.append(detail)
    final = bank_path(root, layout, total)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.assembling.json")
    bank = assemble_bank(layout, str(tmp), dirs, name=f"{layout}_bank{total}", bundled=True, planners=("all",))
    kinds = [r.kind for r in bank.records]
    if (len(bank), kinds.count("brdiv"), kinds.count("planner")) != (total, learned, planners):
        tmp.unlink(missing_ok=True)
        raise PilotError(f"{layout}: assembled {len(bank)} partners ({kinds.count('brdiv')} learned, "
                         f"{kinds.count('planner')} planners), expected {total} ({learned}, {planners})")
    return final, tmp, bank


# ---------------------------------------------------------------------------------------------------------
# ego training jobs
# ---------------------------------------------------------------------------------------------------------

def identity_tag(use_task_id: bool, use_multihead: bool) -> str:
    return ("id" if use_task_id else "noid") + "-" + ("mh" if use_multihead else "1h")


def resolve_identity(identity: str, use_task_id: Optional[bool], use_multihead: Optional[bool]):
    """`hidden` removes both the identity input and the oracle head routing (a single head); the two controls
    stay independently overridable. Returns (use_task_id, use_multihead, notes)."""
    base = identity == "available"
    task_id = base if use_task_id is None else use_task_id
    multihead = base if use_multihead is None else use_multihead
    notes = []
    if not task_id and multihead:
        notes.append("use_task_id=False but use_multihead=True: the stage index still selects the head, so the "
                     "identity is NOT hidden from the network")
    if task_id and not multihead:
        notes.append("use_task_id=True with a single head: identity is an input only (no oracle head routing)")
    return task_id, multihead, notes


@dataclass
class Job:
    layout: str
    method: str
    seed: int
    tag: str
    root: Path                     # checkpoint_path of the run_br invocation; run directories are created inside
    overrides: Dict[str, Any]
    notes: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.layout}/{self.method}__{self.tag}/seed{self.seed}"


def shaping_note(horizon: float, updates: int, steps_per_update: int) -> str:
    from experiments.partner_adaptation.train_ego import shaping_coefficient
    end = shaping_coefficient(horizon, updates, steps_per_update)
    per_partner = updates * steps_per_update
    if horizon <= 0:
        return "reward shaping disabled (horizon 0)"
    if per_partner >= horizon:
        done = f"fully annealed after {horizon / per_partner:.0%} of each partner's budget"
    else:
        done = f"never fully annealed within a partner (weight {end:.3f} at the last rollout)"
    return (f"reward shaping: weight 1 -> 0 linearly over {human(horizon)} joint steps, restarting at EVERY partner; "
            f"per-partner budget {human(per_partner)} joint steps -> {done}")


def job_numbers(config, partners: int, learned: int, planners: int) -> Dict[str, Any]:
    from experiments.partner_adaptation.train_ego import eval_plan
    spu = config.num_envs * config.num_steps
    updates = int(config.num_updates)
    kinds = list(eval_plan(config).values())
    pilot = config.eval_schedule == "pilot"
    full = partners * kinds.count("full") + (1 if pilot else 0)
    current = partners * kinds.count("current")
    rows = full * partners * config.num_eval_episodes + current * config.num_eval_episodes
    importance = (partners * config.importance_episodes * config.importance_steps
                  if config.cl_method and config.cl_method.lower() in IMPORTANCE_METHODS else 0)
    return dict(
        partners=partners, learned=learned, planners=planners, updates_per_partner=updates,
        joint_steps_per_update=spu, joint_steps_per_partner=updates * spu, agent_transitions_per_partner=2 * updates * spu,
        train_joint_steps=partners * updates * spu, eval_full_events=full, eval_current_events=current,
        eval_episodes=rows, eval_joint_steps=rows * config.num_steps, importance_joint_steps=importance)


def protocol_changes(config) -> List[str]:
    changes = []
    if config.num_steps != HORIZON:
        changes.append(f"episode horizon {config.num_steps} (reference {HORIZON})")
    if config.num_eval_episodes != REFERENCE_EVAL_EPISODES:
        changes.append(f"{config.num_eval_episodes} evaluation episodes (reference {REFERENCE_EVAL_EPISODES})")
    if config.eval_current_every != REFERENCE_CURRENT_EVERY:
        changes.append(f"current-partner evaluation every {config.eval_current_every} updates "
                       f"(reference {REFERENCE_CURRENT_EVERY})")
    return changes


def inspect_runs(job: Job, fingerprint: Dict[str, Any], partners: int) -> Tuple[str, Optional[Path], str]:
    """("new"|"complete"|"resumable", run dir, detail) from the run directories of this job that were produced
    by exactly this configuration and partner list; other directories are ignored."""
    from experiments.partner_adaptation.run_outputs import normalise
    found = []
    for d in sorted(p for p in job.root.glob("*") if p.is_dir()) if job.root.exists() else []:
        try:
            meta = json.loads((d / "run.json").read_text())
            status = json.loads((d / "status.json").read_text())
        except (OSError, ValueError):
            continue
        if normalise(meta.get("fingerprint")) != normalise(fingerprint):
            continue
        done = (status.get("state") == "complete" and status.get("stages_completed") == partners
                and any(d.glob("params_seed*.pt")))
        found.append((done, d, status))
    for done, d, status in found:
        if done:
            return "complete", d, f"complete, {status['stages_completed']}/{partners} partners"
    if found:
        _, d, status = found[-1]
        return ("resumable", d, f"{status.get('state')}, {status.get('stages_completed', 0)}/{partners} partners "
                                f"done; use `resume {d}`")
    return "new", None, ""


# ---------------------------------------------------------------------------------------------------------
# command definitions
# ---------------------------------------------------------------------------------------------------------

def default_run() -> bool:
    return os.environ.get("RUN") == "1"


@dataclass
class Selection:
    root: str = ""                                          # persistent output root, or $MEAL_CPA_ROOT
    layouts: Tuple[str, ...] = PILOT_LAYOUTS                # pilot layouts; all four layouts are supported
    gen_seeds: Tuple[int, ...] = PILOT_GEN_SEEDS            # generation seeds: one population each (not ego seeds)
    pop_size: int = 3                                       # members per generated population
    gen_total_timesteps: Optional[float] = None             # agent transitions per population; unset = BRDiv default
    gen_mode: str = "online"                                # W&B mode of the generation jobs


@dataclass
class TrainSelection(Selection):
    methods: Tuple[str, ...] = PILOT_METHODS                # ft, online_ewc, online_mas
    seeds: Tuple[int, ...] = (0,)                           # ego seeds: list several explicitly, never implied
    identity: Literal["available", "hidden"] = "available"  # hidden = no identity input AND no oracle head routing
    use_task_id: Optional[bool] = None                      # override the identity input alone
    use_multihead: Optional[bool] = None                    # override the oracle head routing alone
    steps_per_partner: Optional[float] = None               # joint env steps per partner; no default (see `plan`)
    num_envs: int = 2048
    num_steps: int = HORIZON
    update_epochs: int = 8
    num_minibatches: int = 16
    reward_shaping_horizon: float = 2.5e7
    anneal_lr: bool = False
    eval_episodes: int = REFERENCE_EVAL_EPISODES            # per partner and event
    eval_current_every: int = REFERENCE_CURRENT_EVERY       # updates between current-partner evaluations
    importance_episodes: int = 5
    importance_steps: int = 500
    wandb_mode: str = "online"
    wandb_project: str = "MEAL"
    wandb_group: str = "cpa-pilot"
    allow_protocol_change: bool = False                     # accept a horizon other than 400 (labelled in the plan)
    profile: Optional[str] = None                           # ego profile json (from `profile`) for a projection
    brdiv_profile: Optional[str] = None                     # BRDiv profile json (from `profile-brdiv`)
    accept_over_budget: bool = False                        # run even if the projection exceeds the target hours


def make_jobs(sel: TrainSelection, root: Path) -> List[Job]:
    check_layouts(sel.layouts)
    check_methods(sel.methods)
    if not sel.seeds or len(set(sel.seeds)) != len(sel.seeds):
        raise PilotError("seeds must be distinct integers")
    task_id, multihead, notes = resolve_identity(sel.identity, sel.use_task_id, sel.use_multihead)
    if sel.num_steps != HORIZON and not sel.allow_protocol_change:
        raise PilotError(f"the experiment uses episode horizon {HORIZON}; num_steps={sel.num_steps} is a protocol "
                         f"change (pass --allow-protocol-change to accept it)")
    total, _, _ = expected_partners(sel.pop_size, sel.gen_seeds)
    jobs = []
    for layout in sel.layouts:
        for method in sel.methods:
            for seed in sel.seeds:
                tag = identity_tag(task_id, multihead)
                overrides = dict(
                    layout_name=layout, partner_bank=str(bank_path(root, layout, total)), num_heuristic_partners=0,
                    seed=seed, checkpoint_path=str(root / "runs" / layout / f"{method}__{tag}" / f"seed{seed}"),
                    mode=sel.wandb_mode, project=sel.wandb_project, group=sel.wandb_group,
                    tags=["PARTNER_ADAPTATION", "CPA_PILOT", layout, method],
                    total_timesteps=sel.steps_per_partner, num_envs=sel.num_envs, num_steps=sel.num_steps,
                    update_epochs=sel.update_epochs, num_minibatches=sel.num_minibatches,
                    reward_shaping_horizon=sel.reward_shaping_horizon, anneal_lr=sel.anneal_lr,
                    num_eval_episodes=sel.eval_episodes, eval_schedule="pilot",
                    eval_current_every=sel.eval_current_every, use_task_id=task_id, use_multihead=multihead,
                    importance_episodes=sel.importance_episodes, importance_steps=sel.importance_steps,
                    **METHODS[method])
                jobs.append(Job(layout, method, seed, tag, Path(overrides["checkpoint_path"]), overrides, notes))
    return jobs


def build_config(job: Job, budget: Optional[float]):
    """The validated TrainConfig a job runs with; the budget is required to build one."""
    from experiments.partner_adaptation.run_br import TrainConfig, validate_config
    overrides = dict(job.overrides, total_timesteps=budget)
    with quiet():
        config = TrainConfig(**overrides)
    validate_config(config)
    return config


def check_budget(sel: TrainSelection):
    if sel.steps_per_partner is None:
        raise PilotError(
            "--steps-per-partner is required: there is no default budget. Profile first (`profile`, `profile-brdiv`) "
            "and read the suggestion of `plan --profile <ego.json> --brdiv-profile <brdiv.json>`, then choose.")
    update = sel.num_envs * sel.num_steps
    if sel.steps_per_partner < update:
        raise PilotError(f"steps-per-partner={sel.steps_per_partner:g} is less than one update ({update} joint "
                         f"environment steps = num_envs x num_steps): no training would happen")
    if sel.steps_per_partner % update:
        raise PilotError(f"steps-per-partner={sel.steps_per_partner:g} is not a multiple of one update ({update}); "
                         f"the trained amount would silently differ between settings")


def describe_jobs(sel: TrainSelection, root: Path, jobs: List[Job], out=print) -> Dict[str, Any]:
    """Print the dry-run plan of the training jobs; returns {job.name: info} (info has the numbers and the status)."""
    from experiments.partner_adaptation.run_br import resolve_run
    info = {}
    total_expected, learned_expected, planners_expected = expected_partners(sel.pop_size, sel.gen_seeds)
    out(f"ego training jobs ({len(jobs)}), run one after another on one GPU; equal ego budget for every method:")
    for job in jobs:
        config = build_config(job, sel.steps_per_partner)
        partners, learned, planners, fingerprint, state = total_expected, learned_expected, planners_expected, None, ""
        manifest = Path(job.overrides["partner_bank"])
        if manifest.is_file():
            with quiet():
                resolved = resolve_run(config)
            kinds = [i["kind"] for i in resolved.identities]
            partners, learned, planners = len(kinds), kinds.count("brdiv"), kinds.count("planner")
            fingerprint = resolved.fingerprint
            status, run_dir, detail = inspect_runs(job, fingerprint, partners)
            state = {"new": "NEW", "complete": f"REUSE {run_dir}  ({detail})",
                     "resumable": f"INCOMPLETE {run_dir}  ({detail})"}[status]
        else:
            status, run_dir, state = "bank_missing", None, f"bank missing: {manifest} (run `bank`)"
        n = job_numbers(config, partners, learned, planners)
        info[job.name] = dict(job=job, config=config, numbers=n, status=status, run_dir=run_dir,
                              fingerprint=fingerprint)
        out(f"- {job.name}   [{state}]")
        out(f"    partners {partners} ({learned} learned + {planners} planners), one stage each; cl_method="
            f"{config.cl_method}, identity input={config.use_task_id}, multi-head routing={config.use_multihead}")
        out(f"    budget per partner {human(n['joint_steps_per_partner'])} joint env steps = "
            f"{human(n['agent_transitions_per_partner'])} agent transitions ({n['updates_per_partner']} updates of "
            f"{human(n['joint_steps_per_update'])}); training total {human(n['train_joint_steps'])} joint steps")
        out(f"    evaluation: {n['eval_full_events']} full-bank events (initialization + after each stage) and "
            f"{n['eval_current_events']} current-partner events, {config.num_eval_episodes} episodes each = "
            f"{human(n['eval_episodes'])} episodes, {human(n['eval_joint_steps'])} joint steps (not training)")
        if n["importance_joint_steps"]:
            out(f"    importance rollouts: {human(n['importance_joint_steps'])} joint steps (not training)")
        out(f"    {shaping_note(config.reward_shaping_horizon, n['updates_per_partner'], n['joint_steps_per_update'])}")
        out(f"    learning rate {config.lr:g}, " + ("annealed linearly to 0 within each partner" if config.anneal_lr
                                                      else "constant (no annealing)"))
        out(f"    output: {job.root}/<run directory>")
        for note in job.notes:
            out(f"    NOTE: {note}")
        for change in protocol_changes(config):
            out(f"    PROTOCOL CHANGE: {change}")
    return info


def require_banks(jobs: List[Job]):
    missing = sorted({j.overrides["partner_bank"] for j in jobs if not Path(j.overrides["partner_bank"]).is_file()})
    if missing:
        raise PilotError("partner bank(s) missing, run `bank` first:\n  " + "\n  ".join(missing))


def run_jobs(sel: TrainSelection, root: Path, info, keep_going: bool, restart: bool, run: bool) -> int:
    """Launch the NEW jobs one by one; complete jobs are skipped; an incomplete one needs `resume` (or --restart)."""
    failures, blocked = 0, []
    for name, item in info.items():
        job = item["job"]
        if item["status"] == "complete":
            print(f"[skip] {name}: already complete ({item['run_dir']})")
            continue
        if item["status"] == "resumable" and not restart:
            blocked.append(name)
            print(f"[blocked] {name}: an incomplete run of this exact configuration exists: {item['run_dir']}\n"
                  f"          resume it with `resume {item['run_dir']}` or start a new one with --restart")
            continue
        command = python_module_command("experiments.partner_adaptation.run_br", cli_args(
            type(item["config"]), job.overrides))
        print(f"[{'run' if run else 'dry-run'}] {name}\n    {shlex.join(command)}")
        if not run:
            continue
        code = run_logged(command, root / "logs" / f"{job.layout}_{job.method}_{job.tag}_seed{job.seed}.log")
        if code:
            failures += 1
            print(f"[failed] {name}: exit code {code}")
            if not keep_going:
                return 1
    if blocked:
        return 1
    return 1 if failures else 0


@dataclass
class Smoke:
    """Local CPU smoke checks without W&B: tiny BRDiv generation, bank assembly (bundled + generated + planner)
    and FT / Online EWC / Online MAS ego runs with the pilot evaluation schedule, then a no-op resume."""
    root: str = ""                                          # default: a temporary directory
    layouts: Tuple[str, ...] = PILOT_LAYOUTS
    methods: Tuple[str, ...] = ("ft", "online_ewc", "online_mas")
    generation: bool = True                                 # --no-generation skips the tiny BRDiv run

    def run(self) -> int:
        from experiments.partner_adaptation.pilot_gpu import smoke
        return smoke(self)


@dataclass
class Preflight:
    """GPU preflight (short, a few partners): verifies a real CUDA device, loads the bundled policies, runs a tiny
    optimizer step and evaluation against a learned partner and a planner, checks checkpoint/metric writes and
    W&B authentication. It never starts a full job."""
    root: str = ""
    layout: str = "coord_ring"
    wandb_mode: str = "online"          # online verifies the login; offline/disabled skip it (and say so)
    allow_cpu: bool = False             # run the checks on CPU for a dry rehearsal; NOT a GPU verification
    num_envs: int = 16

    def run(self) -> int:
        from experiments.partner_adaptation.pilot_gpu import preflight
        return preflight(self)


@dataclass
class Profile:
    """Profile the ego path at the real batch settings for a handful of updates: compilation separately from
    synchronized steady-state work, learned and planner partners, evaluation against the whole bank and importance
    rollouts per method. Writes <root>/profiles/ego_*.json; never trains a full sequence."""
    root: str = ""
    layout: str = "coord_ring"
    bank: str = ""                      # an assembled bank; empty = bundled population + 2 planners (provisional)
    methods: Tuple[str, ...] = PILOT_METHODS
    updates: int = 6                    # updates per profiled stage (at most 50)
    num_envs: int = 2048
    num_steps: int = HORIZON
    update_epochs: int = 8
    num_minibatches: int = 16
    eval_episodes: int = REFERENCE_EVAL_EPISODES
    eval_current_every: int = REFERENCE_CURRENT_EVERY
    pop_size: int = 3
    gen_seeds: Tuple[int, ...] = PILOT_GEN_SEEDS            # defines the size of the bank projected to
    allow_cpu: bool = False
    allow_protocol_change: bool = False

    def run(self) -> int:
        from experiments.partner_adaptation.pilot_gpu import profile_ego
        return profile_ego(self)


@dataclass
class ProfileBrdiv:
    """Profile BRDiv generation separately (two short runs: compile and per-update time, fixed overhead)."""
    root: str = ""
    layout: str = "coord_ring"
    updates: Tuple[int, int] = (2, 4)   # two update counts; the difference gives the per-update time
    num_envs_xp: int = 32
    num_envs_sp: int = 32
    num_steps: int = HORIZON
    partner_pop_size: int = 3
    allow_cpu: bool = False

    def run(self) -> int:
        from experiments.partner_adaptation.pilot_gpu import profile_brdiv
        return profile_brdiv(self)


@dataclass
class Plan(TrainSelection):
    """Dry-run plan of the whole pilot: populations, banks, jobs, budgets, interaction units, paths and, with
    profiles, a runtime projection and a clearly labelled suggested budget."""
    suggest_budget: bool = False

    def run(self) -> int:
        return plan(self)


@dataclass
class Generate(Selection):
    """Generate the missing populations (one BRDiv job per layout and generation seed). Valid existing ones are
    reused; incomplete or differently configured ones stop the command and are never touched."""
    run: bool = field(default_factory=default_run)
    keep_going: bool = False

    def execute(self) -> int:
        return generate(self)


@dataclass
class Bank(Selection):
    """Assemble the bank of every layout (bundled + generated populations + the 12 planners) and check it."""
    check_only: bool = False            # validate existing manifests only
    overwrite: bool = False             # replace a manifest that differs from what would be assembled

    def run(self) -> int:
        return bank(self)


@dataclass
class Train(TrainSelection):
    """Train the selected layouts x methods x seeds sequentially. Needs --steps-per-partner."""
    run: bool = field(default_factory=default_run)
    keep_going: bool = False            # continue with the next job after a failure
    restart: bool = False               # start a new run even if an incomplete one of this configuration exists

    def execute(self) -> int:
        return train(self)


@dataclass
class Resume:
    """Resume an interrupted run from its directory (everything else is read from the run's saved configuration)."""
    run_dir: tyro.conf.Positional[str]
    run: bool = field(default_factory=default_run)
    wandb_mode: Optional[str] = None    # change only the W&B mode of the resumed attempt

    def execute(self) -> int:
        return resume(self)


# ---------------------------------------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------------------------------------

def generation_jobs(sel: Selection, root: Path):
    check_layouts(sel.layouts)
    jobs = []
    for layout in sel.layouts:
        for seed in sel.gen_seeds:
            cfg = generation_config(layout, seed, root, sel.pop_size, sel.gen_total_timesteps, sel.gen_mode)
            jobs.append((layout, seed, cfg, *population_state(cfg)))
    return jobs


def describe_generation(sel: Selection, root: Path, out=print):
    from experiments.partner_adaptation.partner_generation.run import interaction_counts
    jobs = generation_jobs(sel, root)
    out(f"BRDiv generation: {len(jobs)} job(s), population size {sel.pop_size}, one per layout and generation seed")
    for layout, seed, cfg, state, detail in jobs:
        ic = interaction_counts(cfg)
        budget = ("repository default" if sel.gen_total_timesteps is None else "selected")
        out(f"- {layout} gseed{seed}: {state.upper()}  {detail if state != 'valid' else population_dir(cfg)}")
        out(f"    budget ({budget}) {human(ic['requested_agent_transitions_per_seed'])} agent transitions = "
            f"{human(ic['joint_env_transitions_per_seed'])} joint steps, {ic['num_updates']} updates of "
            f"{human(ic['joint_env_transitions_per_update'])} joint steps (evaluation episodes not counted)")
    return jobs


def generate(sel: Generate) -> int:
    root = resolve_root(sel.root, create=sel.run)
    jobs = describe_generation(sel, root)
    bad = [(l, s, st, d) for l, s, _, st, d in jobs if st in ("invalid", "conflict")]
    if bad:
        for l, s, st, d in bad:
            print(f"[error] {l} gseed{s}: {st}: {d}")
        print("existing populations are never modified or regenerated; fix the request or choose another root")
        return 1
    failures = 0
    from experiments.partner_adaptation.partner_generation.run import TrainConfig as GenConfig
    for layout, seed, cfg, state, detail in jobs:
        if state == "valid":
            print(f"[skip] {layout} gseed{seed}: valid population exists")
            continue
        overrides = dict(layout_name=layout, seed=seed, num_seeds=1, partner_pop_size=sel.pop_size,
                         checkpoint_path=str(root / "populations"), mode=sel.gen_mode,
                         total_timesteps=sel.gen_total_timesteps)
        command = python_module_command("experiments.partner_adaptation.partner_generation.run",
                                        cli_args(GenConfig, overrides))
        print(f"[{'run' if sel.run else 'dry-run'}] {layout} gseed{seed}\n    {shlex.join(command)}")
        if not sel.run:
            continue
        code = run_logged(command, root / "logs" / f"generate_{layout}_gseed{seed}.log")
        after = population_state(cfg)[0] if code == 0 else "failed"
        if code or after != "valid":
            failures += 1
            print(f"[failed] {layout} gseed{seed}: exit code {code}, population {after}")
            if not sel.keep_going:
                return 1
    return 1 if failures else 0


def bank(sel: Bank) -> int:
    root = resolve_root(sel.root, create=not sel.check_only)
    check_layouts(sel.layouts)
    status = 0
    for layout in sel.layouts:
        try:
            final, tmp, assembled = assemble_checked_bank(root, layout, sel.pop_size, sel.gen_seeds, sel.gen_mode,
                                                          sel.gen_total_timesteps)
        except PilotError as e:
            print(f"[error] {e}")
            status = 1
            continue
        try:
            same = final.is_file() and final.read_bytes() == tmp.read_bytes()
            if same:
                state = "valid, unchanged"
            elif final.is_file() and not sel.overwrite:
                print(f"[error] {layout}: {final} differs from what the current populations and planners would give; "
                      f"pass --overwrite to replace it")
                status = 1
                continue
            elif sel.check_only:
                print(f"[error] {layout}: {final} is missing (run `bank` without --check-only)")
                status = 1
                continue
            else:
                os.replace(tmp, final)
                state = "written"
            kinds = [r.kind for r in assembled.records]
            pops = sorted({r.population for r in assembled.records if r.kind == "brdiv"})
            print(f"[ok] {layout}: {final} {state}: {len(assembled)} partners = {kinds.count('brdiv')} learned "
                  f"({len(pops)} populations: {', '.join(pops)}) + {kinds.count('planner')} planners")
        finally:
            tmp.unlink(missing_ok=True)
    return status


def plan(sel: Plan) -> int:
    root = resolve_root(sel.root)
    check_layouts(sel.layouts)
    check_methods(sel.methods)
    if sel.steps_per_partner is not None:
        check_budget(sel)
    jobs = make_jobs(sel, root)
    print(f"output root: {root}")
    describe_generation(sel, root)
    total, learned, planners = expected_partners(sel.pop_size, sel.gen_seeds)
    print(f"\nbanks: {len(sel.layouts)} x {total} partners = {learned} learned (bundled population + "
          f"{len(sel.gen_seeds)} generated x {sel.pop_size}) + {planners} planners")
    for layout in sel.layouts:
        path = bank_path(root, layout, total)
        print(f"- {layout}: {path} [{'present' if path.is_file() else 'not yet assembled'}]")
    print()
    if sel.steps_per_partner is None:
        print(f"ego training: {len(jobs)} job(s), one after another, budget UNSET (choose --steps-per-partner after "
              f"profiling; `plan --profile ... --brdiv-profile ...` prints a labelled suggestion):")
        for job in jobs:
            print(f"- {job.name}: identity input={job.overrides['use_task_id']}, multi-head routing="
                  f"{job.overrides['use_multihead']}, output {job.root}/<run directory>")
            for note in job.notes:
                print(f"    NOTE: {note}")
        jobs_info = None
    else:
        jobs_info = describe_jobs(sel, root, jobs)
    from experiments.partner_adaptation.pilot_gpu import report_projection
    return report_projection(sel, root, jobs_info)


def train(sel: Train) -> int:
    root = resolve_root(sel.root, create=sel.run)
    check_budget(sel)
    jobs = make_jobs(sel, root)
    info = describe_jobs(sel, root, jobs)
    require_banks(jobs)
    from experiments.partner_adaptation.pilot_gpu import report_projection
    status = report_projection(sel, root, info, gate=sel.run)
    if status:
        return status
    return run_jobs(sel, root, info, sel.keep_going, sel.restart, sel.run)


def resume(sel: Resume) -> int:
    from experiments.partner_adaptation.run_br import TrainConfig, resolve_run
    from experiments.partner_adaptation.run_outputs import check_resume, read_checkpoint, resolve_resume
    run_dir, ckpt = resolve_resume(sel.run_dir)
    saved = json.loads((run_dir / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    overrides = {k: v for k, v in saved.items() if k in fields and k not in ("layouts_path", "save_dir", "resume")}
    overrides["mode"] = sel.wandb_mode or overrides["mode"]
    overrides["resume"] = str(run_dir)
    with quiet():
        config = TrainConfig(**{k: v for k, v in overrides.items()})
        resolved = resolve_run(config)
    meta, _ = read_checkpoint(ckpt)
    check_resume(meta, resolved.fingerprint)
    total = len(resolved.identities)
    print(f"resume {run_dir}: {meta['stages_completed']}/{total} partners done; the interrupted partner restarts "
          f"from its beginning (no mid-stage resume); W&B mode {overrides['mode']}")
    command = python_module_command("experiments.partner_adaptation.run_br", cli_args(TrainConfig, overrides))
    print(f"[{'run' if sel.run else 'dry-run'}]\n    {shlex.join(command)}")
    if not sel.run:
        return 0
    code = run_logged(command, run_dir / "resume.log")
    return 1 if code else 0


Command = Union[
    Annotated[Smoke, tyro.conf.subcommand("smoke")],
    Annotated[Preflight, tyro.conf.subcommand("preflight")],
    Annotated[Profile, tyro.conf.subcommand("profile")],
    Annotated[ProfileBrdiv, tyro.conf.subcommand("profile-brdiv")],
    Annotated[Plan, tyro.conf.subcommand("plan")],
    Annotated[Generate, tyro.conf.subcommand("generate")],
    Annotated[Bank, tyro.conf.subcommand("bank")],
    Annotated[Train, tyro.conf.subcommand("train")],
    Annotated[Resume, tyro.conf.subcommand("resume")],
]


def main(argv: Optional[List[str]] = None) -> int:
    command = tyro.cli(Command, args=argv, description=__doc__)
    from experiments.partner_adaptation.run_outputs import RunOutputError
    try:
        run = getattr(command, "execute", None) or command.run
        return int(run())
    except (PilotError, RunOutputError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    # Re-enter through the package so pilot_gpu and this module share one set of classes (PilotError, ...).
    from experiments.partner_adaptation.pilot import main as _main
    sys.exit(_main())
