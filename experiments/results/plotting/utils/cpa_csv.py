"""Read the local run directories written by `experiments/partner_adaptation/run_br.py` for the CPA plots.

Shared by `cumulative_eval_partners.py` and `eval_per_task_partners.py` (CSV mode, selected with `--run_root`).
Only the standard library and numpy are used: no JAX, no W&B, so figures can be made on a laptop. Nothing under a
run directory is written to.

A run is selected from its *saved metadata* (config.json, run.json, status.json, partner_bank.json), never from the
folder name; its full-bank evaluations come from eval_metrics.csv (`scope == "full"`, completed episodes only).
Anything that cannot be read unambiguously raises `CpaCsvError`; nothing is guessed, padded or averaged over.
"""
import csv
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Plot name -> (saved cl_method, saved importance_mode that must match; None = not part of the identity)
CSV_METHODS = {"FT": ("ft", None), "Online_EWC": ("ewc", "online"), "Online_MAS": ("mas", "online")}
DEFAULT_CSV_PARTNERS = 24
DEFAULT_CSV_FORMATS = ["png"]


class CpaCsvError(RuntimeError):
    """The run directories do not unambiguously provide the requested data."""


# ---------------------------------------------------------------------------------------------------------
# command line shared by both scripts
# ---------------------------------------------------------------------------------------------------------

def add_csv_args(parser) -> None:
    """Additive flags of both CPA plotting scripts (`--run_root` switches to CSV mode)."""
    parser.add_argument("--run_root", default=None,
                        help="CSV mode: the runs directory written by run_br.py (run directories are found below it)")
    parser.add_argument("--run_dirs", nargs="+", default=None,
                        help="CSV mode: explicit run directories to choose from (required when several completed "
                             "runs match a selection)")
    parser.add_argument("--identity", choices=["available", "hidden"], default="available",
                        help="CSV mode: partner-identity input of the runs (saved use_task_id); the head structure "
                             "is --arch (saved use_multihead)")
    parser.add_argument("--output_dir", default=None, help="Where to save figures (default: experiments/results/plots)")
    parser.add_argument("--no_show", action="store_true", help="Save only; do not open a window (headless backend)")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf"], default=None,
                        help="Figure formats (default: png, pdf in legacy mode; png in CSV mode)")


def prepare_output(args, default_dir: Path, default_formats: Sequence[str]) -> Tuple[Path, List[str]]:
    """Output directory and formats; `--no_show` selects the Agg backend (call before any figure exists).

    A directory given with `--output_dir` is created with its parents; the default one is left to `save_plot`.
    """
    if args.no_show:
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
    out_dir = Path(args.output_dir) if args.output_dir else Path(default_dir)
    if args.output_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, list(args.formats or default_formats)


# ---------------------------------------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------------------------------------

@dataclass
class CpaRun:
    path: Path
    layout: str
    cl_method: str
    importance_mode: Optional[str]
    seed: int
    use_task_id: bool
    use_multihead: bool
    partners: List[Dict[str, Any]]          # partner_bank.json, in training order (index == partner_id)
    state: str
    stages_completed: int
    env_steps_per_stage: int
    episode_steps: int
    eval_episodes: int
    planned_full_events: Optional[int]

    @property
    def num_partners(self) -> int:
        return len(self.partners)

    @property
    def is_complete(self) -> bool:
        return self.state == "complete" and self.stages_completed == self.num_partners

    def describe(self) -> str:
        return (f"{self.path}: {self.layout} cl_method={self.cl_method}/{self.importance_mode} seed={self.seed} "
                f"use_task_id={self.use_task_id} use_multihead={self.use_multihead} partners={self.num_partners} "
                f"state={self.state} ({self.stages_completed} stages)")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise CpaCsvError(f"cannot read {path}: {e}") from e


def _flag(name: str, config: Dict[str, Any], run: Dict[str, Any], path: Path) -> bool:
    """A boolean architecture flag from config.json, else run.json (`num_heads` is deliberately not used)."""
    for source in (config, run.get("architecture", {})):
        if isinstance(source.get(name), bool):
            return source[name]
    raise CpaCsvError(f"{path}: no saved boolean `{name}`; the head/identity condition cannot be determined")


def load_run(path) -> CpaRun:
    """Metadata of one run directory (no CSV is read)."""
    path = Path(path)
    config, run = _read_json(path / "config.json"), _read_json(path / "run.json")
    status, bank = _read_json(path / "status.json"), _read_json(path / "partner_bank.json")
    budget = run.get("budget", {})
    missing = [k for k in ("env_steps_per_stage", "num_steps", "eval_episodes_per_partner") if k not in budget]
    if missing:
        raise CpaCsvError(f"{path / 'run.json'}: budget lacks {missing}")
    if not isinstance(bank, list) or [p.get("partner_id") for p in bank] != list(range(len(bank))):
        raise CpaCsvError(f"{path / 'partner_bank.json'}: partners are not listed as ids 0..N-1 in order")
    if run.get("partners") is not None and len(run["partners"]) != len(bank):
        raise CpaCsvError(f"{path}: run.json and partner_bank.json disagree on the number of partners")
    layout = config.get("layout_name_resolved") or config.get("layout_name")
    if not layout or "seed" not in config or "cl_method" not in config:
        raise CpaCsvError(f"{path / 'config.json'}: layout, seed or cl_method missing")
    return CpaRun(
        path=path, layout=layout, cl_method=str(config["cl_method"]).lower(),
        importance_mode=config.get("importance_mode"), seed=int(config["seed"]),
        use_task_id=_flag("use_task_id", config, run, path), use_multihead=_flag("use_multihead", config, run, path),
        partners=bank, state=str(status.get("state")), stages_completed=int(status.get("stages_completed", 0)),
        env_steps_per_stage=int(budget["env_steps_per_stage"]), episode_steps=int(budget["num_steps"]),
        eval_episodes=int(budget["eval_episodes_per_partner"]),
        planned_full_events=budget.get("planned_full_evaluations"))


def discover_runs(run_root) -> Tuple[List[CpaRun], List[str]]:
    """Every run directory below `run_root` (a directory holding run.json) and the ones that could not be read."""
    root = Path(run_root)
    if not root.is_dir():
        raise CpaCsvError(f"--run_root {root} is not a directory")
    runs, unreadable = [], []
    for run_json in sorted(root.rglob("run.json")):
        try:
            runs.append(load_run(run_json.parent))
        except CpaCsvError as e:
            unreadable.append(str(e))
    return runs, unreadable


def select_run(run_root, method: str, layout: str, seed: int, identity: str, arch: str,
               num_partners: int, run_dirs: Optional[Sequence[str]] = None) -> CpaRun:
    """The one completed run of this (method, layout, seed, identity input, head structure, partner count).

    Several matching completed runs, an incomplete match or no match at all is an error; with several matches the
    user has to say which one with `run_dirs`.
    """
    if method not in CSV_METHODS:
        raise CpaCsvError(f"CSV mode supports the methods {sorted(CSV_METHODS)}, got {method!r}")
    cl_method, importance_mode = CSV_METHODS[method]
    unreadable: List[str] = []
    if run_dirs:
        pool = []
        for d in run_dirs:
            p = Path(d)
            p = p if p.is_dir() or not run_root else Path(run_root) / d
            pool.append(load_run(p))
    else:
        pool, unreadable = discover_runs(run_root)

    matches = [r for r in pool if (
        r.layout == layout and r.cl_method == cl_method and r.seed == int(seed)
        and (importance_mode is None or r.importance_mode == importance_mode)
        and r.use_task_id == (identity == "available") and r.use_multihead == (arch == "multihead")
        and r.num_partners == int(num_partners))]
    wanted = (f"{method} (cl_method={cl_method}" + (f", importance_mode={importance_mode}" if importance_mode else "")
              + f") on {layout}, seed {seed}, identity {identity}, {arch}, {num_partners} partners")
    if not matches:
        found = "\n  ".join(r.describe() for r in pool) or "no run directories"
        note = ("\n  unreadable: " + "; ".join(unreadable)) if unreadable else ""
        raise CpaCsvError(f"no run matches {wanted}. Runs found:\n  {found}{note}")
    done = [r for r in matches if r.is_complete]
    if not done:
        raise CpaCsvError(f"the run(s) matching {wanted} are not complete:\n  "
                          + "\n  ".join(r.describe() for r in matches))
    if len(done) > 1:
        raise CpaCsvError(f"{len(done)} completed runs match {wanted}; choose one with --run_dirs:\n  "
                          + "\n  ".join(r.describe() for r in done))
    return done[0]


# ---------------------------------------------------------------------------------------------------------
# full-bank evaluation events
# ---------------------------------------------------------------------------------------------------------

@dataclass
class FullBankEvents:
    """Measured full-bank evaluations of a run, sorted by cumulative training env steps (exactly as recorded)."""
    env_steps: np.ndarray                   # (T,) int64
    event_ids: List[str]
    partner_means: np.ndarray               # (T, P) team soups per episode, mean over the evaluation episodes

    @property
    def mean(self) -> np.ndarray:
        """(T,) equal-weight mean of the P partner means: the average over ALL partners, trained with or not."""
        return self.partner_means.mean(axis=1)


_REQUIRED_COLUMNS = ("event_id", "row_id", "scope", "env_steps", "eval_partner_id", "eval_episode", "soups",
                     "completed")


def load_full_bank_events(run: CpaRun) -> FullBankEvents:
    """Per-partner means (over the evaluation episodes) of every `scope == "full"` event of `run`.

    Only completed episode rows count. Duplicate rows (same `row_id`, e.g. repeated by a resume) are kept once if
    identical and are an error if they disagree. Every event must hold exactly the expected partner ids and
    episode ids, otherwise it is an error (an incomplete event is never averaged).
    """
    path = run.path / "eval_metrics.csv"
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise CpaCsvError(f"{path}: columns {missing} missing")
            rows = [r for r in reader if r["scope"] == "full"]
    except OSError as e:
        raise CpaCsvError(f"cannot read {path}: {e}") from e

    records: Dict[str, Tuple] = {}
    triples: Dict[Tuple[str, int, int], str] = {}
    for r in rows:
        done = str(r["completed"]).strip().lower() in ("1", "1.0", "true")
        soups = float(r["soups"]) if r["soups"].strip() != "" else None
        rec = (r["event_id"], int(float(r["env_steps"])), int(r["eval_partner_id"]), int(r["eval_episode"]), done, soups)
        if r["row_id"] in records and records[r["row_id"]] != rec:
            raise CpaCsvError(f"{path}: row {r['row_id']} appears twice with different measurements "
                              f"({records[r['row_id']]} vs {rec})")
        records[r["row_id"]] = rec
        owner = triples.setdefault(rec[:1] + rec[2:4], r["row_id"])
        if owner != r["row_id"]:
            raise CpaCsvError(f"{path}: rows {owner} and {r['row_id']} are the same episode of event {rec[0]}")

    expected_partners, expected_episodes = list(range(run.num_partners)), list(range(run.eval_episodes))
    events: Dict[str, Dict[str, Any]] = {}
    for event_id, env_steps, pid, ep, done, soups in records.values():
        event = events.setdefault(event_id, {"env_steps": env_steps, "scores": {}})
        if event["env_steps"] != env_steps:
            raise CpaCsvError(f"{path}: event {event_id} is recorded at two env_steps values")
        if not done:
            continue
        if soups is None:
            raise CpaCsvError(f"{path}: completed episode {event_id}/p{pid}/e{ep} has no soups value")
        event["scores"].setdefault(pid, {})[ep] = soups
    if not events:
        raise CpaCsvError(f"{path}: no full-bank evaluation rows")
    if len({e["env_steps"] for e in events.values()}) != len(events):
        raise CpaCsvError(f"{path}: two full-bank events share the same env_steps")

    ordered = sorted(events.items(), key=lambda kv: kv[1]["env_steps"])
    means = np.empty((len(ordered), run.num_partners))
    for t, (event_id, event) in enumerate(ordered):
        if sorted(event["scores"]) != expected_partners:
            raise CpaCsvError(f"{path}: full event {event_id} has partners {sorted(event['scores'])} with completed "
                              f"episodes, expected {run.num_partners} (0..{run.num_partners - 1})")
        for pid in expected_partners:
            got = sorted(event["scores"][pid])
            if got != expected_episodes:
                raise CpaCsvError(f"{path}: full event {event_id}, partner {pid}: completed episodes {got}, "
                                  f"expected {expected_episodes}")
            means[t, pid] = np.mean([event["scores"][pid][e] for e in expected_episodes])
    if run.planned_full_events is not None and len(ordered) != run.planned_full_events:
        raise CpaCsvError(f"{path}: {len(ordered)} full events, run.json planned {run.planned_full_events}")
    return FullBankEvents(np.array([e["env_steps"] for _, e in ordered], dtype=np.int64),
                          [i for i, _ in ordered], means)


# ---------------------------------------------------------------------------------------------------------
# what the figures say about a run (text and structure only; drawing stays in the scripts)
# ---------------------------------------------------------------------------------------------------------

def stage_boundaries(run: CpaRun) -> List[int]:
    """Cumulative env steps at which partner k starts (k = 0..P-1) and the last one ends, from the saved budget."""
    return [k * run.env_steps_per_stage for k in range(run.num_partners + 1)]


def partner_labels(run: CpaRun) -> List[str]:
    """`<position> · <name>` with a 1-based position from the manifest: `1 · BRDiv m0`, `13 · planner P01`."""
    out = []
    for p in run.partners:
        if p.get("planner_id"):
            name = f"planner {p['planner_id']}"
        elif p.get("kind") == "brdiv":
            name = f"BRDiv {str(p.get('label', p['partner_id'])).rsplit('/', 1)[-1]}"
        else:
            name = str(p.get("label", f"partner {p['partner_id']}"))
        out.append(f"{p['partner_id'] + 1} · {name}")
    return out


def partner_groups(run: CpaRun) -> List[Tuple[str, int, int]]:
    """Consecutive partners of the same kind in the manifest: (kind, first index, last index)."""
    groups: List[Tuple[str, int, int]] = []
    for i, p in enumerate(run.partners):
        kind = str(p.get("kind", "partner"))
        if groups and groups[-1][0] == kind:
            groups[-1] = (kind, groups[-1][1], i)
        else:
            groups.append((kind, i, i))
    return groups


GROUP_NAMES = {"brdiv": "BRDiv", "planner": "Planner"}


def condition_text(run: CpaRun) -> str:
    return (f"identity input {'available' if run.use_task_id else 'hidden'}, "
            f"{'multi-head' if run.use_multihead else 'single-head'}")


def smooth_line(y: np.ndarray, sigma: float) -> np.ndarray:
    """Optional smoothing of a drawn line over the measurement index (sigma 0 = the raw values)."""
    if not sigma or sigma <= 0:
        return np.asarray(y, dtype=float)
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.asarray(y, dtype=float), sigma=sigma, mode="nearest")


def caption(run: CpaRun, kind: str, sigma: float = 0.0) -> str:
    """Two-line description of what a figure shows; `kind` is "average" or "per_partner"."""
    t, e, p = run.episode_steps, run.eval_episodes, run.num_partners
    if kind == "average":
        first = (f"Mean team soups per {t}-step episode over all {p} partners (equal weight), including partners not "
                 f"yet trained with: each point is the mean of {e} evaluation episodes per partner, then the mean of "
                 f"the {p} partner means.")
        last = ("Not an average over seen partners, a cumulative sum or forward transfer.")
    else:
        first = (f"One curve per partner: mean team soups per {t}-step episode ({e} evaluation episodes) at the same "
                 f"{p}-partner evaluations. Thick segment and shaded band: that partner's training interval.")
        last = "Dashed vertical lines: partner boundaries; the BRDiv to planner transition is drawn heavier."
    raw = (f"Markers are measured evaluations at recorded cumulative env steps (lines only connect them); raw soups, "
           + (f"line smoothed over measurements (sigma={sigma:g}), markers raw; " if sigma else "no smoothing; ")
           + f"seed {run.seed} only, no confidence band ({e} episodes are not independent training seeds).")
    return f"{first}\n{last} {raw}"


# ---------------------------------------------------------------------------------------------------------
# selection for the scripts
# ---------------------------------------------------------------------------------------------------------

def load_csv_selection(args, layouts: Optional[Sequence[str]]):
    """Resolve the CSV-mode arguments and load the data: `(layout, seed, sigma, [(method, run, events), ...])`.

    CSV mode needs exactly one layout, one seed and the soup metric (all means are raw soups); the defaults are
    all three methods, seed 0, 24 partners and no smoothing.
    """
    if getattr(args, "compare_by", "method") != "method":
        raise CpaCsvError("CSV mode compares methods only: --compare_by arch is not supported (use --arch)")
    if getattr(args, "metric", "soup") != "soup":
        raise CpaCsvError("CSV mode plots raw soups only (--metric soup)")
    if not layouts or len(layouts) != 1:
        raise CpaCsvError("CSV mode needs exactly one layout (--layout_names / --layout_name)")
    seeds = args.seeds or [0]
    if len(seeds) != 1:
        raise CpaCsvError(f"CSV mode plots one seed per figure, got --seeds {seeds}")
    methods = args.methods or list(CSV_METHODS)
    num_partners = args.num_partners or DEFAULT_CSV_PARTNERS
    sigma = 0.0 if args.sigma is None else float(args.sigma)
    selected = []
    for method in methods:
        run = select_run(args.run_root, method, layouts[0], seeds[0], args.identity, args.arch, num_partners,
                         args.run_dirs)
        selected.append((method, run, load_full_bank_events(run)))
    return layouts[0], int(seeds[0]), sigma, selected


def csv_stem(prefix: str, layout: str, methods: Sequence[str], run: CpaRun) -> str:
    return (f"{prefix}_{layout}_{'-'.join(methods)}_identity-{'available' if run.use_task_id else 'hidden'}_"
            f"{'multihead' if run.use_multihead else 'singlehead'}_seed{run.seed}_partners{run.num_partners}")


def common_boundaries(runs: Sequence[CpaRun]) -> List[int]:
    """Stage boundaries of the runs on one axis; runs with different budgets cannot share one."""
    boundaries = stage_boundaries(runs[0])
    if any(stage_boundaries(r) != boundaries for r in runs):
        raise CpaCsvError("the selected runs have different per-partner budgets; plot them one method at a time")
    return boundaries


# ---------------------------------------------------------------------------------------------------------
# shared decoration (the axes and figure passed in are drawn on; nothing is created here)
# ---------------------------------------------------------------------------------------------------------

X_LABEL = "Cumulative training environment steps (joint steps, all partners so far)"


def decorate_partner_axis(ax, run: CpaRun, fontsize: int = 8) -> List[int]:
    """Partner boundaries, the BRDiv/planner transition, top partner numbers and group names; returns boundaries."""
    from .plotting import add_task_boundaries
    boundaries = stage_boundaries(run)
    add_task_boundaries(ax, boundaries, color="grey", linewidth=0.5)
    groups = partner_groups(run)
    for _, first, _ in groups[1:]:
        ax.axvline(boundaries[first], color="black", ls="--", lw=1.4)
    step = max(1, run.num_partners // 4)
    ax.set_xticks(boundaries[::step])
    ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    mids = [(boundaries[i] + boundaries[i + 1]) / 2.0 for i in range(run.num_partners)]
    top = ax.secondary_xaxis("top")
    top.set_xticks(mids)
    top.set_xticklabels([str(i + 1) for i in range(run.num_partners)], fontsize=fontsize)
    top.tick_params(axis="x", length=0, pad=1)
    for kind, first, last in groups:
        ax.annotate(f"{GROUP_NAMES.get(kind, kind)} partners {first + 1}-{last + 1}",
                    xy=((boundaries[first] + boundaries[last + 1]) / 2.0, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 15), textcoords="offset points", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", annotation_clip=False)
    return boundaries


def finish_figure(fig, title: str, text: str) -> None:
    """Title above, wrapped explanatory caption below; the axes fill what is left."""
    width_in, height_in = fig.get_size_inches()
    wrapped = textwrap.fill(" ".join(text.split("\n")), width=int(width_in * 13))
    fig.suptitle(title, fontsize=11, y=0.995, va="top")
    fig.text(0.01, 0.008, wrapped, fontsize=7.5, va="bottom", ha="left", color="#333333")
    bottom = (0.125 * (wrapped.count("\n") + 1) + 0.2) / height_in
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.35 / height_in))
