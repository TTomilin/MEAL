"""What a `run_br` run writes, and how a run is recovered from a completed partner boundary.

Run directory (unique per invocation; see ``allocate_run_dir``)::

    config.json       resolved TrainConfig (readable)            config.pckl  same, pickled (legacy)
    run.json          code revision, device, architecture, budget, schedule, partner manifest, fingerprint
    partner_bank.json exact partner order with checkpoint/planner identities
    status.json       state, completed stages, counters, one entry per attempt (start / resume)
    train_metrics.csv one row per PPO update              eval_metrics.csv  one row per evaluation episode
    stage_timings.csv one row per stage                   latest.ckpt       rolling boundary checkpoint
    params_seed{S}.pt final policy export (unchanged format)

Everything except ``latest.ckpt`` and ``params_seed*.pt`` is text. CSV files are append-only while a run is
live; on resume, rows of a stage that never reached a checkpoint are cut (see ``truncate_csv``).
"""
import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from experiments.partner_adaptation.train_ego import peak_device_memory

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
CHECKPOINT_NAME = "latest.ckpt"
MAGIC = b"MEALCPA1"

TRAIN_COLUMNS = [
    "env_steps", "stage", "partner_id", "partner", "update", "episodes", "soups", "ego_return", "sparse_reward",
    "shaped_reward", "shaping_coef", "actor_loss", "value_loss", "entropy", "cl_penalty", "grad_norm", "attempt"]
EVAL_COLUMNS = [
    "event_id", "row_id", "scope", "env_steps", "stage", "update", "train_episodes", "eval_partner_id",
    "eval_partner", "eval_episode", "episode_key", "soups", "ego_return", "completed", "attempt"]
TIMING_COLUMNS = [
    "stage", "partner_id", "partner", "attempt", "updates", "env_steps_end", "compile_s", "compile_cached",
    "train_eval_s", "eval_events", "eval_event_s", "eval_current_events", "eval_current_event_s", "init_eval_s",
    "importance_s", "importance_compiled_here", "stage_wall_s",
    "backend", "device_kind", "peak_bytes_in_use", "memory_status"]

# Settings that do not change what is trained or evaluated: logging, output location, resume mechanics.
NON_SEMANTIC_CONFIG = frozenset({
    "project", "mode", "group", "entity", "tags", "checkpoint_path", "checkpoint_freq", "save_dir", "resume",
    "save_checkpoints", "record_video", "gif_len", "log_train_out", "partner_bank", "layouts_path"})


class RunOutputError(RuntimeError):
    pass


class CheckpointError(RunOutputError):
    """The checkpoint is missing, truncated, corrupted or does not fit this run's parameter structure."""


class ResumeError(RunOutputError):
    """The checkpoint is valid but belongs to a different experiment than the one being resumed."""


# ---------------------------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _fsync_dir(path: Path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path, data: bytes):
    """Write `data` so that `path` holds either its previous content or all of `data`, never a prefix."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def write_json(path, obj):
    atomic_write(path, (json.dumps(obj, indent=2, sort_keys=True, default=_jsonable) + "\n").encode())


def normalise(obj):
    """Round-trip through JSON so values compare equal whether they were just built or read from disk."""
    return json.loads(json.dumps(obj, sort_keys=True, default=_jsonable))


def allocate_run_dir(root, run_string):
    """Create a new, empty run directory; never reuses an existing one.

    ``mkdir`` is atomic, so concurrent launches with identical layout/method/seed get different directories
    through the random suffix, and a collision is retried instead of overwritten.
    """
    root = Path(root or ".")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for _ in range(100):
        uid = uuid.uuid4().hex[:8]
        path = root / f"{run_string}_{stamp}_{uid}"
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path, uid
    raise RunOutputError(f"could not allocate a run directory under {root}")


def code_revision(root: Optional[Path] = None) -> Dict[str, Any]:
    """Git revision of the code that is running; `available` is False outside a git checkout."""
    root = Path(root) if root else Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)

    try:
        head = git("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"available": False, "commit": None}
        lines = git("status", "--porcelain").stdout.splitlines()
        return {"available": True, "commit": head.stdout.strip(),
                "modified_tracked_files": sum(not line.startswith("??") for line in lines),
                "untracked_files": sum(line.startswith("??") for line in lines)}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None}


def device_info() -> Dict[str, Any]:
    devices = jax.local_devices()
    return {
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "devices": [{"id": d.id, "platform": d.platform, "device_kind": d.device_kind} for d in devices],
        "memory_statistics": "available" if peak_device_memory() is not None else "unavailable on this backend",
        "host_cpu_count": os.cpu_count(),
        "jax": jax.__version__, "flax": flax.__version__, "numpy": np.__version__,
        "python": sys.version.split()[0], "platform": platform.platform(),
    }


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------------------
# append-only CSV records
# ---------------------------------------------------------------------------------------------------------

def _cell(value):
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return format(value, ".9g")
    return value


class CsvLog:
    """Append-only CSV with a fixed header; every write is flushed so a crash loses nothing committed."""

    def __init__(self, path, columns):
        self.path, self.columns = Path(path), list(columns)
        fresh = not self.path.exists() or self.path.stat().st_size == 0
        if not fresh:
            with open(self.path, newline="") as f:
                header = next(csv.reader(f), None)
            if header != self.columns:
                raise ResumeError(f"{self.path.name} has columns {header}, expected {self.columns}")
        self._file = open(self.path, "a", newline="")
        self._writer = csv.writer(self._file)
        if fresh:
            self._writer.writerow(self.columns)
            self.flush()

    def write(self, rows):
        self._writer.writerows([[_cell(row.get(c)) for c in self.columns] for row in rows])
        self.flush()

    def flush(self):
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self):
        if not self._file.closed:
            self.flush()
            self._file.close()


def truncate_csv(path, max_stage: int) -> int:
    """Drop rows whose `stage` exceeds `max_stage` (an interrupted stage); returns how many were dropped."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0
    stage_col = rows[0].index("stage")
    keep = [rows[0]] + [r for r in rows[1:] if int(r[stage_col]) <= max_stage]
    dropped = len(rows) - len(keep)
    if dropped:
        out = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(out, "w", newline="") as f:
            csv.writer(f).writerows(keep)
            f.flush()
            os.fsync(f.fileno())
        os.replace(out, path)
    return dropped


class RunRecorder:
    """Receives the per-update training and evaluation events of one stage and commits them in update order.

    The training scan reports events through unordered host callbacks, so they can arrive out of order. An
    update is committed only when its training event (and its evaluation event, if one is due) has arrived and
    all earlier updates are committed. Each commit appends the CSV rows and sends ONE merged W&B log at
    ``step = cumulative environment interactions``; steps therefore increase strictly and nothing is logged
    for a step after a later one was committed. A stage must commit every update before `end_stage` returns,
    otherwise it raises and no checkpoint is written.

    W&B is a mirror: a failure is counted and reported in the run status, never raised, and local rows are
    written first.
    """

    def __init__(self, run_dir, labels: List[str], seed: int, attempt: int = 0,
                 wandb_log: Optional[Callable[[Dict[str, Any], int], None]] = None):
        run_dir = Path(run_dir)
        self.labels, self.seed, self.attempt = list(labels), int(seed), int(attempt)
        self.train_log = CsvLog(run_dir / "train_metrics.csv", TRAIN_COLUMNS)
        self.eval_log = CsvLog(run_dir / "eval_metrics.csv", EVAL_COLUMNS)
        self.timing_log = CsvLog(run_dir / "stage_timings.csv", TIMING_COLUMNS)
        self.wandb_log = wandb_log
        self.wandb_failures, self.wandb_error = 0, None
        self._lock = threading.Lock()
        self._error: Optional[BaseException] = None
        self._stage = None

    def _label(self, partner_id):
        return self.labels[partner_id] if 0 <= partner_id < len(self.labels) else f"partner{partner_id}"

    def begin_stage(self, stage: int, num_updates: int, eval_plan: Dict[int, str], init_eval: bool = False):
        """`eval_plan` maps update -> "full"/"current" (see train_ego.eval_plan); `init_eval` adds the evaluation
        before the first update (update 0), which has no training event."""
        with self._lock:
            self._stage, self._num_updates = int(stage), int(num_updates)
            self._eval_updates = set(eval_plan) | ({0} if init_eval else set())
            self._pending, self._next = {}, 0 if init_eval else 1
            self._summary = {"updates": 0, "train_episodes": 0, "eval_events": 0, "eval_full_events": 0,
                             "eval_current_events": 0, "eval_episodes": 0, "env_steps_end": None}

    def handle(self, kind: str, record: Dict[str, Any], log_dict: Dict[str, Any]):
        """Host-callback entry point; must not raise into JAX, so errors are kept for `end_stage`."""
        try:
            with self._lock:
                if self._stage is None or record["stage_id"] != self._stage:
                    raise RunOutputError(f"{kind} event for stage {record['stage_id']} outside stage {self._stage}")
                slot = self._pending.setdefault(record["update"], {})
                if kind in slot:
                    raise RunOutputError(f"duplicate {kind} event for stage {self._stage} update {record['update']}")
                slot[kind] = (record, log_dict)
                self._drain()
        except BaseException as e:  # noqa: BLE001 - re-raised from end_stage
            if self._error is None:
                self._error = e

    def _drain(self):
        while self._next <= self._num_updates:
            slot = self._pending.get(self._next, {})
            if (self._next > 0 and "train" not in slot) or (self._next in self._eval_updates and "eval" not in slot):
                return
            self._commit(self._next, slot)
            del self._pending[self._next]
            self._next += 1

    def _commit(self, update, slot):
        merged, stage, env_steps = {}, self._stage, None
        if "train" in slot:
            train, train_log = slot["train"]
            stage, env_steps = train["stage_id"], train["env_steps"]
            self.train_log.write([dict(train, stage=stage, partner_id=stage, partner=self._label(stage),
                                       attempt=self.attempt)])
            merged.update(train_log)
            self._summary["updates"] += 1
            self._summary["train_episodes"] += train["episodes"]
            self._summary["env_steps_end"] = env_steps
        if "eval" in slot:
            ev, ev_log = slot["eval"]
            env_steps = ev["env_steps"]
            scope = ev.get("scope", "full")
            event_id = f"s{stage:03d}-u{update:05d}"
            rows = []
            for row, pid in enumerate(ev["partner_ids"]):
                for ep in ev["episode_ids"]:
                    done = bool(ev["completed"][row][ep])
                    rows.append({
                        "event_id": event_id, "row_id": f"{event_id}-p{pid:03d}-e{ep:03d}", "scope": scope,
                        "env_steps": ev["env_steps"], "stage": stage, "update": update,
                        "train_episodes": ev["train_episodes"], "eval_partner_id": pid,
                        "eval_partner": self._label(pid), "eval_episode": ep,
                        "episode_key": f"seed{self.seed}/stage{stage}/update{update}/episode{ep}",
                        "soups": float(ev["team_soups"][row][ep]) if done else None,
                        "ego_return": float(ev["team_returns"][row][ep]) if done else None,
                        "completed": done, "attempt": self.attempt})
            self.eval_log.write(rows)
            merged.update(ev_log)
            self._summary["eval_events"] += 1
            self._summary["eval_full_events" if scope == "full" else "eval_current_events"] += 1
            self._summary["eval_episodes"] += len(rows)
        merged["env_steps"] = env_steps
        self._mirror(merged, env_steps)

    def _mirror(self, log_dict, step):
        if self.wandb_log is None or self.wandb_failures:
            return
        try:
            self.wandb_log(log_dict, step)
        except Exception as e:  # noqa: BLE001 - local records are the source of truth
            self.wandb_failures, self.wandb_error = 1, f"{type(e).__name__}: {e}"
            log.warning("W&B logging failed (%s); continuing with local records only.", self.wandb_error)

    def end_stage(self) -> Dict[str, Any]:
        with self._lock:
            if self._error is not None:
                raise RunOutputError(f"recording stage {self._stage} failed: {self._error}") from self._error
            if self._next != self._num_updates + 1 or self._pending:
                raise RunOutputError(
                    f"stage {self._stage} recorded {self._next - 1} of {self._num_updates} updates "
                    f"(incomplete events: {sorted(self._pending)})")
            summary, self._stage = dict(self._summary), None
        return summary

    def write_timing(self, row: Dict[str, Any]):
        self.timing_log.write([row])

    def close(self):
        for csv_log in (self.train_log, self.eval_log, self.timing_log):
            csv_log.close()


# ---------------------------------------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------------------------------------

class RunStatus:
    """status.json: where the run is, plus one entry per attempt (the first start and every resume)."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "status.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"state": "created", "attempts": []}

    def begin_attempt(self, info: Dict[str, Any]) -> int:
        self.data["attempts"].append(dict(info, started_utc=now_utc()))
        self.data["state"] = "running"
        self.save()
        return len(self.data["attempts"]) - 1

    def update(self, **fields):
        self.data.update(fields)
        self.save()

    def update_attempt(self, index: int, **fields):
        self.data["attempts"][index].update(fields)
        self.save()

    def save(self):
        self.data["updated_utc"] = now_utc()
        write_json(self.path, self.data)


# ---------------------------------------------------------------------------------------------------------
# compatibility and checkpoints
# ---------------------------------------------------------------------------------------------------------

def build_fingerprint(config_dict: Dict[str, Any], partners: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Everything a resumed run must reproduce: every training/evaluation setting and the exact partner list."""
    config = {k: v for k, v in config_dict.items() if k not in NON_SEMANTIC_CONFIG}
    return normalise({"config": config, "partners": partners})


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k in sorted(obj):
            yield from _flatten(obj[k], f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        yield f"{prefix}.length", len(obj)
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def fingerprint_differences(saved: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    a, b = dict(_flatten(saved)), dict(_flatten(current))
    return [f"{k}: checkpoint={a.get(k, '<absent>')!r} current={b.get(k, '<absent>')!r}"
            for k in sorted(set(a) | set(b)) if a.get(k, "<absent>") != b.get(k, "<absent>")]


def check_resume(meta: Dict[str, Any], fingerprint: Dict[str, Any]):
    diffs = fingerprint_differences(meta["fingerprint"], fingerprint)
    if diffs:
        hint = ""
        if any(d.startswith(("config.total_timesteps", "config.num_envs", "config.num_steps")) for d in diffs):
            hint = ("\nThe training budget is per partner: resuming never extends or rescales the stages that "
                    "were already trained.")
        shown = "\n  ".join(diffs[:12]) + (f"\n  ... and {len(diffs) - 12} more" if len(diffs) > 12 else "")
        raise ResumeError(f"cannot resume: the run differs from the checkpoint in\n  {shown}{hint}")
    n = len(fingerprint["partners"])
    if not 0 <= meta["stages_completed"] <= n:
        raise ResumeError(f"checkpoint reports {meta['stages_completed']} completed stages of {n}")


def save_checkpoint(run_dir, ego_params, cl_state, meta: Dict[str, Any]) -> Path:
    """Atomically replace the rolling `latest.ckpt` (magic + sha256 + msgpack of params, CL state, metadata)."""
    body = serialization.msgpack_serialize({
        "meta": json.dumps(dict(meta, format_version=FORMAT_VERSION, saved_utc=now_utc()), default=_jsonable),
        "ego_params": serialization.to_state_dict(ego_params),
        "cl_state": serialization.to_state_dict(cl_state),
    })
    path = Path(run_dir) / CHECKPOINT_NAME
    atomic_write(path, MAGIC + hashlib.sha256(body).digest() + body)
    return path


def read_checkpoint(path):
    """Verify integrity and return (meta, raw_state); raises CheckpointError for anything that is not intact."""
    path = Path(path)
    try:
        blob = path.read_bytes()
    except OSError as e:
        raise CheckpointError(f"cannot read checkpoint {path}: {e}") from e
    if len(blob) < len(MAGIC) + 32 or blob[:len(MAGIC)] != MAGIC:
        raise CheckpointError(f"{path} is not a CPA checkpoint (bad or truncated header)")
    digest, body = blob[len(MAGIC):len(MAGIC) + 32], blob[len(MAGIC) + 32:]
    if hashlib.sha256(body).digest() != digest:
        raise CheckpointError(f"{path} is corrupted (content hash mismatch)")
    try:
        state = serialization.msgpack_restore(body)
        meta = json.loads(state["meta"])
    except Exception as e:  # noqa: BLE001
        raise CheckpointError(f"{path} could not be decoded: {e}") from e
    if meta.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(f"{path} has format version {meta.get('format_version')}, expected {FORMAT_VERSION}")
    return meta, state


def restore_state(raw, template, what: str):
    """Rebuild `template`'s pytree from checkpoint data, rejecting any difference in structure/shape/dtype."""
    if template is None or raw is None:
        if template is not raw:
            raise CheckpointError(f"{what}: checkpoint and run disagree on whether this state exists")
        return None
    try:
        restored = serialization.from_state_dict(template, raw)
    except Exception as e:  # noqa: BLE001
        raise CheckpointError(f"{what} does not match this run's structure: {e}") from e
    t_leaves, t_def = jax.tree.flatten(template)
    r_leaves, r_def = jax.tree.flatten(restored)
    if t_def != r_def:
        raise CheckpointError(f"{what} does not match this run's structure")
    for t, r in zip(t_leaves, r_leaves):
        if np.shape(t) != np.shape(r) or np.asarray(t).dtype != np.asarray(r).dtype:
            raise CheckpointError(
                f"{what} leaf mismatch: run {np.shape(t)}/{np.asarray(t).dtype}, "
                f"checkpoint {np.shape(r)}/{np.asarray(r).dtype}")
    # python scalars (e.g. sizes stored in a struct) stay python scalars; arrays come back as jax arrays
    return jax.tree.map(
        lambda t, r: jnp.asarray(r) if isinstance(t, (jax.Array, np.ndarray)) else type(t)(np.asarray(r).item()),
        template, restored)


def resolve_resume(path):
    """Accept a run directory or a checkpoint file; returns (run_dir, checkpoint_path)."""
    path = Path(path)
    if path.is_dir():
        run_dir, ckpt = path, path / CHECKPOINT_NAME
    else:
        run_dir, ckpt = path.parent, path
    if not ckpt.is_file():
        raise ResumeError(f"no checkpoint at {ckpt} (runs without a completed partner have nothing to resume)")
    return run_dir, ckpt
