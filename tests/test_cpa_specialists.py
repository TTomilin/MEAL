"""DRY_RUN coverage for scripts/cpa_specialists.sh (CPU only; no training)."""
import json
import os
import subprocess
from pathlib import Path

from experiments.partner_adaptation.partner_bank import load_partner_bank, partner_identity

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "cpa_specialists.sh"
LAYOUT = "coord_ring"
IDS = ("P01", "P02", "P03")


def write_bank(path: Path) -> Path:
    partners = [{"partner_id": i, "kind": "planner", "planner_id": pid} for i, pid in enumerate(IDS)]
    path.write_text(json.dumps({
        "format_version": 2, "name": "specialists_planners", "layout": LAYOUT, "num_partners": len(partners),
        "populations": {}, "partners": partners, "heldout": [],
    }))
    return path


def identities(bank: Path):
    loaded = load_partner_bank(str(bank), LAYOUT)
    return [partner_identity(r) for r in loaded.records]


def run_specialists(tmp_path, bank, **env):
    base = {
        **os.environ,
        "LAYOUT": LAYOUT,
        "BANK": str(bank),
        "MEAL_CPA_ROOT": str(tmp_path),
        "DRY_RUN": "1",
        "SEED": "0",
    }
    base.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", str(SCRIPT)], cwd=str(REPO), env=base, text=True,
                          capture_output=True, check=False)


def fake_run(root: Path, index: int, identity: dict, state="profile", seed=0, params=True):
    tag = "specialists/" + identity["label"].replace("/", "__")
    d = root / "runs" / f"aht_{LAYOUT}" / tag / "br_fake_00000000-000000_abcd1234"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state}))
    bank = [dict(partner_id=j, kind="planner", label=f"{LAYOUT}/{pid}", planner_id=pid,
                 config_sha256="") for j, pid in enumerate(IDS)]
    bank[index] = identity
    (d / "partner_bank.json").write_text(json.dumps(bank))
    if params:
        (d / f"params_seed{seed}.pt").write_bytes(b"x")
    return d


def test_dry_run_one_command_per_partner(tmp_path):
    bank = write_bank(tmp_path / "bank.json")
    result = run_specialists(tmp_path, bank)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    idents = identities(bank)
    for i, ident in enumerate(idents):
        tag = "specialists/" + ident["label"].replace("/", "__")
        assert f"--only-stages {i}" in out
        assert f"RUN_TAG={tag}" in out
        assert f"{i} {ident['label']} trained" in out
    assert out.count("scripts/cpa_aht_bank.sh") == len(IDS)
    assert "--no-record-video" in out
    assert "--env-dynamics aht" in out
    assert f"GROUP=cpa_specialists_{LAYOUT}" in out


def test_partners_limits_the_set(tmp_path):
    bank = write_bank(tmp_path / "bank.json")
    result = run_specialists(tmp_path, bank, PARTNERS="0 2")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "--only-stages 0" in out and "--only-stages 2" in out
    assert "--only-stages 1" not in out
    assert "0 coord_ring/P01 trained" in out
    assert "2 coord_ring/P03 trained" in out
    assert "1 coord_ring/P02" not in out
    assert out.count("scripts/cpa_aht_bank.sh") == 2


def test_matching_done_run_is_skipped_and_mismatch_is_stale(tmp_path):
    bank = write_bank(tmp_path / "bank.json")
    idents = identities(bank)
    fake_run(tmp_path, 0, idents[0], state="complete")
    stale = dict(idents[1], config_sha256="0" * 64)
    fake_run(tmp_path, 1, stale, state="profile")
    result = run_specialists(tmp_path, bank, PARTNERS="0 1")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "0 coord_ring/P01 skipped" in out
    assert "1 coord_ring/P02 stale" in out
    assert "RUN_TAG=specialists/coord_ring__P01" not in out
    assert "RUN_TAG=specialists/coord_ring__P02" in out
    assert out.count("scripts/cpa_aht_bank.sh") == 1


def test_incomplete_run_retrains(tmp_path):
    bank = write_bank(tmp_path / "bank.json")
    idents = identities(bank)
    fake_run(tmp_path, 0, idents[0], state="failed", params=False)
    result = run_specialists(tmp_path, bank, PARTNERS="0")
    assert result.returncode == 0, result.stderr
    assert "0 coord_ring/P01 incomplete" in result.stdout
    assert "RUN_TAG=specialists/coord_ring__P01" in result.stdout


def test_cpa_aht_bank_defaults_to_aht_dynamics_without_duplicating(tmp_path):
    bank = write_bank(tmp_path / "bank.json")
    fake_py = tmp_path / "echo_py.sh"
    fake_py.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\"\n")
    fake_py.chmod(0o755)
    base_env = {
        **os.environ,
        "LAYOUT": LAYOUT,
        "BANK": str(bank),
        "MEAL_CPA_ROOT": str(tmp_path),
        "PYTHON": str(fake_py),
    }
    aht_bank_script = REPO / "scripts" / "cpa_aht_bank.sh"
    default_res = subprocess.run(["bash", str(aht_bank_script)], cwd=str(REPO), env=base_env,
                                 text=True, capture_output=True, check=False)
    assert default_res.returncode == 0, default_res.stderr
    argv = default_res.stdout.splitlines()
    assert argv.count("--env-dynamics") == 1
    assert argv[argv.index("--env-dynamics") + 1] == "aht"

    extra_res = subprocess.run(["bash", str(aht_bank_script)], cwd=str(REPO),
                               env={**base_env, "EXTRA_ARGS": "--only-stages 0 --env-dynamics aht"},
                               text=True, capture_output=True, check=False)
    assert extra_res.returncode == 0, extra_res.stderr
    extra_argv = extra_res.stdout.splitlines()
    assert extra_argv.count("--env-dynamics") == 1


def test_cpa_specialists_exports_python_and_avoids_duplicate_env_dynamics(tmp_path):
    import sys

    bank = write_bank(tmp_path / "bank.json")
    argv_log = tmp_path / "run_br_argv.log"
    fake_py = tmp_path / "wrap_py.sh"
    fake_py.write_text(
        "#!/bin/bash\n"
        "if [[ \"$1\" == \"-m\" && \"$2\" == \"experiments.partner_adaptation.run_br\" ]]; then\n"
        f"    printf '%s\n' \"$@\" > {str(argv_log)!r}\n"
        "    exit 0\n"
        "fi\n"
        f"exec {sys.executable!r} \"$@\"\n"
    )
    fake_py.chmod(0o755)
    res = run_specialists(
        tmp_path, bank, DRY_RUN="", PARTNERS="0", PYTHON=str(fake_py),
        EXTRA_ARGS="--env-dynamics meal",
    )
    assert res.returncode == 0, res.stderr
    argv = argv_log.read_text().splitlines()
    assert argv.count("--env-dynamics") == 1
    assert argv[argv.index("--env-dynamics") + 1] == "meal"
