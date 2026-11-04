import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def submission(tmp_path):
    source = Path(__file__).resolve().parents[1]
    (tmp_path / "send.sh").write_bytes((source / "send.sh").read_bytes())
    (tmp_path / "body.sh").write_text("true\n")
    (tmp_path / ".config").write_text(
        "HPC_USERNAME=test\nHPC_HOST=cluster\nPARTITION=gpu\nGPUS=1\n"
        "CPUS_PER_TASK=4\nHPC_PROJECT_DIR=/project\nTIME_LIMIT=01:00:00\nMEM=8G\n"
        "ALG=comedi\n")

    def run(*args):
        return subprocess.run(["bash", "send.sh", "test", "body.sh", *args], cwd=tmp_path,
                              env={**os.environ, "DRY": "1"}, text=True, capture_output=True)
    return run


def test_job_assignments_survive_config_and_shell_quoting(submission):
    value = "--name 'two words' $HOME; $(touch should-not-exist)"
    result = submission("00:10:00", "ALG=fcp", "LAYOUT=cramped_room", f"EXTRA_ARGS={value}", "EMPTY=")
    assert result.returncode == 0, result.stderr
    script = result.stdout
    exports = script[script.index("export ALG="):script.index('echo "job')]
    restored = subprocess.run(["bash", "-c", exports + '\nprintf "%s\\n" "$ALG" "$LAYOUT" "$EXTRA_ARGS" "$EMPTY"'],
                             text=True, capture_output=True, check=True)
    assert restored.stdout == f"fcp\ncramped_room\n{value}\n\n"
    assert script.index("source .config") < script.index("export ALG=")


def test_cpu_submission_has_no_gpu_request(submission):
    result = submission("GPUS=0", "PARTITION=cpu")
    assert result.returncode == 0, result.stderr
    assert "#SBATCH --partition=cpu" in result.stdout
    assert "--gres" not in result.stdout


def test_default_submission_still_requests_gpu(submission):
    result = submission()
    assert result.returncode == 0, result.stderr
    assert "#SBATCH --gres=gpu:1" in result.stdout


def test_invalid_assignment_is_rejected(submission):
    assert submission("00:10:00", "BAD-NAME=value").returncode == 2