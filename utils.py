import pathlib
import subprocess

import defs


def run_bash(
    command: str, cwd: pathlib.Path, timeout_sec: int = 120
) -> defs.CommandResult:
    """Executes a bash command and captures its output."""
    cp = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return defs.CommandResult(
        command=command, returncode=cp.returncode, stdout=cp.stdout, stderr=cp.stderr
    )
