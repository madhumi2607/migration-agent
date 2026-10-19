"""Run a target repo's test suite inside the sandbox Docker container."""

import re
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

from mcp_servers.sandbox.report import EXIT_INSTALL_FAILED, EXIT_UPGRADE_FAILED, build_report

# Exit codes that mean "the tests ran (or failed to install)": pytest's 0-5 plus the
# entrypoint's 10/11. Anything else means the sandbox itself broke.
EXPECTED_EXIT_CODES = {0, 1, 2, 3, 4, 5, EXIT_INSTALL_FAILED, EXIT_UPGRADE_FAILED}


class SandboxError(ToolError):
    """A run_tests call we refuse. Subclasses ToolError so MCP shows the message to the LLM."""


@dataclass
class SandboxConfig:
    """How to run one task's tests. Fixed when the server starts; the agent can't change it."""

    repo_root: Path
    install_cmd: str
    test_cmd: str
    upgrade: str = ""  # e.g. "pydantic>=2,<3"; empty means run with the repo's own pins
    image: str = "migration-agent-sandbox:py3.11"
    timeout_seconds: int = 600
    memory: str = "2g"
    cpus: str = "2"


def upgrade_packages(upgrade: str) -> list[str]:
    """Package names from a requirement string: "pydantic>=2,<3 foo==1" -> ["pydantic", "foo"]."""
    return [re.match(r"[A-Za-z0-9_.-]+", req)[0] for req in shlex.split(upgrade)]


def build_docker_command(config: SandboxConfig, out_dir: Path, container_name: str, tests: list[str] | None) -> list[str]:
    """The `docker run` argv for one test run. A list, not a shell string: no quoting bugs on the host."""
    test_cmd = config.test_cmd
    for t in tests or []:
        # Only test ids/paths: flags like --deselect or -k "not x" would let the agent hide failures.
        if t.startswith("-"):
            raise SandboxError(f"'{t}' looks like a pytest option. Pass test ids or paths only, e.g. tests/test_x.py::test_y.")
    if tests:
        # Quoted because it's interpreted by the shell *inside* the container.
        test_cmd = f"{test_cmd} {' '.join(shlex.quote(t) for t in tests)}"
    return [
        "docker", "run", "--rm",
        "--name", container_name,
        "--memory", config.memory,
        "--cpus", config.cpus,
        "-v", f"{config.repo_root.resolve()}:/src:ro",
        "-v", f"{out_dir.resolve()}:/out",
        "-e", f"INSTALL_CMD={config.install_cmd}",
        "-e", f"UPGRADE={config.upgrade}",
        "-e", f"TEST_CMD={test_cmd}",
        config.image,
    ]


def run_tests(config: SandboxConfig, tests: list[str] | None = None) -> dict[str, Any]:
    """Install the repo, apply the upgrade and run pytest in a fresh container; return the summary."""
    container_name = f"migration-sandbox-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix="sandbox-out-") as tmp:
        out_dir = Path(tmp)
        cmd = build_docker_command(config, out_dir, container_name, tests)
        start = time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=config.timeout_seconds)
        except subprocess.TimeoutExpired:
            # Killing the `docker run` client does NOT stop the container; do it explicitly.
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            return {"status": "timeout", "timeout_seconds": config.timeout_seconds}
        except FileNotFoundError:
            return {"status": "sandbox_error", "detail": "The `docker` command was not found. Is Docker installed and on PATH?"}
        duration = round(time.monotonic() - start, 1)

        # The entrypoint writes install.log before anything else, so if it's missing the
        # container never ran. Exit codes alone can't tell: with the Docker engine down,
        # `docker run` exits 1, the same code pytest uses for "some tests failed".
        never_ran = not (out_dir / "install.log").exists()
        if never_ran or proc.returncode not in EXPECTED_EXIT_CODES:
            # Docker couldn't start the container (engine down, 125), the entrypoint couldn't run
            # (126/127, or 255 from a broken script), or it was OOM-killed (137). Not the repo's fault.
            return {"status": "sandbox_error", "exit_code": proc.returncode, "detail": proc.stderr.strip()[-1000:]}

        report = build_report(out_dir, proc.returncode, upgrade_packages(config.upgrade))
        report["duration_seconds"] = duration
        return report
