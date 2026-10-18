"""sandbox MCP server: run the target repo's tests in Docker and summarise the failures.

How tests are installed and run is fixed at startup from environment variables
(see .env.example); the agent only chooses *which* tests to run.

Run (stdio transport):
    uv run python -m mcp_servers.sandbox.server
"""

import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from mcp_servers.sandbox import runner
from mcp_servers.sandbox.runner import SandboxConfig


def config_from_env() -> SandboxConfig:
    """Read the task's sandbox settings from environment variables."""
    def required(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise SystemExit(f"Set {name} (see .env.example).")
        return value

    return SandboxConfig(
        repo_root=Path(required("TARGET_REPO_ROOT")),
        install_cmd=required("SANDBOX_INSTALL_CMD"),
        test_cmd=required("SANDBOX_TEST_CMD"),
        upgrade=os.environ.get("SANDBOX_UPGRADE", ""),
        image=os.environ.get("SANDBOX_IMAGE", SandboxConfig.image),
        timeout_seconds=int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", SandboxConfig.timeout_seconds)),
    )


def build_server(config: SandboxConfig) -> MCPServer:
    """Create the server for one task. Separate from main() so tests can inject a config."""
    server = MCPServer(
        "sandbox",
        instructions="Runs the target repo's test suite in an isolated Docker container with the upgraded dependency.",
    )

    @server.tool()
    def run_tests(tests: list[str] | None = None) -> dict[str, Any]:
        """Install the repo with the upgraded dependency and run its tests in a fresh container.

        tests: optional pytest ids or paths to run (e.g. ["tests/test_x.py::test_y"]); omit to run
        the whole suite. Failures are grouped by root cause, each with error type, message, the
        deepest file:line inside the repo, a count and example test ids. Takes ~30s or more.
        """
        return runner.run_tests(config, tests)

    return server


def main() -> None:
    build_server(config_from_env()).run()


if __name__ == "__main__":
    main()
