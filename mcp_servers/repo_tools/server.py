"""repo_tools MCP server: search, read and edit files inside the target repo.

Run (stdio transport, which is how the agent will launch it):
    TARGET_REPO_ROOT=/path/to/repo uv run python -m mcp_servers.repo_tools.server
"""

import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from mcp_servers.repo_tools import tools


def build_server(root: Path) -> MCPServer:
    """Create the server bound to one target repo. Separate from main() so tests
    can build a server for a temporary repo."""
    if not root.is_dir():
        raise ValueError(f"Target repo root {root} is not a directory.")

    server = MCPServer(
        "repo_tools",
        instructions="Tools for exploring and editing the target repository. All paths are relative to the repo root.",
    )

    # Thin wrappers: the docstrings and type hints become the tool descriptions and
    # JSON schemas the LLM sees, so they're written for the model.

    @server.tool()
    def list_files(glob: str = "**/*.py", limit: int = 200) -> dict[str, Any]:
        """List files in the repo matching a glob pattern (e.g. '**/*.py', 'src/**/models.py')."""
        return tools.list_files(root, glob, limit)

    @server.tool()
    def search_code(pattern: str, glob: str = "**/*.py", max_results: int = 50) -> dict[str, Any]:
        """Search file contents with a Python regex. Returns path, line number and the matching line."""
        return tools.search_code(root, pattern, glob, max_results)

    @server.tool()
    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> dict[str, Any]:
        """Read a file, optionally only lines start_line..end_line (1-based, inclusive). Lines are numbered."""
        return tools.read_file(root, path, start_line, end_line)

    @server.tool()
    def edit_file(path: str, old_text: str, new_text: str) -> dict[str, Any]:
        """Replace one exact occurrence of old_text with new_text in a file.

        old_text must match the file exactly (including indentation) and appear exactly once;
        include surrounding lines to make it unique. Test files cannot be edited.
        """
        return tools.edit_file(root, path, old_text, new_text)

    return server


def main() -> None:
    root = os.environ.get("TARGET_REPO_ROOT")
    if not root:
        raise SystemExit("Set TARGET_REPO_ROOT to the target repo's path.")
    build_server(Path(root)).run()  # stdio by default


if __name__ == "__main__":
    main()
