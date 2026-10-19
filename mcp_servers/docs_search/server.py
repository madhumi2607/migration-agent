"""docs_search MCP server: search the target library's migration guide.

Build the index first:  uv run python -m mcp_servers.docs_search.build
Run (stdio transport):  uv run python -m mcp_servers.docs_search.server
"""

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mcp_servers.docs_search.build import index_path
from mcp_servers.docs_search.index import DocsIndex

MAX_TEXT_CHARS = 1800  # per result: enough for a section with its code example, not a whole page


def build_server(index: DocsIndex) -> MCPServer:
    """Create the server around an index. Separate from main() so tests can inject one."""
    server = MCPServer(
        "docs_search",
        instructions="Search official migration guides for the library being upgraded.",
    )

    @server.tool()
    def search_docs(query: str, k: int = 4) -> dict[str, Any]:
        """Search the official migration guide (e.g. Pydantic V1 -> V2).

        Use exact identifiers from error messages or old code (e.g. "ModelField", "orm_mode",
        "validator pre=True") or describe the change in words. Returns the best-matching
        sections with their heading path, a link and the section text.
        """
        if not query.strip():
            raise ToolError("query must not be empty.")
        hits = index.search(query, k=max(1, min(k, 8)))
        return {
            "results": [
                {"title": c.title, "url": c.url, "text": c.text[:MAX_TEXT_CHARS] + ("…" if len(c.text) > MAX_TEXT_CHARS else "")}
                for c in hits
            ]
        }

    return server


def main() -> None:
    index = DocsIndex(index_path())
    if not index.chunks:
        raise SystemExit("The docs index is empty. Build it with: uv run python -m mcp_servers.docs_search.build")
    build_server(index).run()


if __name__ == "__main__":
    main()
