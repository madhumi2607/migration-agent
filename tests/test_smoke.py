"""Smoke test: the environment is set up and project packages are importable."""


def test_packages_importable() -> None:
    import agent  # noqa: F401
    import mcp_servers.repo_tools  # noqa: F401
    import mcp_servers.sandbox  # noqa: F401


def test_mcp_sdk_installed() -> None:
    from mcp.server.mcpserver import MCPServer

    assert MCPServer("smoke").name == "smoke"
