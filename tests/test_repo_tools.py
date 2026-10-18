"""Tests for the repo_tools server: the pure functions, then the MCP layer end to end."""

import asyncio
from pathlib import Path

import pytest
from mcp import Client

from mcp_servers.repo_tools import tools
from mcp_servers.repo_tools.server import build_server
from mcp_servers.repo_tools.tools import RepoToolError


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tiny fake target repo."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "models.py").write_text(
        "from pydantic import BaseModel\n\n\nclass User(BaseModel):\n    name: str\n\n    class Config:\n        orm_mode = True\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_models.py").write_text("def test_user():\n    assert True\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("orm_mode\n", encoding="utf-8")
    return tmp_path


# --- path safety ---

@pytest.mark.parametrize("bad", ["../outside.py", "pkg/../../outside.py", ".git/config"])
def test_resolve_path_rejects_escapes(repo: Path, bad: str) -> None:
    with pytest.raises(RepoToolError):
        tools.resolve_path(repo, bad)


@pytest.mark.parametrize(
    ("path", "expected"),
    [("tests/test_x.py", True), ("src/pkg/tests/helpers.py", True), ("conftest.py", True),
     ("pkg/test_utils.py", True), ("pkg/models_test.py", True), ("pkg/models.py", False), ("pkg/testing.py", False)],
)
def test_is_test_file(path: str, expected: bool) -> None:
    assert tools.is_test_file(Path(path)) is expected


# --- list / search / read ---

def test_list_files_skips_git(repo: Path) -> None:
    result = tools.list_files(repo, "**/*")
    assert result["files"] == ["pkg/models.py", "tests/test_models.py"]


def test_search_code_finds_line_numbers(repo: Path) -> None:
    result = tools.search_code(repo, r"orm_mode")
    assert result["matches"] == [{"path": "pkg/models.py", "line": 8, "text": "orm_mode = True"}]


def test_search_code_caps_results(repo: Path) -> None:
    result = tools.search_code(repo, r".", max_results=2)
    assert len(result["matches"]) == 2 and result["truncated"]


def test_search_code_bad_regex(repo: Path) -> None:
    with pytest.raises(RepoToolError, match="Invalid regex"):
        tools.search_code(repo, "(")


def test_read_file_line_range(repo: Path) -> None:
    result = tools.read_file(repo, "pkg/models.py", start_line=4, end_line=5)
    assert result["total_lines"] == 8
    assert result["content"] == "    4  class User(BaseModel):\n    5      name: str"


# --- edit ---

def test_edit_file_replaces_unique_match(repo: Path) -> None:
    result = tools.edit_file(
        repo, "pkg/models.py",
        "    class Config:\n        orm_mode = True\n",
        "    model_config = ConfigDict(from_attributes=True)\n",
    )
    assert result["line"] == 7
    assert "+    model_config = ConfigDict(from_attributes=True)" in result["diff"]
    assert "orm_mode" not in (repo / "pkg" / "models.py").read_text(encoding="utf-8")


def test_edit_file_preserves_crlf(repo: Path) -> None:
    f = repo / "pkg" / "win.py"
    f.write_bytes(b"a = 1\r\nb = 2\r\n")
    tools.edit_file(repo, "pkg/win.py", "b = 2\n", "b = 3\n")
    assert f.read_bytes() == b"a = 1\r\nb = 3\r\n"


def test_edit_file_not_found_and_ambiguous(repo: Path) -> None:
    with pytest.raises(RepoToolError, match="not found"):
        tools.edit_file(repo, "pkg/models.py", "does not exist", "x")
    (repo / "pkg" / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    with pytest.raises(RepoToolError, match="appears 2 times"):
        tools.edit_file(repo, "pkg/dup.py", "x = 1", "x = 2")


def test_edit_file_refuses_tests(repo: Path) -> None:
    with pytest.raises(RepoToolError, match="test file"):
        tools.edit_file(repo, "tests/test_models.py", "assert True", "assert 1")


def test_edit_file_refuses_skip_markers(repo: Path) -> None:
    with pytest.raises(RepoToolError, match="not allowed"):
        tools.edit_file(repo, "pkg/models.py", "class User", "@pytest.mark.skip\nclass User")


# --- MCP layer: what the agent will actually see ---

def test_server_over_mcp(repo: Path) -> None:
    async def scenario() -> None:
        async with Client(build_server(repo)) as client:
            listed = await client.list_tools()
            assert {t.name for t in listed.tools} == {"list_files", "search_code", "read_file", "edit_file"}

            ok = await client.call_tool("search_code", {"pattern": "orm_mode"})
            assert not ok.is_error
            assert ok.structured_content["matches"][0]["line"] == 8

            # Refusals come back as tool errors the LLM can read, not as crashes.
            refused = await client.call_tool("edit_file", {"path": "../x.py", "old_text": "a", "new_text": "b"})
            assert refused.is_error
            assert "outside the target repo" in refused.content[0].text

    asyncio.run(scenario())
