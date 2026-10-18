"""Core logic for the repo_tools MCP server.

Every function takes the target repo root explicitly, so it can be unit-tested
without MCP. All paths coming from the LLM are relative to that root and are
checked to stay inside it. Outputs are capped so a single call can't flood
the agent's context window.
"""

import difflib
import re
from pathlib import Path
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

# Directories that are never interesting to the agent and can be huge.
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".tox", ".mypy_cache", ".pytest_cache"}

# Text that would let the agent "fix" a failing test by not running it.
FORBIDDEN_SNIPPETS = ("pytest.mark.skip", "pytest.skip(", "pytest.mark.xfail", "pytest.xfail(")

MAX_LINE_CHARS = 200  # long lines (minified data, etc.) get truncated in search results
MAX_DIFF_LINES = 60


class RepoToolError(ToolError):
    """A tool call the agent made that we refuse or can't satisfy.

    The message is meant for the LLM, so it should say what to do instead.
    Subclassing the SDK's ToolError is what makes MCP pass the message through;
    any other exception is treated as a crash and the model only sees a generic error.
    """


def resolve_path(root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` against `root`, refusing anything that escapes the repo.

    `resolve()` follows `..` and symlinks first, so `../../etc/passwd` or a
    symlink pointing outside the repo are both caught by the containment check.
    """
    root = root.resolve()
    target = (root / rel_path).resolve()
    if not target.is_relative_to(root):
        raise RepoToolError(f"Path '{rel_path}' is outside the target repo.")
    if ".git" in target.relative_to(root).parts:
        raise RepoToolError("The .git directory is off-limits.")
    return target


def is_test_file(rel_path: Path) -> bool:
    """True for files the agent must not edit: anything in a tests/ or test/ dir,
    test_*.py, *_test.py and conftest.py."""
    if any(part in ("tests", "test") for part in rel_path.parts[:-1]):
        return True
    name = rel_path.name
    return name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py")


def _iter_files(root: Path, glob: str) -> list[Path]:
    """Files under `root` matching `glob`, skipping ignored dirs, sorted for stable output."""
    root = root.resolve()
    files = []
    for path in root.glob(glob):
        rel = path.relative_to(root)
        if path.is_file() and not IGNORED_DIRS.intersection(rel.parts):
            files.append(path)
    return sorted(files)


def _rel(root: Path, path: Path) -> str:
    """Repo-relative path with forward slashes (same on Windows and in Docker)."""
    return path.relative_to(root.resolve()).as_posix()


def list_files(root: Path, glob: str = "**/*.py", limit: int = 200) -> dict[str, Any]:
    """List repo files matching `glob`."""
    files = _iter_files(root, glob)
    return {
        "files": [_rel(root, f) for f in files[:limit]],
        "total": len(files),
        "truncated": len(files) > limit,
    }


def search_code(root: Path, pattern: str, glob: str = "**/*.py", max_results: int = 50) -> dict[str, Any]:
    """Regex-search file contents. Returns one entry per matching line."""
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise RepoToolError(f"Invalid regex {pattern!r}: {exc}") from exc

    matches: list[dict[str, Any]] = []
    truncated = False
    for path in _iter_files(root, glob):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable file: not useful to the agent
        for lineno, line in enumerate(lines, start=1):
            if regex.search(line):
                if len(matches) == max_results:
                    truncated = True
                    break
                matches.append({"path": _rel(root, path), "line": lineno, "text": line.strip()[:MAX_LINE_CHARS]})
        if truncated:
            break
    return {"matches": matches, "truncated": truncated}


def read_file(root: Path, path: str, start_line: int = 1, end_line: int | None = None, max_lines: int = 400) -> dict[str, Any]:
    """Read a line range of a file, returned with line numbers (1-based, inclusive)."""
    target = resolve_path(root, path)
    if not target.is_file():
        raise RepoToolError(f"File '{path}' does not exist.")
    lines = target.read_text(encoding="utf-8").splitlines()
    total = len(lines)

    start = max(start_line, 1)
    end = min(end_line if end_line is not None else total, total, start + max_lines - 1)
    if start > total and total > 0:
        raise RepoToolError(f"start_line {start_line} is past the end of the file ({total} lines).")

    numbered = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))
    return {
        "path": _rel(root, target),
        "start_line": start,
        "end_line": end,
        "total_lines": total,
        "content": numbered,
    }


def edit_file(root: Path, path: str, old_text: str, new_text: str) -> dict[str, Any]:
    """Replace exactly one occurrence of `old_text` with `new_text`.

    Exact-match instead of line numbers: line numbers go stale after the first
    edit, while a text match either applies cleanly or fails loudly.
    """
    target = resolve_path(root, path)
    rel = target.relative_to(root.resolve())
    if is_test_file(rel):
        raise RepoToolError(f"'{rel.as_posix()}' is a test file. Tests must not be edited; fix the code under test instead.")
    if not target.is_file():
        raise RepoToolError(f"File '{path}' does not exist.")
    if not old_text:
        raise RepoToolError("old_text must not be empty.")
    for snippet in FORBIDDEN_SNIPPETS:
        if snippet in new_text and snippet not in old_text:
            raise RepoToolError(f"Adding '{snippet}' is not allowed: failing tests must be fixed, not skipped.")

    # Keep the file's own line endings: the LLM always sends "\n".
    raw = target.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    content = raw.replace("\r\n", "\n")
    old_text = old_text.replace("\r\n", "\n")
    new_text = new_text.replace("\r\n", "\n")

    count = content.count(old_text)
    if count == 0:
        raise RepoToolError("old_text was not found. Re-read the file and copy the text exactly, including indentation.")
    if count > 1:
        raise RepoToolError(f"old_text appears {count} times. Include more surrounding lines so it is unique.")

    line = content[: content.index(old_text)].count("\n") + 1
    updated = content.replace(old_text, new_text, 1)
    target.write_bytes(updated.replace("\n", newline).encode("utf-8"))

    diff = list(difflib.unified_diff(
        content.splitlines(), updated.splitlines(),
        fromfile=f"a/{rel.as_posix()}", tofile=f"b/{rel.as_posix()}", lineterm="", n=2,
    ))
    return {
        "path": rel.as_posix(),
        "line": line,
        "diff": "\n".join(diff[:MAX_DIFF_LINES]) + ("\n... (diff truncated)" if len(diff) > MAX_DIFF_LINES else ""),
    }
