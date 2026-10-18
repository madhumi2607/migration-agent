"""Turn the raw output of a sandbox run into a short, structured summary for the LLM.

Two sources, in order of preference:
1. junit.xml - pytest's machine-readable report (normal runs, including per-module
   collection errors).
2. pytest.log - only when junit.xml is missing, which happens when pytest dies
   before it can report anything (e.g. tests/conftest.py fails to import).
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Exit codes set by sandbox/entrypoint.sh (pytest's own codes are 0-5).
EXIT_INSTALL_FAILED = 10
EXIT_UPGRADE_FAILED = 11
PYTEST_NO_TESTS = 5

MAX_GROUPS = 20          # distinct root causes reported
EXAMPLE_TESTS = 3        # test ids listed per root cause
MAX_MESSAGE_CHARS = 300
LOG_TAIL_LINES = 30

# A traceback frame in pytest's output: "pkg/module.py:12: in func" or the final
# "pkg/module.py:12: AttributeError". Relative paths are inside the repo (pytest
# prints them relative to rootdir); absolute ones are site-packages or stdlib.
FRAME_RE = re.compile(r"^(?P<path>[^\s:][^:]*\.py):(?P<line>\d+): (?P<rest>.*)$")
# An exception line: "E   AttributeError: module 'pydantic' has no attribute 'fields'"
E_LINE_RE = re.compile(r"^E\s+(?P<text>.*)$")
EXC_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Exit|Warning|Interrupt)): ?(?P<msg>.*)$")


def _repo_relative(path: str) -> str | None:
    """Map a traceback path to a repo-relative one, or None if it's outside the repo."""
    path = path.replace("\\", "/")  # pytest on Windows prints backslashes
    if path.startswith("../"):
        return None  # pytest may print files outside rootdir relative to it
    if path.startswith("/work/"):
        return path[len("/work/"):]
    if path.startswith("/"):
        return None  # /venv/..., /usr/lib/python3.11/...
    return path


def parse_traceback(text: str) -> dict[str, str | None]:
    """Extract error type, message and the deepest in-repo `file:line` from a pytest traceback.

    The deepest in-repo frame is where the agent can act: the frame below it is
    usually inside the library being upgraded, which the agent can't (and shouldn't) edit.
    """
    location = None
    frame_type = None
    for line in text.splitlines():
        match = FRAME_RE.match(line.strip())
        if not match:
            continue
        rel = _repo_relative(match["path"])
        if rel is not None:
            location = f"{rel}:{match['line']}"
        rest = match["rest"]
        if not rest.startswith("in ") and EXC_RE.match(rest + ": "):
            frame_type = rest  # final frame line names the exception, e.g. "AssertionError"

    error_type, message = frame_type, None
    for line in text.splitlines():
        e_match = E_LINE_RE.match(line)
        if not e_match:
            continue
        e_text = e_match["text"].strip()
        exc = EXC_RE.match(e_text)
        if exc:  # prefer the explicit "Type: message" line
            error_type, message = exc["type"], exc["msg"]
            break
        if message is None:
            message = e_text  # e.g. "assert 1 == 2"

    return {
        "error_type": error_type,
        "message": (message or "")[:MAX_MESSAGE_CHARS],
        "location": location,
    }


def _test_id(case: ET.Element) -> str:
    """Runnable pytest node id from an xunit1 <testcase> (has file=, classname=, name=)."""
    file = (case.get("file") or "").replace("\\", "/")
    classname = case.get("classname", "")
    name = case.get("name", "")
    if not file:
        return f"{classname}::{name}" if classname else name
    module = file.removesuffix(".py").replace("/", ".")
    parts = [file]
    if classname.startswith(module + "."):
        parts.extend(classname[len(module) + 1:].split("."))  # test classes
    if name:
        parts.append(name)
    return "::".join(parts)


def parse_junit(xml_text: str) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Return (counts, failures) from a pytest JUnit XML report."""
    root = ET.fromstring(xml_text)
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")

    counts = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    failures: list[dict[str, Any]] = []
    for suite in suites:
        for case in suite.iter("testcase"):
            counts["total"] += 1
            problem = case.find("failure")
            kind = "failed"
            if problem is None:
                problem = case.find("error")
                kind = "errors"
            if problem is None:
                counts["skipped" if case.find("skipped") is not None else "passed"] += 1
                continue
            counts[kind] += 1
            details = parse_traceback(problem.text or "")
            if not details["error_type"] and not details["message"]:
                details["message"] = (problem.get("message") or "")[:MAX_MESSAGE_CHARS]
            failures.append({"test": _test_id(case), "kind": "failure" if kind == "failed" else "error", **details})
    return counts, failures


def group_failures(failures: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Group failures that share a root cause (same type, message and location).

    Returns (groups sorted by size, number of groups dropped by the cap).
    """
    groups: dict[tuple, dict[str, Any]] = {}
    for f in failures:
        key = (f["kind"], f["error_type"], f["message"], f["location"])
        group = groups.setdefault(key, {
            "kind": f["kind"], "error_type": f["error_type"], "message": f["message"],
            "location": f["location"], "count": 0, "example_tests": [],
        })
        group["count"] += 1
        if len(group["example_tests"]) < EXAMPLE_TESTS:
            group["example_tests"].append(f["test"])
    ordered = sorted(groups.values(), key=lambda g: g["count"], reverse=True)
    return ordered[:MAX_GROUPS], max(len(ordered) - MAX_GROUPS, 0)


def parse_startup_error(log: str) -> dict[str, Any]:
    """Summarise a run where pytest crashed before writing junit.xml."""
    conftest = re.search(r"while loading conftest '(?:/work/)?([^']+)'", log)
    tail = "\n".join(log.splitlines()[-60:])  # the relevant traceback is at the end
    return {
        "kind": "startup_error",
        **parse_traceback(tail),
        "count": 1,
        "example_tests": [conftest[1]] if conftest else [],
    }


def installed_versions(freeze: str, packages: list[str]) -> dict[str, str]:
    """Look up the installed version of each package in `pip freeze` output."""
    wanted = {p.lower().replace("_", "-") for p in packages}
    versions = {}
    for line in freeze.splitlines():
        name, sep, version = line.partition("==")
        if sep and name.lower().replace("_", "-") in wanted:
            versions[name] = version
    return versions


def _tail(text: str) -> str:
    return "\n".join(text.splitlines()[-LOG_TAIL_LINES:])


def build_report(out_dir: Path, exit_code: int, upgrade_packages: list[str]) -> dict[str, Any]:
    """Build the summary returned to the agent from a finished run's /out directory."""
    def read(name: str) -> str:
        path = out_dir / name
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    report: dict[str, Any] = {"exit_code": exit_code}
    if exit_code in (EXIT_INSTALL_FAILED, EXIT_UPGRADE_FAILED):
        report["status"] = "install_failed" if exit_code == EXIT_INSTALL_FAILED else "upgrade_failed"
        report["log_tail"] = _tail(read("install.log"))
        return report

    report["versions"] = installed_versions(read("freeze.txt"), upgrade_packages)
    junit = read("junit.xml")
    if not junit:
        report["status"] = "startup_error"
        report["failure_groups"] = [parse_startup_error(read("pytest.log"))]
        return report

    counts, failures = parse_junit(junit)
    groups, dropped = group_failures(failures)
    if exit_code == 0:
        status = "passed"
    elif exit_code == PYTEST_NO_TESTS:
        status = "no_tests"
    elif counts["failed"] or counts["errors"]:
        status = "failed"
    else:
        status = "pytest_error"  # e.g. bad command-line args: nothing ran
        report["log_tail"] = _tail(read("pytest.log"))
    report.update(status=status, counts=counts, failure_groups=groups)
    if dropped:
        report["groups_not_shown"] = dropped
    return report
