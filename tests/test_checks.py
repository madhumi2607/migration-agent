"""Tests for the automatic review checks (agent/checks.py)."""

import subprocess
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

from agent.checks import check_diff, check_pins, check_removed_dependencies, poetry_to_pep440, read_dependencies, run_checks


@pytest.mark.parametrize(("poetry", "pep440"), [
    ("^1.0", ">=1.0,<2"), ("^1.0,<2.0", ">=1.0,<2,<2.0"), ("^0.3", ">=0.3,<0.4"),
    ("~1.2", ">=1.2,<1.3"), ("*", ""), (">=0.89,<1.0", ">=0.89,<1.0"),
])
def test_poetry_to_pep440(poetry: str, pep440: str) -> None:
    assert poetry_to_pep440(poetry) == pep440


def deps(**specs: str) -> dict[str, SpecifierSet]:
    return {name.replace("_", "-"): SpecifierSet(spec) for name, spec in specs.items()}


def test_pins_that_still_allow_the_old_version_warn() -> None:
    findings = check_pins({}, deps(pydantic=">=1.10,<3", fastapi=">=0.89,<1"), "pydantic>=2,<3 fastapi>=0.100,<0.104")
    assert [(f.severity, f.message.split(" is pinned")[0]) for f in findings] == [("warning", "pydantic"), ("warning", "fastapi")]


def test_good_and_exact_pins_pass() -> None:
    assert check_pins({}, deps(pydantic=">=2.0,<3.0"), "pydantic>=2,<3") == []
    assert check_pins({}, deps(pydantic="==2.5.0"), "pydantic>=2,<3") == []  # exact v2 pin is fine


def test_pin_excluding_the_new_version_is_an_error() -> None:
    [finding] = check_pins({}, deps(pydantic=">=1.5,<2"), "pydantic>=2,<3")
    assert finding.severity == "error" and "excludes" in finding.message


def test_unpinned_dependency_warns_and_dropped_dependency_errors() -> None:
    [unpinned] = check_pins({}, deps(pydantic=""), "pydantic>=2,<3")
    assert unpinned.severity == "warning" and "any version" in unpinned.message
    [dropped] = check_pins(deps(pydantic="<2"), {}, "pydantic>=2,<3")
    assert dropped.severity == "error" and "missing" in dropped.message


def test_removed_dependencies() -> None:
    [f] = check_removed_dependencies(deps(pydantic="<2", pydantic_settings=">=2"), deps(pydantic=">=2"))
    assert "pydantic-settings" in f.message


def test_check_diff_flags_shims_skips_and_test_edits() -> None:
    diff = "\n".join([
        "diff --git a/pkg/models.py b/pkg/models.py", "--- a/pkg/models.py", "+++ b/pkg/models.py",
        "+from pydantic.v1 import BaseModel", "+@pytest.mark.skip", "+x = 1",
        "diff --git a/tests/test_models.py b/tests/test_models.py", "--- a/tests/test_models.py", "+++ b/tests/test_models.py",
        "+assert True",
    ])
    found = {(f.check, f.file) for f in check_diff(diff, "pydantic")}
    assert found == {("compat_shim", "pkg/models.py"), ("test_integrity", "pkg/models.py"), ("test_integrity", "tests/test_models.py")}


def test_read_dependencies_from_poetry_and_requirements(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.poetry.dependencies]\npython = "^3.7"\npydantic = "^1.0,<2.0"\nsqlalchemy = { version = ">=1.4", optional = true }\n',
        encoding="utf-8")
    (tmp_path / "REQUIREMENTS.txt").write_text("requests >= 2  # comment\n", encoding="utf-8")
    (tmp_path / "REQUIREMENTS-TEST.txt").write_text("pytest\n", encoding="utf-8")  # ignored
    found = read_dependencies(tmp_path)
    assert set(found) == {"pydantic", "sqlalchemy", "requests"}
    assert not found["pydantic"].contains("2.0") and found["pydantic"].contains("1.10")


def test_run_checks_end_to_end(tmp_path: Path) -> None:
    """Before = the commit, after = commit + the run's diff, rebuilt in a temporary clone."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["pydantic>=1.8,<2", "attrs"]\n', encoding="utf-8")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(repo)]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "v1"], check=True)
    commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    (repo / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["pydantic>=1.10,<3"]\n', encoding="utf-8")
    diff = subprocess.run([*git, "diff"], capture_output=True, text=True).stdout
    subprocess.run([*git, "checkout", "-q", "."], check=True)  # like a reset checkout after the run

    findings = run_checks(repo, commit, diff, "pydantic>=2,<3", "pydantic")
    assert {(f.check, f.severity) for f in findings} == {("pins", "warning"), ("removed_dependency", "warning")}
    assert (repo / "pyproject.toml").read_text(encoding="utf-8").count("attrs") == 1  # the real checkout is untouched
