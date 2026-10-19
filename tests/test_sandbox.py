"""Tests for the sandbox server: report parsing, docker command building, and the MCP layer.

The end-to-end test against the real target repo needs Docker and is opt-in:
    uv run pytest -m docker
"""

import asyncio
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from mcp import Client

from mcp_servers.sandbox import report, runner
from mcp_servers.sandbox.runner import SandboxConfig, SandboxError
from mcp_servers.sandbox.server import build_server

FIXTURES = Path(__file__).parent / "fixtures"
PROJECT_ROOT = Path(__file__).parent.parent


# --- startup errors (no junit.xml): the real Pydantic v2 run of pydantic-argparse ---

def test_startup_error_from_real_log() -> None:
    log = (FIXTURES / "pytest_conftest_import_error.log").read_text(encoding="utf-8")
    group = report.parse_startup_error(log)
    assert group["kind"] == "startup_error"
    assert group["error_type"] == "AttributeError"
    assert group["message"] == "module 'pydantic' has no attribute 'fields'"
    # The deepest frame *in the repo*, not the one inside site-packages/pydantic.
    assert group["location"] == "pydantic_argparse/utils/arguments.py:12"
    assert group["example_tests"] == ["tests/conftest.py"]


def test_build_report_without_junit(tmp_path: Path) -> None:
    (tmp_path / "pytest.log").write_text((FIXTURES / "pytest_conftest_import_error.log").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "freeze.txt").write_text("pydantic==2.14.1\npydantic_core==2.50.1\n", encoding="utf-8")
    result = report.build_report(tmp_path, exit_code=4, upgrade_packages=["pydantic"])
    assert result["status"] == "startup_error"
    assert result["versions"] == {"pydantic": "2.14.1"}
    assert len(result["failure_groups"]) == 1


# --- junit.xml produced by a real pytest run on a tiny fake repo ---

@pytest.fixture(scope="module")
def junit_xml(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Run pytest on a small repo whose tests fail in different ways; return its junit.xml."""
    repo = tmp_path_factory.mktemp("fake_repo")
    lib = tmp_path_factory.mktemp("site_packages")  # stands in for an installed library
    (lib / "fakelib.py").write_text("def removed_api():\n    raise AttributeError('removed_api was removed in v2')\n", encoding="utf-8")
    (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (repo / "app.py").write_text("import fakelib\n\n\ndef call():\n    return fakelib.removed_api()\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_app.py").write_text(textwrap.dedent("""
        import pytest
        import app

        def test_ok():
            assert True

        def test_uses_removed_api_1():
            app.call()

        def test_uses_removed_api_2():
            app.call()

        class TestMath:
            def test_wrong(self):
                assert 1 + 1 == 3

        @pytest.mark.skip(reason="demo")
        def test_skipped():
            pass
    """), encoding="utf-8")
    (repo / "tests" / "test_broken_import.py").write_text("import does_not_exist\n", encoding="utf-8")

    out = repo / "junit.xml"
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests", f"--junitxml={out}", "-o", "junit_family=xunit1",
         "--continue-on-collection-errors", "-p", "no:cacheprovider"],
        cwd=repo, env={"PYTHONPATH": str(lib), "SYSTEMROOT": "C:\\Windows", "PATH": ""}, capture_output=True,
    )
    return out.read_text(encoding="utf-8")


def test_parse_junit_counts(junit_xml: str) -> None:
    counts, _ = report.parse_junit(junit_xml)
    assert counts == {"total": 6, "passed": 1, "failed": 3, "errors": 1, "skipped": 1}


def test_parse_junit_failures(junit_xml: str) -> None:
    _, failures = report.parse_junit(junit_xml)
    by_test = {f["test"]: f for f in failures}

    # Exception raised inside the library: location is the repo frame that called it.
    removed = by_test["tests/test_app.py::test_uses_removed_api_1"]
    assert removed["error_type"] == "AttributeError"
    assert removed["message"] == "removed_api was removed in v2"
    assert removed["location"] == "app.py:5"

    # Class-based test gets a runnable id; plain assertion has no "Type: msg" line.
    wrong = by_test["tests/test_app.py::TestMath::test_wrong"]
    assert wrong["error_type"] == "AssertionError"
    assert wrong["message"].startswith("assert (1 + 1) == 3")

    # Import error while collecting a module is reported as an error, not a failure.
    broken = next(f for f in failures if f["kind"] == "error")
    assert broken["error_type"] == "ModuleNotFoundError"
    assert "does_not_exist" in broken["message"]


def test_group_failures_merges_same_root_cause(junit_xml: str) -> None:
    _, failures = report.parse_junit(junit_xml)
    groups, dropped = report.group_failures(failures)
    assert dropped == 0
    top = groups[0]
    assert top["count"] == 2 and top["location"] == "app.py:5"
    assert top["example_tests"] == ["tests/test_app.py::test_uses_removed_api_1", "tests/test_app.py::test_uses_removed_api_2"]


def test_build_report_statuses(tmp_path: Path, junit_xml: str) -> None:
    (tmp_path / "install.log").write_text("ERROR: No matching distribution found for nope\n", encoding="utf-8")
    failed_install = report.build_report(tmp_path, exit_code=10, upgrade_packages=[])
    assert failed_install["status"] == "install_failed"
    assert "No matching distribution" in failed_install["log_tail"]

    (tmp_path / "junit.xml").write_text(junit_xml, encoding="utf-8")
    failed = report.build_report(tmp_path, exit_code=1, upgrade_packages=[])
    assert failed["status"] == "failed" and failed["counts"]["failed"] == 3


# --- docker command ---

def test_upgrade_packages() -> None:
    assert runner.upgrade_packages('pydantic>=2,<3 "email-validator==2.*"') == ["pydantic", "email-validator"]
    assert runner.upgrade_packages("") == []


def test_build_docker_command(tmp_path: Path) -> None:
    config = SandboxConfig(repo_root=tmp_path, install_cmd="pip install .", test_cmd="pytest tests", upgrade="pydantic>=2",
                           test_deps='pytest "shapely<2.1"')
    cmd = runner.build_docker_command(config, tmp_path / "out", "c1", ["tests/test_a.py::test_x[a b]"])
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert f"{tmp_path.resolve()}:/src:ro" in cmd  # host repo is read-only
    assert "TEST_CMD=pytest tests 'tests/test_a.py::test_x[a b]'" in cmd  # quoted for the container shell
    assert 'TEST_DEPS=pytest "shapely<2.1"' in cmd
    assert cmd[-1] == config.image


def test_build_docker_command_refuses_flags(tmp_path: Path) -> None:
    config = SandboxConfig(repo_root=tmp_path, install_cmd="x", test_cmd="pytest")
    with pytest.raises(SandboxError, match="pytest option"):
        runner.build_docker_command(config, tmp_path, "c1", ["--deselect=tests/test_a.py::test_x"])


# --- MCP layer ---

def test_server_over_mcp(tmp_path: Path) -> None:
    config = SandboxConfig(repo_root=tmp_path, install_cmd="x", test_cmd="pytest")

    async def scenario() -> None:
        async with Client(build_server(config)) as client:
            listed = await client.list_tools()
            assert [t.name for t in listed.tools] == ["run_tests"]
            refused = await client.call_tool("run_tests", {"tests": ["-k", "not slow"]})
            assert refused.is_error and "pytest option" in refused.content[0].text

    asyncio.run(scenario())


# --- end to end: real Docker, real target repo (opt-in) ---

@pytest.mark.docker
def test_geojson_pydantic_under_v2() -> None:
    """The benchmark's first task (benchmark/tasks.yaml): one root cause blocks every test module."""
    repo = PROJECT_ROOT / "workspace" / "geojson-pydantic"
    if not repo.is_dir():
        pytest.fail(f"Clone the target first: git clone https://github.com/developmentseed/geojson-pydantic {repo} "
                    "&& git -C {repo} checkout 8a07ab86373fb289c6a4ce34c2105a777debe380")
    config = SandboxConfig(
        repo_root=repo,
        install_cmd="pip install -e .",
        test_deps='pytest "shapely>=2.0,<2.1"',
        test_cmd="pytest tests",
        upgrade="pydantic>=2,<3",
    )
    result = runner.run_tests(config)
    assert result["status"] == "failed", result
    assert result["versions"]["pydantic"].startswith("2.")
    [group] = result["failure_groups"]  # editable install: all 3 import errors share one in-repo location
    assert group["count"] == 3 and group["location"] == "geojson_pydantic/geometries.py:9"


def test_broken_container_is_sandbox_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exit codes pytest never uses (here 255 from an unrunnable entrypoint) must not look like test results."""
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 255, stdout="", stderr="exec /usr/local/bin/run-tests: no such file or directory")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    config = SandboxConfig(repo_root=tmp_path, install_cmd="x", test_cmd="pytest")
    result = runner.run_tests(config)
    assert result["status"] == "sandbox_error"
    assert "no such file" in result["detail"]


def test_docker_engine_down_is_sandbox_error_despite_exit_code_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the engine down, `docker run` exits 1 (same as pytest's "tests failed") and writes nothing."""
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_tests(SandboxConfig(repo_root=tmp_path, install_cmd="x", test_cmd="pytest"))
    assert result["status"] == "sandbox_error" and result["exit_code"] == 1
    assert "docker API" in result["detail"]
