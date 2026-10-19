"""Tests for the dashboard backend: run summaries and the JSON API."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent.pricing import estimate_cost
from dashboard import api
from dashboard.runs import list_run_ids, summarize

FAILED_REPORT = {"status": "failed", "counts": {"total": 3, "passed": 0, "failed": 0, "errors": 3, "skipped": 0}}
PASSED_REPORT = {"status": "passed", "counts": {"total": 134, "passed": 134, "failed": 0, "errors": 0, "skipped": 0}}


def write_run(runs: Path, run_id: str, events: list[dict[str, Any]], result: dict[str, Any] | None = None) -> Path:
    run = runs / run_id
    run.mkdir(parents=True)
    meta = {"run_id": run_id, "started_at": "2026-10-19T10:00:00+00:00", "repo": "/x/geojson-pydantic",
            "repo_commit": "8a07ab86373fb289c6a4ce34c2105a777debe380", "provider": "claude", "model": "claude-opus-5-5"}
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    if result is not None:
        (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (run / "diff.patch").write_text("--- a/x.py\n+++ b/x.py\n", encoding="utf-8")
    return run


def llm_call(step: int, inp: int, cached: int, out: int) -> dict[str, Any]:
    return {"type": "llm_call", "step": step, "input_tokens": inp, "cache_read_tokens": cached, "output_tokens": out,
            "text": None, "tool_calls": [], "provider_state": {"provider": "claude", "content": ["big blob"]}}


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runs = tmp_path / "runs"
    write_run(runs, "20261019-100000-geojson-pydantic", [
        {"type": "start"}, {"type": "test_run", "purpose": "baseline", "report": FAILED_REPORT},
        llm_call(1, 2_000, 0, 100), llm_call(2, 10_000, 2_000, 400),
        {"type": "test_run", "purpose": "verification", "report": PASSED_REPORT},
    ], result={"success": True, "stop_reason": "finished", "summary": "Migrated.", "finished_at": "2026-10-19T10:07:30+00:00"})
    (runs / "20261019-100000-geojson-pydantic" / "checks.json").write_text(json.dumps([
        {"severity": "warning", "check": "pins", "message": "fastapi pin allows <0.100", "file": None}]), encoding="utf-8")
    # A run still in progress: no result.json, and a half-written last line.
    live = write_run(runs, "20261019-110000-geojson-pydantic", [
        {"type": "start"}, {"type": "test_run", "purpose": "baseline", "report": FAILED_REPORT}, llm_call(1, 2_000, 0, 50)])
    with (live / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"type": "llm_ca')
    (runs / "not-a-run").mkdir()  # no meta.json: ignored
    monkeypatch.setattr(api, "RUNS_DIR", runs)
    return runs


def test_list_run_ids_newest_first(runs: Path) -> None:
    assert list_run_ids(runs) == ["20261019-110000-geojson-pydantic", "20261019-100000-geojson-pydantic"]


def test_summary_of_finished_run(runs: Path) -> None:
    s = summarize(runs / "20261019-100000-geojson-pydantic")
    assert (s["status"], s["stop_reason"], s["steps"]) == ("succeeded", "finished", 2)
    assert (s["input_tokens"], s["cache_read_tokens"], s["output_tokens"]) == (12_000, 2_000, 500)
    assert s["tests_before"] == {"status": "failed", "passed": 0, "total": 3}
    assert s["tests_after"] == {"status": "passed", "passed": 134, "total": 134}
    assert s["duration_seconds"] == 450 and s["repo"] == "geojson-pydantic" and s["repo_commit"] == "8a07ab8"
    assert s["cost_usd"] == pytest.approx((10_000 * 4 + 2_000 * 0.2 + 500 * 20) / 1e6)
    assert (s["check_errors"], s["check_warnings"]) == (0, 1)


def test_summary_of_live_run(runs: Path) -> None:
    s = summarize(runs / "20261019-110000-geojson-pydantic")
    assert s["status"] == "running" and s["steps"] == 1 and s["tests_after"] is None and s["duration_seconds"] is None
    assert s["check_errors"] is None  # not checked yet


def test_cost_only_for_known_models() -> None:
    assert estimate_cost("gemini-3.8-flash", 1000, 0, 10) is None
    assert estimate_cost("claude-haiku-5-5", 1_000_000, 0, 0) == pytest.approx(0.10)


def test_api_lists_and_details_runs(runs: Path) -> None:
    client = TestClient(api.app)
    listed = client.get("/api/runs").json()
    assert [r["status"] for r in listed] == ["running", "succeeded"]

    detail = client.get("/api/runs/20261019-100000-geojson-pydantic").json()
    assert detail["summary"]["status"] == "succeeded" and detail["diff"].startswith("--- a/x.py")
    assert len(detail["events"]) == 5 and all("provider_state" not in e for e in detail["events"])
    assert detail["checks"][0]["check"] == "pins"


@pytest.mark.parametrize("bad_id", ["nope", "not-a-run", "..%2F..%2F.env", "..%5C..%5C.env"])
def test_api_rejects_unknown_or_traversal_ids(runs: Path, bad_id: str) -> None:
    assert TestClient(api.app).get(f"/api/runs/{bad_id}").status_code == 404


def test_unknown_api_paths_are_404_not_the_app(runs: Path) -> None:
    """With the React build present, unknown paths serve index.html, except under /api/."""
    client = TestClient(api.app)
    if not api.WEB_DIST.is_dir():
        pytest.skip("frontend not built (npm run build in dashboard/web)")
    assert client.get("/runs/anything").headers["content-type"].startswith("text/html")
    for path in ("/api/nope", "/api/runs/..%2F..%2F.env"):
        response = client.get(path)
        assert response.status_code == 404 and response.headers["content-type"] == "application/json"
