"""Tests for the benchmark runner's bookkeeping (the agent runs themselves are not executed)."""

import json
from pathlib import Path

from benchmark.runner import diff_size, failure_category, result_row
from benchmark.tasks import load_tasks


def summary(**overrides):
    base = {"status": "succeeded", "stop_reason": "finished", "check_errors": 0, "check_warnings": 0}
    return {**base, **overrides}


def test_failure_categories() -> None:
    assert failure_category(summary()) is None
    assert failure_category(summary(check_warnings=1)) == "passed_with_findings"
    assert failure_category(summary(status="failed", stop_reason="max_steps")) == "budget_exhausted"
    assert failure_category(summary(status="failed", stop_reason="token_budget")) == "budget_exhausted"
    assert failure_category(summary(status="failed", stop_reason="sandbox_error")) == "infrastructure"
    assert failure_category(summary(status="failed", stop_reason="finished")) == "needs_review"


def test_result_row_from_a_recorded_run(tmp_path: Path) -> None:
    run = tmp_path / "20261019-100000-pydantic-cli"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"run_id": run.name, "started_at": "2026-10-19T10:00:00+00:00",
                                               "repo": "/x/pydantic-cli", "model": "claude-haiku-5-5"}), encoding="utf-8")
    events = [
        {"type": "test_run", "purpose": "baseline", "report": {"status": "failed", "counts": {"total": 15, "passed": 0}}},
        {"type": "llm_call", "step": 1, "input_tokens": 1000, "cache_read_tokens": 0, "output_tokens": 100},
        {"type": "test_run", "purpose": "verification", "report": {"status": "passed", "counts": {"total": 35, "passed": 35}}},
    ]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    (run / "result.json").write_text(json.dumps({"success": True, "stop_reason": "finished",
                                                 "finished_at": "2026-10-19T10:05:00+00:00"}), encoding="utf-8")
    (run / "diff.patch").write_text("+++ b/a.py\n+x = 2\n-x = 1\n+++ b/setup.py\n+y\n", encoding="utf-8")
    (run / "checks.json").write_text(json.dumps([{"severity": "warning", "check": "pins", "message": "m", "file": None}]),
                                     encoding="utf-8")

    row = result_row(load_tasks()["pydantic-cli"], "claude-haiku-5-5", True, run)
    assert row["success"] and row["failure_category"] == "passed_with_findings"
    assert row["tests_after"] == {"status": "passed", "passed": 35, "total": 35}
    assert (row["files_changed"], row["lines_added"], row["lines_removed"]) == (2, 2, 1)
    assert row["cost_usd"] is not None and row["check_findings"] == ["[warning] pins: m"]
    assert diff_size(tmp_path / "missing") == {"files_changed": 0, "lines_added": 0, "lines_removed": 0}
