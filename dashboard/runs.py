"""Read recorded runs (runs/<run-id>/, written by agent/trace.py) for the dashboard.

Plain functions over files, so they can be tested without a web server. Summaries are
computed from events.jsonl rather than result.json, so a run that is still in progress
(no result.json yet) shows live numbers.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.pricing import estimate_cost


def list_run_ids(runs_dir: Path) -> list[str]:
    """Run ids, newest first (ids start with a UTC timestamp, so name order is time order)."""
    if not runs_dir.is_dir():
        return []
    return sorted((p.name for p in runs_dir.iterdir() if (p / "meta.json").is_file()), reverse=True)


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "events.jsonl"
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a run in progress may have a half-written last line
    return events


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _tests(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Passed/total from a sandbox report, or None if there was no test run yet."""
    if report is None:
        return None
    counts = report.get("counts") or {}
    return {"status": report.get("status"), "passed": counts.get("passed", 0), "total": counts.get("total")}


def summarize(run_dir: Path) -> dict[str, Any]:
    """The numbers shown for one run in the runs list and run header."""
    meta = _read_json(run_dir / "meta.json") or {}
    result = _read_json(run_dir / "result.json")
    events = read_events(run_dir)

    llm_calls = [e for e in events if e["type"] == "llm_call"]
    test_runs = [e for e in events if e["type"] == "test_run"]
    tokens = {key: sum(e.get(key, 0) for e in llm_calls) for key in ("input_tokens", "cache_read_tokens", "output_tokens")}

    if result is None:
        status = "running"
    else:
        status = "succeeded" if result.get("success") else "failed"
    finished_at = result.get("finished_at") if result else None
    duration = None
    if finished_at and meta.get("started_at"):
        duration = (datetime.fromisoformat(finished_at) - datetime.fromisoformat(meta["started_at"])).total_seconds()

    return {
        "run_id": meta.get("run_id", run_dir.name),
        "started_at": meta.get("started_at"),
        "finished_at": finished_at,
        "duration_seconds": duration,
        "repo": Path(meta["repo"]).name if meta.get("repo") else None,
        "repo_commit": (meta.get("repo_commit") or "")[:7] or None,
        "provider": meta.get("provider"),
        "model": meta.get("model"),
        "effort": meta.get("effort"),
        "status": status,
        "stop_reason": result.get("stop_reason") if result else None,
        "error": result.get("error") if result else None,
        "summary": result.get("summary") if result else None,
        "steps": len(llm_calls),
        **tokens,
        "cost_usd": estimate_cost(meta.get("model"), **tokens),
        "tests_before": _tests(test_runs[0]["report"] if test_runs else None),
        "tests_after": _tests(test_runs[-1]["report"] if len(test_runs) > 1 else None),
    }


def run_detail(run_dir: Path) -> dict[str, Any]:
    """Everything the run page shows: settings, summary, the event timeline and the diff."""
    diff_path = run_dir / "diff.patch"
    events = [{k: v for k, v in e.items() if k != "provider_state"}  # opaque signed blobs: no use in the UI
              for e in read_events(run_dir)]
    return {
        "meta": _read_json(run_dir / "meta.json"),
        "summary": summarize(run_dir),
        "events": events,
        "diff": diff_path.read_text(encoding="utf-8") if diff_path.is_file() else None,
    }
