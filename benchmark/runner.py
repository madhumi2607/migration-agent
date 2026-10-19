"""Run benchmark experiments: every task x model x docs setting, one agent run each.

    uv run --env-file .env python -m benchmark.runner --name haiku-docs \\
        --models claude-haiku-5-5 --docs both [--tasks geojson-pydantic,pydantic-cli]

Each run is a separate `python -m agent.loop --task ...` process (the same code path as a
manual run; a crash can't leak into the next run). The model is chosen through the
environment, which takes precedence over .env. Results go to
benchmark/results/<name>-<timestamp>.json, one row per run, linking to runs/<run-id>/.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.tasks import Task, load_tasks
from dashboard.runs import read_checks, summarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RUNS_DIR = PROJECT_ROOT / "runs"


def reset_checkout(task: Task) -> None:
    """Discard the previous run's edits. Safe: every run's diff is already in runs/<id>/diff.patch."""
    if not (task.checkout / ".git").is_dir():
        return  # agent.loop clones it
    if subprocess.run(["git", "-C", str(task.checkout), "status", "--porcelain"], capture_output=True, text=True).stdout.strip():
        print(f"  resetting {task.checkout.name} (previous edits are kept in runs/<id>/diff.patch)")
        subprocess.run(["git", "-C", str(task.checkout), "checkout", "--quiet", "."], check=True)
        subprocess.run(["git", "-C", str(task.checkout), "clean", "--quiet", "-fd"], check=True)


def failure_category(summary: dict[str, Any]) -> str | None:
    """Automatic part of CLAUDE.md's failure categories. Missed file / misread docs / behavior
    change need a human reading the trace, so those start as needs_review."""
    if summary["status"] == "succeeded":
        return "passed_with_findings" if (summary["check_errors"] or summary["check_warnings"]) else None
    if summary["stop_reason"] in ("max_steps", "token_budget"):
        return "budget_exhausted"
    if summary["stop_reason"] in ("llm_error", "sandbox_error"):
        return "infrastructure"
    return "needs_review"


def diff_size(run_dir: Path) -> dict[str, int]:
    diff = (run_dir / "diff.patch").read_text(encoding="utf-8") if (run_dir / "diff.patch").is_file() else ""
    lines = diff.splitlines()
    return {
        "files_changed": sum(1 for l in lines if l.startswith("+++ b/")),
        "lines_added": sum(1 for l in lines if l.startswith("+") and not l.startswith("+++")),
        "lines_removed": sum(1 for l in lines if l.startswith("-") and not l.startswith("---")),
    }


def result_row(task: Task, model: str, docs: bool, run_dir: Path) -> dict[str, Any]:
    s = summarize(run_dir)
    checks = read_checks(run_dir) or []
    return {
        "task": task.id, "difficulty": task.difficulty, "model": model, "docs_search": docs,
        "run_id": s["run_id"], "success": s["status"] == "succeeded", "stop_reason": s["stop_reason"],
        "failure_category": failure_category(s),
        "steps": s["steps"], "input_tokens": s["input_tokens"], "cache_read_tokens": s["cache_read_tokens"],
        "output_tokens": s["output_tokens"], "cost_usd": s["cost_usd"], "duration_seconds": s["duration_seconds"],
        "tests_before": s["tests_before"], "tests_after": s["tests_after"],
        **diff_size(run_dir),
        "check_errors": s["check_errors"], "check_warnings": s["check_warnings"],
        "check_findings": [f"[{f['severity']}] {f['check']}: {f['message']}" for f in checks],
    }


def run_one(task: Task, model: str, docs: bool, max_steps: int) -> Path | None:
    """One agent run in a subprocess; returns its runs/<id> directory (None if it never started)."""
    reset_checkout(task)
    cmd = [sys.executable, "-m", "agent.loop", "--task", task.id, "--max-steps", str(max_steps)]
    if not docs:
        cmd.append("--no-docs")
    env = {**os.environ, "LLM_PROVIDER": "claude", "AGENT_CLAUDE_MODEL": model}
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    match = re.search(r"Recording run to (.+)", proc.stderr)
    if not match:
        print(f"  run did not start (exit {proc.returncode}):\n{proc.stderr[-1500:]}")
        return None
    return Path(match[1].strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark experiment.")
    parser.add_argument("--name", required=True, help="experiment name, used in the results file name")
    parser.add_argument("--models", default="claude-opus-5-5", help="comma-separated Claude model ids")
    parser.add_argument("--docs", choices=["on", "off", "both"], default="on")
    parser.add_argument("--tasks", default="all", help="comma-separated task ids, or 'all'")
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()

    all_tasks = load_tasks()
    tasks = list(all_tasks.values()) if args.tasks == "all" else [all_tasks[t] for t in args.tasks.split(",")]
    models = args.models.split(",")
    docs_settings = {"on": [True], "off": [False], "both": [False, True]}[args.docs]
    plan = [(t, m, d) for t in tasks for m in models for d in docs_settings]

    started = datetime.now(timezone.utc)
    out = RESULTS_DIR / f"{args.name}-{started:%Y%m%d-%H%M%S}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    experiment: dict[str, Any] = {
        "name": args.name, "started_at": started.isoformat(),
        "config": {"models": models, "docs": args.docs, "tasks": [t.id for t in tasks], "max_steps": args.max_steps},
        "rows": [],
    }
    for i, (task, model, docs) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {task.id} · {model} · docs {'on' if docs else 'off'}", flush=True)
        run_dir = run_one(task, model, docs, args.max_steps)
        if run_dir is None:
            experiment["rows"].append({"task": task.id, "model": model, "docs_search": docs, "success": False,
                                       "failure_category": "infrastructure", "run_id": None})
        else:
            row = result_row(task, model, docs, run_dir)
            experiment["rows"].append(row)
            cost = f"${row['cost_usd']:.2f}" if row["cost_usd"] is not None else "?"
            print(f"  -> {'OK' if row['success'] else 'FAIL'} · {row['steps']} steps · {cost} · {row['run_id']}", flush=True)
        out.write_text(json.dumps(experiment, indent=1, default=str), encoding="utf-8")  # saved after every run
    for task in tasks:
        reset_checkout(task)

    rows = experiment["rows"]
    total = sum(r.get("cost_usd") or 0 for r in rows)
    print(f"\n{sum(r['success'] for r in rows)}/{len(rows)} succeeded · est. ${total:.2f} · {out}")


if __name__ == "__main__":
    main()
