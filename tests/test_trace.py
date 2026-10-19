"""Tests for run tracing: the event stream from run_agent, replay, and RunTrace's files."""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from agent.llm import assistant_message, LLMResponse, ToolCall, tool_result_message
from agent.loop import RunResult, ToolBox, run_agent
from agent.trace import RunTrace
from mcp_servers.repo_tools.server import build_server as build_repo_tools
from tests.test_loop import FAILING, PASSING, TASK, ScriptedLLM, call, fake_sandbox, repo  # noqa: F401  (repo is a fixture)


def run_with_events(llm: ScriptedLLM, repo_path: Path, reports: list[dict[str, Any]], on_event: Any) -> RunResult:
    sandbox, _ = fake_sandbox(reports)

    async def go() -> RunResult:
        async with ToolBox([build_repo_tools(repo_path), sandbox]) as toolbox:
            return await run_agent(llm, toolbox, TASK, on_event=on_event)

    return asyncio.run(go())


def scripted() -> ScriptedLLM:
    return ScriptedLLM(
        LLMResponse(text="Let me look first."),  # no tool call -> nudge
        call("read_file", {"path": "pkg/models.py"}, call_id="r1"),
        call("finish", {"summary": "too early"}, call_id="f1"),  # rejected: tests still fail
        call("finish", {"summary": "done"}, call_id="f2"),
    )


def test_event_stream_order(repo: Path) -> None:
    events: list[dict[str, Any]] = []
    result = run_with_events(scripted(), repo, [FAILING, FAILING, PASSING], events.append)
    assert result.success
    summary = [(e["type"], e.get("purpose") or e.get("name")) for e in events]
    assert summary == [
        ("start", None), ("test_run", "baseline"), ("user_message", None),
        ("llm_call", None), ("user_message", None),  # nudge
        ("llm_call", None), ("tool_call", "read_file"),
        ("llm_call", None), ("test_run", "verification"), ("tool_call", "finish"),
        ("llm_call", None), ("test_run", "verification"), ("tool_call", "finish"),
    ]
    assert all("ts" in e for e in events)
    assert events[0]["system"] and {t["name"] for t in events[0]["tools"]} >= {"read_file", "finish"}


def test_replaying_events_rebuilds_the_exact_history(repo: Path) -> None:
    """The trace must be enough to reconstruct what the model saw on every call."""
    llm = scripted()
    events: list[dict[str, Any]] = []
    run_with_events(llm, repo, [FAILING, FAILING, PASSING], events.append)

    rebuilt: list[dict[str, Any]] = []
    for e in events:
        if e["type"] == "user_message":
            rebuilt.append({"role": "user", "content": e["content"]})
        elif e["type"] == "llm_call":
            calls = [ToolCall(c["id"], c["name"], c["arguments"], c["error"]) for c in e["tool_calls"]]
            rebuilt.append(assistant_message(LLMResponse(text=e["text"], tool_calls=calls, provider_state=e["provider_state"])))
        elif e["type"] == "tool_call":
            rebuilt.append(tool_result_message(ToolCall(e["id"], e["name"], e["arguments"]), e["result"]))

    last_seen = llm.seen[-1]  # history sent on the final model call
    assert rebuilt[: len(last_seen)] == last_seen


def test_run_trace_writes_all_files(repo: Path, tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], check=True)

    trace = RunTrace.create(tmp_path / "runs", repo=repo, meta={"model": "scripted"})
    llm = ScriptedLLM(
        call("edit_file", {"path": "pkg/models.py", "old_text": "ModelField", "new_text": "FieldInfo"}),
        call("finish", {"summary": "done"}),
    )
    result = run_with_events(llm, repo, [FAILING, PASSING], trace)
    trace.finish(result)

    meta = json.loads((trace.path / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "scripted" and len(meta["repo_commit"]) == 40 and trace.path.name == meta["run_id"]
    events = [json.loads(line) for line in (trace.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "start" and events[-1]["name"] == "finish"
    assert json.loads((trace.path / "result.json").read_text(encoding="utf-8"))["success"] is True
    assert "+from pydantic.fields import FieldInfo" in (trace.path / "diff.patch").read_text(encoding="utf-8")
