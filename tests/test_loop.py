"""Tests for the agent loop.

Uses a scripted fake LLM, the real repo_tools server on a temp repo, and a fake
sandbox server whose run_tests returns scripted reports - all over in-process MCP.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from mcp.server.mcpserver import MCPServer

from agent.llm import LLMResponse, Message, ToolCall, ToolSpec
from agent.loop import LoopConfig, RunResult, TaskSpec, ToolBox, run_agent
from mcp_servers.repo_tools.server import build_server as build_repo_tools

TASK = TaskSpec("pydantic", "1", "2")
FAILING = {"status": "startup_error", "failure_groups": [{"error_type": "AttributeError", "location": "pkg/models.py:1"}]}
PASSING = {"status": "passed", "counts": {"total": 3, "passed": 3}}


class ScriptedLLM:
    """Returns pre-written responses in order and records every conversation it was sent."""

    model = "scripted"

    def __init__(self, *responses: LLMResponse) -> None:
        self.responses = list(responses)
        self.seen: list[list[Message]] = []

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        self.seen.append([dict(m) for m in messages])
        self.tool_names = [t.name for t in tools]
        if not self.responses:
            return call("list_files", {})  # keep "working" forever, to exercise budgets
        return self.responses.pop(0)


def call(name: str, arguments: dict[str, Any], *, call_id: str | None = None, error: str | None = None) -> LLMResponse:
    return LLMResponse(text=None, tool_calls=[ToolCall(call_id or f"id_{name}", name, arguments, error)], input_tokens=100, output_tokens=10)


def fake_sandbox(reports: list[dict[str, Any]]) -> tuple[MCPServer, list[int]]:
    """A sandbox server whose run_tests returns the given reports in order (the last one repeats)."""
    server = MCPServer("sandbox")
    calls: list[int] = []

    @server.tool()
    def run_tests(tests: list[str] | None = None) -> dict[str, Any]:
        """Run tests."""
        calls.append(1)
        return reports[min(len(calls) - 1, len(reports) - 1)]

    return server, calls


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "models.py").write_text("from pydantic.fields import ModelField\n", encoding="utf-8")
    return tmp_path


def run(llm: ScriptedLLM, repo: Path, reports: list[dict[str, Any]], config: LoopConfig | None = None) -> tuple[RunResult, list[int]]:
    sandbox, sandbox_calls = fake_sandbox(reports)

    async def go() -> RunResult:
        async with ToolBox([build_repo_tools(repo), sandbox]) as toolbox:
            return await run_agent(llm, toolbox, TASK, config)

    return asyncio.run(go()), sandbox_calls


def test_happy_path_edits_file_and_verifies(repo: Path) -> None:
    llm = ScriptedLLM(
        call("read_file", {"path": "pkg/models.py"}),
        call("edit_file", {"path": "pkg/models.py", "old_text": "from pydantic.fields import ModelField", "new_text": "from pydantic.fields import FieldInfo"}),
        call("finish", {"summary": "Replaced ModelField with FieldInfo."}),
    )
    result, sandbox_calls = run(llm, repo, [FAILING, PASSING])

    assert result.success and result.stop_reason == "finished"
    assert result.summary == "Replaced ModelField with FieldInfo."
    assert result.steps == 3 and result.input_tokens == 300
    assert len(sandbox_calls) == 2  # baseline + verification
    assert "FieldInfo" in (repo / "pkg" / "models.py").read_text(encoding="utf-8")
    # The model saw the baseline failures in its first message, and has the finish tool.
    assert "startup_error" in llm.seen[0][0]["content"]
    assert "finish" in llm.tool_names and "run_tests" in llm.tool_names


def test_finish_is_rejected_while_tests_fail(repo: Path) -> None:
    llm = ScriptedLLM(call("finish", {"summary": "done?"}, call_id="f1"), call("finish", {"summary": "now done"}, call_id="f2"))
    result, sandbox_calls = run(llm, repo, [FAILING, FAILING, PASSING])

    assert result.success and result.summary == "now done"
    rejection = next(m for m in llm.seen[1] if m.get("tool_call_id") == "f1")
    assert "finish was rejected" in rejection["content"]
    assert len(sandbox_calls) == 3


def test_stops_at_max_steps_and_runs_final_report(repo: Path) -> None:
    result, sandbox_calls = run(ScriptedLLM(), repo, [FAILING], LoopConfig(max_steps=3))
    assert not result.success and result.stop_reason == "max_steps"
    assert result.steps == 3
    assert result.final_report["status"] == "startup_error"
    assert len(sandbox_calls) == 2  # baseline + final


def test_stops_at_token_budget(repo: Path) -> None:
    result, _ = run(ScriptedLLM(), repo, [FAILING], LoopConfig(max_total_tokens=250))
    assert result.stop_reason == "token_budget" and result.steps == 3  # 110 tokens per step


def test_already_passing_does_not_call_llm(repo: Path) -> None:
    llm = ScriptedLLM()
    result, _ = run(llm, repo, [PASSING])
    assert result.success and result.stop_reason == "already_passing" and llm.seen == []


def test_mistakes_are_fed_back_to_the_model(repo: Path) -> None:
    llm = ScriptedLLM(
        LLMResponse(text="I think I should look around."),                              # no tool call
        call("delete_everything", {}, call_id="u1"),                                     # unknown tool
        call("read_file", {}, call_id="j1", error="Tool arguments were not valid JSON"),  # bad JSON
        call("edit_file", {"path": "tests/test_x.py", "old_text": "a", "new_text": "b"}, call_id="t1"),  # refused by repo_tools
        call("finish", {"summary": "ok"}),
    )
    result, _ = run(llm, repo, [FAILING, PASSING])
    assert result.success

    history = llm.seen[-1]
    by_id = {m["tool_call_id"]: m["content"] for m in history if m["role"] == "tool"}
    assert history[2] == {"role": "user", "content": "Continue by calling a tool. If every test passes, call finish with a summary of your changes."}
    assert by_id["u1"].startswith("ERROR: Unknown tool 'delete_everything'")
    assert "not valid JSON" in by_id["j1"]
    assert by_id["t1"].startswith("ERROR:") and "test file" in by_id["t1"]


def test_long_tool_results_are_truncated(repo: Path) -> None:
    (repo / "pkg" / "big.py").write_text("x = 1\n" * 5000, encoding="utf-8")
    llm = ScriptedLLM(call("read_file", {"path": "pkg/big.py"}, call_id="r1"), call("finish", {"summary": "ok"}))
    run(llm, repo, [FAILING, PASSING], LoopConfig(max_tool_result_chars=500))
    content = next(m["content"] for m in llm.seen[-1] if m.get("tool_call_id") == "r1")
    assert len(content) < 600 and "[truncated" in content


def test_unknown_and_missing_arguments_are_reported(repo: Path) -> None:
    llm = ScriptedLLM(
        call("search_code", {"path": "**/*.py", "pattern": "x"}, call_id="s1"),  # `path` is not an argument
        call("read_file", {"start_line": 1}, call_id="r1"),                      # `path` is required
        call("finish", {"summary": "ok"}),
    )
    run(llm, repo, [FAILING, PASSING])
    by_id = {m["tool_call_id"]: m["content"] for m in llm.seen[-1] if m["role"] == "tool"}
    assert "unknown argument(s) ['path']" in by_id["s1"] and "'glob'" in by_id["s1"]
    assert "missing required ['path']" in by_id["r1"]


def test_llm_error_stops_cleanly_with_final_report(repo: Path) -> None:
    class FailingLLM(ScriptedLLM):
        def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
            if len(self.seen) == 1:
                raise RuntimeError("413 Request too large")
            return super().chat(system, messages, tools)

    result, sandbox_calls = run(FailingLLM(call("list_files", {})), repo, [FAILING])
    assert result.stop_reason == "llm_error" and "413" in result.error
    assert result.steps == 1 and result.final_report["status"] == "startup_error"
    assert len(sandbox_calls) == 2
