"""The hand-written agent loop: call LLM -> run the tool it picks -> append result -> repeat.

Run against the target repo (needs .env with GROQ_API_KEY, TARGET_REPO_ROOT and SANDBOX_*):
    uv run --env-file .env python -m agent.loop --library pydantic --old-version 1 --new-version 2
"""

import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer

from agent import prompts
from agent.llm import LLM, ToolCall, ToolSpec, assistant_message, get_llm, tool_result_message

FINISH_TOOL = ToolSpec(
    name="finish",
    description="Call when the migration is complete and all tests pass. Triggers a full verification run.",
    input_schema={
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "Short summary of the changes made."}},
        "required": ["summary"],
    },
)


@dataclass
class TaskSpec:
    library: str
    old_version: str
    new_version: str


@dataclass
class LoopConfig:
    max_steps: int = 40                 # LLM calls
    # Billable tokens: input NOT served from the prompt cache, plus output, summed over all calls.
    # Cache reads are excluded because they cost ~1/20 of normal input; counting them would end
    # cheap, well-cached runs early (a 29-step Claude run read 978K of its 1.03M input from cache).
    max_billable_tokens: int = 500_000
    max_tool_result_chars: int = 8_000  # longer tool results are truncated before the LLM sees them


@dataclass
class RunResult:
    success: bool = False
    # finished | already_passing | max_steps | token_budget | llm_error | sandbox_error
    stop_reason: str = ""
    error: str | None = None  # set when stop_reason is llm_error or sandbox_error
    steps: int = 0
    input_tokens: int = 0  # all prompt tokens, cached or not
    cache_read_tokens: int = 0  # the part of input_tokens served from the prompt cache
    output_tokens: int = 0
    summary: str | None = None
    initial_report: dict[str, Any] = field(default_factory=dict)
    final_report: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)  # one per tool call, for tracing


class ToolBox:
    """Connects to several MCP servers and routes each tool call to the server that owns it."""

    def __init__(self, servers: list[MCPServer | StdioServerParameters]) -> None:
        self._servers = servers
        self._stack = AsyncExitStack()
        self._routes: dict[str, Client] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self.specs: list[ToolSpec] = []

    async def __aenter__(self) -> "ToolBox":
        for server in self._servers:
            # Generous read timeout: a run_tests call installs deps and runs the suite.
            client = await self._stack.enter_async_context(Client(server, read_timeout_seconds=900))
            for tool in (await client.list_tools()).tools:
                if tool.name in self._routes:
                    raise ValueError(f"Two MCP servers both define a tool named {tool.name!r}.")
                self._routes[tool.name] = client
                self._schemas[tool.name] = tool.input_schema
                self.specs.append(ToolSpec(tool.name, tool.description or "", tool.input_schema))
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    def has(self, name: str) -> bool:
        return name in self._routes

    def check_arguments(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Reject unknown or missing argument names before calling the tool.

        MCP servers silently drop unknown arguments, so a model that guesses `path=`
        instead of `glob=` would get a result for the defaults and never learn its mistake.
        """
        schema = self._schemas[name]
        allowed = set(schema.get("properties", {}))
        unknown = sorted(set(arguments) - allowed)
        missing = sorted(set(schema.get("required", [])) - set(arguments))
        if not unknown and not missing:
            return None
        problems = [f"unknown argument(s) {unknown}" if unknown else "", f"missing required {missing}" if missing else ""]
        return f"{name}: {'; '.join(p for p in problems if p)}. Valid arguments: {sorted(allowed)}."

    async def call(self, name: str, arguments: dict[str, Any]) -> tuple[bool, Any]:
        """Run a tool. Returns (is_error, payload): structured content if any, else the text."""
        result = await self._routes[name].call_tool(name, arguments)
        if result.structured_content is not None and not result.is_error:
            return False, result.structured_content
        text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
        return result.is_error, text

    async def run_tests(self) -> dict[str, Any]:
        """Full test-suite run, used by the loop itself (baseline and verification)."""
        is_error, payload = await self.call("run_tests", {})
        if is_error or not isinstance(payload, dict):
            return {"status": "sandbox_error", "detail": str(payload)}
        return payload


def render(is_error: bool, payload: Any, limit: int) -> str:
    """Tool result as the string the LLM sees: compact JSON, errors prefixed, length capped."""
    text = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    if is_error:
        text = f"ERROR: {text}"
    if len(text) > limit:
        text = text[:limit] + f"... [truncated {len(text) - limit} chars]"
    return text


async def run_agent(llm: LLM, toolbox: ToolBox, task: TaskSpec, config: LoopConfig | None = None) -> RunResult:
    """Run the migration loop until the model finishes with a passing suite or a budget runs out."""
    config = config or LoopConfig()
    result = RunResult(initial_report=await toolbox.run_tests())
    if result.initial_report.get("status") == "passed":
        result.success, result.stop_reason, result.final_report = True, "already_passing", result.initial_report
        return result
    if result.initial_report.get("status") in ("sandbox_error", "timeout"):
        # The tests can't run at all (e.g. Docker is down). Don't pay for LLM calls the
        # model can't verify: it would be editing code blind.
        result.stop_reason, result.final_report = "sandbox_error", result.initial_report
        result.error = str(result.initial_report.get("detail") or result.initial_report.get("status"))[:1000]
        return result

    system = prompts.system_prompt(task.library, task.old_version, task.new_version)
    messages = [{"role": "user", "content": prompts.initial_message(task.library, task.new_version, result.initial_report)}]
    tools = [*toolbox.specs, FINISH_TOOL]

    while True:
        if result.steps >= config.max_steps:
            result.stop_reason = "max_steps"
            break
        if result.input_tokens - result.cache_read_tokens + result.output_tokens >= config.max_billable_tokens:
            result.stop_reason = "token_budget"
            break

        # The LLM client is synchronous; run it in a thread so MCP connections stay serviced.
        try:
            response = await asyncio.to_thread(llm.chat, system, messages, tools)
        except Exception as exc:  # adapters already retried; e.g. a request over the provider's size limit
            result.stop_reason, result.error = "llm_error", f"{type(exc).__name__}: {exc}"[:1000]
            break
        result.steps += 1
        result.input_tokens += response.input_tokens
        result.cache_read_tokens += response.cache_read_tokens
        result.output_tokens += response.output_tokens
        messages.append(assistant_message(response))

        if not response.tool_calls:
            messages.append({"role": "user", "content": prompts.NUDGE})
            continue

        for call in response.tool_calls:
            if call.name == FINISH_TOOL.name and call.error is None:
                report = await toolbox.run_tests()
                result.events.append(_event(result.steps, call, report.get("status")))
                if report.get("status") == "passed":
                    result.success, result.stop_reason = True, "finished"
                    result.summary = call.arguments.get("summary")
                    result.final_report = report
                    return result
                content = prompts.verification_failed(report)
            else:
                content = await _run_tool_call(toolbox, call, config.max_tool_result_chars)
                result.events.append(_event(result.steps, call, content[:200]))
            messages.append(tool_result_message(call, content))

    # Budget exhausted or LLM failed: one last full run so the metrics reflect where the repo ended up.
    result.final_report = await toolbox.run_tests()
    return result


async def _run_tool_call(toolbox: ToolBox, call: ToolCall, limit: int) -> str:
    if call.error:
        return f"ERROR: {call.error}"
    if not toolbox.has(call.name):
        return f"ERROR: Unknown tool {call.name!r}. Available: {', '.join(s.name for s in toolbox.specs)}, finish."
    if problem := toolbox.check_arguments(call.name, call.arguments):
        return f"ERROR: {problem}"
    is_error, payload = await toolbox.call(call.name, call.arguments)
    return render(is_error, payload, limit)


def _event(step: int, call: ToolCall, outcome: Any) -> dict[str, Any]:
    return {"step": step, "tool": call.name, "arguments": call.arguments, "outcome": outcome}


def _stdio_server(module: str) -> StdioServerParameters:
    """Launch one of our MCP servers as a subprocess, inheriting this process's env (.env values, PATH)."""
    return StdioServerParameters(command=sys.executable, args=["-m", module], env=dict(os.environ))


async def _main(args: argparse.Namespace) -> RunResult:
    servers = [_stdio_server("mcp_servers.repo_tools.server"), _stdio_server("mcp_servers.sandbox.server")]
    task = TaskSpec(args.library, args.old_version, args.new_version)
    async with ToolBox(servers) as toolbox:
        return await run_agent(get_llm(), toolbox, task, LoopConfig(max_steps=args.max_steps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the migration agent on TARGET_REPO_ROOT.")
    parser.add_argument("--library", default="pydantic")
    parser.add_argument("--old-version", default="1")
    parser.add_argument("--new-version", default="2")
    parser.add_argument("--max-steps", type=int, default=LoopConfig.max_steps)
    result = asyncio.run(_main(parser.parse_args()))
    print(json.dumps(asdict(result), indent=1))


if __name__ == "__main__":
    main()
