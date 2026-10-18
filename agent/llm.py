"""Provider-neutral LLM interface for the agent loop.

The loop only ever sees the types in this file. Each provider is one small adapter
class with a `chat()` method; `get_llm()` picks one from the LLM_PROVIDER env var.

Conversation history uses the OpenAI-style chat format, which Groq accepts as-is:
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "..." | None, "tool_calls": [...]}   # see assistant_message()
    {"role": "tool", "tool_call_id": "...", "content": "..."}             # see tool_result_message()
The system prompt is passed separately, because providers disagree on where it goes.

An assistant message may also carry "provider_state": the provider's own record of that
turn, as plain JSON. Claude and Gemini need it (their turns carry signed thinking data that
must be sent back unchanged); other adapters ignore or strip it. Because those signatures
cover the whole conversation before them, the history must stay append-only.

List available models for the configured provider:
    uv run --env-file .env python -m agent.llm --list-models
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass
class ToolSpec:
    """A tool the model may call. Built from an MCP server's tool listing."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """One tool call requested by the model.

    `error` is set when the model produced arguments that aren't valid JSON; the loop
    should send that error back to the model instead of running the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    error: str | None = None


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None  # provider's own value, kept for tracing
    provider_state: dict[str, Any] | None = None  # provider-native turn, JSON-serialisable
    cache_read_tokens: int = 0  # part of input_tokens served from the prompt cache (billed far lower)


class LLM(Protocol):
    """What the agent loop needs from a model provider."""

    model: str

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        """Send the conversation and return the model's next turn."""
        ...


def assistant_message(response: LLMResponse) -> Message:
    """The history entry for a model turn, so the next call sees its own tool calls."""
    message: Message = {"role": "assistant", "content": response.text}
    if response.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in response.tool_calls
        ]
    if response.provider_state is not None:
        message["provider_state"] = response.provider_state
    return message


def tool_result_message(call: ToolCall, content: str) -> Message:
    """The history entry carrying a tool's result back to the model."""
    return {"role": "tool", "tool_call_id": call.id, "content": content}


def parse_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    """Parse a tool call's JSON argument string. Returns (arguments, error)."""
    if not raw:
        return {}, None
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"Tool arguments were not valid JSON ({exc}). Call the tool again with a valid JSON object."
    if not isinstance(args, dict):
        return {}, "Tool arguments must be a JSON object."
    return args, None


class GroqLLM:
    """Groq chat completions (OpenAI-compatible API)."""

    # Groq rejects a request with `tool_use_failed` when the model emits a malformed
    # tool call. Sampling again usually fixes it, so retry a couple of times.
    TOOL_FORMAT_RETRIES = 2

    def __init__(self, model: str, client: Any = None, temperature: float = 0.0, max_tokens: int = 4096) -> None:
        if client is None:
            import groq  # imported here so other providers don't need the groq package

            client = groq.Groq(max_retries=5)  # reads GROQ_API_KEY; retries 429/5xx with backoff
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        # Groq rejects unknown message fields, so drop other providers' state.
        history = [{k: v for k, v in m.items() if k != "provider_state"} for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *history],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        completion = self._create(kwargs)
        choice = completion.choices[0]
        calls = []
        for c in choice.message.tool_calls or []:
            arguments, error = parse_arguments(c.function.arguments)
            calls.append(ToolCall(id=c.id, name=c.function.name, arguments=arguments, error=error))
        usage = completion.usage
        return LLMResponse(
            text=choice.message.content or None,
            tool_calls=calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=choice.finish_reason,
        )

    def _create(self, kwargs: dict[str, Any]) -> Any:
        import groq

        for attempt in range(self.TOOL_FORMAT_RETRIES + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except groq.BadRequestError as exc:
                if "tool_use_failed" not in str(exc) or attempt == self.TOOL_FORMAT_RETRIES:
                    raise
        raise AssertionError("unreachable")

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list().data)


class GeminiLLM:
    """Google Gemini via the google-genai SDK (generateContent)."""

    PROVIDER = "gemini"
    # Prefix for tool-call ids we invent when Gemini doesn't return one; never sent back.
    LOCAL_ID_PREFIX = "local-"

    def __init__(self, model: str, client: Any = None, temperature: float | None = None, max_tokens: int = 8192) -> None:
        if client is None:
            from google import genai
            from google.genai import types

            # Reads GEMINI_API_KEY. Retries 429/5xx with exponential backoff.
            client = genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=6)))
        self.client = client
        self.model = model
        # None = the model's default. Google advises against lowering it for Gemini 3+ models.
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._calls = 0

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            # We run tools ourselves (through MCP); never let the SDK call anything.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.input_schema)
                for t in tools
            ])] if tools else None,
        )
        response = self.client.models.generate_content(model=self.model, contents=self.to_contents(messages), config=config)
        return self.from_response(response)

    def to_contents(self, messages: list[Message]) -> list[Any]:
        """Translate our OpenAI-style history into Gemini Contents."""
        from google.genai import types

        names_by_id: dict[str, str] = {}  # tool_call_id -> tool name (Gemini responses need the name)
        contents: list[Any] = []
        for m in messages:
            if m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif m["role"] == "assistant":
                for c in m.get("tool_calls", []):
                    names_by_id[c["id"]] = c["function"]["name"]
                state = m.get("provider_state")
                if state and state.get("provider") == self.PROVIDER:
                    contents.append(types.Content.model_validate(state["content"]))  # exact turn, signatures intact
                else:
                    contents.append(self._rebuild_model_turn(m))
            elif m["role"] == "tool":
                call_id = m["tool_call_id"]
                part = types.Part(function_response=types.FunctionResponse(
                    id=None if call_id.startswith(self.LOCAL_ID_PREFIX) else call_id,
                    name=names_by_id.get(call_id, "unknown"),
                    response={"result": m["content"]},
                ))
                # All results for one model turn must go back together in a single user turn.
                previous = contents[-1] if contents else None
                if previous is not None and previous.role == "user" and previous.parts and previous.parts[0].function_response:
                    previous.parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))
        return contents

    def _rebuild_model_turn(self, m: Message) -> Any:
        """A model turn from another provider (or a test): rebuilt without signatures."""
        from google.genai import types

        parts = [types.Part(text=m["content"])] if m.get("content") else []
        for c in m.get("tool_calls", []):
            parts.append(types.Part(function_call=types.FunctionCall(
                id=None if c["id"].startswith(self.LOCAL_ID_PREFIX) else c["id"],
                name=c["function"]["name"],
                args=json.loads(c["function"]["arguments"] or "{}"),
            )))
        return types.Content(role="model", parts=parts)

    def from_response(self, response: Any) -> LLMResponse:
        self._calls += 1
        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        # Thinking tokens are billed as output, so count them toward the budget.
        output_tokens = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0

        candidate = response.candidates[0] if response.candidates else None
        finish = getattr(candidate.finish_reason, "name", str(candidate.finish_reason)) if candidate and candidate.finish_reason else None
        content = candidate.content if candidate else None
        if content is None or not content.parts:
            # e.g. MALFORMED_FUNCTION_CALL or a safety stop: no usable turn. The loop will nudge.
            return LLMResponse(text=None, input_tokens=input_tokens, output_tokens=output_tokens, stop_reason=finish)

        texts, calls = [], []
        for i, part in enumerate(content.parts):
            if part.text and not part.thought:
                texts.append(part.text)
            if part.function_call:
                fc = part.function_call
                call_id = fc.id or f"{self.LOCAL_ID_PREFIX}{self._calls}-{i}"
                calls.append(ToolCall(id=call_id, name=fc.name, arguments=dict(fc.args or {})))
        return LLMResponse(
            text="\n".join(texts) or None,
            tool_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=finish,
            provider_state={"provider": self.PROVIDER, "content": content.model_dump(mode="json", exclude_none=True)},
        )

    def list_models(self) -> list[str]:
        return sorted(m.name.removeprefix("models/") for m in self.client.models.list()
                      if "generateContent" in (m.supported_actions or []))


class ClaudeLLM:
    """Anthropic Claude via the official SDK (Messages API, streamed)."""

    PROVIDER = "claude"
    # Server-side refusal fallback: if a safety classifier declines a request, the API re-runs it
    # on a fallback model inside the same call instead of failing the agent run.
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str, client: Any = None, effort: str = "high", max_tokens: int = 64_000) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=5)  # reads ANTHROPIC_API_KEY; retries 429/5xx
        self.client = client
        self.model = model
        self.effort = effort
        # Thinking counts toward max_tokens; 64K leaves room for long agentic turns.
        self.max_tokens = max_tokens

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self.to_messages(messages),
            "thinking": {"type": "adaptive"},  # always on for current models; effort sets the depth
            "output_config": {"effort": self.effort},
            # Cache the longest stable prefix. Our history only grows at the end, so each
            # request re-reads the previous one's prefix from cache at a fraction of the price.
            "cache_control": {"type": "ephemeral"},
            "betas": [self.FALLBACK_BETA],
            "fallbacks": "default",
        }
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]
        # Streamed because large max_tokens values exceed the SDK's non-streaming timeout.
        with self.client.beta.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
        return self.from_message(message)

    def to_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Translate our OpenAI-style history into Anthropic messages."""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                state = m.get("provider_state")
                if state and state.get("provider") == self.PROVIDER:
                    content = state["content"]  # exact blocks, thinking signatures intact
                else:
                    content = self._rebuild_blocks(m)
                out.append({"role": "assistant", "content": content})
            elif m["role"] == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                    "is_error": m["content"].startswith("ERROR:"),
                }
                # All results for one assistant turn go back together in a single user message.
                if out and _is_tool_results(out[-1]):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    @staticmethod
    def _rebuild_blocks(m: Message) -> list[dict[str, Any]]:
        """An assistant turn written by another provider (or a test): text + tool_use blocks."""
        blocks: list[dict[str, Any]] = [{"type": "text", "text": m["content"]}] if m.get("content") else []
        for c in m.get("tool_calls", []):
            blocks.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                           "input": json.loads(c["function"]["arguments"] or "{}")})
        return blocks or [{"type": "text", "text": "(no response)"}]

    def from_message(self, message: Any) -> LLMResponse:
        texts, calls = [], []
        for block in message.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
        if message.stop_reason == "max_tokens":
            # A call cut off mid-generation may have truncated arguments: never run it.
            for c in calls:
                c.error = "Your response hit the output limit before this tool call was complete. Make a smaller call."

        usage = message.usage
        cache_read = usage.cache_read_input_tokens or 0
        return LLMResponse(
            text="\n".join(texts) or None,
            tool_calls=calls,
            # Total prompt size, comparable across providers; cache_read_tokens says how much was cached.
            input_tokens=usage.input_tokens + (usage.cache_creation_input_tokens or 0) + cache_read,
            output_tokens=usage.output_tokens,
            stop_reason=message.stop_reason,
            provider_state={"provider": self.PROVIDER,
                            "content": [b.model_dump(mode="json", exclude_none=True) for b in message.content]},
            cache_read_tokens=cache_read,
        )

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())


def _is_tool_results(message: dict[str, Any]) -> bool:
    content = message["content"]
    return message["role"] == "user" and isinstance(content, list) and bool(content) and content[0].get("type") == "tool_result"


def get_llm() -> LLM:
    """Build the provider named by LLM_PROVIDER (default: claude)."""
    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    if provider == "claude":
        _require_key("ANTHROPIC_API_KEY")
        return ClaudeLLM(model=os.environ.get("AGENT_CLAUDE_MODEL", "claude-opus-5-5"),
                         effort=os.environ.get("AGENT_CLAUDE_EFFORT", "high"))
    if provider == "groq":
        _require_key("GROQ_API_KEY")
        return GroqLLM(model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"))
    if provider == "gemini":
        _require_key("GEMINI_API_KEY")
        return GeminiLLM(model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"))
    raise SystemExit(f"Unknown LLM_PROVIDER {provider!r}. Use one of: claude, gemini, groq.")


def _require_key(name: str) -> None:
    if not os.environ.get(name):
        raise SystemExit(f"Set {name} in .env (see .env.example).")


if __name__ == "__main__":
    if sys.argv[1:] == ["--list-models"]:
        print("\n".join(get_llm().list_models()))
