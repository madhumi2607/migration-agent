"""Tests for agent/llm.py. The Groq client is faked, so no API key or network is needed."""

import json
from types import SimpleNamespace as NS
from typing import Any

import groq
import httpx
import pytest

from agent import llm
from agent.llm import GroqLLM, LLMResponse, ToolCall, ToolSpec

READ_FILE = ToolSpec("read_file", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})


def completion(content: str | None = None, tool_calls: list[tuple[str, str, str]] | None = None, finish: str = "stop") -> NS:
    """A fake Groq ChatCompletion with the fields the adapter reads."""
    calls = [NS(id=i, function=NS(name=n, arguments=a)) for i, n, a in tool_calls or []] or None
    return NS(
        choices=[NS(message=NS(content=content, tool_calls=calls), finish_reason=finish)],
        usage=NS(prompt_tokens=120, completion_tokens=30),
    )


class FakeClient:
    """Records requests and returns queued responses (or raises queued exceptions)."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def bad_request(message: str) -> groq.BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return groq.BadRequestError(message, response=response, body=None)


def test_request_shape() -> None:
    client = FakeClient(completion("hi"))
    GroqLLM("m", client=client).chat("You are a migration agent.", [{"role": "user", "content": "go"}], [READ_FILE])
    req = client.requests[0]
    assert req["model"] == "m" and req["temperature"] == 0.0
    assert req["messages"][0] == {"role": "system", "content": "You are a migration agent."}
    assert req["messages"][1] == {"role": "user", "content": "go"}
    assert req["tools"][0]["function"]["name"] == "read_file"
    assert req["tools"][0]["function"]["parameters"] == READ_FILE.input_schema


def test_no_tools_means_no_tool_params() -> None:
    client = FakeClient(completion("hi"))
    GroqLLM("m", client=client).chat("s", [], [])
    assert "tools" not in client.requests[0] and "tool_choice" not in client.requests[0]


def test_parses_tool_calls_and_usage() -> None:
    client = FakeClient(completion(None, [("call_1", "read_file", '{"path": "a.py"}')], finish="tool_calls"))
    resp = GroqLLM("m", client=client).chat("s", [], [READ_FILE])
    assert resp.text is None
    assert resp.tool_calls == [ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})]
    assert (resp.input_tokens, resp.output_tokens, resp.stop_reason) == (120, 30, "tool_calls")


def test_invalid_json_arguments_become_an_error_not_a_crash() -> None:
    client = FakeClient(completion(None, [("call_1", "read_file", '{"path": "a.py"')]))
    call = GroqLLM("m", client=client).chat("s", [], [READ_FILE]).tool_calls[0]
    assert call.arguments == {} and "not valid JSON" in call.error


def test_retries_tool_use_failed_then_succeeds() -> None:
    client = FakeClient(bad_request("Error code: 400 - {'error': {'code': 'tool_use_failed'}}"), completion("ok"))
    assert GroqLLM("m", client=client).chat("s", [], [READ_FILE]).text == "ok"
    assert len(client.requests) == 2


def test_other_bad_requests_are_raised() -> None:
    client = FakeClient(bad_request("Error code: 400 - context_length_exceeded"))
    with pytest.raises(groq.BadRequestError):
        GroqLLM("m", client=client).chat("s", [], [])
    assert len(client.requests) == 1


def test_history_round_trip() -> None:
    """assistant_message + tool_result_message produce the history Groq expects next turn."""
    resp = LLMResponse(text=None, tool_calls=[ToolCall("call_1", "read_file", {"path": "a.py"})])
    msg = llm.assistant_message(resp)
    assert msg["tool_calls"][0]["id"] == "call_1"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}
    assert llm.tool_result_message(resp.tool_calls[0], "contents") == {"role": "tool", "tool_call_id": "call_1", "content": "contents"}


def test_get_llm_selects_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    assert llm.get_llm().model == "llama-3.3-70b-versatile"

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(SystemExit):
        llm.get_llm()


def test_groq_strips_other_providers_state() -> None:
    client = FakeClient(completion("hi"))
    history = [{"role": "assistant", "content": "x", "provider_state": {"provider": "gemini", "content": {}}}]
    GroqLLM("m", client=client).chat("s", history, [])
    assert "provider_state" not in client.requests[0]["messages"][1]


# --- Gemini (real google-genai types, fake client) ---

from google.genai import types  # noqa: E402

from agent.llm import GeminiLLM  # noqa: E402


class FakeGeminiClient:
    def __init__(self, *responses: types.GenerateContentResponse) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.models = NS(generate_content=self._generate)

    def _generate(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.requests.append(kwargs)
        return self.responses.pop(0)


def gemini_response(*parts: types.Part, finish: str = "STOP", thoughts: int = 0) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)), finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=200, candidates_token_count=20, thoughts_token_count=thoughts),
    )


SIGNED_CALL = types.Part(function_call=types.FunctionCall(name="read_file", args={"path": "a.py"}), thought_signature=b"\x01secret-sig")


def test_gemini_request_shape() -> None:
    client = FakeGeminiClient(gemini_response(types.Part(text="hi")))
    GeminiLLM("gemini-x", client=client).chat("You migrate code.", [{"role": "user", "content": "go"}], [READ_FILE])
    req = client.requests[0]
    assert req["model"] == "gemini-x"
    assert req["contents"] == [types.Content(role="user", parts=[types.Part(text="go")])]
    config = req["config"]
    assert config.system_instruction == "You migrate code."
    assert config.automatic_function_calling.disable is True
    decl = config.tools[0].function_declarations[0]
    assert decl.name == "read_file" and decl.parameters_json_schema == READ_FILE.input_schema


def test_gemini_parses_calls_tokens_and_skips_thoughts() -> None:
    client = FakeGeminiClient(gemini_response(types.Part(text="thinking...", thought=True), types.Part(text="Reading."), SIGNED_CALL, thoughts=50))
    resp = GeminiLLM("g", client=client).chat("s", [], [READ_FILE])
    assert resp.text == "Reading."  # thought summaries are not the answer
    assert resp.tool_calls[0].name == "read_file" and resp.tool_calls[0].arguments == {"path": "a.py"}
    assert resp.tool_calls[0].id.startswith("local-")  # Gemini gave no id, so we made one up
    assert (resp.input_tokens, resp.output_tokens) == (200, 70)  # thinking tokens count as output
    json.dumps(resp.provider_state)  # history must stay JSON-serialisable for traces


def test_gemini_thought_signature_survives_the_round_trip() -> None:
    """Turn 1 returns a signed call; turn 2 must send that exact part back, plus the result."""
    client = FakeGeminiClient(gemini_response(SIGNED_CALL), gemini_response(types.Part(text="done")))
    gemini = GeminiLLM("g", client=client)
    first = gemini.chat("s", [{"role": "user", "content": "go"}], [READ_FILE])

    history = [{"role": "user", "content": "go"}, llm.assistant_message(first), llm.tool_result_message(first.tool_calls[0], "file text")]
    gemini.chat("s", json.loads(json.dumps(history)), [READ_FILE])  # through JSON, like a saved trace

    model_turn, result_turn = client.requests[1]["contents"][1:]
    assert model_turn.parts[0].thought_signature == b"\x01secret-sig"
    response = result_turn.parts[0].function_response
    assert result_turn.role == "user" and response.name == "read_file" and response.response == {"result": "file text"}
    assert response.id is None  # our invented id is never sent to Gemini


def test_gemini_merges_parallel_results_and_rebuilds_foreign_turns() -> None:
    """A Groq-written assistant turn (no provider_state) is rebuilt; its two results go back in one turn."""
    client = FakeGeminiClient(gemini_response(types.Part(text="ok")))
    groq_turn = llm.assistant_message(LLMResponse(text=None, tool_calls=[ToolCall("c1", "read_file", {"path": "a"}), ToolCall("c2", "list_files", {})]))
    history = [groq_turn, llm.tool_result_message(ToolCall("c1", "read_file", {}), "A"), llm.tool_result_message(ToolCall("c2", "list_files", {}), "B")]
    GeminiLLM("g", client=client).chat("s", history, [READ_FILE])

    model_turn, result_turn = client.requests[0]["contents"]
    assert [p.function_call.id for p in model_turn.parts] == ["c1", "c2"]
    assert [(p.function_response.id, p.function_response.name) for p in result_turn.parts] == [("c1", "read_file"), ("c2", "list_files")]


def test_gemini_empty_candidate_is_not_a_crash() -> None:
    empty = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="MALFORMED_FUNCTION_CALL")])
    resp = GeminiLLM("g", client=FakeGeminiClient(empty)).chat("s", [], [READ_FILE])
    assert resp.text is None and resp.tool_calls == [] and "MALFORMED_FUNCTION_CALL" in resp.stop_reason


def test_get_llm_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    assert isinstance(llm.get_llm(), GeminiLLM) and llm.get_llm().model == "gemini-test"


# --- Claude (real anthropic types, fake client) ---

from anthropic.types.beta import BetaMessage  # noqa: E402

from agent.llm import ClaudeLLM  # noqa: E402


def claude_message(*content: dict[str, Any], stop: str = "tool_use", cache_read: int = 0) -> BetaMessage:
    return BetaMessage.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": list(content), "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": 50, "output_tokens": 40, "cache_creation_input_tokens": 10, "cache_read_input_tokens": cache_read},
    })


class FakeStream:
    def __init__(self, message: BetaMessage) -> None:
        self.message = message

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> BetaMessage:
        return self.message


class FakeClaudeClient:
    def __init__(self, *messages: BetaMessage) -> None:
        self.messages = list(messages)
        self.requests: list[dict[str, Any]] = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kwargs: Any) -> FakeStream:
        self.requests.append(kwargs)
        return FakeStream(self.messages.pop(0))


THINKING = {"type": "thinking", "thinking": "", "signature": "c2lnbmVkLXJlYXNvbmluZw=="}
TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}}


def test_claude_request_shape() -> None:
    client = FakeClaudeClient(claude_message({"type": "text", "text": "hi"}, stop="end_turn"))
    ClaudeLLM("claude-opus-5-5", client=client, effort="high").chat("You migrate code.", [{"role": "user", "content": "go"}], [READ_FILE])
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["system"] == "You migrate code."
    assert req["thinking"] == {"type": "adaptive"} and req["output_config"] == {"effort": "high"}
    assert req["cache_control"] == {"type": "ephemeral"}
    assert req["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in req["betas"]

    haiku = FakeClaudeClient(claude_message({"type": "text", "text": "hi"}, stop="end_turn"))
    ClaudeLLM("claude-haiku-5-5", client=haiku).chat("s", [], [])
    assert "fallbacks" not in haiku.requests[0] and "betas" not in haiku.requests[0]  # Haiku 5.5 has none
    assert req["tools"] == [{"name": "read_file", "description": "Read a file.", "input_schema": READ_FILE.input_schema}]
    assert req["messages"] == [{"role": "user", "content": "go"}]


def test_claude_parses_calls_and_usage() -> None:
    client = FakeClaudeClient(claude_message(THINKING, {"type": "text", "text": "Reading it."}, TOOL_USE, cache_read=900))
    resp = ClaudeLLM("m", client=client).chat("s", [], [READ_FILE])
    assert resp.text == "Reading it."
    assert resp.tool_calls == [ToolCall("toolu_1", "read_file", {"path": "a.py"})]
    assert (resp.input_tokens, resp.cache_read_tokens, resp.output_tokens) == (960, 900, 40)  # 50 + 10 + 900
    json.dumps(resp.provider_state)


def test_claude_thinking_blocks_round_trip_unchanged() -> None:
    """The signed thinking block must go back exactly as received, with results merged into one message."""
    first = claude_message(THINKING, TOOL_USE, {"type": "tool_use", "id": "toolu_2", "name": "list_files", "input": {}})
    client = FakeClaudeClient(first, claude_message({"type": "text", "text": "done"}, stop="end_turn"))
    claude = ClaudeLLM("m", client=client)
    r1 = claude.chat("s", [{"role": "user", "content": "go"}], [READ_FILE])
    history = [{"role": "user", "content": "go"}, llm.assistant_message(r1),
               llm.tool_result_message(r1.tool_calls[0], "file text"), llm.tool_result_message(r1.tool_calls[1], "ERROR: boom")]
    claude.chat("s", json.loads(json.dumps(history)), [READ_FILE])

    _, assistant, results = client.requests[1]["messages"]
    assert assistant["content"][0] == THINKING
    assert results["role"] == "user"
    assert [(b["tool_use_id"], b["is_error"]) for b in results["content"]] == [("toolu_1", False), ("toolu_2", True)]


def test_claude_rebuilds_turns_from_other_providers() -> None:
    client = FakeClaudeClient(claude_message({"type": "text", "text": "ok"}, stop="end_turn"))
    groq_turn = llm.assistant_message(LLMResponse(text="Let me look.", tool_calls=[ToolCall("call_9", "read_file", {"path": "b.py"})]))
    ClaudeLLM("m", client=client).chat("s", [groq_turn, llm.tool_result_message(ToolCall("call_9", "read_file", {}), "B")], [READ_FILE])
    assistant = client.requests[0]["messages"][0]
    assert assistant["content"] == [{"type": "text", "text": "Let me look."},
                                    {"type": "tool_use", "id": "call_9", "name": "read_file", "input": {"path": "b.py"}}]


def test_claude_truncated_tool_call_is_not_run() -> None:
    client = FakeClaudeClient(claude_message(TOOL_USE, stop="max_tokens"))
    call = ClaudeLLM("m", client=client).chat("s", [], [READ_FILE]).tool_calls[0]
    assert call.error and "output limit" in call.error


def test_get_llm_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_CLAUDE_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("AGENT_CLAUDE_EFFORT", "low")
    model = llm.get_llm()
    assert isinstance(model, ClaudeLLM) and model.model == "claude-haiku-5-5" and model.effort == "low"
