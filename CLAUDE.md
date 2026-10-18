# Migration Agent

An autonomous AI agent that upgrades Python codebases across breaking library releases
(first target: Pydantic v1 -> v2; later: SQLAlchemy 1.4 -> 2.0). Given a repo pinned to the
old version, the agent upgrades the dependency, runs the tests, reads the migration guide,
fixes the breakages, verifies with the full test suite, and opens a pull request after a
human approves the diff.

This is a portfolio project. The owner (Anirudh) must be able to explain every design
decision in a technical interview, so clarity and understanding matter more than speed.

## Working style for Claude Code

- Before writing code for a new component, briefly explain the design and the main
  tradeoff, then build it. Ask before any decision that changes the architecture below.
- Keep changes small and reviewable: one component or feature per step, with tests.
- After each step, summarize what was built, how to run it, and one thing worth
  understanding about it (a concept, a tradeoff, or a likely interview question).
- Prefer simple, readable code over clever code. No unnecessary abstractions.
- Never add a dependency without saying why it's needed.

## Architecture

1. **Agent loop** (`agent/`): hand-written in plain Python. No agent framework
   (no LangChain/LangGraph) in v1. Each step: build context -> call LLM -> LLM picks a
   tool -> run it -> append the result -> repeat until done or a budget is hit.
   A LangGraph version comes later (week 5) for comparison; keep v1 untouched.
2. **MCP tool servers** (`mcp_servers/`), built with the official MCP Python SDK.
   The agent can only act through these:
   - `repo_tools`: search code, read files (with line ranges), make targeted edits.
   - `sandbox`: install deps and run pytest inside Docker; return a structured summary
     of failures (test name, error type, message, file:line), not raw logs.
   - `docs_search`: RAG over migration guides and changelogs (ChromaDB). Week 3.
   - `git_tools`: create branch, commit, produce diff, open PR. Week 4.
3. **Working memory** (`agent/memory.py`): records failing tests, attempted fixes and
   their outcomes, so the agent doesn't repeat failed attempts.
4. **Guardrails** (`agent/guardrails.py`): max steps, token/cost budget, and a hard rule
   that the agent may NOT edit, delete or skip tests (no `@pytest.mark.skip`, no editing
   files under `tests/`), and may only touch files inside the target repo.
5. **Human review**: the agent stops and shows the diff plus a change summary; the PR is
   only opened after explicit approval.
6. **Tracing**: log every LLM call and tool call (Langfuse or a simple JSONL trace) so
   any run can be replayed.

The LLM provider must be swappable behind one small interface in `agent/llm.py`
(Claude, Gemini, or Groq), selected by environment variable.

## Planned layout

```
migration-agent/
  CLAUDE.md
  pyproject.toml
  .env.example
  agent/            # loop.py, llm.py, memory.py, guardrails.py, prompts.py
  mcp_servers/      # repo_tools/, sandbox/, docs_search/, git_tools/
  sandbox/          # Dockerfile for running target repos' tests
  benchmark/        # tasks.yaml, runner.py, results/
  tests/            # tests for this project's own code
  dashboard/        # FastAPI + React run viewer (week 6)
```

## Tech stack

Python 3.11+, uv for env management, official MCP Python SDK, Docker, pytest,
ChromaDB, Langfuse (or JSONL traces), FastAPI. Optional later: Hugging Face, PEFT/QLoRA,
vLLM for fine-tuning a small model on successful runs.

## Benchmark and metrics

Benchmark = 30-50 open-source repos that use Pydantic v1 and have meaningful test
coverage. Each task in `benchmark/tasks.yaml`: repo URL, pinned commit, old/new version,
test command. Metrics per run: full success (all tests pass), % of initially failing
tests fixed, steps, tokens, cost, diff size, and a failure category
(missed file, misread docs, behavior change, budget exhausted, other).
Results are written to `benchmark/results/` as JSON so they can be compared across
experiments (with/without docs search, hand-written loop vs LangGraph, model A vs B).

## Roadmap

- [ ] Week 1: project setup; `repo_tools` and `sandbox` MCP servers; run a target repo's
      tests in Docker and get a structured failure summary.
- [ ] Week 2: hand-written agent loop with memory and retries; first successful
      Pydantic migration on one small repo.
- [ ] Week 3: `docs_search` (RAG over the Pydantic migration guide); build the benchmark.
- [ ] Week 4: guardrails, tracing, human review, `git_tools`.
- [ ] Week 5: LangGraph version, experiments, second library.
- [ ] Week 6: dashboard, README with results tables, demo video, write-up.
- [ ] Optional: fine-tune a small open model on successful trajectories.

## Conventions

- Type hints everywhere; small functions; docstrings on public functions.
- Every MCP tool returns structured, concise output (the LLM's context is precious).
- Tests for this project's code live in `tests/` and run with `pytest`.
- Secrets only via `.env` (never committed); keep `.env.example` up to date.
