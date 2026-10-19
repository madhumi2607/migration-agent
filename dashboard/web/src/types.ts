// Shapes returned by dashboard/api.py (see dashboard/runs.py).

export type TestCounts = { status: string; passed: number; total: number | null };

export type RunSummary = {
  run_id: string;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
  repo: string | null;
  repo_commit: string | null;
  provider: string | null;
  model: string | null;
  effort: string | null;
  status: "running" | "succeeded" | "failed";
  stop_reason: string | null;
  error: string | null;
  summary: string | null;
  steps: number;
  input_tokens: number;
  cache_read_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
  tests_before: TestCounts | null;
  tests_after: TestCounts | null;
};

export type FailureGroup = {
  kind: string;
  error_type: string | null;
  message: string;
  location: string | null;
  count: number;
  example_tests: string[];
};

export type TestReport = {
  status: string;
  counts?: { total: number; passed: number; failed: number; errors: number; skipped: number };
  versions?: Record<string, string>;
  failure_groups?: FailureGroup[];
  duration_seconds?: number;
  detail?: string;
};

export type ToolCallRef = { id: string; name: string; arguments: Record<string, unknown>; error: string | null };

// One line of events.jsonl (agent/loop.py run_agent).
export type RunEvent =
  | { type: "start"; ts: number; system: string; tools: { name: string }[]; config: Record<string, unknown> }
  | { type: "test_run"; ts: number; purpose: "baseline" | "verification" | "final"; report: TestReport }
  | { type: "user_message"; ts: number; step: number; content: string }
  | {
      type: "llm_call"; ts: number; step: number; seconds: number; text: string | null; stop_reason: string | null;
      input_tokens: number; cache_read_tokens: number; output_tokens: number; tool_calls: ToolCallRef[];
    }
  | {
      type: "tool_call"; ts: number; step: number; id: string; name: string; arguments: Record<string, unknown>;
      result: string; is_error: boolean; seconds: number;
    }
  | { type: "llm_error"; ts: number; step: number; error: string };

export type RunDetail = {
  meta: Record<string, unknown>;
  summary: RunSummary;
  events: RunEvent[];
  diff: string | null;
};
