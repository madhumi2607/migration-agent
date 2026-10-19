import type { RunEvent, TestReport } from "./types";
import { formatDuration, formatTokens } from "./format";

const MAX_SHOWN_CHARS = 6000; // long tool results are cut in the UI (the trace keeps them in full)

export default function Timeline({ events }: { events: RunEvent[] }) {
  return (
    <ol className="timeline">
      {events.map((e, i) => (
        <li key={i} className={`event event-${e.type}`}>
          <EventView event={e} />
        </li>
      ))}
    </ol>
  );
}

function EventView({ event: e }: { event: RunEvent }) {
  switch (e.type) {
    case "start":
      return (
        <details>
          <summary><b>Run started</b> <span className="muted">· tools: {e.tools.map((t) => t.name).join(", ")}</span></summary>
          <pre className="code">{e.system}</pre>
        </details>
      );
    case "test_run":
      return <TestRunView purpose={e.purpose} report={e.report} />;
    case "user_message":
      return (
        <details>
          <summary>
            <b>{e.step === 0 ? "Task given to the model" : "Nudge"}</b>{" "}
            <span className="muted">{e.step === 0 ? "(baseline test results)" : "· the model replied without calling a tool"}</span>
          </summary>
          <pre className="code">{e.content}</pre>
        </details>
      );
    case "llm_call":
      return (
        <div>
          <div className="event-title">
            <b>Step {e.step}</b> <span className="muted">· model · {e.seconds}s · {formatTokens(e.input_tokens)} in
              ({formatTokens(e.cache_read_tokens)} cached) / {formatTokens(e.output_tokens)} out
              {e.tool_calls.length > 0 && ` · ${e.tool_calls.length} tool call${e.tool_calls.length > 1 ? "s" : ""}`}
              {e.stop_reason && !["tool_use", "end_turn", "STOP", "tool_calls", "stop"].includes(e.stop_reason) && ` · stop: ${e.stop_reason}`}
            </span>
          </div>
          {e.text && <p className="prewrap model-text">{e.text}</p>}
        </div>
      );
    case "tool_call":
      return (
        <details className={e.is_error ? "is-error" : ""}>
          <summary>
            <span className="tool-name mono">{e.name}</span> <span className="mono small">{describeArgs(e.name, e.arguments)}</span>
            <span className="muted small"> · {e.seconds}s</span>
            {e.is_error && <span className="badge badge-failed small">error</span>}
          </summary>
          <div className="small muted">arguments</div>
          <pre className="code">{JSON.stringify(e.arguments, null, 2)}</pre>
          <div className="small muted">result</div>
          <pre className="code">{truncate(prettyJson(e.result))}</pre>
        </details>
      );
    case "llm_error":
      return <div className="error prewrap"><b>Model API error</b> at step {e.step}: {e.error}</div>;
  }
}

function TestRunView({ purpose, report }: { purpose: string; report: TestReport }) {
  const c = report.counts;
  const label = { baseline: "Baseline test run", verification: "Verification (finish requested)", final: "Final test run" }[purpose] ?? purpose;
  return (
    <div className="card test-card">
      <div className="event-title">
        <b>{label}</b>{" "}
        <span className={`badge badge-${report.status === "passed" ? "succeeded" : "failed"}`}>{report.status.replace("_", " ")}</span>
        <span className="muted small">
          {c && ` · ${c.passed}/${c.total} passed${c.failed ? `, ${c.failed} failed` : ""}${c.errors ? `, ${c.errors} errors` : ""}`}
          {report.versions && Object.entries(report.versions).map(([k, v]) => ` · ${k} ${v}`)}
          {report.duration_seconds !== undefined && ` · ${formatDuration(report.duration_seconds)}`}
        </span>
      </div>
      {report.detail && <pre className="code">{report.detail}</pre>}
      {report.failure_groups && report.failure_groups.length > 0 && (
        <ul className="failures">
          {report.failure_groups.map((g, i) => (
            <li key={i}>
              <span className="mono">×{g.count}</span> <b>{g.error_type ?? g.kind}</b> at{" "}
              <span className="mono">{g.location ?? "?"}</span>
              <div className="small">{g.message}</div>
              {g.example_tests.length > 0 && <div className="muted small mono">e.g. {g.example_tests.join(", ")}</div>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** One-line summary of a tool call's arguments, e.g. "geometries.py:1-120" for read_file. */
function describeArgs(name: string, a: Record<string, unknown>): string {
  const str = (v: unknown) => (typeof v === "string" ? v : JSON.stringify(v));
  switch (name) {
    case "read_file":
      return `${str(a.path)}${a.start_line || a.end_line ? `:${a.start_line ?? 1}-${a.end_line ?? "end"}` : ""}`;
    case "edit_file":
      return str(a.path);
    case "search_code":
      return `/${str(a.pattern)}/${a.glob ? ` in ${str(a.glob)}` : ""}`;
    case "list_files":
      return str(a.glob ?? "**/*.py");
    case "run_tests":
      return Array.isArray(a.tests) && a.tests.length ? a.tests.join(" ") : "(full suite)";
    case "finish":
      return "";
    default:
      return JSON.stringify(a);
  }
}

function prettyJson(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

const truncate = (text: string) =>
  text.length > MAX_SHOWN_CHARS ? `${text.slice(0, MAX_SHOWN_CHARS)}\n… (${text.length - MAX_SHOWN_CHARS} more chars)` : text;
