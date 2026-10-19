import { useState } from "react";
import { navigate } from "./App";
import { useRun } from "./api";
import { cachedShare, formatCost, formatDuration, formatTests, formatTime, formatTokens } from "./format";
import RunCharts from "./RunCharts";
import { StatusBadge } from "./StatusBadge";
import Timeline from "./Timeline";
import type { Finding } from "./types";

type Tab = "overview" | "timeline" | "diff" | "settings";

export default function RunPage({ runId }: { runId: string }) {
  const { data: run, error } = useRun(runId);
  const [tab, setTab] = useState<Tab>("overview");

  if (error) return <p className="error">Could not load run: {error}</p>;
  if (!run) return <p className="muted">Loading…</p>;
  const s = run.summary;
  const tabs: [Tab, string][] = [
    ["overview", "Overview"], ["timeline", `Timeline · ${run.events.length}`], ["diff", "Diff"], ["settings", "Settings"],
  ];

  return (
    <>
      <div className="crumbs">
        <a href="/" onClick={(e) => { e.preventDefault(); navigate("/"); }}>Runs</a>
        <span className="muted"> / </span>
        <span className="mono small">{s.run_id}</span>
      </div>

      <div className="page-head">
        <div>
          <h1>
            {s.repo} <span className="muted mono commit">@{s.repo_commit}</span>
          </h1>
          <div className="muted small">
            <span className="mono">{s.model}</span>
            {s.effort && <> · effort {s.effort}</>} · {formatTime(s.started_at)}
          </div>
        </div>
        <StatusBadge status={s.status} stopReason={s.stop_reason} large />
      </div>

      <div className="stats">
        <Stat label="Tests passing" value={formatTests(s.tests_after ?? s.tests_before)} note={`baseline ${formatTests(s.tests_before)}`} />
        <Stat label="Steps" value={String(s.steps)} note="model calls" />
        <Stat label="Estimated cost" value={formatCost(s.cost_usd)} note={s.cost_usd === null ? "no prices for this model" : "list prices"} />
        <Stat label="Input tokens" value={formatTokens(s.input_tokens)} note={`${cachedShare(s.input_tokens, s.cache_read_tokens)}% from cache · ${formatTokens(s.output_tokens)} out`} />
        <Stat label="Duration" value={formatDuration(s.duration_seconds)} note={s.status === "running" ? "in progress" : "wall clock"} />
      </div>

      {s.summary && (
        <div className="card summary-card">
          <div className="card-label">Agent's summary</div>
          <p className="prewrap">{s.summary}</p>
        </div>
      )}
      {s.error && <div className="card error-card prewrap"><div className="card-label">Error</div>{s.error}</div>}
      <ChecksCard checks={run.checks} running={s.status === "running"} />

      <nav className="tabs">
        {tabs.map(([t, label]) => (
          <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>{label}</button>
        ))}
      </nav>

      {tab === "overview" && <RunCharts events={run.events} model={s.model} />}
      {tab === "timeline" && <Timeline events={run.events} />}
      {tab === "diff" && <Diff diff={run.diff} running={s.status === "running"} />}
      {tab === "settings" && <pre className="code">{JSON.stringify(run.meta, null, 2)}</pre>}
    </>
  );
}

const CHECK_LABEL: Record<string, string> = {
  pins: "Dependency pins", removed_dependency: "Removed dependency",
  compat_shim: "Compatibility shim", test_integrity: "Test integrity",
};

/** Review checks on the final diff: things the tests can't catch (agent/checks.py). */
function ChecksCard({ checks, running }: { checks: Finding[] | null; running: boolean }) {
  if (checks == null) // null or missing (an older API): not checked yet
    return running ? null : <div className="card muted small">Review checks haven't run for this run (<code>python -m agent.checks runs/&lt;id&gt;</code>).</div>;
  return (
    <div className={`card checks-card ${checks.some((f) => f.severity === "error") ? "has-errors" : ""}`}>
      <div className="card-label">Review checks · things the tests can't catch</div>
      {checks.length === 0 ? (
        <div className="check-ok"><span className="icon-ok" aria-hidden>✓</span> No findings: pins, dependencies, shims and test integrity all look right.</div>
      ) : (
        <ul className="findings">
          {checks.map((f, i) => (
            <li key={i}>
              <span className={f.severity === "error" ? "icon-bad" : "icon-warn"} aria-hidden>{f.severity === "error" ? "✕" : "!"}</span>
              <span className={`sev sev-${f.severity}`}>{f.severity}</span>
              <span className="muted small">{CHECK_LABEL[f.check] ?? f.check}</span>
              <div className="finding-msg">{f.message}{f.file && <span className="mono small muted"> · {f.file}</span>}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {note && <div className="muted small">{note}</div>}
    </div>
  );
}

function Diff({ diff, running }: { diff: string | null; running: boolean }) {
  if (diff === null) return <p className="muted">{running ? "The diff is written when the run ends." : "No diff recorded."}</p>;
  if (diff.trim() === "") return <p className="muted">The agent made no changes.</p>;
  const files = diff.split("\n").filter((l) => l.startsWith("+++ b/")).length;
  const added = diff.split("\n").filter((l) => l.startsWith("+") && !l.startsWith("+++")).length;
  const removed = diff.split("\n").filter((l) => l.startsWith("-") && !l.startsWith("---")).length;
  return (
    <>
      <p className="muted small">
        {files} file{files === 1 ? "" : "s"} changed · <span className="diff-add-text">+{added}</span>{" "}
        <span className="diff-del-text">−{removed}</span>
      </p>
      <pre className="code diff">
        {diff.split("\n").map((line, i) => {
          const kind = line.startsWith("+++") || line.startsWith("---") || line.startsWith("diff ") ? "file"
            : line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : line.startsWith("@@") ? "hunk" : "";
          return <div key={i} className={kind}>{line || " "}</div>;
        })}
      </pre>
    </>
  );
}
