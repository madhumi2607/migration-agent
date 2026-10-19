import { useState } from "react";
import { navigate } from "./App";
import { useRun } from "./api";
import { cachedShare, formatCost, formatDuration, formatTests, formatTime, formatTokens } from "./format";
import { StatusBadge } from "./StatusBadge";
import Timeline from "./Timeline";

type Tab = "timeline" | "diff" | "settings";

export default function RunPage({ runId }: { runId: string }) {
  const { data: run, error } = useRun(runId);
  const [tab, setTab] = useState<Tab>("timeline");

  if (error) return <p className="error">Could not load run: {error}</p>;
  if (!run) return <p className="muted">Loading…</p>;
  const s = run.summary;

  return (
    <>
      <p>
        <a href="/" onClick={(e) => { e.preventDefault(); navigate("/"); }}>← All runs</a>
      </p>
      <div className="run-header">
        <h1>
          {s.repo} <span className="muted mono small">@{s.repo_commit}</span>
        </h1>
        <StatusBadge status={s.status} stopReason={s.stop_reason} />
      </div>
      <p className="muted">
        {s.model}
        {s.effort && ` · effort ${s.effort}`} · started {formatTime(s.started_at)}
      </p>

      <div className="stats">
        <Stat label="Tests" value={`${formatTests(s.tests_before)} → ${formatTests(s.tests_after)}`} />
        <Stat label="Steps" value={String(s.steps)} />
        <Stat
          label="Tokens in / out"
          value={`${formatTokens(s.input_tokens)} / ${formatTokens(s.output_tokens)}`}
          note={`${cachedShare(s.input_tokens, s.cache_read_tokens)}% of input from cache`}
        />
        <Stat label="Est. cost" value={formatCost(s.cost_usd)} />
        <Stat label="Duration" value={formatDuration(s.duration_seconds)} />
      </div>

      {s.summary && (
        <div className="card">
          <h3>Agent's summary</h3>
          <p className="prewrap">{s.summary}</p>
        </div>
      )}
      {s.error && <div className="card error-card prewrap">{s.error}</div>}

      <nav className="tabs">
        {(["timeline", "diff", "settings"] as Tab[]).map((t) => (
          <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>
            {t === "timeline" ? `Timeline (${run.events.length})` : t === "diff" ? "Diff" : "Settings"}
          </button>
        ))}
      </nav>

      {tab === "timeline" && <Timeline events={run.events} />}
      {tab === "diff" && <Diff diff={run.diff} running={s.status === "running"} />}
      {tab === "settings" && <pre className="code">{JSON.stringify(run.meta, null, 2)}</pre>}
    </>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="stat">
      <div className="muted small">{label}</div>
      <div className="stat-value">{value}</div>
      {note && <div className="muted small">{note}</div>}
    </div>
  );
}

function Diff({ diff, running }: { diff: string | null; running: boolean }) {
  if (diff === null) return <p className="muted">{running ? "The diff is written when the run ends." : "No diff recorded."}</p>;
  if (diff.trim() === "") return <p className="muted">The agent made no changes.</p>;
  return (
    <pre className="code diff">
      {diff.split("\n").map((line, i) => {
        const kind = line.startsWith("+++") || line.startsWith("---") ? "file"
          : line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : line.startsWith("@@") ? "hunk" : "";
        return <div key={i} className={kind}>{line || " "}</div>;
      })}
    </pre>
  );
}
