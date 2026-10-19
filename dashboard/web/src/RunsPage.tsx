import { navigate } from "./App";
import { useRuns } from "./api";
import { cachedShare, formatCost, formatDuration, formatTests, formatTime, formatTokens } from "./format";
import { StatusBadge } from "./StatusBadge";

export default function RunsPage() {
  const { data: runs, error } = useRuns();

  if (error) return <p className="error">Could not load runs: {error}</p>;
  if (!runs) return <p className="muted">Loading…</p>;
  if (runs.length === 0)
    return (
      <p className="muted">
        No runs yet. Start one with <code>uv run --env-file .env python -m agent.loop</code>.
      </p>
    );

  return (
    <>
      <h1>Runs</h1>
      <div className="table-wrap">
        <table className="runs">
          <thead>
            <tr>
              <th>Started</th>
              <th>Repo</th>
              <th>Model</th>
              <th>Status</th>
              <th className="num">Tests</th>
              <th className="num">Steps</th>
              <th className="num">Tokens in / out</th>
              <th className="num">Cost</th>
              <th className="num">Duration</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id} onClick={() => navigate(`/runs/${encodeURIComponent(r.run_id)}`)}>
                <td>{formatTime(r.started_at)}</td>
                <td>
                  {r.repo} <span className="muted mono">@{r.repo_commit}</span>
                </td>
                <td>
                  {r.model}
                  {r.effort && <span className="muted"> · {r.effort}</span>}
                </td>
                <td>
                  <StatusBadge status={r.status} stopReason={r.stop_reason} />
                </td>
                <td className="num mono">
                  {formatTests(r.tests_before)} → {formatTests(r.tests_after)}
                </td>
                <td className="num">{r.steps}</td>
                <td className="num">
                  {formatTokens(r.input_tokens)} / {formatTokens(r.output_tokens)}
                  <div className="muted small">{cachedShare(r.input_tokens, r.cache_read_tokens)}% cached</div>
                </td>
                <td className="num">{formatCost(r.cost_usd)}</td>
                <td className="num">{formatDuration(r.duration_seconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
