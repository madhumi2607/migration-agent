import { navigate } from "./App";
import { useRuns } from "./api";
import { cachedShare, formatCost, formatDuration, formatTests, formatTime, formatTokens } from "./format";
import { StatusBadge } from "./StatusBadge";
import type { RunSummary } from "./types";

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function Headline({ runs }: { runs: RunSummary[] }) {
  const finished = runs.filter((r) => r.status !== "running");
  const succeeded = finished.filter((r) => r.status === "succeeded");
  const costs = finished.map((r) => r.cost_usd).filter((c): c is number => c !== null);
  const medianSteps = median(succeeded.map((r) => r.steps));
  return (
    <div className="stats">
      <div className="stat">
        <div className="stat-label">Success rate</div>
        <div className="stat-value">{finished.length ? `${Math.round((100 * succeeded.length) / finished.length)}%` : "—"}</div>
        <div className="muted small">{succeeded.length} of {finished.length} finished runs</div>
      </div>
      <div className="stat">
        <div className="stat-label">Median cost</div>
        <div className="stat-value">{formatCost(median(costs))}</div>
        <div className="muted small">per run, where prices are known</div>
      </div>
      <div className="stat">
        <div className="stat-label">Median steps to success</div>
        <div className="stat-value">{medianSteps ?? "—"}</div>
        <div className="muted small">model calls</div>
      </div>
      <div className="stat">
        <div className="stat-label">Runs</div>
        <div className="stat-value">{runs.length}</div>
        <div className="muted small">{runs.length - finished.length} in progress</div>
      </div>
    </div>
  );
}

function ChecksCell({ errors, warnings }: { errors: number | null; warnings: number | null }) {
  if (errors == null || warnings == null) return <span className="muted">—</span>;
  if (errors === 0 && warnings === 0) return <span className="check-pill ok"><span aria-hidden>✓</span> clean</span>;
  return (
    <span className="check-pills">
      {errors > 0 && <span className="check-pill err"><span aria-hidden>✕</span> {errors} error{errors > 1 ? "s" : ""}</span>}
      {warnings > 0 && <span className="check-pill warn"><span aria-hidden>!</span> {warnings} warning{warnings > 1 ? "s" : ""}</span>}
    </span>
  );
}

export default function RunsPage() {
  const { data: runs, error } = useRuns();

  if (error) return <p className="error">Could not load runs: {error}</p>;
  if (!runs) return <p className="muted">Loading…</p>;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Runs</h1>
          <div className="muted small">Every agent run recorded in runs/, newest first</div>
        </div>
      </div>
      {runs.length === 0 ? (
        <div className="card empty">
          No runs yet. Start one with <code>uv run --env-file .env python -m agent.loop</code>.
        </div>
      ) : (
        <>
          <Headline runs={runs} />
          <div className="table-wrap">
            <table className="runs">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Model</th>
                  <th>Status</th>
                  <th className="num">Tests</th>
                  <th>Checks</th>
                  <th className="num">Steps</th>
                  <th className="num">Input</th>
                  <th className="num">Cost</th>
                  <th className="num">Duration</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id} onClick={() => navigate(`/runs/${encodeURIComponent(r.run_id)}`)}>
                    <td>
                      <div className="strong">{r.repo} <span className="muted mono small">@{r.repo_commit}</span></div>
                      <div className="muted small">{formatTime(r.started_at)}</div>
                    </td>
                    <td>
                      <span className="mono small">{r.model}</span>
                      {r.effort && <div className="muted small">effort {r.effort}</div>}
                    </td>
                    <td><StatusBadge status={r.status} stopReason={r.stop_reason} /></td>
                    <td className="num mono">{formatTests(r.tests_before)} → {formatTests(r.tests_after)}</td>
                    <td><ChecksCell errors={r.check_errors} warnings={r.check_warnings} /></td>
                    <td className="num">{r.steps}</td>
                    <td className="num">
                      {formatTokens(r.input_tokens)}
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
      )}
    </>
  );
}
