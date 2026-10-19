import { ChartCard, CumulativeLine, DataTable, StackedColumns, TestMeters, type Series } from "./charts";
import { stepSeries, testRuns, type StepPoint } from "./derive";
import { formatTokens } from "./format";
import type { RunEvent } from "./types";

// Categorical slots 1-3 of the validated dark palette (worst adjacent CVD dE 9.4 on #1a1a19).
const BLUE = "var(--series-1)";
const ORANGE = "var(--series-2)";
const AQUA = "var(--series-3)";

const TOKEN_SERIES: Series<StepPoint>[] = [
  { key: "cachedInput", label: "Input from cache", color: BLUE },
  { key: "uncachedInput", label: "Input not cached", color: ORANGE },
];

const TIME_SERIES: Series<StepPoint>[] = [
  { key: "modelSeconds", label: "Model", color: BLUE },
  { key: "testSeconds", label: "Test runs", color: ORANGE },
  { key: "toolSeconds", label: "Other tools", color: AQUA },
];

// Seconds until values get large, so axis ticks never mix units (0s, 20s, 40s, 60s).
const seconds = (v: number) => (v >= 300 ? `${(v / 60).toFixed(0)}m` : `${Math.round(v)}s`);
const usd = (v: number) => (v < 1 ? `$${v.toFixed(2)}` : `$${v.toFixed(1)}`);

export default function RunCharts({ events, model }: { events: RunEvent[]; model: string | null }) {
  const steps = stepSeries(events, model);
  const tests = testRuns(events);
  const hasCost = steps.length > 0 && steps[0].cumulativeCost !== null;

  return (
    <div className="chart-grid">
      <ChartCard
        title="Tests"
        subtitle="Every full or partial test run, in order"
        table={() => (
          <DataTable
            columns={["Test run", "Status", "Passed", "Total", "Failed", "Errors"]}
            rows={tests.map((t) => [t.label, t.status, t.passed, t.total ?? "—", t.failed, t.errors])}
          />
        )}
      >
        {tests.length ? <TestMeters rows={tests} /> : <p className="muted">No test runs yet.</p>}
      </ChartCard>

      {hasCost && (
        <ChartCard
          title="Cumulative cost"
          subtitle={`Estimated USD after each model call (${model})`}
          table={() => (
            <DataTable columns={["Step", "Cumulative cost"]} rows={steps.map((s) => [s.step, usd(s.cumulativeCost ?? 0)])} />
          )}
        >
          <CumulativeLine data={steps} value={(s) => s.cumulativeCost ?? 0} label="cost so far" color={BLUE} format={usd} />
        </ChartCard>
      )}

      <ChartCard
        title="Context size per step"
        subtitle="Input tokens sent with each model call, by step; cached input is billed at ~1/20"
        legend={TOKEN_SERIES.map((s) => ({ label: s.label, color: s.color }))}
        table={() => (
          <DataTable
            columns={["Step", "Input from cache", "Input not cached", "Output"]}
            rows={steps.map((s) => [s.step, s.cachedInput, s.uncachedInput, s.output])}
          />
        )}
      >
        <StackedColumns data={steps} series={TOKEN_SERIES} format={formatTokens} />
      </ChartCard>

      <ChartCard
        title="Time per step"
        subtitle="Where each step's wall-clock time went, by step"
        legend={TIME_SERIES.map((s) => ({ label: s.label, color: s.color }))}
        table={() => (
          <DataTable
            columns={["Step", "Model (s)", "Test runs (s)", "Other tools (s)"]}
            rows={steps.map((s) => [s.step, s.modelSeconds.toFixed(1), s.testSeconds.toFixed(1), s.toolSeconds.toFixed(1)])}
          />
        )}
      >
        <StackedColumns data={steps} series={TIME_SERIES} format={seconds} />
      </ChartCard>
    </div>
  );
}
