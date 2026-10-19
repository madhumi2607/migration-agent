// Turn a run's event stream into the series the charts plot. Pure functions, no React.
import type { RunEvent, TestReport } from "./types";

// USD per million tokens (uncached input, cache read, output); mirrors agent/pricing.py.
const PRICES: Record<string, [number, number, number]> = {
  "claude-opus-5-5": [4.0, 0.2, 20.0],
  "claude-sonnet-5-5": [2.0, 0.2, 10.0],
  "claude-haiku-5-5": [0.1, 0.01, 0.5],
};

export type StepPoint = {
  step: number;
  cachedInput: number;
  uncachedInput: number;
  output: number;
  cumulativeCost: number | null; // null when the model has no known prices
  modelSeconds: number;
  testSeconds: number; // run_tests calls and finish verifications
  toolSeconds: number; // every other tool
};

export function stepSeries(events: RunEvent[], model: string | null): StepPoint[] {
  const prices = model ? PRICES[model] : undefined;
  const points: StepPoint[] = [];
  let cost = 0;
  for (const e of events) {
    if (e.type === "llm_call") {
      const uncached = e.input_tokens - e.cache_read_tokens;
      if (prices) cost += (uncached * prices[0] + e.cache_read_tokens * prices[1] + e.output_tokens * prices[2]) / 1e6;
      points.push({
        step: e.step, cachedInput: e.cache_read_tokens, uncachedInput: uncached, output: e.output_tokens,
        cumulativeCost: prices ? cost : null, modelSeconds: e.seconds, testSeconds: 0, toolSeconds: 0,
      });
    } else if (e.type === "tool_call") {
      const point = points.find((p) => p.step === e.step);
      if (!point) continue;
      if (e.name === "run_tests" || e.name === "finish") point.testSeconds += e.seconds;
      else point.toolSeconds += e.seconds;
    }
  }
  return points;
}

export type TestRunPoint = {
  label: string;
  afterStep: number; // 0 = before the first model call
  passed: number;
  total: number | null;
  failed: number;
  errors: number;
  status: string;
  report: TestReport;
};

const PURPOSE_LABEL: Record<string, string> = { baseline: "Baseline", verification: "Verification", final: "Final" };

/** Every full-suite run the loop did itself, plus run_tests calls the model made. */
export function testRuns(events: RunEvent[]): TestRunPoint[] {
  const runs: TestRunPoint[] = [];
  let step = 0;
  for (const e of events) {
    if (e.type === "llm_call") step = e.step;
    let report: TestReport | null = null;
    let label = "";
    if (e.type === "test_run") {
      report = e.report;
      label = `${PURPOSE_LABEL[e.purpose] ?? e.purpose}${e.purpose === "baseline" ? "" : ` · step ${step}`}`;
    } else if (e.type === "tool_call" && e.name === "run_tests") {
      report = parseReport(e.result);
      const subset = Array.isArray(e.arguments.tests) && e.arguments.tests.length > 0;
      label = `run_tests${subset ? " (subset)" : ""} · step ${step}`;
    }
    if (!report) continue;
    const c = report.counts;
    runs.push({
      label, afterStep: step, report, status: report.status,
      passed: c?.passed ?? 0, total: c?.total ?? null, failed: c?.failed ?? 0, errors: c?.errors ?? 0,
    });
  }
  return runs;
}

function parseReport(result: string): TestReport | null {
  try {
    const parsed = JSON.parse(result);
    return parsed && typeof parsed.status === "string" ? (parsed as TestReport) : null;
  } catch {
    return null; // e.g. an "ERROR: ..." refusal
  }
}
