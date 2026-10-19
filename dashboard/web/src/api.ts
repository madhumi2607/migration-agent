import { useEffect, useState } from "react";
import type { RunDetail, RunSummary } from "./types";

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

/** Fetch `url` now and then every `intervalMs` while `keepPolling` is true (live runs). */
function usePolled<T>(url: string, intervalMs: number, keepPolling: (data: T) => boolean) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let timer: number | undefined;
    let cancelled = false;
    const load = async () => {
      try {
        const next = await getJson<T>(url);
        if (cancelled) return;
        setData(next);
        setError(null);
        if (keepPolling(next)) timer = window.setTimeout(load, intervalMs);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    };
    load();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // keepPolling is a pure predicate defined inline by callers, so only url and interval trigger reloads.
  }, [url, intervalMs]);

  return { data, error };
}

export const useRuns = () =>
  usePolled<RunSummary[]>("/api/runs", 5000, (runs) => runs.some((r) => r.status === "running"));

export const useRun = (runId: string) =>
  usePolled<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`, 3000, (run) => run.summary.status === "running");
