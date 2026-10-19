import type { TestCounts } from "./types";

export const formatTokens = (n: number) =>
  n >= 1_000_000 ? `${(n / 1_000_000).toFixed(2)}M` : n >= 1_000 ? `${Math.round(n / 1_000)}K` : String(n);

export const formatCost = (usd: number | null) => (usd === null ? "—" : `$${usd.toFixed(2)}`);

export function formatDuration(seconds: number | null): string {
  if (seconds === null) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export const formatTime = (iso: string | null) =>
  iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—";

export const formatTests = (t: TestCounts | null) =>
  t === null ? "—" : t.total === null ? t.status.replace("_", " ") : `${t.passed}/${t.total}`;

export const cachedShare = (input: number, cached: number) => (input ? Math.round((100 * cached) / input) : 0);
