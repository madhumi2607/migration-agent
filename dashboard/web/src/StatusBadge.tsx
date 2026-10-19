import type { RunSummary } from "./types";

export function StatusBadge({ status, stopReason }: { status: RunSummary["status"]; stopReason: string | null }) {
  const label = status === "running" ? "running" : status === "succeeded" ? "succeeded" : `failed · ${stopReason ?? "?"}`;
  return <span className={`badge badge-${status}`}>{label}</span>;
}
