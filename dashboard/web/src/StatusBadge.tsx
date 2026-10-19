import type { RunSummary } from "./types";

// Status always pairs an icon and a word with its color, so it never depends on color alone.
const ICON = { running: "●", succeeded: "✓", failed: "✕" } as const;

export function StatusBadge(props: { status: RunSummary["status"]; stopReason: string | null; large?: boolean }) {
  const { status, stopReason } = props;
  const label = status === "failed" ? `failed · ${(stopReason ?? "?").replace("_", " ")}` : status;
  return (
    <span className={`badge badge-${status}${props.large ? " badge-lg" : ""}`}>
      <span className={status === "running" ? "pulse" : ""} aria-hidden>{ICON[status]}</span> {label}
    </span>
  );
}
