// Hand-rolled SVG charts (no chart library). Specs follow the data-viz rules used for
// this dashboard: thin marks, 2px surface gaps, hairline grid, one y-axis per chart,
// text in ink (never series color), a legend for 2+ series, hover tooltips that lead
// with the value, and a table view for every chart so no value is hover-only.
import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

export type Series<T> = { key: keyof T & string; label: string; color: string };

// ---------- shared pieces ----------

function useWidth<E extends HTMLElement>() {
  const ref = useRef<E>(null);
  const [width, setWidth] = useState(600);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/** ~4 round tick values from 0 to at least max. */
function niceTicks(max: number): number[] {
  if (max <= 0) return [0, 1];
  const raw = max / 4;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw)!;
  const ticks = [];
  for (let v = 0; v <= max + step * 0.001; v += step) ticks.push(v);
  if (ticks[ticks.length - 1] < max) ticks.push(ticks[ticks.length - 1] + step);
  return ticks;
}

export function ChartCard(props: {
  title: string;
  subtitle?: string;
  legend?: { label: string; color: string; kind?: "rect" | "line" }[];
  table: () => ReactNode;
  children: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <section className="chart-card">
      <header className="chart-head">
        <div>
          <h3>{props.title}</h3>
          {props.subtitle && <div className="muted small">{props.subtitle}</div>}
        </div>
        <button className="ghost-btn" onClick={() => setShowTable(!showTable)}>
          {showTable ? "Chart" : "Table"}
        </button>
      </header>
      {props.legend && props.legend.length > 1 && !showTable && (
        <ul className="legend">
          {props.legend.map((l) => (
            <li key={l.label}>
              <span className={l.kind === "line" ? "key-line" : "key-rect"} style={{ background: l.color }} />
              {l.label}
            </li>
          ))}
        </ul>
      )}
      {showTable ? <div className="chart-table">{props.table()}</div> : props.children}
    </section>
  );
}

type Tip = { x: number; y: number; title: string; rows: { label: string; value: string; color?: string }[] };

function Tooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null;
  return (
    <div className="tooltip" style={{ left: tip.x, top: tip.y }}>
      <div className="tooltip-title">{tip.title}</div>
      {tip.rows.map((r) => (
        <div className="tooltip-row" key={r.label}>
          {r.color && <span className="key-line" style={{ background: r.color }} />}
          <b>{r.value}</b> <span className="muted">{r.label}</span>
        </div>
      ))}
    </div>
  );
}

const PAD = { top: 10, right: 16, bottom: 26, left: 52 };

// ---------- stacked columns ----------

export function StackedColumns<T extends { step: number }>(props: {
  data: T[];
  series: Series<T>[];
  format: (v: number) => string;
  height?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<Tip | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const height = props.height ?? 200;
  const plotW = Math.max(width - PAD.left - PAD.right, 10);
  const plotH = height - PAD.top - PAD.bottom;

  const totals = props.data.map((d) => props.series.reduce((sum, s) => sum + Number(d[s.key]), 0));
  const ticks = niceTicks(Math.max(...totals, 0));
  const yMax = ticks[ticks.length - 1];
  const y = (v: number) => PAD.top + plotH - (v / yMax) * plotH;
  const band = plotW / Math.max(props.data.length, 1);
  const barW = Math.min(24, Math.max(band * 0.6, 2));
  const labelEvery = Math.ceil(props.data.length / Math.max(Math.floor(plotW / 28), 1));

  return (
    <div className="chart" ref={ref} onPointerLeave={() => { setTip(null); setHovered(null); }}>
      <svg width={width} height={height} role="img">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={PAD.left + plotW} y1={y(t)} y2={y(t)} className={t === 0 ? "axis" : "grid"} />
            <text x={PAD.left - 8} y={y(t) + 4} textAnchor="end" className="tick">{props.format(t)}</text>
          </g>
        ))}
        {props.data.map((d, i) => {
          const cx = PAD.left + band * i + band / 2;
          let base = 0;
          const visible = props.series.filter((s) => Number(d[s.key]) > 0);
          return (
            <g key={d.step} opacity={hovered === null || hovered === i ? 1 : 0.55}>
              {visible.map((s, j) => {
                const v = Number(d[s.key]);
                const top = y(base + v);
                const bottom = y(base);
                base += v;
                const isTop = j === visible.length - 1;
                // 2px surface gap between stacked segments; 4px rounded data-end on the top segment only.
                const h = Math.max(bottom - top - (j > 0 ? 2 : 0), 0.5);
                return isTop ? (
                  <path key={s.key} d={roundedTop(cx - barW / 2, top, barW, h, Math.min(4, h, barW / 2))} fill={s.color} />
                ) : (
                  <rect key={s.key} x={cx - barW / 2} y={top} width={barW} height={h} fill={s.color} />
                );
              })}
              {i % labelEvery === 0 && (
                <text x={cx} y={height - 8} textAnchor="middle" className="tick">{d.step}</text>
              )}
              {/* Hit target: the whole band, not just the painted bar. */}
              <rect
                x={PAD.left + band * i} y={PAD.top} width={band} height={plotH} fill="transparent"
                tabIndex={0}
                onPointerMove={(e) => {
                  setHovered(i);
                  setTip(tipAt(e.currentTarget, cx, `Step ${d.step}`, props.series.map((s) => ({
                    label: s.label, value: props.format(Number(d[s.key])), color: s.color,
                  })).reverse()));
                }}
                onFocus={(e) => {
                  setHovered(i);
                  setTip(tipAt(e.currentTarget, cx, `Step ${d.step}`, props.series.map((s) => ({
                    label: s.label, value: props.format(Number(d[s.key])), color: s.color,
                  })).reverse()));
                }}
              />
            </g>
          );
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

function roundedTop(x: number, y: number, w: number, h: number, r: number): string {
  return `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h} Z`;
}

function tipAt(target: Element, cx: number, title: string, rows: Tip["rows"]): Tip {
  const svg = target.closest("svg")!;
  const width = svg.getBoundingClientRect().width;
  // Keep the tooltip inside the card: flip to the left of the mark in the right third.
  const x = cx > width * 0.66 ? cx - 170 : cx + 14;
  return { x, y: 8, title, rows };
}

// ---------- cumulative line ----------

export function CumulativeLine<T extends { step: number }>(props: {
  data: T[];
  value: (d: T) => number;
  label: string;
  color: string;
  format: (v: number) => string;
  height?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const height = props.height ?? 200;
  const plotW = Math.max(width - PAD.left - PAD.right - 40, 10); // room for the end label
  const plotH = height - PAD.top - PAD.bottom;
  const values = props.data.map(props.value);
  const ticks = niceTicks(Math.max(...values, 0));
  const yMax = ticks[ticks.length - 1];
  const n = props.data.length;
  const x = (i: number) => PAD.left + (n <= 1 ? plotW / 2 : (plotW * i) / (n - 1));
  const y = (v: number) => PAD.top + plotH - (v / yMax) * plotH;
  const line = values.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(" ");
  const area = n ? `${line} L${x(n - 1)},${y(0)} L${x(0)},${y(0)} Z` : "";
  const last = n - 1;

  const onMove = (clientX: number, svg: SVGSVGElement) => {
    const px = clientX - svg.getBoundingClientRect().left;
    const i = Math.round(((px - PAD.left) / plotW) * (n - 1));
    setHover(Math.min(Math.max(i, 0), n - 1));
  };

  return (
    <div className="chart" ref={ref} onPointerLeave={() => setHover(null)}>
      <svg
        width={width} height={height} role="img"
        onPointerMove={(e) => onMove(e.clientX, e.currentTarget)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={PAD.left + plotW} y1={y(t)} y2={y(t)} className={t === 0 ? "axis" : "grid"} />
            <text x={PAD.left - 8} y={y(t) + 4} textAnchor="end" className="tick">{props.format(t)}</text>
          </g>
        ))}
        <path d={area} fill={props.color} opacity={0.1} />
        <path d={line} fill="none" stroke={props.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {n > 0 && (
          <>
            {/* End marker with a 2px surface ring, and the one direct label: the final value. */}
            <circle cx={x(last)} cy={y(values[last])} r={4} fill={props.color} stroke="var(--surface)" strokeWidth={2} />
            <text x={x(last) + 8} y={y(values[last]) + 4} className="end-label">{props.format(values[last])}</text>
          </>
        )}
        {props.data.map((d, i) => (i % Math.ceil(n / Math.max(Math.floor(plotW / 28), 1)) === 0 ? (
          <text key={d.step} x={x(i)} y={height - 8} textAnchor="middle" className="tick">{d.step}</text>
        ) : null))}
        {hover !== null && (
          <>
            <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + plotH} className="crosshair" />
            <circle cx={x(hover)} cy={y(values[hover])} r={4} fill={props.color} stroke="var(--surface)" strokeWidth={2} />
          </>
        )}
      </svg>
      {hover !== null && (
        <Tooltip tip={{
          x: x(hover) > width * 0.66 ? x(hover) - 170 : x(hover) + 14, y: 8,
          title: `After step ${props.data[hover].step}`,
          rows: [{ label: props.label, value: props.format(values[hover]), color: props.color }],
        }} />
      )}
    </div>
  );
}

// ---------- test-run meters ----------

export type MeterRow = { label: string; passed: number; total: number | null; failed: number; errors: number; status: string };

/** One meter per test run: passed share as a status-green fill on a neutral track, always with
 * an icon and the counts in text, so the state never depends on color. */
export function TestMeters({ rows }: { rows: MeterRow[] }) {
  return (
    <ul className="meters">
      {rows.map((r, i) => {
        const ok = r.status === "passed";
        const share = r.total ? r.passed / r.total : 0;
        const problems = [r.failed && `${r.failed} failed`, r.errors && `${r.errors} errors`].filter(Boolean).join(" · ");
        return (
          <li key={i}>
            <div className="meter-label">
              <span className={ok ? "icon-ok" : "icon-bad"} aria-hidden>{ok ? "✓" : "✕"}</span>
              <span>{r.label}</span>
            </div>
            <div className="meter-track" role="meter" aria-valuemin={0} aria-valuemax={r.total ?? 0} aria-valuenow={r.passed}>
              <div className="meter-fill" style={{ width: `${share * 100}%` }} />
            </div>
            <div className="meter-value">
              <b>{r.total === null ? r.status.replace("_", " ") : `${r.passed}/${r.total}`}</b>
              <span className="muted"> {r.total === null ? "" : "passed"}{problems && ` · ${problems}`}</span>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

// ---------- table view ----------

export function DataTable({ columns, rows }: { columns: string[]; rows: (string | number)[][] }) {
  return (
    <table className="data-table">
      <thead>
        <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>{r.map((v, j) => <td key={j} className={typeof v === "number" ? "num" : ""}>{v}</td>)}</tr>
        ))}
      </tbody>
    </table>
  );
}
