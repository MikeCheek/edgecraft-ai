import { Download, X, ZoomIn } from 'lucide-react';
import React, { useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts';

// Categorical slots (validated dark-surface steps): train, val, third series.
export const SERIES_COLORS = { blue: '#3987e5', orange: '#d95926', aqua: '#199e70', violet: '#9085e9' };

export interface Series {
  key: string;
  name: string;
  color: string;
  dashed?: boolean;
}

export interface ChartMarker {
  x: number;
  label: string;
  color?: string;
}

type Row = Record<string, number | null | undefined>;

interface SeriesChartProps {
  data: Row[];
  series: Series[];
  xKey?: string;
  xLabel?: string;
  height?: number;
  formatY?: (v: number) => string;
  logY?: boolean;
  domainY?: [number | 'auto' | ((v: number) => number), number | 'auto' | ((v: number) => number)];
  markers?: ChartMarker[];
  large?: boolean;
}

const fmtDefault = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3));

export function SeriesChart({
  data, series, xKey = 'epoch', xLabel = 'Epoch', height = 160, formatY, logY, domainY, markers, large,
}: SeriesChartProps) {
  const fmt = formatY ?? fmtDefault;
  const fs = large ? 11 : 10;
  const xs = data.map((d) => d[xKey]).filter((v): v is number => typeof v === 'number');
  const midX = xs.length ? (Math.min(...xs) + Math.max(...xs)) / 2 : 0;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#1e293b" vertical={false} />
        <XAxis
          dataKey={xKey}
          type="number"
          domain={[(d: number) => Math.floor(d), (d: number) => Math.ceil(d)]}
          allowDecimals={false}
          tickFormatter={(v: number) => String(Math.round(v))}
          tick={{ fill: '#94a3b8', fontSize: fs }}
          tickLine={false}
          axisLine={{ stroke: '#334155' }}
          label={large ? { value: xLabel, position: 'insideBottomRight', offset: -4, fill: '#64748b', fontSize: fs } : undefined}
        />
        <YAxis
          scale={logY ? 'log' : 'auto'}
          domain={domainY ?? (logY ? ['auto', 'auto'] : ['auto', 'auto'])}
          allowDataOverflow={false}
          tick={{ fill: '#94a3b8', fontSize: fs }}
          tickLine={false}
          axisLine={false}
          tickFormatter={fmt}
          width={large ? 64 : 58}
        />
        <Tooltip
          contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: '#94a3b8' }}
          itemStyle={{ color: '#e2e8f0' }}
          labelFormatter={(x: number) => `${xLabel} ${Number.isInteger(x) ? x : x.toFixed(2)}`}
          formatter={(v: number, name: string) => [v == null ? '—' : fmt(v), name]}
          cursor={{ stroke: '#475569', strokeWidth: 1 }}
        />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: fs, paddingTop: 2, color: '#cbd5e1' }} iconType="plainline" />}
        {markers?.map((m) => (
          <ReferenceLine key={`${m.label}-${m.x}`} x={m.x} stroke={m.color ?? '#64748b'} strokeDasharray="3 3"
            label={{ value: m.label, position: m.x > midX ? 'insideTopRight' : 'insideTopLeft', fill: '#94a3b8', fontSize: 9 }} />
        ))}
        {series.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.name}
            stroke={s.color}
            strokeWidth={2}
            strokeDasharray={s.dashed ? '5 3' : undefined}
            dot={data.length <= 12 ? { r: 2.5, strokeWidth: 0, fill: s.color } : false}
            activeDot={{ r: 4, stroke: '#0f172a', strokeWidth: 2 }}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Titled card around a SeriesChart with an expand button. */
export function ChartCard({
  title, subtitle, onExpand, children, className = '',
}: { title: string; subtitle?: string; onExpand?: () => void; children: React.ReactNode; className?: string }) {
  return (
    <div className={`p-4 bg-slate-900/50 rounded-xl border border-slate-700/70 ${className}`}>
      <div className="flex items-start justify-between mb-2 gap-2">
        <div className="min-w-0">
          <p className="text-xs text-gray-300 font-medium">{title}</p>
          {subtitle && <p className="text-[10px] text-gray-500 mt-0.5">{subtitle}</p>}
        </div>
        {onExpand && (
          <button onClick={onExpand} title="Expand chart" aria-label={`Expand ${title}`}
            className="p-1 -m-1 text-gray-500 hover:text-white hover:bg-slate-700 rounded-md transition shrink-0">
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Backwards-compatible train/val chart (used by the past-session popup)
// ---------------------------------------------------------------------------

interface MetricChartProps {
  data: { epoch: number; train: number; val: number }[];
  label: string;
  color: string;
  valColor: string;
  formatY?: (v: number) => string;
  expanded?: boolean;
}

function trainValSeries(color: string, valColor: string): Series[] {
  return [
    { key: 'train', name: 'Train', color },
    { key: 'val', name: 'Val', color: valColor, dashed: true },
  ];
}

function MetricChart({ data, label, color, valColor, formatY, expanded }: MetricChartProps) {
  return (
    <div className={`p-4 bg-slate-900/50 rounded-xl border border-slate-700 ${expanded ? 'col-span-2' : ''}`}>
      <p className="text-xs text-gray-400 mb-3 font-medium">{label}</p>
      <SeriesChart data={data} series={trainValSeries(color, valColor)} formatY={formatY} height={expanded ? 320 : 140} />
    </div>
  );
}

// Expanded chart modal with export
interface ChartModalProps {
  label: string;
  data: Row[];
  series?: Series[];
  color?: string;
  valColor?: string;
  formatY?: (v: number) => string;
  logY?: boolean;
  xKey?: string;
  xLabel?: string;
  markers?: ChartMarker[];
  onClose: () => void;
}

function ChartModal({ label, color, valColor, series, data, formatY, logY, xKey, xLabel, markers, onClose }: ChartModalProps) {
  const svgRef = useRef<HTMLDivElement>(null);
  const lines = series ?? trainValSeries(color ?? SERIES_COLORS.blue, valColor ?? SERIES_COLORS.orange);

  const handleExport = () => {
    const svgEl = svgRef.current?.querySelector('svg');
    if (!svgEl) return;
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svgEl);
    const blob = new Blob([svgStr], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${label.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <div
        className="w-full max-w-4xl bg-slate-900 rounded-2xl border border-slate-700 shadow-2xl p-6 animate-slideIn"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-white">{label}</h3>
          <div className="flex gap-2">
            <button
              onClick={handleExport}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-gray-300 text-sm rounded-lg transition"
            >
              <Download className="w-4 h-4" /> Export SVG
            </button>
            <button onClick={onClose} className="p-2 text-gray-400 hover:text-white hover:bg-slate-700 rounded-lg transition" aria-label="Close">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>
        <div ref={svgRef}>
          <SeriesChart data={data} series={lines} formatY={formatY} logY={logY} xKey={xKey} xLabel={xLabel}
            markers={markers} height={380} large />
        </div>
      </div>
    </div>
  );
}

export { MetricChart, ChartModal };
