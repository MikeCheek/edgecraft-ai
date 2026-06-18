import { Download, X } from 'lucide-react';
import React, { useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';


interface MetricChartProps {
  data: { epoch: number; train: number; val: number }[];
  label: string;
  color: string;
  valColor: string;
  formatY?: (v: number) => string;
  expanded?: boolean;
}

function MetricChart({ data, label, color, valColor, formatY, expanded }: MetricChartProps) {
  const height = expanded ? 320 : 140;
  return (
    <div className={`p-4 bg-slate-900/50 rounded-xl border border-slate-700 ${expanded ? 'col-span-2' : ''}`}>
      <p className="text-xs text-gray-400 mb-3 font-medium">{label}</p>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="epoch"
            tick={{ fill: '#94a3b8', fontSize: 10 }}
            tickLine={false}
            label={{ value: 'Epoch', position: 'insideBottomRight', offset: -4, fill: '#64748b', fontSize: 10 }}
          />
          <YAxis
            tick={{ fill: '#94a3b8', fontSize: 10 }}
            tickLine={false}
            tickFormatter={formatY}
          />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#94a3b8' }}
            formatter={(v: number, name: string) => [
              formatY ? formatY(v) : v.toFixed(4),
              name,
            ]}
          />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 4 }} />
          <Line type="monotone" dataKey="train" name="Train" stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="val" name="Val" stroke={valColor} strokeWidth={2} dot={false} strokeDasharray="4 2" isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// Expanded chart modal with export
interface ChartModalProps {
  label: string;
  color: string;
  valColor: string;
  data: { epoch: number; train: number; val: number }[];
  formatY?: (v: number) => string;
  onClose: () => void;
}

function ChartModal({ label, color, valColor, data, formatY, onClose }: ChartModalProps) {
  const svgRef = useRef<HTMLDivElement>(null);

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
            <button onClick={onClose} className="p-2 text-gray-400 hover:text-white hover:bg-slate-700 rounded-lg transition">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>
        <div ref={svgRef}>
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="epoch"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                tickLine={false}
                label={{ value: 'Epoch', position: 'insideBottomRight', offset: -4, fill: '#64748b', fontSize: 11 }}
              />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickLine={false} tickFormatter={formatY} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #475569', borderRadius: 8, fontSize: 13 }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={(v: number, name: string) => [
                  formatY ? formatY(v) : v.toFixed(4),
                  name,
                ]}
              />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Line type="monotone" dataKey="train" name="Train" stroke={color} strokeWidth={2.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="val" name="Val" stroke={valColor} strokeWidth={2.5} dot={false} strokeDasharray="5 3" isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export { MetricChart, ChartModal };