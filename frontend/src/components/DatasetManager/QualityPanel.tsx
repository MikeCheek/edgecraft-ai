// Dataset quality report: issues that silently hurt models (imbalance, tiny
// classes, duplicates leaking across splits, unreadable files, missing boxes).

import { useCallback, useEffect, useState } from 'react';
import { AlertOctagon, AlertTriangle, Info, RefreshCw, ShieldCheck, Trash2 } from 'lucide-react';
import { apiFetch } from '../../config';
import { useToast } from '../../context/ToastContext';

interface Issue {
  severity: 'error' | 'warning' | 'info';
  code: string;
  message: string;
}

interface QualityReport {
  total_samples: number;
  class_counts: Record<string, number>;
  split_counts: Record<string, Record<string, number>>;
  imbalance_ratio: number | null;
  duplicate_group_count: number;
  leakage_group_count: number;
  unreadable: { id: string; filename: string; reason: string }[];
  unreadable_count: number;
  issues: Issue[];
  score: number;
}

const ICON = {
  error: <AlertOctagon className="w-4 h-4 text-red-400 shrink-0" />,
  warning: <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />,
  info: <Info className="w-4 h-4 text-sky-400 shrink-0" />,
};

export default function QualityPanel({ datasetId, onChanged }: { datasetId: string; onChanged?: () => void }) {
  const { toast } = useToast();
  const [report, setReport] = useState<QualityReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [removing, setRemoving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await apiFetch(`/datasets/${datasetId}/quality`);
      const j = await r.json();
      if (j.status === 'success') setReport(j.report);
      else toast('error', j.detail ?? 'Quality check failed');
    } catch {
      toast('error', 'Quality check failed');
    } finally {
      setLoading(false);
    }
  }, [datasetId, toast]);

  useEffect(() => { load(); }, [load]);

  const removeDuplicates = async () => {
    setRemoving(true);
    try {
      const j = await apiFetch(`/datasets/${datasetId}/remove_duplicates`, { method: 'POST' }).then(r => r.json());
      toast('success', `Removed ${j.removed ?? 0} duplicate file(s)`);
      onChanged?.();
      load();
    } finally {
      setRemoving(false);
    }
  };

  if (loading && !report) {
    return <p className="text-sm text-gray-400 flex items-center gap-2"><RefreshCw className="w-4 h-4 animate-spin" /> Analyzing every file…</p>;
  }
  if (!report) return null;

  const maxCount = Math.max(1, ...Object.values(report.class_counts));
  const scoreColor = report.score >= 85 ? 'text-emerald-400' : report.score >= 60 ? 'text-amber-300' : 'text-red-400';

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <ShieldCheck className={`w-5 h-5 ${scoreColor}`} />
        <span className="text-sm text-white font-semibold">Quality score <span className={`font-mono ${scoreColor}`}>{report.score}/100</span></span>
        <span className="text-xs text-gray-500">{report.total_samples} samples · imbalance {report.imbalance_ratio ?? '—'}×</span>
        <div className="ml-auto flex gap-2">
          {report.duplicate_group_count > 0 && (
            <button onClick={removeDuplicates} disabled={removing}
              className="text-xs px-2.5 py-1 rounded-lg bg-red-600/20 text-red-300 hover:bg-red-600/30 flex items-center gap-1.5 disabled:opacity-50">
              <Trash2 className="w-3.5 h-3.5" /> Remove duplicates
            </button>
          )}
          <button onClick={load} disabled={loading} className="text-xs px-2.5 py-1 rounded-lg bg-slate-700 hover:bg-slate-600 text-gray-200 flex items-center gap-1.5">
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} /> Re-check
          </button>
        </div>
      </div>

      {report.issues.length === 0 ? (
        <p className="text-sm text-emerald-300">No problems found.</p>
      ) : (
        <ul className="space-y-1.5">
          {report.issues.map((i) => (
            <li key={i.code} className="flex items-start gap-2 text-sm text-gray-300">{ICON[i.severity]} {i.message}</li>
          ))}
        </ul>
      )}

      <div className="space-y-1">
        <p className="text-[11px] uppercase tracking-wide text-gray-500">Samples per class (train / val / test)</p>
        {Object.entries(report.class_counts).sort((a, b) => b[1] - a[1]).map(([label, count]) => {
          const sp = report.split_counts[label] ?? {};
          return (
            <div key={label} className="flex items-center gap-2 text-xs">
              <span className="w-28 truncate text-gray-300" title={label}>{label}</span>
              <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-2 bg-purple-500/70" style={{ width: `${(count / maxCount) * 100}%` }} />
              </div>
              <span className="w-36 text-right font-mono text-gray-400">{count} ({sp.train ?? 0}/{sp.val ?? 0}/{sp.test ?? 0})</span>
            </div>
          );
        })}
      </div>

      {report.unreadable_count > 0 && (
        <details className="text-xs text-gray-400">
          <summary className="cursor-pointer">Unreadable files ({report.unreadable_count})</summary>
          <ul className="mt-1 space-y-0.5 font-mono">
            {report.unreadable.map((u) => <li key={u.id}>{u.filename} - {u.reason}</li>)}
          </ul>
        </details>
      )}
    </div>
  );
}
