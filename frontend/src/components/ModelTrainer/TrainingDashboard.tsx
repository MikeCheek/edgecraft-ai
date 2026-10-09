// Live training dashboard: progress (epoch + batch), KPI tiles with
// sparklines, automatic diagnostics, a grid of per-epoch charts, the latest
// validation confusion matrix / per-class scores and run facts.

import { useMemo, useState } from 'react';
import {
  AlertTriangle, CheckCircle2, Info, XCircle, Clock, Timer, Gauge, Layers, Cpu, CalendarClock, Zap,
} from 'lucide-react';
import { TrainingMetrics, TrainingStatus } from '../../types';
import { ChartCard, ChartMarker, ChartModal, SERIES_COLORS, Series, SeriesChart } from './Chart';
import { ConfusionMatrix } from '../EvaluationReport';
import { formatTime } from './constants';

const { blue, orange, aqua } = SERIES_COLORS;

const pct = (v: number | null | undefined, d = 1) => (v == null || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`);
const num = (v: number | null | undefined, d = 4) => (v == null || !isFinite(v) ? '—' : v.toFixed(d));
const compact = (v: number | null | undefined) =>
  v == null ? '—' : v >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)}k` : `${Math.round(v)}`;
const sci = (v: number | null | undefined) => (v == null ? '—' : v.toExponential(1));

// ---------------------------------------------------------------------------
// Small building blocks
// ---------------------------------------------------------------------------

function Sparkline({ values, color }: { values: (number | null | undefined)[]; color: string }) {
  const pts = values.filter((v): v is number => v != null && isFinite(v));
  if (pts.length < 2) return <div className="h-6" />;
  const min = Math.min(...pts);
  const max = Math.max(...pts);
  const span = max - min || 1;
  const w = 100;
  const h = 24;
  const d = pts.map((v, i) => `${(i / (pts.length - 1)) * w},${h - 2 - ((v - min) / span) * (h - 4)}`).join(' ');
  const [lx, ly] = d.split(' ').pop()!.split(',').map(Number);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="w-full h-6" aria-hidden>
      <polyline points={d} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
      <circle cx={lx} cy={ly} r={2} fill={color} />
    </svg>
  );
}

function Delta({ value, goodWhenUp, format }: { value: number | null; goodWhenUp: boolean; format: (v: number) => string }) {
  if (value == null || !isFinite(value) || Math.abs(value) < 1e-9) return null;
  const good = goodWhenUp ? value > 0 : value < 0;
  return (
    <span className={`text-[10px] font-mono ${good ? 'text-emerald-400' : 'text-red-400'}`}>
      {value > 0 ? '▲' : '▼'} {format(Math.abs(value))}
    </span>
  );
}

interface KpiProps {
  label: string;
  value: string;
  sub?: React.ReactNode;
  delta?: React.ReactNode;
  spark?: (number | null | undefined)[];
  color?: string;
}

function Kpi({ label, value, sub, delta, spark, color = blue }: KpiProps) {
  return (
    <div className="p-3 bg-slate-800/70 rounded-xl border border-slate-700/70 min-w-0">
      <div className="flex items-center justify-between gap-1">
        <span className="text-[11px] text-gray-400 truncate">{label}</span>
        {delta}
      </div>
      <div className="text-xl font-bold text-white font-mono tabular-nums mt-0.5 truncate">{value}</div>
      {spark && <Sparkline values={spark} color={color} />}
      {sub && <div className="text-[10px] text-gray-500 truncate mt-0.5">{sub}</div>}
    </div>
  );
}

function Bar({ value, className }: { value: number; className: string }) {
  return (
    <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden shadow-inner">
      <div className={`h-2.5 rounded-full transition-all duration-500 ${className}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

function Stat({ icon: Icon, label, value }: { icon: typeof Clock; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 min-w-0">
      <Icon className="w-4 h-4 text-gray-500 shrink-0" />
      <div className="min-w-0">
        <div className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</div>
        <div className="text-sm text-gray-200 font-mono tabular-nums truncate">{value}</div>
      </div>
    </div>
  );
}

type Severity = 'good' | 'info' | 'warning' | 'critical';
interface Insight { severity: Severity; text: string }

const SEVERITY = {
  good: { icon: CheckCircle2, cls: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30', label: 'Good' },
  info: { icon: Info, cls: 'text-sky-300 bg-sky-500/10 border-sky-500/30', label: 'Info' },
  warning: { icon: AlertTriangle, cls: 'text-amber-300 bg-amber-500/10 border-amber-500/30', label: 'Warning' },
  critical: { icon: XCircle, cls: 'text-red-300 bg-red-500/10 border-red-500/30', label: 'Problem' },
} as const;

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

function bestIndex(ms: TrainingMetrics[], key: keyof TrainingMetrics, mode: 'max' | 'min'): number {
  let best = -1;
  ms.forEach((m, i) => {
    const v = m[key] as number | undefined;
    if (v == null || !isFinite(v)) return;
    if (best < 0 || (mode === 'max' ? v > (ms[best][key] as number) : v < (ms[best][key] as number))) best = i;
  });
  return best;
}

function diagnose(status: TrainingStatus, metricLabel: string): Insight[] {
  const ms = status.metrics ?? [];
  const out: Insight[] = [];
  if (!ms.length) return out;
  const last = ms[ms.length - 1];

  if (!isFinite(last.loss) || (ms.length > 2 && last.loss > ms[0].loss * 3)) {
    out.push({ severity: 'critical', text: 'Training loss is diverging. Lower the learning rate.' });
  }

  // Val loss rising while train loss keeps falling -> overfitting.
  let rising = 0;
  for (let i = ms.length - 1; i > 0; i--) {
    if (ms[i].val_loss > ms[i - 1].val_loss && ms[i].loss <= ms[i - 1].loss) rising++;
    else break;
  }
  if (rising >= 3) {
    out.push({ severity: 'warning', text: `Val loss has risen for ${rising} epochs while train loss falls: the model is overfitting. Early stopping, augmentation or more dropout would help.` });
  }

  const gap = last.accuracy - last.val_accuracy;
  if (ms.length >= 3 && gap > 0.1) {
    out.push({ severity: 'warning', text: `Train ${metricLabel.toLowerCase()} is ${(gap * 100).toFixed(1)} points above validation (generalization gap).` });
  }

  const monitor = status.early_stopping_monitor || 'val_loss';
  const bi = monitor.includes('acc')
    ? bestIndex(ms, 'val_accuracy', 'max')
    : bestIndex(ms, 'val_loss', 'min');
  const since = ms.length - 1 - bi;
  if (status.early_stopping && bi >= 0) {
    const patience = status.early_stopping_patience ?? 5;
    out.push({
      severity: since >= patience - 1 && since > 0 ? 'warning' : 'info',
      text: `Early stopping on ${monitor}: ${since}/${patience} epochs without improvement (best at epoch ${ms[bi].epoch}; best weights are restored).`,
    });
  }

  const bestAcc = bestIndex(ms, 'val_accuracy', 'max');
  if (bestAcc === ms.length - 1 && ms.length > 1) {
    out.push({ severity: 'good', text: `New best val ${metricLabel.toLowerCase()}: ${pct(last.val_accuracy)} at epoch ${last.epoch}.` });
  }

  if (ms.length >= 6) {
    const recent = ms.slice(-5).map((m) => m.val_accuracy);
    if (Math.max(...recent) - Math.min(...recent) < 0.005 && rising < 3) {
      out.push({ severity: 'info', text: `Val ${metricLabel.toLowerCase()} has plateaued over the last 5 epochs.` });
    }
  }

  const ft = ms.findIndex((m) => m.phase === 'fine-tune');
  if (ft > 0) out.push({ severity: 'info', text: `Backbone fine-tuning started at epoch ${ms[ft].epoch} with the learning rate ×0.1.` });
  else if (last.phase === 'head') out.push({ severity: 'info', text: 'Phase 1: training only the new head on a frozen pretrained backbone.' });

  if (last.val_ece != null && last.val_ece > 0.15) {
    out.push({ severity: 'info', text: `Confidence is poorly calibrated (ECE ${pct(last.val_ece)}): mean confidence ${pct(last.val_confidence)} vs val accuracy ${pct(last.val_accuracy)}.` });
  }
  if (last.update_ratio != null && last.update_ratio < 1e-5 && ms.length > 2) {
    out.push({ severity: 'info', text: 'Weights barely change between epochs; training has effectively stopped learning.' });
  }

  const order: Severity[] = ['critical', 'warning', 'good', 'info'];
  return out.sort((a, b) => order.indexOf(a.severity) - order.indexOf(b.severity));
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

interface Props {
  status: TrainingStatus;
  isRunning: boolean;
  isOD: boolean;
  metricLabel: string;
}

interface ChartDef {
  id: string;
  title: string;
  subtitle?: string;
  data: Record<string, number | null | undefined>[];
  series: Series[];
  formatY?: (v: number) => string;
  domainY?: [(v: number) => number, (v: number) => number];
  logY?: boolean;
  xKey?: string;
  markers?: ChartMarker[];
  show: boolean;
}

export function TrainingDashboard({ status, isRunning, isOD, metricLabel }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const ms = status.metrics ?? [];
  const last = ms.length ? ms[ms.length - 1] : null;
  const prev = ms.length > 1 ? ms[ms.length - 2] : null;
  const info = status.run_info;
  const live = isRunning && status.live && status.live.epoch === status.current_epoch + 1 ? status.live : null;

  const rows = useMemo(() => ms.map((m) => ({
    epoch: m.epoch,
    acc: m.accuracy * 100,
    val_acc: m.val_accuracy * 100,
    loss: m.loss,
    val_loss: m.val_loss,
    p: m.val_precision != null ? m.val_precision * 100 : null,
    r: m.val_recall != null ? m.val_recall * 100 : null,
    f1: m.val_f1 != null ? m.val_f1 * 100 : null,
    lr: m.learning_rate ?? null,
    gap: (m.accuracy - m.val_accuracy) * 100,
    conf: m.val_confidence != null ? m.val_confidence * 100 : null,
    sps: m.samples_per_sec ?? null,
    epoch_s: m.time_ms != null ? m.time_ms / 1000 : null,
    upd: m.update_ratio && m.update_ratio > 0 ? m.update_ratio : null,
    mem: m.memory_mb ?? null,
    gpu_mem: m.gpu_memory_mb ?? null,
  })), [ms]);

  const bestAccI = bestIndex(ms, 'val_accuracy', 'max');
  const bestLossI = bestIndex(ms, 'val_loss', 'min');
  const ftEpoch = ms.find((m) => m.phase === 'fine-tune')?.epoch;
  const phaseMarker: ChartMarker[] = ftEpoch && ftEpoch > ms[0].epoch ? [{ x: ftEpoch - 0.5, label: 'fine-tune' }] : [];
  // "best" marker, unlabeled when it would collide with the phase marker's label.
  const bestMarker = (i: number): ChartMarker[] => {
    if (i < 0 || ms.length < 2) return [];
    const near = ftEpoch != null && Math.abs(ms[i].epoch - (ftEpoch - 0.5)) < Math.max(1.5, ms.length / 12);
    return [{ x: ms[i].epoch, label: near ? ' ' : 'best', color: '#475569' }];
  };
  const insights = useMemo(() => diagnose(status, metricLabel), [status, metricLabel]);

  // Timing
  const elapsed = status.elapsed_seconds ?? 0;
  const remaining = status.remaining_seconds ?? 0;
  const epochTimes = ms.map((m) => (m.time_ms ?? 0) / 1000).filter((t) => t > 0);
  const avgEpoch = epochTimes.length ? epochTimes.reduce((a, b) => a + b, 0) / epochTimes.length : 0;
  const eta = isRunning && remaining > 0 ? new Date(Date.now() + remaining * 1000) : null;
  const batchPct = live && live.batches ? (live.batch / live.batches) * 100 : 0;
  const epochPct = (status.progress || 0) + (live ? batchPct / Math.max(1, status.total_epochs) : 0);

  const pctFmt = (v: number) => `${+v.toFixed(1)}%`;
  const pctDomain: [(v: number) => number, (v: number) => number] = [
    (v) => Math.max(0, Math.floor(v - 1)), (v) => Math.min(100, Math.ceil(v + 1)),
  ];
  const ptsFmt = (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(1)}`;
  const hasVal = ms.some((m) => m.val_f1 != null);
  const hasCalib = ms.some((m) => m.val_confidence != null);
  const batchHistory = status.batch_history ?? [];

  const charts: ChartDef[] = [
    {
      id: 'acc', title: `${metricLabel} (%)`, subtitle: bestAccI >= 0 ? `Best val ${pctFmt(rows[bestAccI].val_acc)} at epoch ${ms[bestAccI].epoch}` : undefined,
      data: rows, formatY: pctFmt, domainY: pctDomain, show: true,
      series: [{ key: 'acc', name: 'Train', color: blue }, { key: 'val_acc', name: 'Val', color: orange, dashed: true }],
      markers: [...phaseMarker, ...bestMarker(bestAccI)],
    },
    {
      id: 'loss', title: isOD ? 'Weighted grid loss' : 'Loss (cross-entropy)',
      subtitle: bestLossI >= 0 ? `Lowest val ${num(ms[bestLossI].val_loss)} at epoch ${ms[bestLossI].epoch}` : undefined,
      data: rows, show: true,
      series: [{ key: 'loss', name: 'Train', color: blue }, { key: 'val_loss', name: 'Val', color: orange, dashed: true }],
      markers: [...phaseMarker, ...bestMarker(bestLossI)],
    },
    {
      id: 'prf', title: isOD ? 'Val detection precision / recall / F1 (%)' : 'Val precision / recall / macro F1 (%)',
      subtitle: isOD ? 'Object-level, centroid matching' : 'Macro-averaged over classes',
      data: rows, formatY: pctFmt, domainY: pctDomain, show: hasVal, markers: phaseMarker,
      series: [{ key: 'p', name: 'Precision', color: blue }, { key: 'r', name: 'Recall', color: orange }, { key: 'f1', name: 'F1', color: aqua }],
    },
    {
      id: 'batch', title: 'Live batch loss', subtitle: 'Running mean within each epoch, sampled every 0.5 s',
      data: batchHistory, xKey: 'step', show: batchHistory.length >= 2,
      series: [{ key: 'loss', name: 'Train loss', color: blue }],
    },
    {
      id: 'gap', title: `Generalization gap (train − val ${metricLabel.toLowerCase()}, pts)`, subtitle: 'Growing gap = overfitting',
      data: rows, formatY: ptsFmt, show: true, markers: phaseMarker,
      series: [{ key: 'gap', name: 'Gap', color: blue }],
    },
    {
      id: 'lr', title: 'Learning rate', subtitle: 'Log scale', data: rows, logY: true, formatY: sci,
      show: rows.some((r) => r.lr != null), markers: phaseMarker,
      series: [{ key: 'lr', name: 'LR', color: blue }],
    },
    {
      id: 'calib', title: 'Calibration: val confidence vs accuracy (%)', subtitle: 'Confidence above accuracy = overconfident',
      data: rows, formatY: pctFmt, domainY: pctDomain, show: hasCalib,
      series: [{ key: 'val_acc', name: 'Val accuracy', color: blue }, { key: 'conf', name: 'Mean confidence', color: orange, dashed: true }],
    },
    {
      id: 'upd', title: 'Weight update ratio ‖Δw‖ / ‖w‖', subtitle: 'How much the weights moved each epoch (log scale)',
      data: rows, logY: true, formatY: sci, show: rows.some((r) => r.upd != null), markers: phaseMarker,
      series: [{ key: 'upd', name: 'Update ratio', color: blue }],
    },
    {
      id: 'sps', title: 'Throughput (samples/s)', subtitle: avgEpoch ? `Average ${avgEpoch.toFixed(1)} s per epoch` : undefined,
      data: rows, formatY: (v) => compact(v), show: rows.some((r) => r.sps != null),
      series: [{ key: 'sps', name: 'Samples/s', color: blue }],
    },
  ];
  const visible = charts.filter((c) => c.show);
  const modal = visible.find((c) => c.id === expanded);
  const valEval = status.live_eval;

  return (
    <div className="space-y-5">
      {/* Progress */}
      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-xs mb-1 text-gray-400">
            <span>Epoch <span className="text-white font-mono">{status.current_epoch}</span> / {status.total_epochs}</span>
            <span className="font-mono">{epochPct.toFixed(0)}%</span>
          </div>
          <Bar value={epochPct} className={status.status === 'completed' ? 'bg-emerald-500' : status.status === 'failed' ? 'bg-red-500' : 'bg-gradient-to-r from-purple-500 to-pink-500'} />
        </div>
        {live && (
          <div>
            <div className="flex justify-between text-xs mb-1 text-gray-400">
              <span>
                Epoch {live.epoch} · batch <span className="text-white font-mono">{live.batch}</span> / {live.batches}
                {live.phase && live.phase !== 'train' && <span className="ml-2 px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] uppercase">{live.phase}</span>}
              </span>
              <span className="font-mono">
                loss {num(live.loss)}{live.accuracy != null && ` · acc ${pct(live.accuracy)}`}
              </span>
            </div>
            <Bar value={batchPct} className="bg-sky-500" />
          </div>
        )}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 pt-1">
          <Stat icon={Clock} label="Elapsed" value={elapsed > 0 ? formatTime(elapsed) : '—'} />
          <Stat icon={Timer} label="Remaining" value={isRunning && ms.length ? formatTime(remaining) : '—'} />
          <Stat icon={CalendarClock} label="ETA" value={eta ? eta.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'} />
          <Stat icon={Gauge} label="Avg epoch" value={avgEpoch ? `${avgEpoch.toFixed(1)} s` : '—'} />
          <Stat icon={Zap} label="Throughput" value={last?.samples_per_sec ? `${compact(last.samples_per_sec)} samples/s` : '—'} />
          <Stat icon={Layers} label="Steps / epoch" value={info ? `${info.steps_per_epoch} × ${status.batch_size ?? '?'}` : '—'} />
        </div>
      </div>

      {/* KPI tiles */}
      {last && (
        <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
          <Kpi label={`Val ${metricLabel}`} value={pct(last.val_accuracy)} spark={ms.map((m) => m.val_accuracy)} color={orange}
            delta={<Delta value={prev ? last.val_accuracy - prev.val_accuracy : null} goodWhenUp format={(v) => `${(v * 100).toFixed(1)}`} />}
            sub={bestAccI >= 0 ? `best ${pct(ms[bestAccI].val_accuracy)} @ ep ${ms[bestAccI].epoch}` : undefined} />
          <Kpi label={`Train ${metricLabel}`} value={pct(last.accuracy)} spark={ms.map((m) => m.accuracy)}
            delta={<Delta value={prev ? last.accuracy - prev.accuracy : null} goodWhenUp format={(v) => `${(v * 100).toFixed(1)}`} />} />
          <Kpi label="Val loss" value={num(last.val_loss)} spark={ms.map((m) => m.val_loss)} color={orange}
            delta={<Delta value={prev ? last.val_loss - prev.val_loss : null} goodWhenUp={false} format={(v) => v.toFixed(3)} />}
            sub={bestLossI >= 0 ? `lowest ${num(ms[bestLossI].val_loss)} @ ep ${ms[bestLossI].epoch}` : undefined} />
          <Kpi label="Train loss" value={num(last.loss)} spark={ms.map((m) => m.loss)}
            delta={<Delta value={prev ? last.loss - prev.loss : null} goodWhenUp={false} format={(v) => v.toFixed(3)} />} />
          <Kpi label={isOD ? 'Val F1 (objects)' : 'Val macro F1'} value={pct(last.val_f1)} spark={ms.map((m) => m.val_f1)} color={aqua}
            delta={<Delta value={prev?.val_f1 != null && last.val_f1 != null ? last.val_f1 - prev.val_f1 : null} goodWhenUp format={(v) => `${(v * 100).toFixed(1)}`} />} />
          <Kpi label="Val precision" value={pct(last.val_precision)} spark={ms.map((m) => m.val_precision)} />
          <Kpi label="Val recall" value={pct(last.val_recall)} spark={ms.map((m) => m.val_recall)} />
          <Kpi label="Generalization gap" value={`${((last.accuracy - last.val_accuracy) * 100).toFixed(1)} pts`}
            spark={ms.map((m) => m.accuracy - m.val_accuracy)}
            sub={last.accuracy - last.val_accuracy > 0.1 ? 'overfitting risk' : 'train − val'} />
          <Kpi label="Learning rate" value={sci(last.learning_rate)} sub={last.phase && last.phase !== 'train' ? `phase: ${last.phase}` : 'Adam'} />
          {!isOD
            ? <Kpi label="Calibration error (ECE)" value={pct(last.val_ece)} spark={ms.map((m) => m.val_ece)}
                sub={last.val_confidence != null ? `mean confidence ${pct(last.val_confidence)}` : undefined} />
            : <Kpi label="Epoch time" value={last.time_ms ? `${(last.time_ms / 1000).toFixed(1)} s` : '—'} spark={ms.map((m) => m.time_ms)} />}
        </div>
      )}

      {/* Diagnostics */}
      {insights.length > 0 && (
        <ul className="grid gap-2 md:grid-cols-2" aria-label="Training diagnostics">
          {insights.map((ins, i) => {
            const sev = SEVERITY[ins.severity];
            const Icon = sev.icon;
            return (
              <li key={i} className={`flex items-start gap-2 px-3 py-2 rounded-lg border text-xs ${sev.cls}`}>
                <Icon className="w-4 h-4 shrink-0 mt-px" aria-label={sev.label} />
                <span className="text-gray-200">{ins.text}</span>
              </li>
            );
          })}
        </ul>
      )}

      {/* Charts */}
      {ms.length >= 1 || batchHistory.length >= 2 ? (
        <div className="grid gap-4 grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
          {visible.map((c) => (
            <ChartCard key={c.id} title={c.title} subtitle={c.subtitle} onExpand={() => setExpanded(c.id)}>
              {c.data.length >= 2
                ? <SeriesChart data={c.data} series={c.series} formatY={c.formatY} domainY={c.domainY} logY={c.logY} xKey={c.xKey}
                    markers={c.markers?.filter((m) => m.label)} />
                : <div className="h-40 flex items-center justify-center text-xs text-gray-500">Waiting for the second epoch…</div>}
            </ChartCard>
          ))}
        </div>
      ) : (
        isRunning && (
          <div className="p-6 text-center text-sm text-gray-400 bg-slate-900/40 border border-dashed border-slate-700 rounded-xl">
            Charts appear as soon as the first batches finish.
          </div>
        )
      )}

      {/* Latest validation breakdown + run facts */}
      {(valEval || info) && (
        <div className="grid gap-4 grid-cols-1 lg:grid-cols-3">
          {valEval?.confusion_matrix && valEval.labels && (
            <ChartCard title={`Val confusion matrix (epoch ${valEval.epoch})`} subtitle={`${valEval.num_samples ?? ''} validation samples · rows = true class`}>
              <ConfusionMatrix matrix={valEval.confusion_matrix} labels={valEval.labels} />
            </ChartCard>
          )}
          {valEval?.per_class && valEval.per_class.length > 0 && (
            <ChartCard title={`Per-class val F1 (epoch ${valEval.epoch})`} subtitle="Precision / recall in the tooltip">
              <ul className="space-y-1.5 max-h-64 overflow-y-auto custom-scrollbar pr-1">
                {[...valEval.per_class].sort((a, b) => a.f1 - b.f1).map((c, i) => (
                  <li key={c.label ?? i} className="text-xs" title={`precision ${pct(c.precision)} · recall ${pct(c.recall)}${c.support != null ? ` · ${c.support} samples` : ''}`}>
                    <div className="flex justify-between text-gray-300 mb-0.5">
                      <span className="truncate pr-2">{c.label ?? `class ${i}`}</span>
                      <span className="font-mono text-gray-400">{pct(c.f1)}</span>
                    </div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${c.f1 * 100}%`, background: blue }} />
                    </div>
                  </li>
                ))}
              </ul>
            </ChartCard>
          )}
          {info && (
            <ChartCard title="Run" subtitle={status.base_model}>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs">
                {([
                  ['Train / val samples', `${info.num_train} / ${info.num_val}`],
                  ['Classes', String(info.num_classes)],
                  ['Parameters', compact(info.params_total)],
                  ['Trainable', compact(info.params_trainable)],
                  ['Float32 size', `${((info.params_total * 4) / 1024).toFixed(0)} KB`],
                  ['Input', status.input_shape ? status.input_shape.join('×') : '—'],
                  ['Batch / LR', `${status.batch_size ?? '—'} / ${status.learning_rate ?? '—'}`],
                  ['Device', status.device_used ? status.device_used.toUpperCase() : '—'],
                  ['Early stopping', status.early_stopping ? `${status.early_stopping_monitor}, patience ${status.early_stopping_patience}` : 'off'],
                  ['Frozen warm-up', status.freeze_encoder_epochs ? `${status.freeze_encoder_epochs} epochs` : 'none'],
                ] as [string, string][]).map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-gray-500">{k}</dt>
                    <dd className="text-gray-200 font-mono truncate" title={v}>{v}</dd>
                  </div>
                ))}
              </dl>
              {info.train_class_counts && (
                <div className="mt-3">
                  <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1">Train class balance</p>
                  <ClassBalance counts={info.train_class_counts} />
                </div>
              )}
              {last?.memory_mb != null && (
                <p className="mt-3 text-[10px] text-gray-500 flex items-center gap-1">
                  <Cpu className="w-3 h-3" /> Backend memory {compact(last.memory_mb)} MB
                  {last.gpu_memory_mb != null && ` · GPU ${compact(last.gpu_memory_mb)} MB`}
                </p>
              )}
            </ChartCard>
          )}
        </div>
      )}

      {modal && (
        <ChartModal label={modal.title} data={modal.data} series={modal.series} formatY={modal.formatY}
          logY={modal.logY} xKey={modal.xKey} markers={modal.markers?.filter((m) => m.label)} onClose={() => setExpanded(null)} />
      )}
    </div>
  );
}

function ClassBalance({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map(([, c]) => c));
  return (
    <ul className="space-y-1 max-h-32 overflow-y-auto custom-scrollbar pr-1">
      {entries.map(([label, c]) => (
        <li key={label} className="flex items-center gap-2 text-[11px]">
          <span className="w-20 truncate text-gray-400" title={label}>{label}</span>
          <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full rounded-full" style={{ width: `${(c / max) * 100}%`, background: blue }} />
          </div>
          <span className="w-10 text-right font-mono text-gray-400">{c}</span>
        </li>
      ))}
    </ul>
  );
}
