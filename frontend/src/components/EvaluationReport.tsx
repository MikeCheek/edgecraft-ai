// Held-out evaluation report: headline metrics, per-class precision / recall /
// F1 and (for classifiers) a confusion matrix. Used after training and for
// original-vs-optimized comparisons.

export interface PerClassMetric {
  label?: string
  precision: number
  recall: number
  f1: number
  support?: number
  tp?: number
  fp?: number
  fn?: number
}

export interface EvaluationMetrics {
  accuracy?: number
  macro_f1?: number
  precision?: number
  recall?: number
  f1?: number
  labels?: string[]
  per_class?: PerClassMetric[]
  confusion_matrix?: number[][]
  split?: string
  num_samples?: number
}

const pct = (v: number | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

function f1Color (v: number): string {
  if (v >= 0.9) return 'text-emerald-400'
  if (v >= 0.7) return 'text-yellow-300'
  return 'text-red-400'
}

export function ConfusionMatrix ({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const n = labels.length
  if (!n || matrix.length !== n) return null
  const short = (s: string) => (s.length > 10 ? `${s.slice(0, 9)}…` : s)
  return (
    <div className="overflow-x-auto">
      <table className="text-[10px] border-separate border-spacing-0.5" aria-label="Confusion matrix">
        <thead>
          <tr>
            <th className="text-slate-500 font-normal text-right pr-1">true ↓ / pred →</th>
            {labels.map(l => (
              <th key={l} className="text-slate-400 font-medium px-1 max-w-[4rem] truncate" title={l}>{short(l)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1
            return (
              <tr key={labels[i]}>
                <th className="text-slate-400 font-medium text-right pr-1 max-w-[6rem] truncate" title={labels[i]}>{short(labels[i])}</th>
                {row.map((v, j) => {
                  const frac = v / total
                  const diag = i === j
                  const bg = diag
                    ? `rgba(16, 185, 129, ${0.12 + frac * 0.7})`
                    : v > 0 ? `rgba(239, 68, 68, ${0.12 + frac * 0.7})` : 'rgba(255,255,255,0.03)'
                  return (
                    <td key={j} className="w-9 h-7 text-center rounded font-mono text-white" style={{ background: bg }}
                      title={`${labels[i]} → ${labels[j]}: ${v} (${(frac * 100).toFixed(0)}% of row)`}>
                      {v}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export function EvaluationReport ({
  metrics, title, detection = false, compact = false,
}: { metrics: EvaluationMetrics | null | undefined; title?: string; detection?: boolean; compact?: boolean }) {
  if (!metrics) return null
  const labels = metrics.labels ?? metrics.per_class?.map(c => c.label ?? '') ?? []
  const tiles = detection
    ? [['F1', metrics.f1], ['Precision', metrics.precision], ['Recall', metrics.recall]]
    : [['Accuracy', metrics.accuracy], ['Macro F1', metrics.macro_f1]]

  return (
    <div className="space-y-3">
      {(title || metrics.split) && (
        <div className="flex items-baseline justify-between gap-2">
          {title && <h4 className="text-xs font-semibold uppercase tracking-widest text-slate-400">{title}</h4>}
          {metrics.split && (
            <span className="text-[11px] text-slate-500">
              {metrics.num_samples ?? '?'} samples · {metrics.split} split
            </span>
          )}
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        {tiles.map(([name, value]) => (
          <div key={name as string} className="px-3 py-2 rounded-lg bg-white/5 border border-white/10">
            <div className="text-[10px] uppercase tracking-wide text-slate-500">{name}</div>
            <div className="text-lg font-bold text-white font-mono">{pct(value as number | undefined)}</div>
          </div>
        ))}
      </div>

      {!compact && metrics.per_class && metrics.per_class.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 text-left">
                <th className="py-1 pr-2 font-medium">Class</th>
                <th className="py-1 px-2 font-medium text-right">Precision</th>
                <th className="py-1 px-2 font-medium text-right">Recall</th>
                <th className="py-1 px-2 font-medium text-right">F1</th>
                <th className="py-1 pl-2 font-medium text-right">{detection ? 'TP / FP / FN' : 'Support'}</th>
              </tr>
            </thead>
            <tbody>
              {metrics.per_class.map((c, i) => (
                <tr key={i} className="border-t border-white/5">
                  <td className="py-1 pr-2 text-slate-200 truncate max-w-[10rem]">{c.label ?? labels[i] ?? `class ${i}`}</td>
                  <td className="py-1 px-2 text-right font-mono text-slate-300">{pct(c.precision)}</td>
                  <td className="py-1 px-2 text-right font-mono text-slate-300">{pct(c.recall)}</td>
                  <td className={`py-1 px-2 text-right font-mono ${f1Color(c.f1)}`}>{pct(c.f1)}</td>
                  <td className="py-1 pl-2 text-right font-mono text-slate-500">
                    {detection ? `${c.tp ?? 0} / ${c.fp ?? 0} / ${c.fn ?? 0}` : c.support ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!compact && !detection && metrics.confusion_matrix && labels.length > 0 && (
        <ConfusionMatrix matrix={metrics.confusion_matrix} labels={labels} />
      )}
    </div>
  )
}
