// Experiments: every training run in one table, filterable, with a
// side-by-side comparison of up to four runs (config diff, held-out metrics,
// validation curves).

import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { FlaskConical, RefreshCw, Cpu, Pencil } from 'lucide-react'
import { useAPI } from '../hooks/useAPI'
import { useToast } from '../context/ToastContext'
import { PastTrainingSession } from '../types'
import { taskLabel, TASK_OPTIONS } from '../constants/catalog'
import { EvaluationReport } from './EvaluationReport'

const MAX_COMPARE = 4
const SERIES_COLORS = ['#a78bfa', '#22d3ee', '#f472b6', '#facc15']

const CONFIG_FIELDS: { key: keyof PastTrainingSession | string; label: string }[] = [
  { key: 'base_model', label: 'Model' },
  { key: 'input_shape', label: 'Input' },
  { key: 'epochs', label: 'Epochs' },
  { key: 'batch_size', label: 'Batch' },
  { key: 'learning_rate', label: 'LR' },
  { key: 'dropout_rate', label: 'Dropout' },
  { key: 'l2_reg', label: 'L2' },
  { key: 'freeze_encoder_epochs', label: 'Head warm-up' },
  { key: 'trainable_layers', label: 'Fine-tuned layers' },
  { key: 'augmentation', label: 'Augmentation' },
  { key: 'class_weighting', label: 'Class weights' },
  { key: 'seed', label: 'Seed' },
  { key: 'device_used', label: 'Device' },
]

type Session = PastTrainingSession & Record<string, unknown>

function fmt (v: unknown): string {
  if (v == null || v === '') return '—'
  if (Array.isArray(v)) return v.join('×')
  if (typeof v === 'object') {
    const on = Object.entries(v as Record<string, unknown>).filter(([, x]) => x).map(([k, x]) => (x === true ? k : `${k}=${x}`))
    return on.length ? on.join(', ') : 'none'
  }
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  return String(v)
}

const headline = (s: Session) => {
  const ev = s.evaluation
  if (!ev) return null
  return s.task === 'OBJECT_DETECTION' ? ev.f1 : ev.accuracy
}

export function ExperimentsPage () {
  const { apiClient, request } = useAPI()
  const { toast } = useToast()
  const navigate = useNavigate()
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(false)
  const [taskFilter, setTaskFilter] = useState('')
  const [datasetFilter, setDatasetFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('completed')
  const [selected, setSelected] = useState<string[]>([])
  const [detail, setDetail] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    const raw = await request(() => apiClient.listAllSessions(false))
    if (raw?.sessions) setSessions(raw.sessions)
    setLoading(false)
  }
  useEffect(() => { load() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const datasetIds = useMemo(() => Array.from(new Set(sessions.map(s => s.dataset_id))), [sessions])
  const filtered = sessions.filter(s =>
    (!taskFilter || s.task === taskFilter) &&
    (!datasetFilter || s.dataset_id === datasetFilter) &&
    (!statusFilter || s.status === statusFilter))
  const compared = selected.map(id => sessions.find(s => s.id === id)).filter(Boolean) as Session[]

  const toggle = (id: string) => setSelected(cur =>
    cur.includes(id) ? cur.filter(x => x !== id) : cur.length >= MAX_COMPARE ? cur : [...cur, id])

  const rename = async (s: Session) => {
    const name = window.prompt('Run name', s.name || '')
    if (name == null) return
    await request(() => apiClient.renameTraining(s.id, name))
    toast('success', 'Renamed')
    load()
  }

  const curveData = useMemo(() => {
    const maxEpochs = Math.max(0, ...compared.map(s => s.metrics?.length ?? 0))
    return Array.from({ length: maxEpochs }, (_, i) => {
      const row: Record<string, number> = { epoch: i + 1 }
      compared.forEach((s, k) => {
        const m = s.metrics?.[i]
        if (m) row[`run${k}`] = +(m.val_accuracy * 100).toFixed(2)
      })
      return row
    })
  }, [compared])

  const runName = (s: Session) => s.name || `${s.base_model} · ${new Date(s.created_at * 1000).toLocaleDateString()}`
  const sel = 'px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm'

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2 items-center">
        <select value={taskFilter} onChange={e => setTaskFilter(e.target.value)} className={sel} aria-label="Filter by task">
          <option value="">All tasks</option>
          {TASK_OPTIONS.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
        </select>
        <select value={datasetFilter} onChange={e => setDatasetFilter(e.target.value)} className={sel} aria-label="Filter by dataset">
          <option value="">All datasets</option>
          {datasetIds.map(id => <option key={id} value={id}>{id.slice(0, 8)}</option>)}
        </select>
        <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)} className={sel} aria-label="Filter by status">
          <option value="">Any status</option>
          {['completed', 'failed', 'cancelled', 'running', 'queued'].map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <button onClick={load} className="ml-auto px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-sm text-white flex items-center gap-2">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </div>

      <div className="rounded-2xl border border-slate-700 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-slate-900 text-xs text-slate-400 text-left">
            <tr>
              <th className="p-3 w-8"><span className="sr-only">Compare</span></th>
              <th className="p-3">Run</th>
              <th className="p-3">Task</th>
              <th className="p-3 text-right">Val (last)</th>
              <th className="p-3 text-right">Held-out</th>
              <th className="p-3 text-right">Epochs</th>
              <th className="p-3">Status</th>
              <th className="p-3"><span className="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr><td colSpan={8} className="p-6 text-center text-slate-500">No runs match these filters.</td></tr>
            )}
            {filtered.map(s => {
              const last = s.metrics?.[s.metrics.length - 1]
              const h = headline(s)
              const checked = selected.includes(s.id)
              return (
                <tr key={s.id} className={`border-t border-slate-800 ${checked ? 'bg-violet-500/5' : 'hover:bg-slate-800/40'}`}>
                  <td className="p-3">
                    <input type="checkbox" checked={checked} onChange={() => toggle(s.id)} aria-label={`Compare ${runName(s)}`}
                      disabled={!checked && selected.length >= MAX_COMPARE} className="accent-violet-500" />
                  </td>
                  <td className="p-3">
                    <button onClick={() => setDetail(detail === s.id ? null : s.id)} className="text-left">
                      <span className="text-white font-medium block">{runName(s)}</span>
                      <span className="text-[11px] text-slate-500">{s.base_model} · {fmt(s.input_shape)}</span>
                    </button>
                  </td>
                  <td className="p-3 text-slate-300 text-xs">{taskLabel(s.task)}</td>
                  <td className="p-3 text-right font-mono text-slate-300">{last ? `${(last.val_accuracy * 100).toFixed(1)}%` : '—'}</td>
                  <td className="p-3 text-right font-mono text-emerald-300">{h != null ? `${(h * 100).toFixed(1)}%` : '—'}</td>
                  <td className="p-3 text-right font-mono text-slate-400">{s.current_epoch}/{s.total_epochs}</td>
                  <td className="p-3 text-xs text-slate-400">{s.status}</td>
                  <td className="p-3 whitespace-nowrap">
                    <button onClick={() => rename(s)} className="p-1.5 rounded hover:bg-slate-700 text-slate-400" aria-label="Rename run"><Pencil className="w-3.5 h-3.5" /></button>
                    {s.status === 'completed' && (
                      <button onClick={() => navigate(`/optimize?model=${s.id}`)} className="p-1.5 rounded hover:bg-slate-700 text-cyan-400" aria-label="Optimize this model"><Cpu className="w-3.5 h-3.5" /></button>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {detail && (() => {
        const s = sessions.find(x => x.id === detail)
        if (!s) return null
        return (
          <div className="rounded-2xl border border-slate-700 bg-slate-900/60 p-5 space-y-3">
            <h3 className="text-sm font-semibold text-white">{runName(s)}</h3>
            {s.error && <p className="text-xs text-red-300">{s.error}</p>}
            {s.evaluation ? <EvaluationReport metrics={s.evaluation} detection={s.task === 'OBJECT_DETECTION'} title="Held-out evaluation" />
              : <p className="text-xs text-slate-500">No held-out evaluation recorded for this run.</p>}
          </div>
        )
      })()}

      {compared.length >= 2 ? (
        <div className="rounded-2xl border border-slate-700 bg-slate-900/60 p-5 space-y-5">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2"><FlaskConical className="w-4 h-4 text-violet-400" /> Comparing {compared.length} runs</h3>
          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr>
                  <th className="text-left text-slate-500 font-medium p-2">Setting</th>
                  {compared.map((s, k) => (
                    <th key={s.id} className="text-left p-2 font-semibold" style={{ color: SERIES_COLORS[k] }}>{runName(s)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-slate-800">
                  <td className="p-2 text-slate-400">Held-out</td>
                  {compared.map(s => <td key={s.id} className="p-2 font-mono text-emerald-300">{headline(s) != null ? `${((headline(s) ?? 0) * 100).toFixed(1)}%` : '—'}</td>)}
                </tr>
                {CONFIG_FIELDS.map(f => {
                  const values = compared.map(s => fmt(s[f.key as string]))
                  const differs = new Set(values).size > 1
                  return (
                    <tr key={String(f.key)} className="border-t border-slate-800">
                      <td className="p-2 text-slate-400">{f.label}</td>
                      {values.map((v, i) => (
                        <td key={i} className={`p-2 font-mono ${differs ? 'text-yellow-200' : 'text-slate-500'}`}>{v}</td>
                      ))}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curveData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
                <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} unit="%" domain={[0, 100]} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {compared.map((s, k) => (
                  <Line key={s.id} type="monotone" dataKey={`run${k}`} name={`${runName(s)} (val)`}
                    stroke={SERIES_COLORS[k]} dot={false} strokeWidth={2} connectNulls />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <p className="text-xs text-slate-500">Tick two to four runs to compare their settings and validation curves.</p>
      )}
    </div>
  )
}
