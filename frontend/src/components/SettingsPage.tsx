// Settings: backend connection (URL + optional API key), AI assistant
// provider/model, studio defaults, and what the connected backend supports.

import { useEffect, useState } from 'react'
import { Server, KeyRound, Lightbulb, Cloud, Cpu, CheckCircle2, XCircle, RefreshCw, Info } from 'lucide-react'
import { API_BASE, getApiKey, getSavedApiBase, saveConnection } from '../config'
import { useAppContext } from '../context/AppContext'
import { useBackendInfo } from '../hooks/useBackendInfo'
import { useAPI } from '../hooks/useAPI'
import { useToast } from '../context/ToastContext'
import { BOARD_OPTIONS, OPENROUTER_MODELS, TASK_OPTIONS } from '../constants/catalog'
import { TargetBoard, TinyMLTask } from '../types'

const card = 'rounded-2xl border border-slate-700 bg-slate-900/60 p-5 space-y-4'
const inputCls = 'w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:border-purple-500 focus:outline-none'

export function SettingsPage () {
  const { state, dispatch } = useAppContext()
  const { apiClient } = useAPI()
  const { toast } = useToast()
  const info = useBackendInfo()

  const [apiBase, setApiBase] = useState(getSavedApiBase())
  const [apiKey, setApiKey] = useState(getApiKey())
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; msg: string } | null>(null)
  const [devices, setDevices] = useState<{ gpu_available: boolean; gpus: { name: string }[] } | null>(null)
  const [customModel, setCustomModel] = useState(
    OPENROUTER_MODELS.some(m => m.id === state.llmModel) ? '' : state.llmModel
  )

  useEffect(() => {
    apiClient.getAvailableDevices().then(r => { if (r.status === 'success') setDevices(r.devices) }).catch(() => {})
  }, [apiClient])

  const testConnection = async () => {
    setTesting(true)
    setTestResult(null)
    const base = (apiBase.trim() || API_BASE).replace(/\/+$/, '')
    try {
      const health = await fetch(`${base}/health`).then(r => r.json())
      const headers: Record<string, string> = apiKey.trim() ? { 'X-API-Key': apiKey.trim() } : {}
      const probe = await fetch(`${base}/datasets/list_datasets`, { headers })
      if (probe.status === 401) {
        setTestResult({ ok: false, msg: 'Backend reachable, but it requires a valid API key.' })
      } else {
        setTestResult({ ok: true, msg: `Connected to EdgeCraft backend ${health.version ?? ''}${health.auth_required ? ' (API key accepted)' : ''}.` })
      }
    } catch {
      setTestResult({ ok: false, msg: `Could not reach ${base}.` })
    } finally {
      setTesting(false)
    }
  }

  const saveAndReload = () => {
    saveConnection(apiBase, apiKey)
    toast('success', 'Connection saved - reloading')
    setTimeout(() => window.location.reload(), 400)
  }

  const llm = state.llmConfig
  return (
    <div className="space-y-6 max-w-3xl">
      {/* Connection */}
      <section className={card} aria-labelledby="conn-title">
        <h3 id="conn-title" className="text-sm font-semibold text-white flex items-center gap-2"><Server className="w-4 h-4 text-cyan-400" /> Backend connection</h3>
        <label className="block text-xs text-gray-400">
          API base URL
          <input value={apiBase} onChange={e => setApiBase(e.target.value)} placeholder={API_BASE} className={`${inputCls} mt-1 font-mono`} />
          <span className="text-[11px] text-gray-500">Currently using <code className="text-cyan-300">{API_BASE}</code>. Leave empty for the default.</span>
        </label>
        <label className="block text-xs text-gray-400">
          <span className="flex items-center gap-1"><KeyRound className="w-3.5 h-3.5" /> API key</span>
          <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="only if the backend sets EDGECRAFT_API_TOKEN" className={`${inputCls} mt-1 font-mono`} autoComplete="off" />
        </label>
        <div className="flex gap-2">
          <button onClick={testConnection} disabled={testing} className="px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-sm text-white flex items-center gap-2 disabled:opacity-50">
            {testing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Info className="w-4 h-4" />} Test
          </button>
          <button onClick={saveAndReload} className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-sm text-white font-semibold">Save &amp; reload</button>
        </div>
        {testResult && (
          <p className={`text-xs flex items-center gap-1.5 ${testResult.ok ? 'text-emerald-300' : 'text-red-300'}`}>
            {testResult.ok ? <CheckCircle2 className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />} {testResult.msg}
          </p>
        )}
      </section>

      {/* Defaults */}
      <section className={card} aria-labelledby="defaults-title">
        <h3 id="defaults-title" className="text-sm font-semibold text-white flex items-center gap-2"><Cpu className="w-4 h-4 text-purple-400" /> Studio defaults</h3>
        <div className="grid sm:grid-cols-2 gap-3">
          <label className="text-xs text-gray-400">Task
            <select value={state.currentTask ?? 'IMAGE_CLASSIFICATION'} onChange={e => dispatch({ type: 'SET_TASK', payload: e.target.value as TinyMLTask })} className={`${inputCls} mt-1`}>
              {TASK_OPTIONS.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
            </select>
          </label>
          <label className="text-xs text-gray-400">Target board
            <select value={state.currentBoard ?? 'ESP32_S3_N16R8'} onChange={e => dispatch({ type: 'SET_BOARD', payload: e.target.value as TargetBoard })} className={`${inputCls} mt-1`}>
              {BOARD_OPTIONS.map(b => <option key={b.id} value={b.id}>{b.label} - {b.specs}</option>)}
            </select>
          </label>
        </div>
      </section>

      {/* AI assistant */}
      <section className={card} aria-labelledby="ai-title">
        <h3 id="ai-title" className="text-sm font-semibold text-white flex items-center gap-2"><Lightbulb className="w-4 h-4 text-yellow-400" /> AI assistant</h3>
        {!llm ? (
          <p className="text-xs text-gray-500">Loading provider configuration…</p>
        ) : !llm.openrouter_available && !llm.ollama_available ? (
          <p className="text-xs text-amber-300">
            No LLM provider configured. Set <code>OPENROUTER_API_KEY</code> or <code>OLLAMA_ENABLED=true</code> in the backend
            <code> .env</code>. Suggestions fall back to built-in rules meanwhile.
          </p>
        ) : (
          <>
            <div className="flex gap-2">
              {([['openrouter', 'OpenRouter', Cloud, llm.openrouter_available], ['ollama', 'Ollama (local)', Server, llm.ollama_available]] as const).map(([id, label, Icon, available]) => (
                <button key={id} disabled={!available} onClick={() => dispatch({ type: 'SET_LLM_PROVIDER', payload: id })}
                  className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg border text-sm font-semibold disabled:opacity-40 ${state.llmProvider === id ? 'border-yellow-500/50 bg-yellow-500/10 text-yellow-200' : 'border-slate-700 text-slate-300 hover:border-slate-500'}`}>
                  <Icon className="w-4 h-4" /> {label}{!available && ' (not configured)'}
                </button>
              ))}
            </div>
            {state.llmProvider === 'ollama' ? (
              <p className="text-xs text-gray-400">Model: <span className="font-mono text-yellow-200">{llm.ollama_model}</span> (OLLAMA_MODEL in the backend .env)</p>
            ) : (
              <div className="space-y-2">
                <div className="grid sm:grid-cols-2 gap-2">
                  {OPENROUTER_MODELS.map(m => (
                    <button key={m.id} onClick={() => { setCustomModel(''); dispatch({ type: 'SET_LLM_MODEL', payload: m.id }) }}
                      className={`text-left p-2.5 rounded-lg border text-xs ${state.llmModel === m.id ? 'border-yellow-500/50 bg-yellow-500/10' : 'border-slate-700 hover:border-slate-500'}`}>
                      <span className="block text-white font-semibold">{m.label}</span>
                      <span className="block text-slate-500 font-mono">{m.id}</span>
                    </button>
                  ))}
                </div>
                <label className="block text-xs text-gray-400">Or any OpenRouter model id
                  <input value={customModel} placeholder="vendor/model-name" className={`${inputCls} mt-1 font-mono`}
                    onChange={e => setCustomModel(e.target.value)}
                    onBlur={() => customModel.trim() && dispatch({ type: 'SET_LLM_MODEL', payload: customModel.trim() })} />
                </label>
              </div>
            )}
          </>
        )}
      </section>

      {/* Backend capabilities */}
      <section className={card} aria-labelledby="caps-title">
        <h3 id="caps-title" className="text-sm font-semibold text-white flex items-center gap-2"><Info className="w-4 h-4 text-emerald-400" /> Backend capabilities</h3>
        {!info ? <p className="text-xs text-gray-500">Backend unreachable.</p> : (
          <dl className="grid sm:grid-cols-2 gap-x-6 gap-y-2 text-xs">
            <dt className="text-gray-500">Version</dt><dd className="text-white font-mono">{info.version}</dd>
            <dt className="text-gray-500">GPU</dt><dd className="text-white">{devices ? (devices.gpu_available ? devices.gpus.map(g => g.name).join(', ') : 'none - training on CPU') : '…'}</dd>
            <dt className="text-gray-500">TensorFlow Lite Micro check</dt>
            <dd className={info.tflite_micro_available ? 'text-emerald-300' : 'text-amber-300'}>
              {info.tflite_micro_available ? 'available (exact arena measurement)' : 'not installed - pip install -r requirements-optional.txt'}
            </dd>
            <dt className="text-gray-500">Upload limits</dt><dd className="text-white font-mono">{info.limits.max_upload_mb} MB zip · {info.limits.max_sample_mb} MB sample</dd>
          </dl>
        )}
      </section>
    </div>
  )
}
