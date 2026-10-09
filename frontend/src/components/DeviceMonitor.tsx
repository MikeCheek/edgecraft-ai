// Talks to a flashed board over Web Serial (Chrome / Edge): shows its output,
// extracts the real on-device latency, arena usage and predictions the
// generated sketch prints, and can send a test input to Serial-harness builds.

import { useEffect, useRef, useState } from 'react'
import { Usb, Unplug, Send, Gauge } from 'lucide-react'
import { blobToPcm16 } from '../utils/audio'

// Minimal Web Serial typings (not in TypeScript's DOM lib yet).
interface SerialPortLike {
  open: (opts: { baudRate: number }) => Promise<void>
  close: () => Promise<void>
  readable: ReadableStream<Uint8Array> | null
  writable: WritableStream<Uint8Array> | null
}
type SerialNavigator = Navigator & { serial?: { requestPort: () => Promise<SerialPortLike> } }

export interface DeviceStats {
  latenciesUs: number[]
  arenaUsed?: number
  arenaTotal?: number
  lastPrediction?: string
}

/** Pure parser for the lines the generated sketch prints (unit-tested). */
export function parseDeviceLine (line: string, stats: DeviceStats): DeviceStats {
  const next = { ...stats }
  const inf = line.match(/inference:\s*(\d+)\s*us/i)
  if (inf) next.latenciesUs = [...stats.latenciesUs, Number(inf[1])].slice(-200)
  const arena = line.match(/Tensor arena used:\s*(\d+)\s*of\s*(\d+)/i)
  if (arena) { next.arenaUsed = Number(arena[1]); next.arenaTotal = Number(arena[2]) }
  const best = line.match(/^Best:\s*(.+)$/)
  if (best) next.lastPrediction = best[1].trim()
  const objects = line.match(/^(\d+) object\(s\)/)
  if (objects) next.lastPrediction = `${objects[1]} object(s)`
  return next
}

const median = (xs: number[]) => {
  if (!xs.length) return null
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.floor(s.length / 2)]
}

interface Props {
  /** 'audio' streams a WAV clip; 'image' sends one raw frame ('I' + pixels). */
  inputKind: 'audio' | 'image' | null
  inputShape?: number[]
  audioSamples?: number
  /** True when the sketch reads its input from Serial (no camera / mic). */
  serialInput: boolean
}

export function DeviceMonitor ({ inputKind, inputShape, audioSamples, serialInput }: Props) {
  const supported = typeof navigator !== 'undefined' && 'serial' in navigator
  const [port, setPort] = useState<SerialPortLike | null>(null)
  const [lines, setLines] = useState<string[]>([])
  const [stats, setStats] = useState<DeviceStats>({ latenciesUs: [] })
  const [error, setError] = useState<string | null>(null)
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null)
  const logRef = useRef<HTMLPreElement>(null)

  useEffect(() => { logRef.current?.scrollTo(0, logRef.current.scrollHeight) }, [lines])
  useEffect(() => () => { readerRef.current?.cancel().catch(() => {}) }, [])

  const connect = async () => {
    setError(null)
    try {
      const p = await (navigator as SerialNavigator).serial!.requestPort()
      await p.open({ baudRate: 115200 })
      setPort(p)
      setStats({ latenciesUs: [] })
      const reader = p.readable!.getReader()
      readerRef.current = reader
      const decoder = new TextDecoder()
      let buffer = ''
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split(/\r?\n/)
        buffer = parts.pop() ?? ''
        if (parts.length) {
          setLines(prev => [...prev, ...parts].slice(-500))
          setStats(prev => parts.reduce((acc, l) => parseDeviceLine(l, acc), prev))
        }
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      readerRef.current?.releaseLock?.()
      readerRef.current = null
    }
  }

  const disconnect = async () => {
    await readerRef.current?.cancel().catch(() => {})
    await port?.close().catch(() => {})
    setPort(null)
  }

  const write = async (bytes: Uint8Array) => {
    if (!port?.writable) return
    const writer = port.writable.getWriter()
    try { await writer.write(bytes) } finally { writer.releaseLock() }
  }

  const sendFile = async (file: File) => {
    setError(null)
    try {
      if (inputKind === 'audio' && audioSamples) {
        const pcm = await blobToPcm16(file, audioSamples)
        const bytes = new Uint8Array(1 + pcm.byteLength)
        bytes[0] = 'A'.charCodeAt(0)
        bytes.set(new Uint8Array(pcm.buffer), 1)
        await write(bytes)
      } else if (inputKind === 'image' && inputShape) {
        const [h, w, c] = inputShape
        const bitmap = await createImageBitmap(file)
        const canvas = document.createElement('canvas')
        canvas.width = w
        canvas.height = h
        const ctx = canvas.getContext('2d')!
        ctx.drawImage(bitmap, 0, 0, w, h)
        const px = ctx.getImageData(0, 0, w, h).data
        const frame = new Uint8Array(1 + w * h * c)
        frame[0] = 'I'.charCodeAt(0)
        for (let i = 0; i < w * h; i++) {
          const [r, g, b] = [px[i * 4], px[i * 4 + 1], px[i * 4 + 2]]
          if (c === 1) frame[1 + i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b)
          else frame.set([r, g, b], 1 + i * 3)
        }
        await write(frame)
      }
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const med = median(stats.latenciesUs)
  return (
    <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700 space-y-3">
      <div className="flex items-center gap-2">
        <Gauge className="w-4 h-4 text-emerald-400" />
        <h4 className="text-sm font-semibold text-white">Device monitor</h4>
        {port ? (
          <button onClick={disconnect} className="ml-auto text-xs px-2.5 py-1 rounded-lg bg-slate-700 hover:bg-slate-600 text-gray-200 flex items-center gap-1.5">
            <Unplug className="w-3.5 h-3.5" /> Disconnect
          </button>
        ) : (
          <button onClick={connect} disabled={!supported} className="ml-auto text-xs px-2.5 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white flex items-center gap-1.5">
            <Usb className="w-3.5 h-3.5" /> Connect board
          </button>
        )}
      </div>
      {!supported && <p className="text-xs text-gray-500">Needs a browser with Web Serial (Chrome or Edge on desktop).</p>}
      {supported && !port && (
        <p className="text-xs text-gray-500">Flash the exported project, then connect at 115200 baud to read the board's real latency and memory use.</p>
      )}
      {port && (
        <>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="rounded-lg bg-white/5 p-2">
              <div className="text-gray-500">Latency (median)</div>
              <div className="font-mono text-white">{med != null ? `${(med / 1000).toFixed(1)} ms` : '—'}</div>
              <div className="text-[10px] text-gray-500">{stats.latenciesUs.length} runs</div>
            </div>
            <div className="rounded-lg bg-white/5 p-2">
              <div className="text-gray-500">Arena used</div>
              <div className="font-mono text-white">{stats.arenaUsed != null ? `${(stats.arenaUsed / 1024).toFixed(1)} KB` : '—'}</div>
              {stats.arenaTotal != null && <div className="text-[10px] text-gray-500">of {(stats.arenaTotal / 1024).toFixed(0)} KB reserved</div>}
            </div>
            <div className="rounded-lg bg-white/5 p-2">
              <div className="text-gray-500">Last result</div>
              <div className="text-white truncate" title={stats.lastPrediction}>{stats.lastPrediction ?? '—'}</div>
            </div>
          </div>
          {serialInput && inputKind && (
            <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer">
              <Send className="w-3.5 h-3.5 text-cyan-400" />
              Send a test {inputKind === 'audio' ? 'audio clip' : 'image'}:
              <input type="file" accept={inputKind === 'audio' ? 'audio/*' : 'image/*'}
                onChange={e => e.target.files?.[0] && sendFile(e.target.files[0])} className="text-xs" />
            </label>
          )}
          <pre ref={logRef} className="max-h-48 overflow-auto text-[11px] font-mono text-gray-300 bg-black/40 rounded p-2 whitespace-pre-wrap">
            {lines.join('\n') || 'Waiting for output…'}
          </pre>
        </>
      )}
      {error && <p className="text-xs text-red-300">{error}</p>}
    </div>
  )
}
