import { useEffect, useState } from 'react'
import { apiFetch } from '../config'

export interface BackendInfo {
  version: string
  tasks: string[]
  boards: string[]
  models: Record<string, string[]>
  model_info: Record<string, { tier: string; approx_params_m: number; note: string }>
  default_input_shapes: Record<string, number[]>
  audio_frontend: Record<string, Record<string, number>>
  tflite_micro_available: boolean
  limits: { max_upload_mb: number; max_chunk_mb: number; max_sample_mb: number }
}

let cached: Promise<BackendInfo | null> | null = null

/** GET /api/info, fetched once per page load and shared by every caller. */
export function loadBackendInfo (): Promise<BackendInfo | null> {
  if (!cached) {
    cached = apiFetch('/info')
      .then(r => (r.ok ? r.json() : null))
      .catch(() => null)
      .then(info => {
        if (!info) cached = null // retry on the next call
        return info
      })
  }
  return cached
}

export function useBackendInfo (): BackendInfo | null {
  const [info, setInfo] = useState<BackendInfo | null>(null)
  useEffect(() => {
    let alive = true
    loadBackendInfo().then(i => { if (alive) setInfo(i) })
    return () => { alive = false }
  }, [])
  return info
}
