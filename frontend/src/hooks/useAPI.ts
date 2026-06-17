// ═══════════════════════════════════════════════════════════════════════════════
// FILE: useApi.ts
// ═══════════════════════════════════════════════════════════════════════════════

import { useCallback, useState } from 'react'
import axios, { AxiosInstance } from 'axios'
import { ApiResponse } from '../types'

const API_BASE = 'http://localhost:8000/api'

class APIClient {
  private client: AxiosInstance
  private uploadClient: AxiosInstance

  constructor() {
    this.client = axios.create({ baseURL: API_BASE, timeout: 30_000 })
    this.uploadClient = axios.create({ baseURL: API_BASE, timeout: 600_000 })
  }

  async health() {
    return this.client.get<ApiResponse<any>>('/health')
  }

  // ── Datasets ──────────────────────────────────────────────────────────────

  async createDataset(name: string, task: string) {
    return this.client.post<ApiResponse<any>>('/datasets/create', { name, task })
  }

  async listDatasets(task?: string) {
    return this.client.get<ApiResponse<any>>('/datasets/list_datasets', { params: { task } })
  }

  async renameDataset(datasetId: string, newName: string) {
    return this.client.put<ApiResponse<any>>(`/datasets/rename/${datasetId}`, { new_name: newName })
  }

  async deleteDataset(datasetId: string) {
    return this.client.delete<ApiResponse<any>>(`/datasets/dataset/${datasetId}`)
  }

  async clearDataset(datasetId: string) {
    return this.client.delete<ApiResponse<any>>(`/datasets/clear_dataset/${datasetId}`)
  }

  // ── Single-file upload ────────────────────────────────────────────────────

  async uploadSample(datasetId: string, label: string, task: string, file: File) {
    const fd = new FormData()
    fd.append('dataset_id', datasetId)
    fd.append('label', label)
    fd.append('task', task)
    fd.append('file', file)
    return this.uploadClient.post<ApiResponse<any>>('/datasets/upload', fd)
  }

  // ── Chunked ZIP upload ────────────────────────────────────────────────────

  async initZipUpload(params: {
    dataset_id: string
    task: string
    filename: string
    total_chunks: number
    file_size: number
    chunk_size: number
  }): Promise<{ status: string; upload_id: string; message?: string }> {
    const res = await this.client.post('/datasets/upload_zip/init', params)
    return res.data
  }

  async putZipChunk(
    uploadId: string,
    chunkIndex: number,
    blob: Blob,
    signal?: AbortSignal,
  ): Promise<void> {
    const res = await fetch(
      `${API_BASE}/datasets/upload_zip/chunk/${uploadId}/${chunkIndex}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: blob,
        signal,
      },
    )
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      throw new Error(`HTTP ${res.status}: ${text}`)
    }
  }

  async getZipUploadStatus(
    uploadId: string,
  ): Promise<{ status: string; upload_id: string; received_chunks: number[] }> {
    const res = await this.client.get(`/datasets/upload_zip/status/${uploadId}`)
    return res.data
  }

  async finalizeZipUpload(params: {
    upload_id: string
    dataset_id: string
    task: string
    total_chunks: number
  }): Promise<{ status: string; sample_ids?: string[]; count?: number; message?: string }> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 30 * 60 * 1000)
    try {
      const res = await fetch(`${API_BASE}/datasets/upload_zip/finalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
        signal: controller.signal,
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return await res.json()
    } finally {
      clearTimeout(timer)
    }
  }

  async abortZipUpload(uploadId: string): Promise<void> {
    await this.client.delete(`/datasets/upload_zip/${uploadId}`)
  }

  // ── Samples ───────────────────────────────────────────────────────────────

  async listSamples(datasetId?: string) {
    return this.client.get<ApiResponse<any>>('/datasets/list', { params: { dataset_id: datasetId } })
  }

  async getDatasetStats() {
    return this.client.get<ApiResponse<any>>('/datasets/stats')
  }

  async deleteSample(sampleId: string) {
    return this.client.delete<ApiResponse<any>>(`/datasets/${sampleId}`)
  }

  async relabelSample(sampleId: string, label: string) {
    return this.client.patch<ApiResponse<any>>(`/datasets/relabel/${sampleId}`, { label })
  }

  async getDatasetLabels(datasetId: string) {
    return this.client.get<ApiResponse<any>>(`/datasets/labels/${datasetId}`)
  }

  async getSampleImage(sampleId: string): Promise<string> {
    const resp = await this.client.get(`/datasets/image/${sampleId}`, { responseType: 'blob' })
    return URL.createObjectURL(resp.data as Blob)
  }

  async addLabel(datasetId: string, label: string) {
    return this.client.post<ApiResponse<any>>(`/datasets/labels/${datasetId}/add`, { label })
  }

  async renameLabel(datasetId: string, oldLabel: string, newLabel: string) {
    return this.client.post<ApiResponse<any>>(`/datasets/labels/${datasetId}/rename`, {
      old_label: oldLabel, new_label: newLabel,
    })
  }

  async deleteLabel(datasetId: string, label: string) {
    return this.client.delete<ApiResponse<any>>(
      `/datasets/labels/${datasetId}/${encodeURIComponent(label)}`,
    )
  }

  async autoSplitDataset(datasetId: string, trainPct: number, valPct: number, testPct: number) {
    return this.client.post<ApiResponse<any>>(`/datasets/split/${datasetId}`, {
      train_pct: trainPct, val_pct: valPct, test_pct: testPct,
    })
  }

  async setSampleSplit(sampleId: string, split: string) {
    return this.client.patch<ApiResponse<any>>(`/datasets/sample/${sampleId}/split`, { split })
  }

  async exportFullDataset(datasetId: string, name: string) {
    const resp = await this.client.get(`/datasets/export/full/${datasetId}`, { responseType: 'blob' })
    const url = URL.createObjectURL(resp.data as Blob)
    const a = document.createElement('a')
    a.href = url; a.download = `${name.replace(/\s+/g, '_')}_full.zip`
    document.body.appendChild(a); a.click(); document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  async exportSplitDataset(datasetId: string, name: string) {
    const resp = await this.client.get(`/datasets/export/split/${datasetId}`, { responseType: 'blob' })
    const url = URL.createObjectURL(resp.data as Blob)
    const a = document.createElement('a')
    a.href = url; a.download = `${name.replace(/\s+/g, '_')}_split.zip`
    document.body.appendChild(a); a.click(); document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  // ── Remote Datasets (URL / Kaggle / HuggingFace) ────────────────────────────

  async getRemoteTokenStatus(): Promise<{ status: string; kaggle_configured: boolean; huggingface_configured: boolean }> {
    const res = await this.client.get('/remote_datasets/token_status')
    return res.data
  }

  async searchKaggle(query: string, page: number = 1): Promise<{ status: string; datasets: any[] }> {
    const res = await this.client.get('/remote_datasets/kaggle/search', { params: { query, page } })
    return res.data
  }

  async searchHuggingFace(query: string, limit: number = 20): Promise<{ status: string; datasets: any[] }> {
    const res = await this.client.get('/remote_datasets/huggingface/search', { params: { query, limit } })
    return res.data
  }

  /**
   * Start a remote download (URL, Kaggle, or HuggingFace).
   * Returns an SSE EventSource that streams progress events.
   * Events: { type: 'progress', downloaded: number, total: number } | { type: 'complete', count: number } | { type: 'error', detail: string }
   */
  createRemoteDownloadSSE(params: {
    source: 'url' | 'kaggle' | 'huggingface'
    url?: string
    dataset_ref?: string
    repo_id?: string
    dataset_id: string
    task: string
  }): EventSource {
    const query = new URLSearchParams()
    query.set('source', params.source)
    query.set('dataset_id', params.dataset_id)
    query.set('task', params.task)
    if (params.url) query.set('url', params.url)
    if (params.dataset_ref) query.set('dataset_ref', params.dataset_ref)
    if (params.repo_id) query.set('repo_id', params.repo_id)
    return new EventSource(`${API_BASE}/remote_datasets/download_stream?${query.toString()}`)
  }

  async cancelRemoteDownload(downloadId: string): Promise<void> {
    await this.client.post('/remote_datasets/cancel', { download_id: downloadId })
  }

  // ── Training ──────────────────────────────────────────────────────────────

  async startTraining(config: any) {
    return this.client.post<ApiResponse<any>>('/training/start', {
      ...config, input_shape: config.input_shape || [224, 224, 3],
    })
  }

  async getTrainingStatus(trainingId: string) {
    return this.client.get<ApiResponse<any>>(`/training/status/${trainingId}`)
  }

  async getTrainingMetrics(trainingId: string) {
    return this.client.get<ApiResponse<any>>(`/training/metrics/${trainingId}`)
  }

  async cancelTraining(trainingId: string) {
    return this.client.post<ApiResponse<any>>(`/training/cancel/${trainingId}`)
  }

  async listModels() {
    return this.client.get<ApiResponse<any>>('/training/models')
  }

  async listAllSessions() {
    return this.client.get<ApiResponse<any>>('/training/sessions')
  }

  // ── Optimization ──────────────────────────────────────────────────────────

  async quantizeModel(config: any) {
    return this.client.post<ApiResponse<any>>('/optimization/quantize', config)
  }

  async getOptimizationStatus(optimizationId: string) {
    return this.client.get<ApiResponse<any>>(`/optimization/status/${optimizationId}`)
  }

  async getOptimizationResult(optimizationId: string) {
    return this.client.get<ApiResponse<any>>(`/optimization/result/${optimizationId}`)
  }

  async exportCArray(optimizationId: string) {
    return this.client.post<ApiResponse<any>>(`/optimization/to-c-array/${optimizationId}`)
  }

  async evaluateBoard(optimizationId: string, board: string) {
    return this.client.post<ApiResponse<any>>('/optimization/evaluate-board', {
      optimization_id: optimizationId, board,
    })
  }

  async getLLMStatus() {
    return this.client.get<ApiResponse<any>>('/optimization/llm-status')
  }

  async getLLMSuggestions(trainingId: string, provider: 'ollama' | 'openrouter', modelName: string, apiKey?: string) {
    return this.client.post<ApiResponse<any>>('/optimization/llm-suggest', {
      training_id: trainingId, provider, model_name: modelName, api_key: apiKey,
    })
  }

  async getLLMOptimizationAdvice(optimizationId: string, board: string, useLocalLLM = false) {
    return this.client.post<ApiResponse<any>>('/optimization/llm-optimize', {
      optimization_id: optimizationId, board, use_local_llm: useLocalLLM,
    })
  }
}

const apiClient = new APIClient()

export function useAPI() {
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const request = useCallback(
    async <T>(fn: () => Promise<{ data: ApiResponse<T> }>) => {
      setLoading(true)
      setError(null)
      try {
        const response = await fn()
        if (response.data.status === 'error') {
          setError(response.data.message || 'An error occurred')
          return null
        }
        return response.data.data !== undefined ? response.data.data : response.data
      } catch (err: any) {
        setError(err.message || 'Network error')
        return null
      } finally {
        setLoading(false)
      }
    },
    [],
  )

  return { request, error, loading, apiClient }
}

