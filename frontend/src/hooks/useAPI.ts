// ----------------------------------------
// FILE: useApi.ts
// ----------------------------------------

import { useCallback, useState } from 'react'
import axios, { AxiosInstance } from 'axios'
import { ApiResponse } from '../types'

const API_BASE = 'http://localhost:8000/api'

class APIClient {
  private client: AxiosInstance // For standard metadata (Fast)
  private heavyClient: AxiosInstance // For processing/disk scans
  private uploadClient: AxiosInstance // For stream pipes

  constructor() {
    // Basic structural requests fail fast if backend stalls
    this.client = axios.create({ baseURL: API_BASE, timeout: 5000 })

    // Disk scans & processing queues get breathing room
    this.heavyClient = axios.create({ baseURL: API_BASE, timeout: 30000 })

    // Massive archive operations
    this.uploadClient = axios.create({ baseURL: API_BASE, timeout: 600000 })
  }

  async health() {
    return this.client.get<ApiResponse<any>>('/health')
  }

  async getStorageOverview() {
    return this.heavyClient.get<ApiResponse<any>>('/storage/overview')
  }

  async getModelTree(includeArchived: boolean = false) {
    const res = await this.client.get('/models/tree', {
      params: { include_archived: includeArchived }
    })
    return res.data
  }

  /** Detects a training job already in progress (used to reattach after
   * navigating away/reloading instead of losing track of it). */
  async getActiveTraining(task?: string) {
    const res = await this.client.get('/training/active', { params: { task } })
    return res.data
  }

  /** Same idea for optimization jobs. */
  async getActiveOptimization(trainingId?: string) {
    const res = await this.client.get('/optimization/active', {
      params: { training_id: trainingId }
    })
    return res.data
  }

  // --- Datasets ---

  async createDataset(name: string, task: string, description?: string) {
    return this.client.post<ApiResponse<any>>('/datasets/create', {
      name,
      task,
      description: description ?? ''
    })
  }

  async listDatasets(task?: string) {
    return this.client.get<ApiResponse<any>>('/datasets/list_datasets', {
      params: { task }
    })
  }

  async renameDataset(datasetId: string, newName: string) {
    return this.client.put<ApiResponse<any>>(`/datasets/rename/${datasetId}`, {
      new_name: newName
    })
  }

  async deleteDataset(datasetId: string) {
    return this.client.delete<ApiResponse<any>>(
      `/datasets/dataset/${datasetId}`
    )
  }

  async clearDataset(datasetId: string) {
    return this.client.delete<ApiResponse<any>>(
      `/datasets/clear_dataset/${datasetId}`
    )
  }

  async getSplitSummary(datasetId: string): Promise<{
    status: string
    summary: { train: number; val: number; test: number; unassigned: number }
  }> {
    const res = await this.client.get(`/datasets/split_summary/${datasetId}`)
    return res.data
  }

  // --- Dataset metadata (description + auto-computed image/storage stats) ---
  // NEW: backs the "Dataset Info" panel in DatasetExplorer.

  /** PATCH /api/datasets/{dataset_id}/metadata - update the free-text
   * description and/or the metadata bag. Both fields are optional so the
   * caller can update just one (e.g. just `{ description }`). */
  async updateDatasetMetadata(
    datasetId: string,
    body: { description?: string; metadata?: Record<string, any> }
  ) {
    return this.client.patch<ApiResponse<any>>(
      `/datasets/${datasetId}/metadata`,
      body
    )
  }

  /** GET /api/datasets/{dataset_id}/image_stats - aggregate width/height/
   * aspect-ratio/format/size stats computed from the dataset's samples. */
  async getDatasetImageStats(datasetId: string) {
    return this.client.get<ApiResponse<any>>(
      `/datasets/${datasetId}/image_stats`
    )
  }

  // --- Single-file upload ---

  async uploadSample(
    datasetId: string,
    label: string,
    task: string,
    file: File
  ) {
    const fd = new FormData()
    fd.append('dataset_id', datasetId)
    fd.append('label', label)
    fd.append('task', task)
    fd.append('file', file)
    return this.uploadClient.post<ApiResponse<any>>('/datasets/upload', fd)
  }

  // --- Chunked ZIP upload ---

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
    signal?: AbortSignal
  ): Promise<void> {
    const res = await fetch(
      `${API_BASE}/datasets/upload_zip/chunk/${uploadId}/${chunkIndex}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: blob,
        signal
      }
    )
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      throw new Error(`HTTP ${res.status}: ${text}`)
    }
  }

  async getZipUploadStatus(
    uploadId: string
  ): Promise<{ status: string; upload_id: string; received_chunks: number[] }> {
    const res = await this.client.get(`/datasets/upload_zip/status/${uploadId}`)
    return res.data
  }

  async finalizeZipUpload(params: {
    upload_id: string
    total_chunks: number
  }): Promise<{
    status: string
    tree?: any[]
    upload_id: string
    message?: string
  }> {
    const res = await this.client.post('/datasets/upload_zip/finalize', params)
    return res.data
  }

  async processZipUpload(
    uploadId: string,
    datasetId: string,
    task: string,
    mapping: any[],
    labelStrategy: string = 'folder',
    regexPattern?: string
  ) {
    // BUGFIX: this used the default 300s (`this.client`) timeout. Large
    // archives - especially now that extraction also probes each image's
    // width/height for the dataset-stats feature - can legitimately take
    // longer than that, and the request has no way to resume (unlike the
    // chunked upload), so a lengthy real extraction shouldn't be aborted
    // client-side while it's still working. 30 minutes gives real headroom
    // without being unbounded.
    //
    // Also NEW: forwards `label_strategy`/`regex_pattern` (previously
    // dropped here even though ZipTreeMapper already collected them),
    // which is what actually applies the filename-regex labeling on the
    // backend instead of silently falling back to folder labels.
    const res = await this.client.post(
      '/datasets/upload_zip/process',
      {
        upload_id: uploadId,
        dataset_id: datasetId,
        task,
        mapping,
        label_strategy: labelStrategy,
        regex_pattern: regexPattern
      },
      { timeout: 1_800_000 }
    )
    return res.data
  }

  async processRemoteZip(
    downloadId: string,
    datasetId: string,
    task: string,
    mapping: any[]
  ) {
    const res = await this.client.post(
      '/remote_datasets/process',
      {
        download_id: downloadId,
        dataset_id: datasetId,
        task,
        mapping
      },
      { timeout: 1_800_000 }
    )
    return res.data
  }

  async abortZipUpload(uploadId: string): Promise<void> {
    await this.client.delete(`/datasets/upload_zip/${uploadId}`)
  }

  // --- Samples ---

  async listSamples(datasetId?: string) {
    return this.client.get<ApiResponse<any>>('/datasets/list', {
      params: { dataset_id: datasetId }
    })
  }

  async getDatasetStats() {
    return this.client.get<ApiResponse<any>>('/datasets/stats')
  }

  async deleteSample(sampleId: string) {
    return this.client.delete<ApiResponse<any>>(`/datasets/${sampleId}`)
  }

  async relabelSample(sampleId: string, label: string) {
    return this.client.patch<ApiResponse<any>>(
      `/datasets/relabel/${sampleId}`,
      { label }
    )
  }

  async getDatasetLabels(datasetId: string) {
    return this.client.get<ApiResponse<any>>(`/datasets/labels/${datasetId}`)
  }

  async getSampleImage(sampleId: string): Promise<string> {
    const resp = await this.client.get(`/datasets/image/${sampleId}`, {
      responseType: 'blob'
    })
    return URL.createObjectURL(resp.data as Blob)
  }

  async addLabel(datasetId: string, label: string) {
    return this.client.post<ApiResponse<any>>(
      `/datasets/labels/${datasetId}/add`,
      { label }
    )
  }

  async renameLabel(datasetId: string, oldLabel: string, newLabel: string) {
    return this.client.post<ApiResponse<any>>(
      `/datasets/labels/${datasetId}/rename`,
      {
        old_label: oldLabel,
        new_label: newLabel
      }
    )
  }

  async deleteLabel(datasetId: string, label: string) {
    return this.client.delete<ApiResponse<any>>(
      `/datasets/labels/${datasetId}/${encodeURIComponent(label)}`
    )
  }

  async autoSplitDataset(
    datasetId: string,
    trainPct: number,
    valPct: number,
    testPct: number
  ) {
    return this.client.post<ApiResponse<any>>(`/datasets/split/${datasetId}`, {
      train_pct: trainPct,
      val_pct: valPct,
      test_pct: testPct
    })
  }

  async setSampleSplit(sampleId: string, split: string) {
    return this.client.patch<ApiResponse<any>>(
      `/datasets/sample/${sampleId}/split`,
      { split }
    )
  }

  async exportFullDataset(datasetId: string, name: string) {
    const resp = await this.client.get(`/datasets/export/full/${datasetId}`, {
      responseType: 'blob'
    })
    const url = URL.createObjectURL(resp.data as Blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${name.replace(/\s+/g, '_')}_full.zip`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  async exportSplitDataset(datasetId: string, name: string) {
    const resp = await this.client.get(`/datasets/export/split/${datasetId}`, {
      responseType: 'blob'
    })
    const url = URL.createObjectURL(resp.data as Blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${name.replace(/\s+/g, '_')}_split.zip`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  // --- Remote Datasets (URL / Kaggle / HuggingFace) ---

  async getRemoteTokenStatus(): Promise<{
    status: string
    kaggle_configured: boolean
    huggingface_configured: boolean
  }> {
    const res = await this.client.get('/remote_datasets/token_status')
    return res.data
  }

  async searchKaggle(
    query: string,
    page: number = 1
  ): Promise<{ status: string; datasets: any[] }> {
    const res = await this.client.get('/remote_datasets/kaggle/search', {
      params: { query, page }
    })
    return res.data
  }

  async searchHuggingFace(
    query: string,
    limit: number = 20
  ): Promise<{ status: string; datasets: any[] }> {
    const res = await this.client.get('/remote_datasets/huggingface/search', {
      params: { query, limit }
    })
    return res.data
  }

  /**
   * Start a remote download (URL, Kaggle, or HuggingFace).
   * Returns an SSE EventSource that streams progress events.
// Events: { type: 'start', download_id: string }
//       | { type: 'progress', downloaded: number, total: number }
//       | { type: 'processing', message: string }
//       | { type: 'complete', count: number }
//       | { type: 'error', message: string }   // <-- was 'detail'
//       | { type: 'canceled' }
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
    return new EventSource(
      `${API_BASE}/remote_datasets/download_stream?${query.toString()}`
    )
  }

  async cancelRemoteDownload(downloadId: string): Promise<void> {
    await this.client.post('/remote_datasets/cancel', {
      download_id: downloadId
    })
  }

  // --- Training ---

  async startTraining(config: any) {
    return this.client.post<ApiResponse<any>>('/training/start', {
      ...config,
      input_shape: config.input_shape || [224, 224, 3],
      device: config.device || 'auto'
    })
  }

  async getAvailableDevices(): Promise<{
    status: string
    devices: {
      cpu_available: boolean
      gpu_available: boolean
      gpus: { name: string; compute_capability: any }[]
    }
  }> {
    const res = await this.client.get('/training/devices')
    return res.data
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

  async listAllSessions(includeArchived: boolean = false) {
    return this.client.get<ApiResponse<any>>('/training/sessions', {
      params: { include_archived: includeArchived }
    })
  }

  async archiveTraining(trainingId: string) {
    return this.client.post<ApiResponse<any>>(`/training/archive/${trainingId}`)
  }

  async unarchiveTraining(trainingId: string) {
    return this.client.post<ApiResponse<any>>(
      `/training/unarchive/${trainingId}`
    )
  }

  async deleteTrainingSession(trainingId: string) {
    return this.client.delete<ApiResponse<any>>(
      `/training/session/${trainingId}`
    )
  }

  /**
   * Ask the LLM (with a rule-based fallback) for a suggested base_model +
   * hyperparameters BEFORE starting a training run, given the task, the
   * dataset actually selected, and the board you plan to deploy to.
   */
  async getTrainingRecommendation(params: {
    task: string
    dataset_id: string
    target_board?: string
    provider?: 'ollama' | 'openrouter'
    model_name?: string
  }) {
    return this.client.post<ApiResponse<any>>('/training/recommend', params, {
      timeout: 60_000
    })
  }

  // --- Optimization ---

  async quantizeModel(config: any) {
    return this.client.post<ApiResponse<any>>('/optimization/quantize', config)
  }

  async getOptimizationStatus(optimizationId: string) {
    return this.client.get<ApiResponse<any>>(
      `/optimization/status/${optimizationId}`
    )
  }

  async getOptimizationResult(optimizationId: string) {
    return this.client.get<ApiResponse<any>>(
      `/optimization/result/${optimizationId}`
    )
  }

  async exportCArray(optimizationId: string) {
    return this.client.post<ApiResponse<any>>(
      `/optimization/to-c-array/${optimizationId}`
    )
  }

  /**
   * NEW: catalog of known camera/display module presets (id, label,
   * default pins, style) for populating the Deployment tab's dropdowns.
   */
  async getHardwarePresets() {
    return this.client.get<ApiResponse<any>>('/optimization/hardware_presets')
  }

  /**
   * NEW: live ".ino" text preview (no zip/download) for the Deployment tab,
   * so the sketch can be shown as the user edits board/camera/display
   * choices without triggering a file download each time.
   */
  async previewExportSketch(
    optimizationId: string,
    board: string,
    config?: {
      camera_pins?: Record<string, number>
      camera_config?: Record<string, any>
      display_config?: Record<string, any>
    }
  ) {
    return this.client.post<ApiResponse<any>>(
      `/optimization/export-preview/${optimizationId}`,
      {
        board,
        camera_pins: config?.camera_pins,
        camera_config: config?.camera_config,
        display_config: config?.display_config
      }
    )
  }

  /**
   * Download a full Arduino/C++ project (model_data.h + sketch.ino + README.md)
   * for the given board, built from the REAL optimized model bytes.
   */
  async exportProject(
    optimizationId: string,
    board: string,
    config?: {
      camera_pins?: Record<string, number>
      // BUGFIX: `camera_config` (module_preset, enabled, module_type,
      // pins_override) was accepted in the type signature's neighbourhood
      // but never actually read here, so there was no way to select a
      // camera module (e.g. "ESP32-S3-CAM (OV3660)") or opt a non-ESP32_CAM
      // board into an external camera - it silently always fell back to
      // the board's bare default. Now forwarded to the backend like
      // camera_pins/display_config already were.
      camera_config?: Record<string, any>
      display_config?: Record<string, any>
    }
  ) {
    const resp = await this.client.post(
      `/optimization/export/${optimizationId}`,
      {
        board,
        camera_pins: config?.camera_pins,
        camera_config: config?.camera_config,
        display_config: config?.display_config
      },
      { responseType: 'blob' }
    )
    const url = URL.createObjectURL(resp.data as Blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `edgecraft_export_${optimizationId.slice(0, 8)}_${board}.zip`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  async evaluateBoard(optimizationId: string, board: string) {
    return this.client.post<ApiResponse<any>>('/optimization/evaluate-board', {
      optimization_id: optimizationId,
      board
    })
  }

  async getLLMStatus() {
    return this.client.get<ApiResponse<any>>('/optimization/llm-status')
  }

  async getLLMSuggestions(
    trainingId: string,
    provider: 'ollama' | 'openrouter',
    modelName: string
  ) {
    return this.client.post<ApiResponse<any>>(
      '/optimization/llm-suggest',
      {
        training_id: trainingId,
        provider,
        model_name: modelName
      },
      {
        timeout: 120_000 // Override: Allow up to 2 minutes for LLM generation
      }
    )
  }

  async getLLMOptimizationAdvice(
    optimizationId: string,
    board: string,
    useLocalLLM = false
  ) {
    return this.client.post<ApiResponse<any>>(
      '/optimization/llm-optimize',
      {
        optimization_id: optimizationId,
        board,
        use_local_llm: useLocalLLM
      },
      {
        timeout: 120_000 // Override: Allow up to 2 minutes for LLM generation
      }
    )
  }

  // ── Inference ──────────────────────────────────────────────────────────────

  async getInferenceHistory(limit: number = 100) {
    return this.client.get<ApiResponse<any>>(
      `/inference/history?limit=${limit}`
    )
  }

  async clearInferenceHistory() {
    return this.client.delete<ApiResponse<any>>('/inference/history')
  }

  async getOptimizationHistory() {
    return this.client.get<ApiResponse<any>>('/optimization/history')
  }

  async previewZipRegex(uploadId: string, regexPattern: string) {
    const res = await axios.post(
      `${API_BASE}/datasets/upload_zip/preview_regex`,
      {
        upload_id: uploadId,
        regex_pattern: regexPattern
      }
    )
    return res.data
  }

  async relabelDatasetBulkRegex(datasetId: string, regexPattern: string) {
    const res = await axios.post(
      `${API_BASE}/datasets/${datasetId}/relabel_bulk_regex`,
      {
        regex_pattern: regexPattern
      }
    )
    return res.data
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
        return response.data.data !== undefined
          ? response.data.data
          : response.data
      } catch (err: any) {
        setError(err.message || 'Network error')
        console.error('API request error:', err)
        return null
      } finally {
        setLoading(false)
      }
    },
    []
  )

  return { request, error, loading, apiClient }
}
