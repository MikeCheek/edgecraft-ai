import { useCallback, useState } from 'react'
import axios, { AxiosInstance } from 'axios'
import { ApiResponse } from '../types'

const API_BASE = 'http://localhost:8000/api'

class APIClient {
  private client: AxiosInstance
  constructor() {
    this.client = axios.create({ baseURL: API_BASE, timeout: 30000 })
  }

  async health() {
    return this.client.get<ApiResponse<any>>('/health')
  }

  // ?? Datasets ??????????????????????????????????????????????????????????????

  async createDataset(name: string, task: string) {
    return this.client.post<ApiResponse<any>>('/datasets/create', {
      name,
      task
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

  async uploadSample(
    datasetId: string,
    label: string,
    task: string,
    file: File
  ) {
    const formData = new FormData()
    formData.append('dataset_id', datasetId)
    formData.append('label', label)
    formData.append('task', task)
    formData.append('file', file)
    return this.client.post<ApiResponse<any>>('/datasets/upload', formData)
  }

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

  // ?? Training ??????????????????????????????????????????????????????????????

  async startTraining(config: any) {
    return this.client.post<ApiResponse<any>>('/training/start', {
      ...config,
      input_shape: config.input_shape || [224, 224, 3]
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

  // ?? Optimization ??????????????????????????????????????????????????????????

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
    metrics: any,
    useLocalLLM = false
  ) {
    return this.client.post<ApiResponse<any>>('/optimization/llm-suggest', {
      training_id: trainingId,
      metrics,
      use_local_llm: useLocalLLM
    })
  }

  async getLLMOptimizationAdvice(
    optimizationId: string,
    board: string,
    useLocalLLM = false
  ) {
    return this.client.post<ApiResponse<any>>('/optimization/llm-optimize', {
      optimization_id: optimizationId,
      board,
      use_local_llm: useLocalLLM
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
        return response.data.data !== undefined
          ? response.data.data
          : response.data
      } catch (err: any) {
        setError(err.message || 'Network error')
        return null
      } finally {
        setLoading(false)
      }
    },
    []
  )

  return { request, error, loading, apiClient }
}
