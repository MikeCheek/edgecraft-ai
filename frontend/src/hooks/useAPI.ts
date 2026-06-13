import { useCallback, useState, useEffect } from 'react'
import axios, { AxiosInstance } from 'axios'
import { ApiResponse } from '../types'

const API_BASE = '/api'

class APIClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE,
      timeout: 30000
    })
  }

  // Health
  async health() {
    return this.client.get<ApiResponse<any>>('/health')
  }

  // Dataset endpoints
  async uploadSample(label: string, task: string, file: File) {
    const formData = new FormData()
    formData.append('label', label)
    formData.append('task', task)
    formData.append('file', file)
    return this.client.post<ApiResponse<any>>('/datasets/upload', formData)
  }

  async listDatasets(task?: string) {
    return this.client.get<ApiResponse<any>>('/datasets/list', {
      params: { task }
    })
  }

  async getDatasetStats() {
    return this.client.get<ApiResponse<any>>('/datasets/stats')
  }

  async deleteSample(sampleId: string) {
    return this.client.delete<ApiResponse<any>>(`/datasets/${sampleId}`)
  }

  // Training endpoints
  async startTraining(config: any) {
    return this.client.post<ApiResponse<any>>('/training/start', config)
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

  // Optimization endpoints
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

  async evaluateBoard(trainingId: string, board: string) {
    return this.client.post<ApiResponse<any>>('/optimization/evaluate-board', {
      training_id: trainingId,
      board
    })
  }

  async getSupportedBoards() {
    return this.client.get<ApiResponse<any>>('/optimization/boards')
  }

  // LLM endpoints
  async getLLMSuggestions(trainingId: string, metrics: any) {
    return this.client.post<ApiResponse<any>>('/optimization/llm-suggest', {
      training_id: trainingId,
      metrics
    })
  }

  async getLLMOptimizationAdvice(optimizationId: string, board: string) {
    return this.client.post<ApiResponse<any>>('/optimization/llm-optimize', {
      optimization_id: optimizationId,
      board
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
        return response.data.data
      } catch (err) {
        const message = err instanceof Error ? err.message : 'An error occurred'
        setError(message)
        return null
      } finally {
        setLoading(false)
      }
    },
    []
  )

  return { request, error, loading, apiClient }
}
