import axios from 'axios'
import { API_BASE } from '../hooks/useAPI'

const client = axios.create({
  baseURL: `${API_BASE}/remote_datasets`,
  timeout: 600000 // 10 minutes for large downloads
})

export interface TokenStatus {
  kaggle_configured: boolean
  huggingface_configured: boolean
}

export interface KaggleDataset {
  ref: string
  title: string
  size: number
  last_updated: string
  download_count: number
  description: string
  vote_count: number
  usability_rating: number
  license: string
  tags: string[]
}

export interface HuggingFaceDataset {
  id: string
  author: string
  title: string
  downloads: number
  likes: number
  last_modified: string
  created_at: string
  tags: string[]
  pipeline_tag: string
  description: string
}

export interface DownloadProgress {
  downloaded: number
  total: number
  /** 0—1, or -1 when total is unknown */
  fraction: number
  /** bytes/sec */
  speed: number
  /** seconds remaining, or null when unknown */
  eta: number | null
}

export interface DownloadCallbacks {
  onProgress?: (p: DownloadProgress) => void
  onProcessing?: (msg: string) => void
  onDatasetCreated?: (info: { dataset_id: string; dataset_name: string }) => void
}

/**
 * Open an SSE stream to /download_stream and resolve when the download
 * completes. Rejects on error or cancellation.
 *
 * Returns the number of imported samples.
 */
function streamDownload (
  params: Record<string, string>,
  callbacks: DownloadCallbacks,
  signal?: AbortSignal
): Promise<number> {
  return new Promise((resolve, reject) => {
    const qs = new URLSearchParams(params).toString()
    const url = `${API_BASE}/download_stream?${qs}`

    const es = new EventSource(url)

    let downloadId: string | null = null
    let startTime = Date.now()
    let lastDownloaded = 0
    let lastTime = Date.now()

    // Allow external cancellation via AbortSignal
    const onAbort = () => {
      es.close()
      if (downloadId) {
        // Fire-and-forget cancel request to the backend
        client.post('/cancel', { download_id: downloadId }).catch(() => {})
      }
      reject(new Error('Download cancelled'))
    }
    signal?.addEventListener('abort', onAbort, { once: true })

    es.onmessage = event => {
      let data: any
      try {
        data = JSON.parse(event.data)
      } catch {
        return
      }

      switch (data.type) {
        case 'start':
          downloadId = data.download_id ?? null
          startTime = Date.now()
          break

        case 'dataset_created':
          // Emit event when a dataset is auto-created
          callbacks.onDatasetCreated?.({
            dataset_id: data.dataset_id,
            dataset_name: data.dataset_name,
          })
          break

        case 'progress': {
          const downloaded: number = data.downloaded ?? 0
          const total: number = data.total ?? 0

          const now = Date.now()
          const dt = (now - lastTime) / 1000 // seconds
          const dd = downloaded - lastDownloaded
          const speed = dt > 0 ? dd / dt : 0
          lastDownloaded = downloaded
          lastTime = now

          const fraction = total > 0 ? downloaded / total : -1
          const eta =
            total > 0 && speed > 0
              ? Math.max(0, (total - downloaded) / speed)
              : null

          callbacks.onProgress?.({ downloaded, total, fraction, speed, eta })
          break
        }

        case 'processing':
          callbacks.onProcessing?.(data.message ?? 'Processing…')
          break

        case 'complete':
          es.close()
          signal?.removeEventListener('abort', onAbort)
          resolve(data.count ?? 0)
          break

        case 'canceled':
          es.close()
          signal?.removeEventListener('abort', onAbort)
          reject(new Error('Download cancelled'))
          break

        case 'error':
          es.close()
          signal?.removeEventListener('abort', onAbort)
          reject(new Error(data.message ?? 'Unknown download error'))
          break
      }
    }

    es.onerror = () => {
      es.close()
      signal?.removeEventListener('abort', onAbort)
      reject(new Error('SSE connection error — check the backend is running'))
    }
  })
}

export const remoteDatasetApi = {
  async getTokenStatus (): Promise<TokenStatus> {
    const res = await client.get('/token_status')
    return res.data
  },

  /**
   * Download a dataset from a direct URL with real-time progress.
   */
  downloadFromUrl (
    url: string,
    datasetId: string | null,
    task: string,
    callbacks: DownloadCallbacks = {},
    signal?: AbortSignal
  ): Promise<number> {
    const params: Record<string, string> = { source: 'url', task, url }
    if (datasetId) params.dataset_id = datasetId
    return streamDownload(params, callbacks, signal)
  },

  async searchKaggle (
    query: string,
    page: number = 1
  ): Promise<KaggleDataset[]> {
    const res = await client.get('/kaggle/search', { params: { query, page } })
    return res.data.datasets
  },

  /**
   * Download a Kaggle dataset with real-time progress.
   */
  downloadKaggle (
    datasetRef: string,
    datasetId: string | null,
    task: string,
    callbacks: DownloadCallbacks = {},
    signal?: AbortSignal
  ): Promise<number> {
    const params: Record<string, string> = {
      source: 'kaggle',
      task,
      dataset_ref: datasetRef,
    }
    if (datasetId) params.dataset_id = datasetId
    return streamDownload(params, callbacks, signal)
  },

  async searchHuggingFace (
    query: string,
    limit: number = 20
  ): Promise<HuggingFaceDataset[]> {
    const res = await client.get('/huggingface/search', {
      params: { query, limit }
    })
    return res.data.datasets
  },

  /**
   * Download a HuggingFace dataset with real-time progress.
   */
  downloadHuggingFace (
    repoId: string,
    datasetId: string | null,
    task: string,
    callbacks: DownloadCallbacks = {},
    signal?: AbortSignal
  ): Promise<number> {
    const params: Record<string, string> = {
      source: 'huggingface',
      task,
      repo_id: repoId,
    }
    if (datasetId) params.dataset_id = datasetId
    return streamDownload(params, callbacks, signal)
  }
}
