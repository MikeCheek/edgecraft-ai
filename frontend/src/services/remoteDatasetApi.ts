import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000/api/remote_datasets';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 600000, // 10 minutes for large downloads
});

export interface TokenStatus {
  kaggle_configured: boolean;
  huggingface_configured: boolean;
}

export interface KaggleDataset {
  ref: string;
  title: string;
  size: number;
  last_updated: string;
  download_count: number;
  description: string;
}

export interface HuggingFaceDataset {
  id: string;
  author: string;
  title: string;
  downloads: number;
  last_modified: string;
  tags: string[];
  description: string;
}

export const remoteDatasetApi = {
  async getTokenStatus(): Promise<TokenStatus> {
    const res = await client.get('/token_status');
    return res.data;
  },

  async downloadFromUrl(url: string, datasetId: string, task: string): Promise<{ status: string; count?: number }> {
    const res = await client.post('/download_url', { url, dataset_id: datasetId, task });
    return res.data;
  },

  async searchKaggle(query: string, page: number = 1): Promise<KaggleDataset[]> {
    const res = await client.get('/kaggle/search', { params: { query, page } });
    return res.data.datasets;
  },

  async downloadKaggle(datasetRef: string, datasetId: string, task: string): Promise<{ status: string; count?: number }> {
    const res = await client.post('/kaggle/download', {
      dataset_ref: datasetRef,
      dataset_id: datasetId,
      task,
    });
    return res.data;
  },

  async searchHuggingFace(query: string, limit: number = 20): Promise<HuggingFaceDataset[]> {
    const res = await client.get('/huggingface/search', { params: { query, limit } });
    return res.data.datasets;
  },

  async downloadHuggingFace(repoId: string, datasetId: string, task: string): Promise<{ status: string; count?: number }> {
    const res = await client.post('/huggingface/download', {
      repo_id: repoId,
      dataset_id: datasetId,
      task,
    });
    return res.data;
  },
};