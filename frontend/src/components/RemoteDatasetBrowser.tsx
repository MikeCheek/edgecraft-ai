import { useState, useEffect } from 'react';
import {
  remoteDatasetApi,
  TokenStatus,
  KaggleDataset,
  HuggingFaceDataset,
} from '../services/remoteDatasetApi';

interface Props {
  datasetId: string;
  task: string;
  onImportComplete: () => void;
}

export function RemoteDatasetBrowser({ datasetId, task, onImportComplete }: Props) {
  const [activeTab, setActiveTab] = useState<'url' | 'kaggle' | 'huggingface'>('url');
  const [tokenStatus, setTokenStatus] = useState<TokenStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // URL state
  const [url, setUrl] = useState('');

  // Kaggle state
  const [kaggleQuery, setKaggleQuery] = useState('');
  const [kaggleResults, setKaggleResults] = useState<KaggleDataset[]>([]);

  // HuggingFace state
  const [hfQuery, setHfQuery] = useState('');
  const [hfResults, setHfResults] = useState<HuggingFaceDataset[]>([]);

  useEffect(() => {
    remoteDatasetApi.getTokenStatus().then(setTokenStatus).catch(() => { });
  }, []);

  const clearMessages = () => {
    setError(null);
    setSuccessMsg(null);
  };

  // ─── URL Download ──────────────────────────────────────────────────────────
  const handleUrlDownload = async () => {
    if (!url.trim()) return;
    clearMessages();
    setLoading(true);
    try {
      const res = await remoteDatasetApi.downloadFromUrl(url.trim(), datasetId, task);
      setSuccessMsg(`Successfully imported ${res.count ?? 0} samples from URL.`);
      onImportComplete();
      setUrl('');
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || 'Download failed');
    } finally {
      setLoading(false);
    }
  };

  // ─── Kaggle Search & Download ──────────────────────────────────────────────
  const handleKaggleSearch = async () => {
    if (!kaggleQuery.trim()) return;
    clearMessages();
    setLoading(true);
    try {
      const results = await remoteDatasetApi.searchKaggle(kaggleQuery.trim());
      setKaggleResults(results);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || 'Kaggle search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleKaggleDownload = async (ref: string) => {
    clearMessages();
    setLoading(true);
    try {
      const res = await remoteDatasetApi.downloadKaggle(ref, datasetId, task);
      setSuccessMsg(`Successfully imported ${res.count ?? 0} samples from Kaggle.`);
      onImportComplete();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || 'Kaggle download failed');
    } finally {
      setLoading(false);
    }
  };

  // ─── HuggingFace Search & Download ─────────────────────────────────────────
  const handleHfSearch = async () => {
    if (!hfQuery.trim()) return;
    clearMessages();
    setLoading(true);
    try {
      const results = await remoteDatasetApi.searchHuggingFace(hfQuery.trim());
      setHfResults(results);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || 'HuggingFace search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleHfDownload = async (repoId: string) => {
    clearMessages();
    setLoading(true);
    try {
      const res = await remoteDatasetApi.downloadHuggingFace(repoId, datasetId, task);
      setSuccessMsg(`Successfully imported ${res.count ?? 0} samples from HuggingFace.`);
      onImportComplete();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || 'HuggingFace download failed');
    } finally {
      setLoading(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
    return `${(bytes / 1073741824).toFixed(2)} GB`;
  };

  return (
    <div className="remote-dataset-browser">
      {/* Token Status */}
      {tokenStatus && (
        <div
          className="token-status"
          style={{ marginBottom: 12, fontSize: '0.85em', color: '#666' }}
        >
          <span style={{ marginRight: 16 }}>
            Kaggle: {tokenStatus.kaggle_configured ? '✅ Configured' : '❌ Not configured'}
          </span>
          <span>
            HuggingFace: {tokenStatus.huggingface_configured ? '✅ Configured' : '⚠️ Optional'}
          </span>
        </div>
      )}

      {/* Tabs */}
      <div className="tabs" style={{ display: 'flex', gap: 4, marginBottom: 16 }}>
        {(['url', 'kaggle', 'huggingface'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => {
              setActiveTab(tab);
              clearMessages();
            }}
            style={{
              padding: '8px 16px',
              border: 'none',
              borderBottom: activeTab === tab ? '2px solid #4f46e5' : '2px solid transparent',
              background: 'none',
              fontWeight: activeTab === tab ? 600 : 400,
              cursor: 'pointer',
            }}
          >
            {tab === 'url' ? '🔗 Download URL' : tab === 'kaggle' ? '📊 Kaggle' : '🤗 HuggingFace'}
          </button>
        ))}
      </div>

      {/* Messages */}
      {error && (
        <div style={{ color: '#dc2626', marginBottom: 8, padding: 8, background: '#fef2f2', borderRadius: 4 }}>
          {error}
        </div>
      )}
      {successMsg && (
        <div style={{ color: '#16a34a', marginBottom: 8, padding: 8, background: '#f0fdf4', borderRadius: 4 }}>
          {successMsg}
        </div>
      )}
      {loading && (
        <div style={{ color: '#4f46e5', marginBottom: 8 }}>
          ⏳ Processing... This may take a while for large datasets.
        </div>
      )}

      {/* URL Tab */}
      {activeTab === 'url' && (
        <div>
          <p style={{ fontSize: '0.9em', color: '#555', marginBottom: 8 }}>
            Provide a direct download link to a .zip dataset. The backend will download, extract, and
            import it.
          </p>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/dataset.zip"
              style={{ flex: 1, padding: '8px 12px', border: '1px solid #d1d5db', borderRadius: 6 }}
              disabled={loading}
              onKeyDown={(e) => e.key === 'Enter' && handleUrlDownload()}
            />
            <button
              onClick={handleUrlDownload}
              disabled={loading || !url.trim()}
              style={{
                padding: '8px 16px',
                background: '#4f46e5',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
              }}
            >
              Download & Import
            </button>
          </div>
        </div>
      )}

      {/* Kaggle Tab */}
      {activeTab === 'kaggle' && (
        <div>
          {!tokenStatus?.kaggle_configured && (
            <p style={{ color: '#dc2626', fontSize: '0.9em' }}>
              ⚠️ Set KAGGLE_USERNAME and KAGGLE_KEY in your backend .env file.
            </p>
          )}
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <input
              type="text"
              value={kaggleQuery}
              onChange={(e) => setKaggleQuery(e.target.value)}
              placeholder="Search Kaggle datasets..."
              style={{ flex: 1, padding: '8px 12px', border: '1px solid #d1d5db', borderRadius: 6 }}
              disabled={loading}
              onKeyDown={(e) => e.key === 'Enter' && handleKaggleSearch()}
            />
            <button
              onClick={handleKaggleSearch}
              disabled={loading || !kaggleQuery.trim()}
              style={{
                padding: '8px 16px',
                background: '#20beff',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
              }}
            >
              Search
            </button>
          </div>
          {kaggleResults.length > 0 && (
            <div style={{ maxHeight: 300, overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: 6 }}>
              {kaggleResults.map((ds) => (
                <div
                  key={ds.ref}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 12px',
                    borderBottom: '1px solid #f3f4f6',
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 500 }}>{ds.title}</div>
                    <div style={{ fontSize: '0.8em', color: '#666' }}>
                      {formatBytes(ds.size)} • {ds.download_count.toLocaleString()} downloads •{' '}
                      {ds.ref}
                    </div>
                  </div>
                  <button
                    onClick={() => handleKaggleDownload(ds.ref)}
                    disabled={loading}
                    style={{
                      padding: '6px 12px',
                      background: '#16a34a',
                      color: '#fff',
                      border: 'none',
                      borderRadius: 4,
                      cursor: 'pointer',
                      fontSize: '0.85em',
                    }}
                  >
                    Import
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* HuggingFace Tab */}
      {activeTab === 'huggingface' && (
        <div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <input
              type="text"
              value={hfQuery}
              onChange={(e) => setHfQuery(e.target.value)}
              placeholder="Search HuggingFace datasets..."
              style={{ flex: 1, padding: '8px 12px', border: '1px solid #d1d5db', borderRadius: 6 }}
              disabled={loading}
              onKeyDown={(e) => e.key === 'Enter' && handleHfSearch()}
            />
            <button
              onClick={handleHfSearch}
              disabled={loading || !hfQuery.trim()}
              style={{
                padding: '8px 16px',
                background: '#ff9d00',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
              }}
            >
              Search
            </button>
          </div>
          {hfResults.length > 0 && (
            <div style={{ maxHeight: 300, overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: 6 }}>
              {hfResults.map((ds) => (
                <div
                  key={ds.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 12px',
                    borderBottom: '1px solid #f3f4f6',
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 500 }}>{ds.id}</div>
                    <div style={{ fontSize: '0.8em', color: '#666' }}>
                      {ds.downloads.toLocaleString()} downloads
                      {ds.tags.length > 0 && ` • ${ds.tags.join(', ')}`}
                    </div>
                  </div>
                  <button
                    onClick={() => handleHfDownload(ds.id)}
                    disabled={loading}
                    style={{
                      padding: '6px 12px',
                      background: '#16a34a',
                      color: '#fff',
                      border: 'none',
                      borderRadius: 4,
                      cursor: 'pointer',
                      fontSize: '0.85em',
                    }}
                  >
                    Import
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}