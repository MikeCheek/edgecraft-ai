import { useState, useEffect, useRef } from 'react';
import {
  remoteDatasetApi,
  TokenStatus,
  KaggleDataset,
  HuggingFaceDataset,
  DownloadProgress,
} from '../../services/remoteDatasetApi';

interface Props {
  datasetId: string;
  task: string;
  onImportComplete: () => void;
}

// ─── helpers ────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1_048_576) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1_073_741_824) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  return `${(bytes / 1_073_741_824).toFixed(2)} GB`;
}

function formatSpeed(bps: number): string {
  return `${formatBytes(bps)}/s`;
}

function formatEta(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}

// ─── ProgressBar ─────────────────────────────────────────────────────────────

function ProgressBar({ progress, processing }: { progress: DownloadProgress | null; processing: string | null }) {
  if (!progress && !processing) return null;

  const isIndeterminate = processing !== null || (progress !== null && progress.fraction < 0);
  const pct = progress && progress.fraction >= 0 ? Math.round(progress.fraction * 100) : null;

  return (
    <div style={{ marginTop: 10 }}>
      <div
        style={{
          height: 8,
          borderRadius: 4,
          background: '#e5e7eb',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        {isIndeterminate ? (
          <div
            style={{
              position: 'absolute',
              height: '100%',
              width: '40%',
              background: '#4f46e5',
              borderRadius: 4,
              animation: 'indeterminate 1.4s ease infinite',
            }}
          />
        ) : (
          <div
            style={{
              height: '100%',
              width: `${pct}%`,
              background: '#4f46e5',
              borderRadius: 4,
              transition: 'width 0.3s ease',
            }}
          />
        )}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78em', color: '#555', marginTop: 4 }}>
        {processing ? (
          <span>{processing}</span>
        ) : progress ? (
          <>
            <span>
              {formatBytes(progress.downloaded)}
              {progress.total > 0 && ` / ${formatBytes(progress.total)}`}
              {progress.speed > 0 && ` · ${formatSpeed(progress.speed)}`}
            </span>
            <span>
              {pct !== null && `${pct}%`}
              {progress.eta !== null && ` · ETA ${formatEta(progress.eta)}`}
            </span>
          </>
        ) : null}
      </div>

      <style>{`
        @keyframes indeterminate {
          0%   { left: -40%; }
          100% { left: 100%; }
        }
      `}</style>
    </div>
  );
}

// ─── DetailsModal ─────────────────────────────────────────────────────────────

type ModalData =
  | { kind: 'kaggle'; dataset: KaggleDataset }
  | { kind: 'huggingface'; dataset: HuggingFaceDataset };

function DetailsModal({ data, onClose }: { data: ModalData; onClose: () => void }) {
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.45)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: '#fff',
          borderRadius: 10,
          padding: 24,
          maxWidth: 520,
          width: '90%',
          boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
          position: 'relative',
        }}
      >
        <button
          onClick={onClose}
          style={{
            position: 'absolute', top: 12, right: 12,
            background: 'none', border: 'none', fontSize: 18,
            cursor: 'pointer', color: '#888',
          }}
        >
          ✕
        </button>

        {data.kind === 'kaggle' && (
          <>
            <h3 style={{ margin: '0 0 4px', fontSize: '1.05em' }}>{data.dataset.title}</h3>
            <div style={{ fontSize: '0.8em', color: '#888', marginBottom: 12 }}>{data.dataset.ref}</div>
            <table style={{ width: '100%', fontSize: '0.88em', borderCollapse: 'collapse' }}>
              <tbody>
                {[
                  ['Size', formatBytes(data.dataset.size)],
                  ['Downloads', data.dataset.download_count.toLocaleString()],
                  ['Last updated', data.dataset.last_updated],
                ].map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ padding: '4px 0', color: '#555', width: 120 }}>{k}</td>
                    <td style={{ padding: '4px 0', fontWeight: 500 }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {data.dataset.description && (
              <p style={{ marginTop: 12, fontSize: '0.87em', color: '#444', lineHeight: 1.5 }}>
                {data.dataset.description}
              </p>
            )}
          </>
        )}

        {data.kind === 'huggingface' && (
          <>
            <h3 style={{ margin: '0 0 4px', fontSize: '1.05em' }}>{data.dataset.id}</h3>
            <div style={{ fontSize: '0.8em', color: '#888', marginBottom: 12 }}>by {data.dataset.author || '—'}</div>
            <table style={{ width: '100%', fontSize: '0.88em', borderCollapse: 'collapse' }}>
              <tbody>
                {[
                  ['Downloads', data.dataset.downloads.toLocaleString()],
                  ['Last modified', data.dataset.last_modified],
                  ['Tags', data.dataset.tags.length > 0 ? data.dataset.tags.join(', ') : '—'],
                ].map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ padding: '4px 0', color: '#555', width: 120 }}>{k}</td>
                    <td style={{ padding: '4px 0', fontWeight: 500 }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {data.dataset.description && (
              <p style={{ marginTop: 12, fontSize: '0.87em', color: '#444', lineHeight: 1.5 }}>
                {data.dataset.description}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ─── Main component ──────────────────────────────────────────────────────────

export function RemoteDatasetBrowser({ datasetId, task, onImportComplete }: Props) {
  const [activeTab, setActiveTab] = useState<'url' | 'kaggle' | 'huggingface'>('url');
  const [tokenStatus, setTokenStatus] = useState<TokenStatus | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [processing, setProcessing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [modal, setModal] = useState<ModalData | null>(null);

  const abortRef = useRef<AbortController | null>(null);

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
    setProgress(null);
    setProcessing(null);
  };

  const startDownload = async (downloadFn: (signal: AbortSignal) => Promise<number>, successLabel: string) => {
    clearMessages();
    setDownloading(true);
    abortRef.current = new AbortController();
    try {
      const count = await downloadFn(abortRef.current.signal);
      setSuccessMsg(`Successfully imported ${count} samples from ${successLabel}.`);
      setProgress(null);
      setProcessing(null);
      onImportComplete();
    } catch (e: any) {
      if (e.message === 'Download cancelled') {
        setError('Download cancelled.');
      } else {
        setError(e.message || 'Download failed');
      }
      setProgress(null);
      setProcessing(null);
    } finally {
      setDownloading(false);
      abortRef.current = null;
    }
  };

  const handleCancelDownload = () => {
    abortRef.current?.abort();
  };

  // ── URL ──────────────────────────────────────────────────────────────────

  const handleUrlDownload = () => {
    if (!url.trim()) return;
    startDownload(
      (signal) =>
        remoteDatasetApi.downloadFromUrl(url.trim(), datasetId, task, {
          onProgress: setProgress,
          onProcessing: setProcessing,
        }, signal),
      'URL',
    );
  };

  // ── Kaggle ───────────────────────────────────────────────────────────────

  const handleKaggleSearch = async () => {
    if (!kaggleQuery.trim()) return;
    clearMessages();
    setSearching(true);
    try {
      const results = await remoteDatasetApi.searchKaggle(kaggleQuery.trim());
      setKaggleResults(results);
    } catch (e: any) {
      setError(e.message || 'Kaggle search failed');
    } finally {
      setSearching(false);
    }
  };

  const handleKaggleDownload = (ref: string) => {
    startDownload(
      (signal) =>
        remoteDatasetApi.downloadKaggle(ref, datasetId, task, {
          onProgress: setProgress,
          onProcessing: setProcessing,
        }, signal),
      'Kaggle',
    );
  };

  // ── HuggingFace ──────────────────────────────────────────────────────────

  const handleHfSearch = async () => {
    if (!hfQuery.trim()) return;
    clearMessages();
    setSearching(true);
    try {
      const results = await remoteDatasetApi.searchHuggingFace(hfQuery.trim());
      setHfResults(results);
    } catch (e: any) {
      setError(e.message || 'HuggingFace search failed');
    } finally {
      setSearching(false);
    }
  };

  const handleHfDownload = (repoId: string) => {
    startDownload(
      (signal) =>
        remoteDatasetApi.downloadHuggingFace(repoId, datasetId, task, {
          onProgress: setProgress,
          onProcessing: setProcessing,
        }, signal),
      'HuggingFace',
    );
  };

  // ── Render ───────────────────────────────────────────────────────────────

  const isLoading = downloading || searching;

  return (
    <div className="remote-dataset-browser">
      {modal && <DetailsModal data={modal} onClose={() => setModal(null)} />}

      {/* Token Status */}
      {tokenStatus && (
        <div style={{ marginBottom: 12, fontSize: '0.85em', color: '#666' }}>
          <span style={{ marginRight: 16 }}>
            Kaggle: {tokenStatus.kaggle_configured ? '✅ Configured' : '❌ Not configured'}
          </span>
          <span>
            HuggingFace: {tokenStatus.huggingface_configured ? '✅ Configured' : '⚪ Optional'}
          </span>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16 }}>
        {(['url', 'kaggle', 'huggingface'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => { setActiveTab(tab); clearMessages(); }}
            style={{
              padding: '8px 16px',
              border: 'none',
              borderBottom: activeTab === tab ? '2px solid #4f46e5' : '2px solid transparent',
              background: 'none',
              fontWeight: activeTab === tab ? 600 : 400,
              cursor: 'pointer',
            }}
          >
            {tab === 'url' ? '🔗 Download URL' : tab === 'kaggle' ? '🐙 Kaggle' : '🤗 HuggingFace'}
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

      {/* Progress */}
      {downloading && (
        <>
          <ProgressBar progress={progress} processing={processing} />
          <button
            onClick={handleCancelDownload}
            style={{
              marginTop: 8,
              padding: '4px 12px',
              background: '#fef2f2',
              color: '#dc2626',
              border: '1px solid #fca5a5',
              borderRadius: 4,
              cursor: 'pointer',
              fontSize: '0.82em',
            }}
          >
            Cancel
          </button>
        </>
      )}

      {/* ── URL Tab ── */}
      {activeTab === 'url' && (
        <div style={{ marginTop: downloading ? 12 : 0 }}>
          <p style={{ fontSize: '0.9em', color: '#555', marginBottom: 8 }}>
            Provide a direct download link to a .zip dataset. The backend will download, extract, and import it.
          </p>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/dataset.zip"
              style={{ flex: 1, padding: '8px 12px', border: '1px solid #d1d5db', borderRadius: 6 }}
              disabled={isLoading}
              onKeyDown={(e) => e.key === 'Enter' && handleUrlDownload()}
            />
            <button
              onClick={handleUrlDownload}
              disabled={isLoading || !url.trim()}
              style={{
                padding: '8px 16px',
                background: isLoading ? '#a5b4fc' : '#4f46e5',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: isLoading ? 'not-allowed' : 'pointer',
              }}
            >
              {downloading ? 'Downloading…' : 'Download & Import'}
            </button>
          </div>
        </div>
      )}

      {/* ── Kaggle Tab ── */}
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
              placeholder="Search Kaggle datasets…"
              style={{ flex: 1, padding: '8px 12px', border: '1px solid #d1d5db', borderRadius: 6 }}
              disabled={isLoading}
              onKeyDown={(e) => e.key === 'Enter' && handleKaggleSearch()}
            />
            <button
              onClick={handleKaggleSearch}
              disabled={isLoading || !kaggleQuery.trim()}
              style={{
                padding: '8px 16px',
                background: '#20beff',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: isLoading ? 'not-allowed' : 'pointer',
              }}
            >
              {searching ? 'Searching…' : 'Search'}
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
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {ds.title}
                    </div>
                    <div style={{ fontSize: '0.8em', color: '#666' }}>
                      {formatBytes(ds.size)} · {ds.download_count.toLocaleString()} downloads · {ds.ref}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 6, marginLeft: 10, flexShrink: 0 }}>
                    <button
                      onClick={() => setModal({ kind: 'kaggle', dataset: ds })}
                      title="Details"
                      style={{
                        padding: '5px 10px',
                        background: '#f3f4f6',
                        color: '#374151',
                        border: '1px solid #d1d5db',
                        borderRadius: 4,
                        cursor: 'pointer',
                        fontSize: '0.82em',
                      }}
                    >
                      Details
                    </button>
                    <a
                      href={`https://www.kaggle.com/datasets/${ds.ref}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="Open on Kaggle"
                      style={{
                        padding: '5px 10px',
                        background: '#f3f4f6',
                        color: '#374151',
                        border: '1px solid #d1d5db',
                        borderRadius: 4,
                        cursor: 'pointer',
                        fontSize: '0.82em',
                        textDecoration: 'none',
                        display: 'inline-flex',
                        alignItems: 'center',
                      }}
                    >
                      ↗
                    </a>
                    <button
                      onClick={() => handleKaggleDownload(ds.ref)}
                      disabled={isLoading}
                      style={{
                        padding: '5px 10px',
                        background: isLoading ? '#bbf7d0' : '#16a34a',
                        color: '#fff',
                        border: 'none',
                        borderRadius: 4,
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        fontSize: '0.82em',
                      }}
                    >
                      Import
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── HuggingFace Tab ── */}
      {activeTab === 'huggingface' && (
        <div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <input
              type="text"
              value={hfQuery}
              onChange={(e) => setHfQuery(e.target.value)}
              placeholder="Search HuggingFace datasets…"
              style={{ flex: 1, padding: '8px 12px', border: '1px solid #d1d5db', borderRadius: 6 }}
              disabled={isLoading}
              onKeyDown={(e) => e.key === 'Enter' && handleHfSearch()}
            />
            <button
              onClick={handleHfSearch}
              disabled={isLoading || !hfQuery.trim()}
              style={{
                padding: '8px 16px',
                background: '#ff9d00',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: isLoading ? 'not-allowed' : 'pointer',
              }}
            >
              {searching ? 'Searching…' : 'Search'}
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
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {ds.id}
                    </div>
                    <div style={{ fontSize: '0.8em', color: '#666' }}>
                      {ds.downloads.toLocaleString()} downloads
                      {ds.tags.length > 0 && ` · ${ds.tags.slice(0, 4).join(', ')}`}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 6, marginLeft: 10, flexShrink: 0 }}>
                    <button
                      onClick={() => setModal({ kind: 'huggingface', dataset: ds })}
                      title="Details"
                      style={{
                        padding: '5px 10px',
                        background: '#f3f4f6',
                        color: '#374151',
                        border: '1px solid #d1d5db',
                        borderRadius: 4,
                        cursor: 'pointer',
                        fontSize: '0.82em',
                      }}
                    >
                      Details
                    </button>
                    <a
                      href={`https://huggingface.co/datasets/${ds.id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="Open on HuggingFace"
                      style={{
                        padding: '5px 10px',
                        background: '#f3f4f6',
                        color: '#374151',
                        border: '1px solid #d1d5db',
                        borderRadius: 4,
                        cursor: 'pointer',
                        fontSize: '0.82em',
                        textDecoration: 'none',
                        display: 'inline-flex',
                        alignItems: 'center',
                      }}
                    >
                      ↗
                    </a>
                    <button
                      onClick={() => handleHfDownload(ds.id)}
                      disabled={isLoading}
                      style={{
                        padding: '5px 10px',
                        background: isLoading ? '#bbf7d0' : '#16a34a',
                        color: '#fff',
                        border: 'none',
                        borderRadius: 4,
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        fontSize: '0.82em',
                      }}
                    >
                      Import
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}