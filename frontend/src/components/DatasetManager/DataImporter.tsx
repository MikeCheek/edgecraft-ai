import { Check, X, AlertTriangle, ExternalLink, Download, Upload, Loader2, Globe, Database, Info, FolderOpen, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useAPI } from '../../hooks/useAPI';
import { TinyMLTask } from '../../types';
import DownloadProgressBar, { DownloadProgress } from './DownloadProgressBar';
import { TreeItem } from '../../types';
import { ZipTreeMapper } from './ZipTreeMapper';

// ============================================================================
// Unified Data Importer Component
// ============================================================================

type ImportTab = 'local' | 'url' | 'kaggle' | 'huggingface';
type SortField = 'downloads' | 'size' | 'votes' | 'rating' | 'name' | 'likes' | 'last_updated';
type SortDir = 'asc' | 'desc';

interface DataImporterProps {
  datasetId?: string; // Now optional - can auto-create
  task: TinyMLTask;
  onImportSuccess: (newDatasetId?: string) => void;
}

function DataImporter({ datasetId, task, onImportSuccess }: DataImporterProps) {
  const { apiClient } = useAPI();
  const [activeTab, setActiveTab] = useState<ImportTab>('local');

  // Token status
  const [tokenStatus, setTokenStatus] = useState<{ kaggle_configured: boolean; huggingface_configured: boolean } | null>(null);
  useEffect(() => {
    apiClient.getRemoteTokenStatus().then(setTokenStatus).catch(() => { });
  }, [apiClient]);

  // --- Local ZIP state ---
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const CHUNK_SIZE = 2 * 1024 * 1024;
  const MAX_CONCURRENT = 3;

  // --- Remote download state ---
  const [url, setUrl] = useState('');
  const [kaggleQuery, setKaggleQuery] = useState('');
  const [kaggleResults, setKaggleResults] = useState<any[]>([]);
  const [hfQuery, setHfQuery] = useState('');
  const [hfResults, setHfResults] = useState<any[]>([]);
  const [searching, setSearching] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Sorting/filtering for search results
  const [kaggleSortField, setKaggleSortField] = useState<SortField>('downloads');
  const [kaggleSortDir, setKaggleSortDir] = useState<SortDir>('desc');
  const [kaggleMinDownloads, setKaggleMinDownloads] = useState(0);
  const [hfSortField, setHfSortField] = useState<SortField>('downloads');
  const [hfSortDir, setHfSortDir] = useState<SortDir>('desc');
  const [hfMinDownloads, setHfMinDownloads] = useState(0);

  const [detailsDataset, setDetailsDataset] = useState<any | null>(null);

  const [mappingTree, setMappingTree] = useState<TreeItem[] | null>(null);
  const [mappingSession, setMappingSession] = useState<{ type: 'local' | 'remote', id: string } | null>(null);
  const [mappingAnnotationFormat, setMappingAnnotationFormat] = useState<string | null>(null);
  const [mappingAnnotationClasses, setMappingAnnotationClasses] = useState<string[]>([]);

  // Track auto-created dataset
  const [currentDatasetId, setCurrentDatasetId] = useState<string | undefined>(datasetId);

  // Update currentDatasetId when prop changes
  useEffect(() => {
    setCurrentDatasetId(datasetId);
  }, [datasetId]);

  // --- Local ZIP upload handler ---
  const handleZipUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);
    setUploadError(null);
    setUploadStatus('Initializing chunked upload...');

    try {
      const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
      const initData = await apiClient.initZipUpload({
        dataset_id: currentDatasetId!, task, filename: file.name,
        total_chunks: totalChunks, file_size: file.size, chunk_size: CHUNK_SIZE,
      });
      if (initData.status !== 'success') throw new Error(initData.message || 'Init failed');
      const { upload_id } = initData;

      setUploadStatus(`Streaming chunks... 0/${totalChunks}`);
      let completed = 0; let active = 0; let current = 0;

      await new Promise<void>((resolve, reject) => {
        const uploadNext = () => {
          if (current >= totalChunks && active === 0) { resolve(); return; }
          while (active < MAX_CONCURRENT && current < totalChunks) {
            const idx = current++; active++;
            const blob = file.slice(idx * CHUNK_SIZE, Math.min((idx + 1) * CHUNK_SIZE, file.size));
            apiClient.putZipChunk(upload_id, idx, blob)
              .then(() => { completed++; setUploadProgress(Math.round((completed / totalChunks) * 100)); setUploadStatus(`Streaming chunks... ${completed}/${totalChunks}`); active--; uploadNext(); })
              .catch(reject);
          }
        };
        uploadNext();
      });

      setUploadStatus('Scanning ZIP structure...');
      const result = await apiClient.finalizeZipUpload({ upload_id, total_chunks: totalChunks });

      // Stop uploading loader, show modal
      setUploading(false);
      setMappingTree(result.tree || []);
      setMappingAnnotationFormat(result.annotation_format || null);
      setMappingAnnotationClasses(result.annotation_classes || []);
      setMappingSession({ type: 'local', id: upload_id });
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      setUploadError(err.message || 'Upload failed');
      setUploading(false);
    } finally {
      setUploading(false);
    }
  };

  // --- Local Folder upload handler ---
  const handleFolderUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    setUploadProgress(0);
    setUploadError(null);
    setUploadStatus(`Uploading ${files.length} files...`);

    try {
      const formData = new FormData();
      formData.append('dataset_id', currentDatasetId!);
      formData.append('task', task);

      // Add all files
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }

      const response = await fetch(`http://localhost:8000/api/datasets/upload_folder`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (result.status === 'success') {
        setUploadStatus(`Uploaded ${result.processed} files${result.errors > 0 ? ` (${result.errors} errors)` : ''}`);
        if (result.error_messages?.length > 0) {
          console.warn('Folder upload errors:', result.error_messages);
        }
        onImportSuccess();
      } else {
        throw new Error(result.message || 'Upload failed');
      }
    } catch (err: any) {
      setUploadError(err.message || 'Folder upload failed');
    } finally {
      setUploading(false);
      if (folderInputRef.current) folderInputRef.current.value = '';
    }
  };

  // --- Remote download via SSE ---
  const startRemoteDownload = (source: 'url' | 'kaggle' | 'huggingface', params: any) => {
    if (eventSourceRef.current) { eventSourceRef.current.close(); eventSourceRef.current = null; }

    const progress: DownloadProgress = {
      downloaded: 0, total: 0, startTime: Date.now(),
      phase: 'downloading',
    };
    setDownloadProgress({ ...progress });

    const es = apiClient.createRemoteDownloadSSE({
      source, dataset_id: currentDatasetId || undefined, task, ...params,
    });
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'dataset_created') {
          // Auto-created dataset - update state
          setCurrentDatasetId(data.dataset_id);
          setDownloadProgress((prev) => prev ? {
            ...prev,
            message: `Created dataset: ${data.dataset_name}`,
          } : prev);
        } else if (data.type === 'start') {
          // FIX: capture download_id for cancellation
          setDownloadProgress((prev) => prev ? {
            ...prev,
            downloadId: data.download_id,
          } : prev);
        } else if (data.type === 'progress') {
          setDownloadProgress((prev) => prev ? {
            ...prev,
            downloaded: data.downloaded || prev.downloaded,
            total: data.total > 0 ? data.total : prev.total,  // don't overwrite known total with 0,
          } : prev);
        } else if (data.type === 'processing') {
          setDownloadProgress((prev) => prev ? {
            ...prev,
            phase: 'processing',
            message: data.message,
          } : prev);
        } else if (data.type === 'ready_to_map') {
          setMappingTree(data.tree);
          setMappingAnnotationFormat(data.annotation_format || null);
          setMappingAnnotationClasses(data.annotation_classes || []);
          setMappingSession({ type: 'remote', id: data.download_id });
          setDownloadProgress(null); // Hide progress bar
          es.close();
        } else if (data.type === 'complete') {
          setDownloadProgress((prev) => prev ? {
            ...prev,
            phase: 'complete',
            count: data.count,
          } : prev);
          es.close();
        } else if (data.type === 'error') {
          setDownloadProgress((prev) => prev ? {
            ...prev,
            phase: 'error',
            message: data.message,  // FIX: was data.detail
          } : prev);
          es.close();
        } else if (data.type === 'canceled') {
          setDownloadProgress((prev) => prev ? {
            ...prev,
            phase: 'cancelled',
          } : prev);
          es.close();
        }
      } catch {
        /* ignore parse errors */
      }
    };

    es.onerror = () => {
      setDownloadProgress(prev => prev ? {
        ...prev, phase: 'error', message: 'Connection lost',
      } : prev);
      es.close();
      eventSourceRef.current = null;
    };
  };

  const handleCancel = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (downloadProgress?.downloadId) {
      apiClient.cancelRemoteDownload(downloadProgress.downloadId).catch(() => { });
    }
    setDownloadProgress(prev => prev ? { ...prev, phase: 'cancelled' } : prev);
  };

  const handleConfirmMapping = async (mapping: TreeItem[]) => {
    setMappingTree(null);
    setMappingAnnotationFormat(null);
    setMappingAnnotationClasses([]);
    setUploading(true);
    setUploadStatus('Extracting explicitly mapped folders...');

    try {
      let result;
      if (mappingSession?.type === 'local') {
        result = await apiClient.processZipUpload(mappingSession.id, currentDatasetId!, task, mapping);
      } else if (mappingSession?.type === 'remote') {
        result = await apiClient.processRemoteZip(mappingSession.id, currentDatasetId!, task, mapping);
      }
      setUploadStatus(`Imported ${result.count ?? 0} samples`);
      onImportSuccess(currentDatasetId);
    } catch (err: any) {
      setUploadError(err.message || 'Extraction failed');
    } finally {
      setUploading(false);
      setMappingSession(null);
    }
  };

  // --- Kaggle search ---
  const handleKaggleSearch = async () => {
    if (!kaggleQuery.trim()) return;
    setSearching(true);
    try {
      const res = await apiClient.searchKaggle(kaggleQuery.trim());
      setKaggleResults(res.datasets || []);
    } catch { setKaggleResults([]); }
    finally { setSearching(false); }
  };

  // --- HuggingFace search ---
  const handleHfSearch = async () => {
    if (!hfQuery.trim()) return;
    setSearching(true);
    try {
      const res = await apiClient.searchHuggingFace(hfQuery.trim());
      setHfResults(res.datasets || []);
    } catch { setHfResults([]); }
    finally { setSearching(false); }
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
    return `${(bytes / 1073741824).toFixed(2)} GB`;
  };

  // --- Sorted & filtered Kaggle results ---
  const sortedKaggleResults = useMemo(() => {
    let filtered = kaggleResults.filter(ds => (ds.download_count || 0) >= kaggleMinDownloads);
    return filtered.sort((a, b) => {
      if (kaggleSortField === 'name') {
        const va = a.title || '', vb = b.title || '';
        return kaggleSortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      if (kaggleSortField === 'last_updated') {
        const va = a.last_updated || '', vb = b.last_updated || '';
        return kaggleSortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      let va: number, vb: number;
      switch (kaggleSortField) {
        case 'downloads': va = a.download_count || 0; vb = b.download_count || 0; break;
        case 'size': va = a.size || 0; vb = b.size || 0; break;
        case 'votes': va = a.vote_count || 0; vb = b.vote_count || 0; break;
        case 'rating': va = a.usability_rating || 0; vb = b.usability_rating || 0; break;
        default: va = 0; vb = 0;
      }
      return kaggleSortDir === 'asc' ? va - vb : vb - va;
    });
  }, [kaggleResults, kaggleSortField, kaggleSortDir, kaggleMinDownloads]);

  // --- Sorted & filtered HuggingFace results ---
  const sortedHfResults = useMemo(() => {
    let filtered = hfResults.filter(ds => (ds.downloads || 0) >= hfMinDownloads);
    return filtered.sort((a, b) => {
      if (hfSortField === 'name') {
        const va = a.title || '', vb = b.title || '';
        return hfSortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      if (hfSortField === 'last_updated') {
        const va = a.last_modified || '', vb = b.last_modified || '';
        return hfSortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      let va: number, vb: number;
      switch (hfSortField) {
        case 'downloads': va = a.downloads || 0; vb = b.downloads || 0; break;
        case 'likes': va = a.likes || 0; vb = b.likes || 0; break;
        default: va = 0; vb = 0;
      }
      return hfSortDir === 'asc' ? va - vb : vb - va;
    });
  }, [hfResults, hfSortField, hfSortDir, hfMinDownloads]);

  const toggleKaggleSort = (field: SortField) => {
    if (kaggleSortField === field) {
      setKaggleSortDir(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setKaggleSortField(field);
      setKaggleSortDir('desc');
    }
  };

  const toggleHfSort = (field: SortField) => {
    if (hfSortField === field) {
      setHfSortDir(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setHfSortField(field);
      setHfSortDir('desc');
    }
  };

  const SortIcon = ({ field, current, dir }: { field: SortField; current: SortField; dir: SortDir }) => {
    if (field !== current) return <ArrowUpDown className="w-2.5 h-2.5 opacity-30" />;
    return dir === 'asc' ? <ArrowUp className="w-2.5 h-2.5 text-indigo-400" /> : <ArrowDown className="w-2.5 h-2.5 text-indigo-400" />;
  };

  const isDownloading = downloadProgress && (downloadProgress.phase === 'downloading' || downloadProgress.phase === 'processing');

  const tabs: { key: ImportTab; icon: React.ReactNode; label: string }[] = [
    { key: 'local', icon: <Upload className="w-3.5 h-3.5" />, label: 'Local Files' },
    { key: 'url', icon: <Globe className="w-3.5 h-3.5" />, label: 'Remote URL' },
    { key: 'kaggle', icon: <Database className="w-3.5 h-3.5" />, label: 'Kaggle' },
    { key: 'huggingface', icon: <Database className="w-3.5 h-3.5" />, label: 'HuggingFace' },
  ];

  return (
    <div className="p-4 bg-slate-900 rounded-xl border border-slate-700/60 mt-2 space-y-3">
      {mappingTree && mappingSession && (
        <ZipTreeMapper
          uploadId={mappingSession.id}
          tree={mappingTree}
          task={task}
          annotationFormat={mappingAnnotationFormat}
          annotationClasses={mappingAnnotationClasses}
          onConfirm={handleConfirmMapping}
          onCancel={() => { setMappingTree(null); setMappingSession(null); setUploading(false); }}
        />
      )}
      {/* Tab bar */}
      <div className="flex gap-1 border-b border-slate-700 pb-2">
        {tabs.map(t => (
          <button key={t.key} onClick={() => setActiveTab(t.key)}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-t-md transition ${activeTab === t.key
              ? 'bg-slate-800 text-white border-b-2 border-indigo-500'
              : 'text-gray-400 hover:text-gray-200'
              }`}>
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {/* Token status for remote tabs */}
      {activeTab !== 'local' && tokenStatus && (
        <div className="flex gap-4 text-[10px] text-gray-500">
          <span className="flex items-center gap-1">
            {tokenStatus.kaggle_configured ? <Check className="w-3 h-3 text-green-400" /> : <X className="w-3 h-3 text-red-400" />}
            Kaggle
          </span>
          <span className="flex items-center gap-1">
            {tokenStatus.huggingface_configured ? <Check className="w-3 h-3 text-green-400" /> : <AlertTriangle className="w-3 h-3 text-yellow-400" />}
            HuggingFace
          </span>
        </div>
      )}

      {/* Auto-creation notice */}
      {!currentDatasetId && (
        <div className="p-2 bg-indigo-900/20 border border-indigo-500/30 rounded-lg text-[11px] text-indigo-300">
          <Info className="w-3.5 h-3.5 inline mr-1" />
          No dataset selected. One will be auto-created using the source name and metadata.
        </div>
      )}

      {detailsDataset && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
          onClick={() => setDetailsDataset(null)}>
          <div className="bg-slate-900 border border-slate-700 rounded-xl p-5 max-w-md w-full mx-4 shadow-xl"
            onClick={e => e.stopPropagation()}>
            <div className="flex items-start justify-between mb-3">
              <h3 className="text-sm font-bold text-white">{detailsDataset.title || detailsDataset.id}</h3>
              <button onClick={() => setDetailsDataset(null)} className="text-gray-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-1.5 text-xs text-gray-300">
              {detailsDataset.ref && <p><span className="text-gray-500">Ref:</span> {detailsDataset.ref}</p>}
              {detailsDataset.id && <p><span className="text-gray-500">ID:</span> {detailsDataset.id}</p>}
              {detailsDataset.description && <p className="text-gray-400 italic">{detailsDataset.description}</p>}
              {detailsDataset.size != null && <p><span className="text-gray-500">Size:</span> {formatBytes(detailsDataset.size)}</p>}
              <p><span className="text-gray-500">Downloads:</span> {(detailsDataset.downloads || detailsDataset.download_count || 0).toLocaleString()}</p>
              {(detailsDataset.vote_count > 0) && <p><span className="text-gray-500">Votes:</span> {detailsDataset.vote_count}</p>}
              {(detailsDataset.likes > 0) && <p><span className="text-gray-500">Likes:</span> {detailsDataset.likes}</p>}
              {(detailsDataset.usability_rating > 0) && <p><span className="text-gray-500">Usability Rating:</span> {detailsDataset.usability_rating}/10</p>}
              {detailsDataset.license && <p><span className="text-gray-500">License:</span> {detailsDataset.license}</p>}
              {detailsDataset.pipeline_tag && <p><span className="text-gray-500">Pipeline:</span> {detailsDataset.pipeline_tag}</p>}
              {(detailsDataset.last_updated || detailsDataset.last_modified) && (
                <p><span className="text-gray-500">Last updated:</span> {detailsDataset.last_updated || detailsDataset.last_modified}</p>
              )}
              {detailsDataset.created_at && <p><span className="text-gray-500">Created:</span> {detailsDataset.created_at}</p>}
              {detailsDataset.author && <p><span className="text-gray-500">Author:</span> {detailsDataset.author}</p>}
              {detailsDataset.tags?.length > 0 && <p><span className="text-gray-500">Tags:</span> {detailsDataset.tags.join(', ')}</p>}
            </div>
            <div className="mt-4 flex gap-2">
              <a href={detailsDataset.ref
                ? `https://www.kaggle.com/datasets/${detailsDataset.ref}`
                : `https://huggingface.co/datasets/${detailsDataset.id}`}
                target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-700 hover:bg-indigo-600 text-white text-xs rounded-lg transition">
                <ExternalLink className="w-3.5 h-3.5" /> Open page
              </a>
              <button onClick={() => {
                startRemoteDownload(detailsDataset.ref ? 'kaggle' : 'huggingface',
                  detailsDataset.ref ? { dataset_ref: detailsDataset.ref } : { repo_id: detailsDataset.id });
                setDetailsDataset(null);
              }} disabled={!!isDownloading}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 text-white text-xs rounded-lg transition">
                <Download className="w-3.5 h-3.5" /> Import
              </button>
            </div>
          </div>
        </div>
      )}

      {/* --- LOCAL FILES TAB --- */}
      {activeTab === 'local' && (
        <div>
          {!uploading && (
            <div className="space-y-3">
              {/* ZIP Upload */}
              <div onClick={() => fileInputRef.current?.click()}
                className="border border-dashed border-slate-700 hover:border-indigo-500 cursor-pointer rounded-lg p-5 text-center bg-slate-950/20 transition group">
                <Upload className="mx-auto mb-2 text-slate-500 group-hover:text-indigo-400 transition" size={24} />
                <span className="block text-xs font-medium mb-0.5">Select ZIP Package</span>
                <span className="text-[10px] text-slate-500">Chunked upload — supports large datasets (10GB+)</span>
                <input type="file" ref={fileInputRef} onChange={handleZipUpload} accept=".zip" className="hidden" />
              </div>

              {/* Folder Upload */}
              <div onClick={() => folderInputRef.current?.click()}
                className="border border-dashed border-slate-700 hover:border-emerald-500 cursor-pointer rounded-lg p-5 text-center bg-slate-950/20 transition group">
                <FolderOpen className="mx-auto mb-2 text-slate-500 group-hover:text-emerald-400 transition" size={24} />
                <span className="block text-xs font-medium mb-0.5">Select Folder</span>
                <span className="text-[10px] text-slate-500">Upload a folder with class subdirectories (e.g., cat/, dog/)</span>
                <input type="file" ref={folderInputRef} onChange={handleFolderUpload}
                  // @ts-ignore - webkitdirectory is not in the type definitions
                  webkitdirectory="" directory="" multiple className="hidden" />
              </div>
            </div>
          )}
          {uploading && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs font-medium text-slate-300">
                <span className="flex items-center gap-1.5">
                  <Loader2 className="animate-spin text-indigo-400" size={14} /> {uploadStatus}
                </span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-indigo-500 transition-all duration-300" style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          )}
          {!uploading && uploadStatus && !uploadError && (
            <div className="mt-2 p-2 bg-slate-950/60 rounded border border-slate-800 text-[11px] font-mono text-emerald-400 flex items-center gap-2">
              <Check size={14} /> {uploadStatus}
            </div>
          )}
          {uploadError && (
            <div className="mt-2 p-2 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-[11px] rounded flex gap-2">
              <AlertTriangle size={14} className="shrink-0" /> {uploadError}
            </div>
          )}
        </div>
      )}

      {/* --- REMOTE URL TAB --- */}
      {activeTab === 'url' && (
        <div>
          <p className="text-[11px] text-gray-500 mb-2">Paste a direct .zip download link. The server will download, extract, and import it.</p>
          <div className="flex gap-2">
            <input type="text" value={url} onChange={e => setUrl(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !isDownloading && url.trim() && startRemoteDownload('url', { url: url.trim() })}
              placeholder="https://example.com/dataset.zip"
              disabled={!!isDownloading}
              className="flex-1 px-3 py-1.5 bg-slate-950 border border-slate-600 rounded-lg text-white text-xs placeholder-gray-500 focus:outline-none focus:border-indigo-500 disabled:opacity-50" />
            <button onClick={() => startRemoteDownload('url', { url: url.trim() })}
              disabled={!!isDownloading || !url.trim()}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition">
              <Globe className="w-3.5 h-3.5" /> Download
            </button>
          </div>
          {downloadProgress && <DownloadProgressBar progress={downloadProgress} onCancel={handleCancel} />}
        </div>
      )}

      {/* --- KAGGLE TAB --- */}
      {activeTab === 'kaggle' && (
        <div>
          {!tokenStatus?.kaggle_configured && (
            <p className="text-[11px] text-amber-400 mb-2">Set KAGGLE_USERNAME and KAGGLE_KEY in backend .env</p>
          )}
          <div className="flex gap-2 mb-3">
            <input type="text" value={kaggleQuery} onChange={e => setKaggleQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleKaggleSearch()}
              placeholder="Search Kaggle datasets..."
              disabled={searching || !!isDownloading}
              className="flex-1 px-3 py-1.5 bg-slate-950 border border-slate-600 rounded-lg text-white text-xs placeholder-gray-500 focus:outline-none focus:border-cyan-500 disabled:opacity-50" />
            <button onClick={handleKaggleSearch} disabled={searching || !kaggleQuery.trim() || !!isDownloading}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition">
              {searching ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Database className="w-3.5 h-3.5" />} Search
            </button>
          </div>
          {kaggleResults.length > 0 && (
            <>
              {/* Sort & Filter controls */}
              <div className="flex items-center gap-2 mb-2 flex-wrap">
                <span className="text-[10px] text-gray-500">Sort:</span>
                {(['downloads', 'size', 'votes', 'rating', 'name', 'last_updated'] as SortField[]).map(f => (
                  <button key={f} onClick={() => toggleKaggleSort(f)}
                    className={`flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] rounded transition ${kaggleSortField === f ? 'bg-slate-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
                    {f === 'last_updated' ? 'date' : f} <SortIcon field={f} current={kaggleSortField} dir={kaggleSortDir} />
                  </button>
                ))}
                <span className="text-[10px] text-gray-500 ml-2">Min downloads:</span>
                <input type="number" value={kaggleMinDownloads} onChange={e => setKaggleMinDownloads(Number(e.target.value) || 0)}
                  className="w-16 px-1.5 py-0.5 text-[10px] bg-slate-950 border border-slate-600 rounded text-white" min="0" step="100" />
              </div>
              <div className="max-h-72 overflow-y-auto border border-slate-700 rounded-lg divide-y divide-slate-800 custom-scrollbar">
                {sortedKaggleResults.length === 0 && (
                  <div className="px-3 py-4 text-center text-[11px] text-gray-500">No results match filters</div>
                )}
                {sortedKaggleResults.map(ds => (
                  <div key={ds.ref} className="px-3 py-2 hover:bg-slate-800/50 transition">
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0 mr-3">
                        <p className="text-xs font-medium text-white truncate">{ds.title}</p>
                        <div className="flex items-center gap-2 text-[10px] text-gray-500 flex-wrap">
                          <span>{formatBytes(ds.size)}</span>
                          <span>{(ds.download_count ?? 0).toLocaleString()} dl</span>
                          {ds.vote_count > 0 && <span>{ds.vote_count} votes</span>}
                          {ds.usability_rating > 0 && <span className="text-amber-400">{ds.usability_rating} rating</span>}
                          {ds.license && <span className="text-gray-600">{ds.license}</span>}
                        </div>
                        {ds.tags?.length > 0 && (
                          <div className="flex gap-1 mt-1 flex-wrap">
                            {ds.tags.slice(0, 3).map((t: string) => (
                              <span key={t} className="px-1 py-0.5 text-[9px] bg-slate-800 text-gray-400 rounded">{t}</span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <a href={`https://www.kaggle.com/datasets/${ds.ref}`} target="_blank" rel="noopener noreferrer"
                          className="p-1 text-gray-500 hover:text-white transition" title="Open on Kaggle">
                          <ExternalLink className="w-3 h-3" />
                        </a>
                        <button onClick={() => setDetailsDataset(ds)} className="p-1 text-gray-500 hover:text-white transition" title="Details">
                          <Info className="w-3 h-3" />
                        </button>
                        <button onClick={() => startRemoteDownload('kaggle', { dataset_ref: ds.ref })}
                          disabled={!!isDownloading}
                          className="px-2 py-1 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 text-white text-[10px] font-semibold rounded-md transition">
                          Import
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {downloadProgress && <DownloadProgressBar progress={downloadProgress} onCancel={handleCancel} />}
        </div>
      )}

      {/* --- HUGGINGFACE TAB --- */}
      {activeTab === 'huggingface' && (
        <div>
          <div className="flex gap-2 mb-3">
            <input type="text" value={hfQuery} onChange={e => setHfQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleHfSearch()}
              placeholder="Search HuggingFace datasets..."
              disabled={searching || !!isDownloading}
              className="flex-1 px-3 py-1.5 bg-slate-950 border border-slate-600 rounded-lg text-white text-xs placeholder-gray-500 focus:outline-none focus:border-amber-500 disabled:opacity-50" />
            <button onClick={handleHfSearch} disabled={searching || !hfQuery.trim() || !!isDownloading}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition">
              {searching ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Database className="w-3.5 h-3.5" />} Search
            </button>
          </div>
          {hfResults.length > 0 && (
            <>
              {/* Sort & Filter controls */}
              <div className="flex items-center gap-2 mb-2 flex-wrap">
                <span className="text-[10px] text-gray-500">Sort:</span>
                {(['downloads', 'likes', 'name', 'last_updated'] as SortField[]).map(f => (
                  <button key={f} onClick={() => toggleHfSort(f)}
                    className={`flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] rounded transition ${hfSortField === f ? 'bg-slate-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
                    {f === 'last_updated' ? 'date' : f} <SortIcon field={f} current={hfSortField} dir={hfSortDir} />
                  </button>
                ))}
                <span className="text-[10px] text-gray-500 ml-2">Min downloads:</span>
                <input type="number" value={hfMinDownloads} onChange={e => setHfMinDownloads(Number(e.target.value) || 0)}
                  className="w-16 px-1.5 py-0.5 text-[10px] bg-slate-950 border border-slate-600 rounded text-white" min="0" step="100" />
              </div>
              <div className="max-h-72 overflow-y-auto border border-slate-700 rounded-lg divide-y divide-slate-800 custom-scrollbar">
                {sortedHfResults.length === 0 && (
                  <div className="px-3 py-4 text-center text-[11px] text-gray-500">No results match filters</div>
                )}
                {sortedHfResults.map(ds => (
                  <div key={ds.id} className="px-3 py-2 hover:bg-slate-800/50 transition">
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0 mr-3">
                        <p className="text-xs font-medium text-white truncate">{ds.id}</p>
                        <div className="flex items-center gap-2 text-[10px] text-gray-500 flex-wrap">
                          <span>{(ds.downloads ?? 0).toLocaleString()} dl</span>
                          {ds.likes > 0 && <span className="text-rose-400">{ds.likes} likes</span>}
                          {ds.pipeline_tag && <span className="text-gray-600">{ds.pipeline_tag}</span>}
                        </div>
                        {ds.tags?.length > 0 && (
                          <div className="flex gap-1 mt-1 flex-wrap">
                            {ds.tags.slice(0, 3).map((t: string) => (
                              <span key={t} className="px-1 py-0.5 text-[9px] bg-slate-800 text-gray-400 rounded">{t}</span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <a href={`https://huggingface.co/datasets/${ds.id}`} target="_blank" rel="noopener noreferrer"
                          className="p-1 text-gray-500 hover:text-white transition" title="Open on HuggingFace">
                          <ExternalLink className="w-3 h-3" />
                        </a>
                        <button onClick={() => setDetailsDataset(ds)} className="p-1 text-gray-500 hover:text-white transition" title="Details">
                          <Info className="w-3 h-3" />
                        </button>
                        <button onClick={() => startRemoteDownload('huggingface', { repo_id: ds.id })}
                          disabled={!!isDownloading}
                          className="px-2 py-1 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 text-white text-[10px] font-semibold rounded-md transition">
                          Import
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {downloadProgress && <DownloadProgressBar progress={downloadProgress} onCancel={handleCancel} />}
        </div>
      )}

      {/* Single-file capture
      <div className="pt-3 border-t border-slate-800">
        <p className="text-[10px] font-semibold text-slate-500 mb-1.5 uppercase tracking-wider">Single File Upload</p>
        <DataCollector datasetId={datasetId} task={task} onSampleAdded={onImportSuccess} />
      </div> */}
    </div>
  );
}

export default DataImporter
