import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Plus, Trash2, RefreshCw, Database, Edit2, Check, X,
  Eye, Tag, ChevronDown, ChevronUp, AlertTriangle, Image as ImageIcon,
  FolderPlus, Tags, Download, Upload, Loader2
} from 'lucide-react';
import { useAPI } from '../hooks/useAPI';
import { DataCollector } from './DataCollector';
import { TinyMLTask, DatasetInfo, DatasetSample } from '../types';

interface DatasetManagerProps {
  task: TinyMLTask;
  onDatasetChanged?: () => void;
}

// ============================================================================
// Sample Card Component
// ============================================================================

interface SampleCardProps {
  sample: DatasetSample;
  allLabels: string[];
  apiBase: string;
  onRelabel: (id: string, newLabel: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  onSplitChange: (id: string, newSplit: string) => Promise<void>;
}

function SampleCard({ sample, allLabels, apiBase, onRelabel, onDelete, onSplitChange }: SampleCardProps) {
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(sample.label);
  const [customLabel, setCustomLabel] = useState('');
  const [useCustom, setUseCustom] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [imgError, setImgError] = useState(false);

  const isAudio = sample.filename?.match(/\.(wav|mp3)$/i);

  const handleSave = async () => {
    const finalLabel = useCustom ? customLabel.trim() : label;
    if (!finalLabel || finalLabel === sample.label) { setEditing(false); return; }
    setSaving(true);
    await onRelabel(sample.id, finalLabel);
    setSaving(false);
    setEditing(false);
  };

  const handleDelete = async () => {
    if (!window.confirm("Are you sure you want to delete this sample?")) return;
    setDeleting(true);
    await onDelete(sample.id);
  };

  return (
    <div className={`bg-slate-800 rounded-xl border border-slate-700 overflow-hidden group hover:border-slate-500 transition-all ${deleting ? 'opacity-50' : ''}`}>
      {/* Thumbnail */}
      <div className="relative h-28 bg-slate-900 flex items-center justify-center">
        {isAudio ? (
          <div className="flex flex-col items-center gap-1 opacity-50">
            <span className="text-2xl">🎵</span>
            <span className="text-xs text-gray-400 truncate px-2 max-w-full">{sample.filename}</span>
          </div>
        ) : imgError ? (
          <div className="flex flex-col items-center gap-1 opacity-40">
            <ImageIcon className="w-8 h-8 text-gray-500" />
            <span className="text-xs text-gray-500">No preview</span>
          </div>
        ) : (
          <img
            src={`${apiBase}/datasets/image/${sample.id}`}
            alt={sample.label}
            loading="lazy"
            decoding="async"
            className="w-full h-full object-cover"
            onError={() => setImgError(true)}
          />
        )}
        {/* Delete button overlay */}
        <button
          onClick={handleDelete}
          disabled={deleting}
          className="absolute top-1 right-1 p-1 bg-red-600/80 hover:bg-red-500 rounded-md opacity-0 group-hover:opacity-100 transition-opacity disabled:cursor-not-allowed"
        >
          {deleting
            ? <RefreshCw className="w-3 h-3 text-white animate-spin" />
            : <Trash2 className="w-3 h-3 text-white" />}
        </button>
      </div>

      {/* Label row */}
      <div className="p-2">
        {editing ? (
          <div className="space-y-1.5">
            {!useCustom && (
              <select
                value={label}
                onChange={e => setLabel(e.target.value)}
                className="w-full px-2 py-1 bg-slate-700 border border-slate-500 rounded text-white text-xs focus:outline-none focus:border-purple-500"
              >
                {allLabels.map(l => <option key={l} value={l}>{l}</option>)}
              </select>
            )}
            <div className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={useCustom}
                onChange={e => setUseCustom(e.target.checked)}
                className="accent-purple-500"
                id={`custom-${sample.id}`}
              />
              <label htmlFor={`custom-${sample.id}`} className="text-xs text-gray-400 cursor-pointer">New label</label>
            </div>
            {useCustom && (
              <input
                autoFocus
                value={customLabel}
                onChange={e => setCustomLabel(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSave()}
                placeholder="Type new label..."
                className="w-full px-2 py-1 bg-slate-700 border border-slate-500 rounded text-white text-xs focus:outline-none focus:border-purple-500"
              />
            )}
            <div className="flex gap-1 pt-0.5">
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex-1 flex items-center justify-center gap-1 py-1 bg-green-700 hover:bg-green-600 disabled:opacity-50 text-white text-xs rounded transition"
              >
                {saving ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
                Save
              </button>
              <button onClick={() => { setEditing(false); setLabel(sample.label); setUseCustom(false); }}
                className="px-2 py-1 bg-slate-700 hover:bg-slate-600 text-gray-400 text-xs rounded transition">
                <X className="w-3 h-3" />
              </button>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-between gap-1">
            <span className="text-xs font-medium text-purple-300 truncate flex items-center gap-1">
              <Tag className="w-3 h-3 flex-shrink-0" />{sample.label}
            </span>
            <button onClick={() => { setEditing(true); setLabel(sample.label); }}
              className="p-1 text-gray-500 hover:text-white hover:bg-slate-700 rounded transition flex-shrink-0">
              <Edit2 className="w-3 h-3" />
            </button>
          </div>
        )}

        <div className="flex items-center justify-between mt-1 pt-1 border-t border-slate-700/50">
          <span className="text-[10px] text-gray-500 uppercase tracking-wider">Split</span>
          <select
            value={sample.split || 'unassigned'}
            onChange={(e) => onSplitChange(sample.id, e.target.value)}
            className={`text-[10px] px-1.5 py-0.5 rounded font-medium focus:outline-none cursor-pointer ${sample.split === 'train' ? 'bg-blue-900/40 text-blue-400' :
              sample.split === 'val' ? 'bg-amber-900/40 text-amber-400' :
                sample.split === 'test' ? 'bg-green-900/40 text-green-400' :
                  'bg-slate-700/50 text-gray-400'
              }`}
          >
            <option value="unassigned">Unassigned</option>
            <option value="train">Train</option>
            <option value="val">Val</option>
            <option value="test">Test</option>
          </select>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Class Manager Panel Component
// ============================================================================

interface ClassManagerProps {
  datasetId: string;
  onChanged: () => void;
}

function ClassManager({ datasetId, onChanged }: ClassManagerProps) {
  const { request, apiClient } = useAPI();
  const [labels, setLabels] = useState<string[]>([]);
  const [sampleCounts, setSampleCounts] = useState<Record<string, number>>({});
  const [newClass, setNewClass] = useState('');
  const [adding, setAdding] = useState(false);
  const [editingLabel, setEditingLabel] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    const [labelsRaw, samplesRaw] = await Promise.all([
      request(() => apiClient.getDatasetLabels(datasetId)),
      request(() => apiClient.listSamples(datasetId)),
    ]);
    if (labelsRaw?.labels) setLabels(labelsRaw.labels);
    if (samplesRaw?.samples) {
      const counts: Record<string, number> = {};
      for (const s of samplesRaw.samples) {
        counts[s.label] = (counts[s.label] ?? 0) + 1;
      }
      setSampleCounts(counts);
    }
    setLoading(false);
  }, [datasetId, request, apiClient]);

  useEffect(() => { refresh(); }, [refresh]);

  const handleAddClass = async () => {
    const trimmed = newClass.trim();
    if (!trimmed) return;
    setAdding(true);
    await request(() => apiClient.addLabel(datasetId, trimmed));
    setNewClass('');
    setAdding(false);
    await refresh();
    onChanged();
  };

  const handleRenameLabel = async (oldLabel: string) => {
    const trimmed = editValue.trim();
    if (!trimmed || trimmed === oldLabel) { setEditingLabel(null); return; }
    await request(() => apiClient.renameLabel(datasetId, oldLabel, trimmed));
    setEditingLabel(null);
    await refresh();
    onChanged();
  };

  const handleDeleteLabel = async (label: string) => {
    const count = sampleCounts[label] ?? 0;
    const msg = count > 0
      ? `Delete class "${label}" and its ${count} sample${count !== 1 ? 's' : ''}?`
      : `Delete empty class "${label}"?`;
    if (!window.confirm(msg)) return;
    await request(() => apiClient.deleteLabel(datasetId, label));
    await refresh();
    onChanged();
  };

  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-400 font-medium flex items-center gap-1.5">
        <Tags className="w-3.5 h-3.5" /> Classes ({labels.length})
      </p>

      {/* Add class */}
      <div className="flex gap-2">
        <input
          type="text"
          value={newClass}
          onChange={e => setNewClass(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleAddClass()}
          placeholder="New class name..."
          className="flex-1 px-3 py-1.5 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-purple-500"
        />
        <button
          onClick={handleAddClass}
          disabled={adding || !newClass.trim()}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-700 hover:bg-purple-600 disabled:opacity-50 text-white text-sm font-semibold rounded-lg transition"
        >
          {adding ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <FolderPlus className="w-3.5 h-3.5" />}
          Add
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-4 gap-2 text-gray-400 text-xs">
          <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Loading...
        </div>
      ) : labels.length === 0 ? (
        <p className="text-xs text-gray-500 text-center py-3">No classes yet.</p>
      ) : (
        <div className="space-y-1.5 max-h-48 overflow-y-auto custom-scrollbar">
          {labels.map(label => (
            <div key={label} className="flex items-center gap-2 p-2 bg-slate-900/50 rounded-lg border border-slate-700 group">
              {editingLabel === label ? (
                <>
                  <input
                    autoFocus
                    value={editValue}
                    onChange={e => setEditValue(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') handleRenameLabel(label); if (e.key === 'Escape') setEditingLabel(null); }}
                    className="flex-1 px-2 py-0.5 bg-slate-700 border border-slate-500 rounded text-white text-xs focus:outline-none focus:border-purple-500"
                  />
                  <button onClick={() => handleRenameLabel(label)} className="text-green-400 hover:text-green-300 p-0.5">
                    <Check className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={() => setEditingLabel(null)} className="text-gray-400 hover:text-gray-200 p-0.5">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </>
              ) : (
                <>
                  <Tag className="w-3.5 h-3.5 text-purple-400 flex-shrink-0" />
                  <span className="flex-1 text-white text-xs font-medium truncate">{label}</span>
                  <span className="text-xs text-gray-500">{sampleCounts[label] ?? 0} samples</span>
                  <button
                    onClick={() => { setEditingLabel(label); setEditValue(label); }}
                    className="p-0.5 text-gray-500 hover:text-white opacity-0 group-hover:opacity-100 transition"
                    title="Rename class"
                  >
                    <Edit2 className="w-3 h-3" />
                  </button>
                  <button
                    onClick={() => handleDeleteLabel(label)}
                    className="p-0.5 text-red-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition"
                    title="Delete class"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Resumable Multi-Part Chunk Upload Controller
// ============================================================================

interface ChunkUploadStudioProps {
  datasetId: string;
  taskType: string;
  onUploadSuccess: () => void;
}

function ChunkUploadStudio({ datasetId, taskType, onUploadSuccess }: ChunkUploadStudioProps) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState('');
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const CHUNK_SIZE = 2 * 1024 * 1024; // 2MB Chunks
  const MAX_CONCURRENT_UPLOADS = 3;

  const handleZipUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setProgress(0);
    setError(null);
    setStatusText('Initializing secure remote multi-part upload stream context...');

    try {
      const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

      const initRes = await fetch('http://127.0.0.1:8000/api/datasets/upload_zip/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          task: taskType,
          filename: file.name,
          total_chunks: totalChunks,
          file_size: file.size
        })
      });

      const initData = await initRes.json();
      if (initData.status !== 'success') throw new Error(initData.message || 'Initialization failed');
      const { upload_id } = initData;

      setStatusText(`Streaming blocks... 0 of ${totalChunks} uploaded`);
      let completedChunks = 0;
      let activeUploads = 0;
      let currentChunkIndex = 0;

      await new Promise<void>((resolve, reject) => {
        const uploadNext = () => {
          if (currentChunkIndex >= totalChunks && activeUploads === 0) {
            resolve();
            return;
          }

          while (activeUploads < MAX_CONCURRENT_UPLOADS && currentChunkIndex < totalChunks) {
            const index = currentChunkIndex;
            currentChunkIndex++;
            activeUploads++;

            const start = index * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, file.size);
            const chunkSlice = file.slice(start, end);

            const reader = new FileReader();
            reader.onload = async (event) => {
              try {
                const arrayBuffer = event.target?.result as ArrayBuffer;

                const chunkRes = await fetch(`http://127.0.0.1:8000/api/datasets/upload_zip/chunk/${upload_id}/${index}`, {
                  method: 'PUT',
                  headers: { 'Content-Type': 'application/octet-stream' },
                  body: arrayBuffer
                });

                if (!chunkRes.ok) throw new Error(`Chunk error on index sequence: ${index}`);

                completedChunks++;
                setProgress(Math.round((completedChunks / totalChunks) * 100));
                setStatusText(`Streaming binary array blocks... ${completedChunks} of ${totalChunks} complete`);

                activeUploads--;
                uploadNext();
              } catch (err: any) {
                reject(err);
              }
            };
            reader.onerror = () => reject(new Error("Local file slicing disk access breach."));
            reader.readAsArrayBuffer(chunkSlice);
          }
        };

        uploadNext();
      });

      setStatusText('Reassembling multi-gigabyte disk buffer and running database unpack pipeline...');
      const finalizeRes = await fetch('http://127.0.0.1:8000/api/datasets/upload_zip/finalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          upload_id,
          dataset_id: datasetId,
          task: taskType,
          total_chunks: totalChunks
        })
      });

      const finalizeData = await finalizeRes.json();
      if (!finalizeRes.ok) throw new Error(finalizeData.detail || 'Post-stream payload unpack execution crash');

      setStatusText(`Successfully compiled data structural mappings! Imported ${finalizeData.count} samples.`);
      if (fileInputRef.current) fileInputRef.current.value = '';
      onUploadSuccess();

    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Stream processing framework pipeline crashed.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="p-4 bg-slate-900 rounded-xl border border-slate-700/60 mt-2">
      <div className="flex items-center gap-2 mb-4">
        <Upload size={18} className="text-indigo-400" />
        <h4 className="text-sm font-semibold text-slate-200">Resumable Gigabyte ZIP Streamer</h4>
      </div>

      {!uploading && (
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border border-dashed border-slate-700 hover:border-indigo-500 cursor-pointer rounded-lg p-5 text-center bg-slate-950/20 transition group"
        >
          <Upload className="mx-auto mb-2 text-slate-500 group-hover:text-indigo-400 transition" size={24} />
          <span className="block text-xs font-medium mb-0.5">Select ZIP Package</span>
          <span className="text-[10px] text-slate-500">Supports massive datasets up to 10GB+ smoothly</span>
          <input type="file" ref={fileInputRef} onChange={handleZipUpload} accept=".zip" className="hidden" />
        </div>
      )}

      {uploading && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs font-medium text-slate-300">
            <span className="flex items-center gap-1.5">
              <Loader2 className="animate-spin text-indigo-400" size={14} />
              Streaming ArrayBuffers...
            </span>
            <span>{progress}%</span>
          </div>
          <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-indigo-500 transition-all duration-300" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {statusText && (
        <div className="mt-3 p-2 bg-slate-950/60 rounded border border-slate-800 text-[11px] font-mono text-slate-300 flex items-center gap-2">
          {!error && !uploading ? <Check className="text-emerald-400 shrink-0" size={14} /> : <Loader2 className="animate-spin text-indigo-400 shrink-0" size={14} />}
          <span>{statusText}</span>
        </div>
      )}

      {error && (
        <div className="mt-3 p-2 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-[11px] rounded flex gap-2">
          <AlertTriangle size={14} className="shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Dataset Explorer Modal Component
// ============================================================================

interface ExplorerProps {
  dataset: DatasetInfo;
  apiBase: string;
  onClose: () => void;
  onChanged: () => void;
}

function DatasetExplorer({ dataset, apiBase, onClose, onChanged }: ExplorerProps) {
  const { request, apiClient } = useAPI();
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [allLabels, setAllLabels] = useState<string[]>([]);
  const [filterLabel, setFilterLabel] = useState<string>('ALL');
  const [filterSplit, setFilterSplit] = useState<string>('ALL');
  const [isLoading, setIsLoading] = useState(true);
  const [showClassManager, setShowClassManager] = useState(false);

  // Split Control
  const [trainPct, setTrainPct] = useState(70);
  const [valPct, setValPct] = useState(20);
  const [testPct, setTestPct] = useState(10);
  const [isSplitting, setIsSplitting] = useState(false);

  const fetchAll = useCallback(async () => {
    setIsLoading(true);
    const [rawSamples, rawLabels] = await Promise.all([
      request(() => apiClient.listSamples(dataset.id)),
      request(() => apiClient.getDatasetLabels(dataset.id)),
    ]);
    if (rawSamples?.samples) setSamples(rawSamples.samples);
    if (rawLabels?.labels) setAllLabels(rawLabels.labels);
    setIsLoading(false);
  }, [dataset.id, request, apiClient]);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const handleRelabel = async (sampleId: string, newLabel: string) => {
    await request(() => apiClient.relabelSample(sampleId, newLabel));
    await fetchAll();
    onChanged();
  };

  const handleSplitChange = async (sampleId: string, newSplit: string) => {
    await request(() => apiClient.setSampleSplit(sampleId, newSplit));
    setSamples(prev => prev.map(s => s.id === sampleId ? { ...s, split: newSplit as any } : s));
    onChanged();
  };

  const handleAutoSplit = async () => {
    if (trainPct + valPct + testPct !== 100) {
      alert("Percentages must sum to exactly 100");
      return;
    }
    setIsSplitting(true);
    await request(() => apiClient.autoSplitDataset(dataset.id, trainPct, valPct, testPct));
    await fetchAll();
    setIsSplitting(false);
    onChanged();
  };

  const handleDelete = async (sampleId: string) => {
    await request(() => apiClient.deleteSample(sampleId));
    setSamples(prev => prev.filter(s => s.id !== sampleId));
    onChanged();
  };

  const visible = samples.filter(s => {
    const passLabel = filterLabel === 'ALL' || s.label === filterLabel;
    const passSplit = filterSplit === 'ALL' || (s.split || 'unassigned') === filterSplit;
    return passLabel && passSplit;
  });

  const countByLabel = allLabels.reduce<Record<string, number>>((acc, l) => {
    acc[l] = samples.filter(s => s.label === l).length;
    return acc;
  }, {});

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div className="w-full max-w-5xl max-h-[90vh] flex flex-col bg-slate-900 rounded-2xl border border-slate-700 shadow-2xl overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700 bg-slate-800/50">
          <div className="flex items-center gap-3">
            <Database className="w-5 h-5 text-purple-400" />
            <div>
              <h2 className="text-lg font-bold text-white">{dataset.name}</h2>
              <p className="text-xs text-gray-400">{samples.length} samples, {allLabels.length} classes</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowClassManager(v => !v)}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg border transition ${showClassManager
                ? 'bg-purple-700 border-purple-500 text-white'
                : 'bg-slate-700 border-slate-600 text-gray-300 hover:bg-slate-600'
                }`}
            >
              <Tags className="w-4 h-4" /> Manage Classes
            </button>
            <button onClick={onClose} className="p-2 text-gray-400 hover:text-white hover:bg-slate-700 rounded-lg transition">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Class manager panel */}
        {showClassManager && (
          <div className="px-6 py-4 border-b border-slate-700 bg-slate-800/20">
            <ClassManager
              datasetId={dataset.id}
              onChanged={() => { fetchAll(); onChanged(); }}
            />
          </div>
        )}

        {/* Auto Split controller */}
        <div className="px-6 py-3 border-b border-slate-700 bg-slate-800/20 flex flex-wrap gap-4 items-end">
          <div className="flex-1 space-y-1">
            <label className="text-xs text-gray-400 font-medium">Auto-Split (Train/Val/Test)</label>
            <div className="flex items-center gap-2">
              <input type="number" min="0" max="100" value={trainPct} onChange={e => setTrainPct(Number(e.target.value))} className="w-16 px-2 py-1 bg-slate-900 border border-slate-600 rounded text-xs text-white" />
              <span className="text-gray-500">%</span>
              <input type="number" min="0" max="100" value={valPct} onChange={e => setValPct(Number(e.target.value))} className="w-16 px-2 py-1 bg-slate-900 border border-slate-600 rounded text-xs text-white" />
              <span className="text-gray-500">%</span>
              <input type="number" min="0" max="100" value={testPct} onChange={e => setTestPct(Number(e.target.value))} className="w-16 px-2 py-1 bg-slate-900 border border-slate-600 rounded text-xs text-white" />
              <span className="text-gray-500">%</span>
            </div>
            {trainPct + valPct + testPct !== 100 && (
              <p className="text-[10px] text-red-400">Sum must be equal to 100%</p>
            )}
          </div>
          <button
            onClick={handleAutoSplit}
            disabled={isSplitting || trainPct + valPct + testPct !== 100 || samples.length === 0}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition flex items-center gap-2"
          >
            {isSplitting ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <FolderPlus className="w-3.5 h-3.5" />}
            Generate Split
          </button>
        </div>

        {/* Filter bars */}
        <div className="flex flex-col border-b border-slate-700 bg-slate-800/30 flex-shrink-0">
          <div className="flex gap-2 px-6 py-2 overflow-x-auto border-b border-slate-700/50">
            <span className="text-xs text-gray-500 py-1 mr-2 flex-shrink-0">Filter Label:</span>
            <button
              onClick={() => setFilterLabel('ALL')}
              className={`px-3 py-1 rounded-full text-[10px] font-semibold uppercase tracking-wider whitespace-nowrap transition ${filterLabel === 'ALL' ? 'bg-purple-600 text-white' : 'bg-slate-700 text-gray-300 hover:bg-slate-600'}`}
            >
              All ({samples.length})
            </button>
            {allLabels.map(l => (
              <button
                key={l}
                onClick={() => setFilterLabel(l)}
                className={`px-3 py-1 rounded-full text-[10px] font-semibold uppercase tracking-wider whitespace-nowrap transition ${filterLabel === l ? 'bg-purple-600 text-white' : 'bg-slate-700 text-gray-300 hover:bg-slate-600'}`}
              >
                {l} ({countByLabel[l] ?? 0})
              </button>
            ))}
          </div>

          <div className="flex gap-2 px-6 py-2 overflow-x-auto">
            <span className="text-xs text-gray-500 py-1 mr-2 flex-shrink-0">Filter Split:</span>
            {['ALL', 'train', 'val', 'test', 'unassigned'].map(sp => (
              <button
                key={sp}
                onClick={() => setFilterSplit(sp)}
                className={`px-3 py-1 rounded-full text-[10px] font-semibold uppercase tracking-wider whitespace-nowrap transition ${filterSplit === sp ? 'bg-blue-600 text-white' : 'bg-slate-700 text-gray-300 hover:bg-slate-600'}`}
              >
                {sp}
              </button>
            ))}
          </div>
        </div>

        {/* Grid */}
        <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
          {isLoading ? (
            <div className="flex items-center justify-center py-20 gap-2 text-gray-400">
              <RefreshCw className="w-5 h-5 animate-spin" /> Loading samples...
            </div>
          ) : visible.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 opacity-40">
              <ImageIcon className="w-12 h-12 text-gray-500 mb-3" />
              <p className="text-gray-400">No samples found for this filter</p>
            </div>
          ) : (
            <>
              {/* Imbalance warning */}
              {allLabels.length > 1 && (() => {
                const counts = allLabels.map(l => countByLabel[l] ?? 0);
                const min = Math.min(...counts), max = Math.max(...counts);
                return max > 0 && min / max < 0.6 ? (
                  <div className="mb-4 flex items-start gap-2 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg text-xs text-yellow-300">
                    <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    Class imbalance detected, some labels have significantly fewer samples. This may affect training accuracy.
                  </div>
                ) : null;
              })()}

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
                {visible.map(s => (
                  <SampleCard
                    key={s.id}
                    sample={s}
                    allLabels={allLabels}
                    apiBase={apiBase}
                    onRelabel={handleRelabel}
                    onSplitChange={handleSplitChange}
                    onDelete={handleDelete}
                  />
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Main DatasetManager Component
// ============================================================================

export function DatasetManager({ task, onDatasetChanged }: DatasetManagerProps) {
  const { request, apiClient, error } = useAPI();
  const apiBase = 'http://localhost:8000/api';

  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [exploringDataset, setExploringDataset] = useState<DatasetInfo | null>(null);
  const [newName, setNewName] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [expandedUpload, setExpandedUpload] = useState<string | null>(null);
  const [expandedClasses, setExpandedClasses] = useState<string | null>(null);
  const [exportingId, setExportingId] = useState<string | null>(null);

  const fetchDatasets = useCallback(async () => {
    setIsLoading(true);
    const raw = await request(() => apiClient.listDatasets(task));
    setIsLoading(false);
    if (raw?.datasets) setDatasets(raw.datasets);
  }, [task, request, apiClient]);

  useEffect(() => {
    fetchDatasets();
    setSelectedId(null);
    setExpandedUpload(null);
    setExpandedClasses(null);
  }, [task, fetchDatasets]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setIsCreating(true);
    const raw = await request(() => apiClient.createDataset(newName.trim(), task));
    setIsCreating(false);
    if (raw?.dataset) {
      setNewName('');
      await fetchDatasets();
      setExpandedUpload(raw.dataset.id);
      onDatasetChanged?.();
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Delete this dataset and all its samples permanently?')) return;
    await request(() => apiClient.deleteDataset(id));
    if (selectedId === id) setSelectedId(null);
    if (expandedUpload === id) setExpandedUpload(null);
    if (expandedClasses === id) setExpandedClasses(null);
    await fetchDatasets();
    onDatasetChanged?.();
  };

  const handleClear = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Clear all samples from this dataset?')) return;
    await request(() => apiClient.clearDataset(id));
    await fetchDatasets();
    onDatasetChanged?.();
  };

  const handleRename = async (id: string) => {
    if (!editName.trim()) return;
    await request(() => apiClient.renameDataset(id, editName.trim()));
    setEditingId(null);
    await fetchDatasets();
  };

  const handleExportFull = async (dataset: DatasetInfo, e: React.MouseEvent) => {
    e.stopPropagation();
    setExportingId(`full-${dataset.id}`);
    try {
      // Force direct browser download standard via URL redirection if dynamic blob compilation finishes
      window.location.href = `${apiBase}/datasets/export/full/${dataset.id}`;
    } catch (err) {
      console.error(err);
    } finally {
      setExportingId(null);
    }
  };

  const handleExportSplit = async (dataset: DatasetInfo, e: React.MouseEvent) => {
    e.stopPropagation();
    setExportingId(`split-${dataset.id}`);
    try {
      window.location.href = `${apiBase}/datasets/export/split/${dataset.id}`;
    } catch (err) {
      console.error(err);
    } finally {
      setExportingId(null);
    }
  };

  const toggleUpload = (id: string) => {
    setExpandedUpload(prev => prev === id ? null : id);
  };

  const toggleClasses = (id: string) => {
    setExpandedClasses(prev => prev === id ? null : id);
  };

  return (
    <div className="space-y-6">
      {/* Explorer modal */}
      {exploringDataset && (
        <DatasetExplorer
          dataset={exploringDataset}
          apiBase={apiBase}
          onClose={() => setExploringDataset(null)}
          onChanged={() => { fetchDatasets(); onDatasetChanged?.(); }}
        />
      )}

      {/* Create New Dataset */}
      <div className="flex gap-3">
        <input
          type="text"
          value={newName}
          onChange={e => setNewName(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleCreate()}
          placeholder={`New ${task.replace(/_/g, ' ').toLowerCase()} dataset...`}
          className="flex-1 px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
        />
        <button
          onClick={handleCreate}
          disabled={isCreating || !newName.trim()}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white font-semibold rounded-lg transition"
        >
          {isCreating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
          Create
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-300 text-sm">{error}</div>
      )}

      {/* Dataset List */}
      {isLoading ? (
        <div className="text-center py-8 text-gray-400 flex items-center justify-center gap-2">
          <RefreshCw className="w-4 h-4 animate-spin" /> Loading datasets...
        </div>
      ) : datasets.length === 0 ? (
        <div className="text-center py-12 opacity-50">
          <Database className="w-12 h-12 text-gray-500 mx-auto mb-3" />
          <p className="text-gray-400">No datasets yet. Create one above.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {datasets.map(dataset => (
            <div key={dataset.id} className="rounded-xl border border-slate-700 overflow-hidden bg-slate-800/50">

              {/* Dataset row */}
              <div className="flex items-center gap-3 px-4 py-3">
                {editingId === dataset.id ? (
                  <div className="flex gap-2 flex-1">
                    <input
                      autoFocus
                      value={editName}
                      onChange={e => setEditName(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && handleRename(dataset.id)}
                      className="flex-1 px-2 py-1 bg-slate-700 rounded text-white text-sm focus:outline-none focus:border-purple-500 border border-slate-600"
                    />
                    <button onClick={() => handleRename(dataset.id)} className="text-green-400 hover:text-green-300 p-1">
                      <Check className="w-4 h-4" />
                    </button>
                    <button onClick={() => setEditingId(null)} className="text-gray-400 hover:text-gray-200 p-1">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-white truncate">{dataset.name}</p>
                      <p className="text-xs text-gray-400">{dataset.sample_count} samples</p>
                    </div>

                    {/* Action buttons */}
                    <div className="flex items-center gap-1 shrink-0">
                      {/* Explore */}
                      <button
                        onClick={() => setExploringDataset(dataset)}
                        className="flex items-center gap-1 px-2 py-1 text-xs text-cyan-400 hover:text-cyan-300 hover:bg-slate-700 rounded-lg transition"
                        title="Explore samples"
                      >
                        <Eye className="w-3.5 h-3.5" /> Explore
                      </button>

                      {/* Export Full */}
                      <button
                        onClick={e => handleExportFull(dataset, e)}
                        disabled={exportingId === `full-${dataset.id}` || dataset.sample_count === 0}
                        className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-emerald-400 hover:text-emerald-300 hover:bg-slate-700 disabled:opacity-40 rounded-lg transition"
                        title={dataset.sample_count === 0 ? 'No samples to export' : 'Download Full ZIP'}
                      >
                        {exportingId === `full-${dataset.id}`
                          ? <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          : <Download className="w-3.5 h-3.5" />}
                        Full ZIP
                      </button>

                      {/* Export Split */}
                      <button
                        onClick={e => handleExportSplit(dataset, e)}
                        disabled={exportingId === `split-${dataset.id}` || dataset.sample_count === 0}
                        className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-emerald-400 hover:text-emerald-300 hover:bg-slate-700 disabled:opacity-40 rounded-lg transition"
                        title={dataset.sample_count === 0 ? 'No samples to export' : 'Download Split ZIP'}
                      >
                        {exportingId === `split-${dataset.id}`
                          ? <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          : <Download className="w-3.5 h-3.5" />}
                        Split ZIP
                      </button>

                      <div className="h-4 w-px bg-slate-700 mx-1" />

                      {/* Expand Classes */}
                      <button
                        onClick={() => toggleClasses(dataset.id)}
                        className={`p-1.5 rounded-lg transition ${expandedClasses === dataset.id ? 'bg-slate-700 text-white' : 'text-gray-400 hover:text-white'}`}
                        title="Quick Classes"
                      >
                        <Tags className="w-4 h-4" />
                      </button>

                      {/* Expand Upload */}
                      <button
                        onClick={() => toggleUpload(dataset.id)}
                        className={`p-1.5 rounded-lg transition ${expandedUpload === dataset.id ? 'bg-slate-700 text-white' : 'text-gray-400 hover:text-white'}`}
                        title="Upload Archive Stream"
                      >
                        <Upload className="w-4 h-4" />
                      </button>

                      {/* Edit Name */}
                      <button
                        onClick={() => { setEditingId(dataset.id); setEditName(dataset.name); }}
                        className="p-1.5 text-gray-400 hover:text-white transition"
                        title="Rename"
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>

                      {/* Clear */}
                      <button
                        onClick={e => handleClear(dataset.id, e)}
                        className="p-1.5 text-amber-500 hover:text-amber-400 transition"
                        title="Clear Samples"
                      >
                        <RefreshCw className="w-4 h-4" />
                      </button>

                      {/* Delete */}
                      <button
                        onClick={e => handleDelete(dataset.id, e)}
                        className="p-1.5 text-red-500 hover:text-red-400 transition"
                        title="Delete Dataset"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </>
                )}
              </div>

              {/* Collapsible Section for Classes */}
              {expandedClasses === dataset.id && (
                <div className="px-4 pb-4 pt-2 border-t border-slate-700 bg-slate-900/20">
                  <ClassManager
                    datasetId={dataset.id}
                    onChanged={() => { fetchDatasets(); onDatasetChanged?.(); }}
                  />
                </div>
              )}

              {/* Collapsible Section for Resumable Upload Streamer */}
              {expandedUpload === dataset.id && (
                <div className="px-4 pb-4 pt-2 border-t border-slate-700 bg-slate-900/20">
                  <ChunkUploadStudio
                    datasetId={dataset.id}
                    taskType={task}
                    onUploadSuccess={() => { fetchDatasets(); onDatasetChanged?.(); }}
                  />
                  <div className="mt-4 pt-4 border-t border-slate-800">
                    <p className="text-xs font-semibold text-slate-400 mb-2">Alternative Single-File Record Target:</p>
                    <DataCollector
                      datasetId={dataset.id}
                      task={task}
                      onSampleAdded={() => { fetchDatasets(); onDatasetChanged?.(); }}
                    />
                  </div>
                </div>
              )}

            </div>
          ))}
        </div>
      )}
    </div>
  );
}