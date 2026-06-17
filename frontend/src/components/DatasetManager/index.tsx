import React, { useState, useEffect, useCallback } from 'react';
import {
  Plus, Trash2, RefreshCw, Database, Edit2, Check, X,
  Eye,
  Tags, Download, Upload
} from 'lucide-react';
import { useAPI } from '../../hooks/useAPI';
import { TinyMLTask, DatasetInfo } from '../../types';
import ClassManager from './ClassManager';
import DataImporter from './DataImporter';
import DatasetExplorer from './DatasetExplorer';
import { RemoteDatasetBrowser } from './RemoteDatasetBrowser';

interface DatasetManagerProps {
  task: TinyMLTask;
  onDatasetChanged?: () => void;
}

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
    await fetchDatasets(); onDatasetChanged?.();
  };

  const handleClear = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Clear all samples from this dataset?')) return;
    await request(() => apiClient.clearDataset(id));
    await fetchDatasets(); onDatasetChanged?.();
  };

  const handleRename = async (id: string) => {
    if (!editName.trim()) return;
    await request(() => apiClient.renameDataset(id, editName.trim()));
    setEditingId(null); await fetchDatasets();
  };

  const handleExportFull = async (dataset: DatasetInfo, e: React.MouseEvent) => {
    e.stopPropagation();
    setExportingId(`full-${dataset.id}`);
    try { window.location.href = `${apiBase}/datasets/export/full/${dataset.id}`; }
    catch { } finally { setExportingId(null); }
  };

  const handleExportSplit = async (dataset: DatasetInfo, e: React.MouseEvent) => {
    e.stopPropagation();
    setExportingId(`split-${dataset.id}`);
    try { window.location.href = `${apiBase}/datasets/export/split/${dataset.id}`; }
    catch { } finally { setExportingId(null); }
  };

  return (
    <div className="space-y-6">
      {exploringDataset && (
        <DatasetExplorer dataset={exploringDataset} apiBase={apiBase}
          onClose={() => setExploringDataset(null)}
          onChanged={() => { fetchDatasets(); onDatasetChanged?.(); }} />
      )}

      {/* Create New Dataset */}
      <div className="flex gap-3">
        <input type="text" value={newName} onChange={e => setNewName(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleCreate()}
          placeholder={`New ${task.replace(/_/g, ' ').toLowerCase()} dataset...`}
          className="flex-1 px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500" />
        <button onClick={handleCreate} disabled={isCreating || !newName.trim()}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white font-semibold rounded-lg transition">
          {isCreating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />} Create
        </button>
      </div>

      {error && <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-300 text-sm">{error}</div>}

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
                    <input autoFocus value={editName} onChange={e => setEditName(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && handleRename(dataset.id)}
                      className="flex-1 px-2 py-1 bg-slate-700 rounded text-white text-sm focus:outline-none focus:border-purple-500 border border-slate-600" />
                    <button onClick={() => handleRename(dataset.id)} className="text-green-400 hover:text-green-300 p-1"><Check className="w-4 h-4" /></button>
                    <button onClick={() => setEditingId(null)} className="text-gray-400 hover:text-gray-200 p-1"><X className="w-4 h-4" /></button>
                  </div>
                ) : (
                  <>
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-white truncate">{dataset.name}</p>
                      <p className="text-xs text-gray-400">{dataset.sample_count} samples</p>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <button onClick={() => setExploringDataset(dataset)}
                        className="flex items-center gap-1 px-2 py-1 text-xs text-cyan-400 hover:text-cyan-300 hover:bg-slate-700 rounded-lg transition" title="Explore samples">
                        <Eye className="w-3.5 h-3.5" /> Explore
                      </button>
                      <button onClick={e => handleExportFull(dataset, e)}
                        disabled={exportingId === `full-${dataset.id}` || dataset.sample_count === 0}
                        className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-emerald-400 hover:text-emerald-300 hover:bg-slate-700 disabled:opacity-40 rounded-lg transition" title="Download Full ZIP">
                        {exportingId === `full-${dataset.id}` ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />} Full ZIP
                      </button>
                      <button onClick={e => handleExportSplit(dataset, e)}
                        disabled={exportingId === `split-${dataset.id}` || dataset.sample_count === 0}
                        className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-emerald-400 hover:text-emerald-300 hover:bg-slate-700 disabled:opacity-40 rounded-lg transition" title="Download Split ZIP">
                        {exportingId === `split-${dataset.id}` ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />} Split ZIP
                      </button>
                      <div className="h-4 w-px bg-slate-700 mx-1" />
                      <button onClick={() => setExpandedClasses(prev => prev === dataset.id ? null : dataset.id)}
                        className={`p-1.5 rounded-lg transition ${expandedClasses === dataset.id ? 'bg-slate-700 text-white' : 'text-gray-400 hover:text-white'}`} title="Quick Classes">
                        <Tags className="w-4 h-4" />
                      </button>
                      <button onClick={() => setExpandedUpload(prev => prev === dataset.id ? null : dataset.id)}
                        className={`p-1.5 rounded-lg transition ${expandedUpload === dataset.id ? 'bg-slate-700 text-white' : 'text-gray-400 hover:text-white'}`} title="Upload & Import">
                        <Upload className="w-4 h-4" />
                      </button>
                      <button onClick={() => { setEditingId(dataset.id); setEditName(dataset.name); }}
                        className="p-1.5 text-gray-400 hover:text-white transition" title="Rename">
                        <Edit2 className="w-4 h-4" />
                      </button>
                      <button onClick={e => handleClear(dataset.id, e)}
                        className="p-1.5 text-amber-500 hover:text-amber-400 transition" title="Clear Samples">
                        <RefreshCw className="w-4 h-4" />
                      </button>
                      <button onClick={e => handleDelete(dataset.id, e)}
                        className="p-1.5 text-red-500 hover:text-red-400 transition" title="Delete Dataset">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </>
                )}
              </div>

              {/* Collapsible Classes */}
              {expandedClasses === dataset.id && (
                <div className="px-4 pb-4 pt-2 border-t border-slate-700 bg-slate-900/20">
                  <ClassManager datasetId={dataset.id} onChanged={() => { fetchDatasets(); onDatasetChanged?.(); }} />
                </div>
              )}

              {expandedClasses === dataset.id && (
                <div className="px-4 pb-4 pt-2 border-t border-slate-700 bg-slate-900/20">
                  <RemoteDatasetBrowser datasetId={dataset.id} onImportComplete={() => { fetchDatasets(); onDatasetChanged?.(); }} task={task} />
                </div>
              )}


              {/* Collapsible Upload & Import */}
              {expandedUpload === dataset.id && (
                <div className="px-4 pb-4 pt-2 border-t border-slate-700 bg-slate-900/20">
                  <DataImporter datasetId={dataset.id} task={task}
                    onImportSuccess={() => { fetchDatasets(); onDatasetChanged?.(); }} />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
