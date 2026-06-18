import { Database, Tags, X, RefreshCw, FolderPlus, ImageIcon, AlertTriangle, ArrowLeft, FileText, Trash2 } from "lucide-react";
import React, { useCallback, useEffect, useState } from "react";
import { useAPI } from "../../hooks/useAPI";
import { DatasetInfo, DatasetSample } from "../../types";
import ClassManager from "./ClassManager";
import SampleCard from "./SampleCard";

interface ExplorerProps {
  dataset: DatasetInfo;
  apiBase: string;
  onClose: () => void;
  onChanged: () => void;
}

// Helper to safely extract file extensions
const getExt = (filename?: string) => {
  if (!filename) return 'unknown';
  const parts = filename.split('.');
  return parts.length > 1 ? `.${parts.pop()?.toLowerCase()}` : 'unknown';
};

function DatasetExplorer({ dataset, apiBase, onClose, onChanged }: ExplorerProps) {
  const { request, apiClient } = useAPI();
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [allLabels, setAllLabels] = useState<string[]>([]);

  // View & Pagination State
  const [viewMode, setViewMode] = useState<'groups' | 'samples'>('groups');
  const [filterLabel, setFilterLabel] = useState<string>('ALL');
  const [filterSplit, setFilterSplit] = useState<string>('ALL');
  const [filterFileType, setFilterFileType] = useState<string>('ALL');
  const [activeFilterType, setActiveFilterType] = useState<'class' | 'split' | 'filetype' | null>(null);
  const [visibleCount, setVisibleCount] = useState<number>(10);

  const [isLoading, setIsLoading] = useState(true);
  const [isBulkDeleting, setIsBulkDeleting] = useState(false);
  const [showClassManager, setShowClassManager] = useState(false);

  // Auto-Split State
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

  // --- Handlers ---

  const handleRelabel = async (sampleId: string, newLabel: string) => {
    await request(() => apiClient.relabelSample(sampleId, newLabel));
    await fetchAll(); onChanged();
  };

  const handleSplitChange = async (sampleId: string, newSplit: string) => {
    await request(() => apiClient.setSampleSplit(sampleId, newSplit));
    setSamples(prev => prev.map(s => s.id === sampleId ? { ...s, split: newSplit as any } : s));
    onChanged();
  };

  const handleAutoSplit = async () => {
    if (trainPct + valPct + testPct !== 100) { alert("Percentages must sum to exactly 100"); return; }
    setIsSplitting(true);
    await request(() => apiClient.autoSplitDataset(dataset.id, trainPct, valPct, testPct));
    await fetchAll(); setIsSplitting(false); onChanged();
  };

  const handleDelete = async (sampleId: string) => {
    await request(() => apiClient.deleteSample(sampleId));
    setSamples(prev => prev.filter(s => s.id !== sampleId)); onChanged();
  };

  // Bulk Delete By File Type
  const handleDeleteByFileType = async (ext: string) => {
    const toDelete = samples.filter(s => getExt(s.filename) === ext);
    if (!window.confirm(`Are you sure you want to PERMANENTLY delete all ${toDelete.length} "${ext}" files in this dataset? This cannot be undone.`)) return;

    setIsBulkDeleting(true);
    try {
      // Sequential processing prevents saturating the connection pool
      for (const s of toDelete) {
        await request(() => apiClient.deleteSample(s.id));
      }
      await fetchAll();
      onChanged();
    } finally {
      setIsBulkDeleting(false);
    }
  };

  // Group Click Handlers
  const handleGroupClick = (type: 'class' | 'split' | 'filetype', value: string) => {
    setFilterLabel(type === 'class' ? value : 'ALL');
    setFilterSplit(type === 'split' ? value : 'ALL');
    setFilterFileType(type === 'filetype' ? value : 'ALL');

    setActiveFilterType(type);
    setVisibleCount(10);
    setViewMode('samples');
  };

  const clearFilter = () => {
    setFilterLabel('ALL');
    setFilterSplit('ALL');
    setFilterFileType('ALL');
    setActiveFilterType(null);
    setViewMode('groups');
  };

  // --- Derived Data ---

  const visible = samples.filter(s => {
    const passLabel = filterLabel === 'ALL' || s.label === filterLabel;
    const passSplit = filterSplit === 'ALL' || (s.split || 'unassigned') === filterSplit;
    const passFileType = filterFileType === 'ALL' || getExt(s.filename) === filterFileType;
    return passLabel && passSplit && passFileType;
  });

  const countByLabel = allLabels.reduce<Record<string, number>>((acc, l) => {
    acc[l] = samples.filter(s => s.label === l).length; return acc;
  }, {});

  const countBySplit = ['train', 'val', 'test', 'unassigned'].reduce<Record<string, number>>((acc, split) => {
    acc[split] = samples.filter(s => (s.split || 'unassigned') === split).length;
    return acc;
  }, {});

  const countByFileType = samples.reduce<Record<string, number>>((acc, s) => {
    const ext = getExt(s.filename);
    acc[ext] = (acc[ext] || 0) + 1;
    return acc;
  }, {});
  const fileTypes = Object.keys(countByFileType).sort();

  // Class Imbalance Check
  const counts = allLabels.map(l => countByLabel[l] ?? 0);
  const minCount = Math.min(...counts);
  const maxCount = Math.max(...counts);
  const hasImbalance = allLabels.length > 1 && maxCount > 0 && (minCount / maxCount) < 0.6;

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
            <button onClick={() => setShowClassManager(v => !v)}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg border transition ${showClassManager ? 'bg-purple-700 border-purple-500 text-white' : 'bg-slate-700 border-slate-600 text-gray-300 hover:bg-slate-600'}`}>
              <Tags className="w-4 h-4" /> Manage Classes
            </button>
            <button onClick={onClose} className="p-2 text-gray-400 hover:text-white hover:bg-slate-700 rounded-lg transition">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {showClassManager && (
          <div className="px-6 py-4 border-b border-slate-700 bg-slate-800/20">
            <ClassManager datasetId={dataset.id} onChanged={() => { fetchAll(); onChanged(); }} />
          </div>
        )}

        {/* Auto Split Toolbar */}
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
            {trainPct + valPct + testPct !== 100 && <p className="text-[10px] text-red-400">Sum must be equal to 100%</p>}
          </div>
          <button onClick={handleAutoSplit}
            disabled={isSplitting || trainPct + valPct + testPct !== 100 || samples.length === 0}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition flex items-center gap-2">
            {isSplitting ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <FolderPlus className="w-3.5 h-3.5" />} Generate Split
          </button>
        </div>

        {/* Main Content Area */}
        {isBulkDeleting ? (
          <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4 text-gray-400">
            <RefreshCw className="w-8 h-8 animate-spin text-red-400" />
            <p className="text-sm font-medium">Purging requested files...</p>
          </div>
        ) : isLoading ? (
          <div className="flex-1 flex items-center justify-center py-20 gap-2 text-gray-400">
            <RefreshCw className="w-5 h-5 animate-spin" /> Fetching indices...
          </div>
        ) : viewMode === 'groups' ? (
          // --- GROUPS OVERVIEW ---
          <div className="flex-1 overflow-y-auto p-6 custom-scrollbar space-y-8">

            {hasImbalance && (
              <div className="flex items-start gap-2 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg text-xs text-yellow-300">
                <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                Class imbalance detected. Some labels have significantly fewer samples than others. This may hurt your model's accuracy.
              </div>
            )}

            {/* File Types Section */}
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4" /> Browse by File Type
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-5 gap-4">
                {fileTypes.map(ext => (
                  <div key={ext} className="relative group">
                    <button
                      onClick={() => handleGroupClick('filetype', ext)}
                      className="w-full p-4 bg-slate-800 rounded-xl border border-slate-700 hover:border-emerald-500 text-left transition"
                    >
                      <div className="text-lg font-bold text-white group-hover:text-emerald-400 truncate uppercase">{ext}</div>
                      <div className="text-xs text-gray-500 mt-1">{countByFileType[ext] ?? 0} items</div>
                    </button>
                    {/* Hover Delete Action */}
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDeleteByFileType(ext); }}
                      className="absolute top-3 right-3 p-2 bg-slate-900/90 hover:bg-red-500 text-gray-400 hover:text-white rounded-md transition-all opacity-0 group-hover:opacity-100 shadow-md border border-slate-700 hover:border-red-500"
                      title={`Delete all ${ext} files globally`}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Splits Section */}
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-3 flex items-center gap-2">
                <FolderPlus className="w-4 h-4" /> Browse by Split
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {['train', 'val', 'test', 'unassigned'].map(sp => (
                  <button key={sp} onClick={() => handleGroupClick('split', sp)} className="p-4 bg-slate-800 rounded-xl border border-slate-700 hover:border-blue-500 text-left transition group uppercase">
                    <div className="text-lg font-bold text-white group-hover:text-blue-400 truncate">{sp}</div>
                    <div className="text-xs text-gray-500 mt-1">{countBySplit[sp] ?? 0} items</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Classes Section */}
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-3 flex items-center gap-2">
                <Tags className="w-4 h-4" /> Browse by Class
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                <button onClick={() => handleGroupClick('class', 'ALL')} className="p-4 bg-slate-800 rounded-xl border border-slate-700 hover:border-purple-500 text-left transition group">
                  <div className="text-lg font-bold text-white group-hover:text-purple-400">All Samples</div>
                  <div className="text-xs text-gray-500 mt-1">{samples.length} items</div>
                </button>
                {allLabels.map(l => (
                  <button key={l} onClick={() => handleGroupClick('class', l)} className="p-4 bg-slate-800 rounded-xl border border-slate-700 hover:border-purple-500 text-left transition group">
                    <div className="text-lg font-bold text-white group-hover:text-purple-400 truncate" title={l}>{l}</div>
                    <div className="text-xs text-gray-500 mt-1">{countByLabel[l] ?? 0} items</div>
                  </button>
                ))}
              </div>
            </div>

          </div>
        ) : (
          // --- SAMPLES DETAIL VIEW ---
          <div className="flex flex-col flex-1 min-h-0">
            {/* Context Navigation Bar */}
            <div className="flex items-center gap-4 px-6 py-3 border-b border-slate-700 bg-slate-800/30">
              <button onClick={clearFilter} className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded-lg text-xs font-semibold text-white transition flex items-center gap-1.5">
                <ArrowLeft className="w-4 h-4" /> Back to Groups
              </button>
              <div className="text-sm text-gray-300">
                Viewing: <span className="font-bold text-white uppercase tracking-wider">
                  {activeFilterType === 'class' ? filterLabel : activeFilterType === 'split' ? filterSplit : filterFileType}
                </span> ({visible.length} items)
              </div>
            </div>

            {/* Paginated Grid Area */}
            <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
              {visible.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 opacity-40">
                  <ImageIcon className="w-12 h-12 text-gray-500 mb-3" />
                  <p className="text-gray-400">No samples found for this category</p>
                </div>
              ) : (
                <>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
                    {visible.slice(0, visibleCount).map(s => (
                      <SampleCard key={s.id} sample={s} allLabels={allLabels} apiBase={apiBase}
                        onRelabel={handleRelabel} onSplitChange={handleSplitChange} onDelete={handleDelete} />
                    ))}
                  </div>

                  {visibleCount < visible.length && (
                    <div className="mt-8 mb-4 flex justify-center">
                      <button
                        onClick={() => setVisibleCount(v => v + 10)}
                        className="px-6 py-2.5 bg-slate-800 hover:bg-slate-700 border border-slate-600 rounded-lg text-sm font-semibold text-white transition flex items-center gap-2"
                      >
                        <RefreshCw className="w-4 h-4" /> Load 10 More
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default DatasetExplorer;