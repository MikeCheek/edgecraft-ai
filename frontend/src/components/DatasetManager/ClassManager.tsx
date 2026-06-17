import { Tags, RefreshCw, FolderPlus, Check, X, Tag, Edit2, Trash2 } from 'lucide-react';
import React, { useCallback, useEffect, useState } from 'react'
import { useAPI } from '../../hooks/useAPI';


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
      for (const s of samplesRaw.samples) counts[s.label] = (counts[s.label] ?? 0) + 1;
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
      <div className="flex gap-2">
        <input type="text" value={newClass} onChange={e => setNewClass(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleAddClass()}
          placeholder="New class name..."
          className="flex-1 px-3 py-1.5 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-purple-500" />
        <button onClick={handleAddClass} disabled={adding || !newClass.trim()}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-700 hover:bg-purple-600 disabled:opacity-50 text-white text-sm font-semibold rounded-lg transition">
          {adding ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <FolderPlus className="w-3.5 h-3.5" />} Add
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
                  <input autoFocus value={editValue} onChange={e => setEditValue(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') handleRenameLabel(label); if (e.key === 'Escape') setEditingLabel(null); }}
                    className="flex-1 px-2 py-0.5 bg-slate-700 border border-slate-500 rounded text-white text-xs focus:outline-none focus:border-purple-500" />
                  <button onClick={() => handleRenameLabel(label)} className="text-green-400 hover:text-green-300 p-0.5"><Check className="w-3.5 h-3.5" /></button>
                  <button onClick={() => setEditingLabel(null)} className="text-gray-400 hover:text-gray-200 p-0.5"><X className="w-3.5 h-3.5" /></button>
                </>
              ) : (
                <>
                  <Tag className="w-3.5 h-3.5 text-purple-400 flex-shrink-0" />
                  <span className="flex-1 text-white text-xs font-medium truncate">{label}</span>
                  <span className="text-xs text-gray-500">{sampleCounts[label] ?? 0} samples</span>
                  <button onClick={() => { setEditingLabel(label); setEditValue(label); }}
                    className="p-0.5 text-gray-500 hover:text-white opacity-0 group-hover:opacity-100 transition" title="Rename class">
                    <Edit2 className="w-3 h-3" />
                  </button>
                  <button onClick={() => handleDeleteLabel(label)}
                    className="p-0.5 text-red-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition" title="Delete class">
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

export default ClassManager