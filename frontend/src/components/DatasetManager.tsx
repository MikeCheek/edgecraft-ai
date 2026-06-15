import { useState, useEffect, useCallback } from 'react';
import { Trash2, ImageIcon, Plus, Settings, FolderOpen, Edit2, ArchiveX } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';
import { TinyMLTask, DatasetInfo, DatasetSample } from '../types';
import { DataCollector } from './DataCollector';

interface DatasetManagerProps { task: TinyMLTask; onDatasetChanged: () => void; }

export function DatasetManager({ task, onDatasetChanged }: DatasetManagerProps) {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [newDatasetName, setNewDatasetName] = useState('');
  const { request, apiClient } = useAPI();

  const fetchDatasets = useCallback(async () => {
    const res = await request(() => apiClient.listDatasets(task));
    if (res && res.datasets) setDatasets(res.datasets);
  }, [task, request, apiClient]);

  const fetchSamples = useCallback(async (dsId: string) => {
    const res = await request(() => apiClient.listSamples(dsId));
    if (res && res.samples) setSamples(res.samples);
  }, [request, apiClient]);

  useEffect(() => { fetchDatasets(); setSelectedDatasetId(null); }, [fetchDatasets]);
  useEffect(() => { if (selectedDatasetId) fetchSamples(selectedDatasetId); }, [selectedDatasetId, fetchSamples]);

  const handleCreateDataset = async () => {
    if (!newDatasetName.trim()) return;
    await request(() => apiClient.createDataset(newDatasetName, task));
    setNewDatasetName('');
    fetchDatasets();
    onDatasetChanged();
  };

  const handleQuickAction = async (action: 'rename' | 'clear' | 'delete', id: string) => {
    if (action === 'rename') {
      const newName = prompt('Enter new dataset name:');
      if (newName) await request(() => apiClient.renameDataset(id, newName));
    } else if (action === 'clear') {
      if (confirm('Clear all samples from this dataset?')) await request(() => apiClient.clearDataset(id));
    } else if (action === 'delete') {
      if (confirm('Delete this dataset entirely?')) {
        await request(() => apiClient.deleteDataset(id));
        if (selectedDatasetId === id) setSelectedDatasetId(null);
      }
    }
    fetchDatasets();
    if (selectedDatasetId === id) fetchSamples(id);
    onDatasetChanged();
  };

  if (!selectedDatasetId) {
    return (
      <div className="space-y-6">
        <div className="flex gap-4 mb-6 p-4 bg-slate-800/50 rounded-xl border border-slate-700">
          <input type="text" value={newDatasetName} onChange={e => setNewDatasetName(e.target.value)} placeholder="New Dataset Name..." className="flex-1 px-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white" />
          <button onClick={handleCreateDataset} className="px-6 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg flex items-center gap-2"><Plus className="w-4 h-4" /> Create Dataset</button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets.map(ds => (
            <div key={ds.id} className="p-5 bg-slate-800 rounded-xl border border-slate-700 hover:border-purple-500 transition-colors group relative">
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center gap-3 cursor-pointer" onClick={() => setSelectedDatasetId(ds.id)}>
                  <FolderOpen className="w-8 h-8 text-purple-400" />
                  <div><h3 className="text-white font-semibold">{ds.name}</h3><span className="text-xs text-gray-400">{ds.sample_count || 0} samples</span></div>
                </div>
                <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button onClick={() => handleQuickAction('rename', ds.id)} title="Rename" className="p-1.5 bg-slate-700 hover:bg-blue-500 text-white rounded"><Edit2 className="w-3 h-3" /></button>
                  <button onClick={() => handleQuickAction('clear', ds.id)} title="Clear Samples" className="p-1.5 bg-slate-700 hover:bg-yellow-500 text-white rounded"><ArchiveX className="w-3 h-3" /></button>
                  <button onClick={() => handleQuickAction('delete', ds.id)} title="Delete Dataset" className="p-1.5 bg-slate-700 hover:bg-red-500 text-white rounded"><Trash2 className="w-3 h-3" /></button>
                </div>
              </div>
              <button onClick={() => setSelectedDatasetId(ds.id)} className="w-full py-2 bg-slate-700/50 hover:bg-slate-600 text-sm text-gray-300 rounded-lg">Open Dataset Space</button>
            </div>
          ))}
          {datasets.length === 0 && <p className="col-span-3 text-center text-gray-500 py-8">No datasets created for {task} yet.</p>}
        </div>
      </div>
    );
  }

  const groupedSamples = samples.reduce((acc, sample) => {
    if (!acc[sample.label]) acc[sample.label] = [];
    acc[sample.label].push(sample);
    return acc;
  }, {} as Record<string, DatasetSample[]>);

  const activeDataset = datasets.find(d => d.id === selectedDatasetId);

  return (
    <div className="space-y-6">
      <button onClick={() => setSelectedDatasetId(null)} className="text-purple-400 hover:text-purple-300 text-sm mb-4 inline-flex items-center gap-2">← Back to Datasets</button>
      <div className="flex justify-between items-center mb-6 border-b border-slate-700 pb-4">
        <h3 className="text-xl font-bold text-white flex items-center gap-2"><FolderOpen className="w-5 h-5 text-purple-400" /> {activeDataset?.name} Space</h3>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h4 className="text-md font-semibold text-gray-300 mb-4 border-b border-slate-700 pb-2">Add Data</h4>
          <DataCollector datasetId={selectedDatasetId} task={task} onSampleAdded={() => { fetchSamples(selectedDatasetId); fetchDatasets(); onDatasetChanged(); }} />
        </div>
        <div>
          <h4 className="text-md font-semibold text-gray-300 mb-4 border-b border-slate-700 pb-2">Database Overview</h4>
          <div className="space-y-6 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
            {Object.keys(groupedSamples).length === 0 ? (
              <p className="text-gray-500 text-center py-8">No samples found.</p>
            ) : (
              Object.entries(groupedSamples).map(([label, items]) => (
                <div key={label} className="space-y-2">
                  <div className="flex justify-between"><span className="text-sm font-bold text-white">{label}</span><span className="text-xs bg-slate-800 px-2 py-1 rounded text-gray-400">{items.length} samples</span></div>
                  <div className="grid grid-cols-4 gap-2">
                    {items.slice(0, 8).map(sample => (
                      <div key={sample.id} className="relative aspect-square bg-slate-800 rounded border border-slate-700 group">
                        {task.includes('IMAGE') || task === 'VISUAL_WAKE_WORDS' ? (
                          <img src={`http://localhost:8000/api/datasets/image/${sample.id}`} alt="sample" className="w-full h-full object-cover rounded" loading="lazy" />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-[10px] text-gray-500 text-center break-all p-1">{sample.filename}</div>
                        )}
                        <button onClick={async () => { await request(() => apiClient.deleteSample(sample.id)); fetchSamples(selectedDatasetId); }} className="absolute inset-0 bg-red-900/80 hidden group-hover:flex items-center justify-center text-white"><Trash2 className="w-4 h-4" /></button>
                      </div>
                    ))}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}