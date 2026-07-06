import { useEffect, useState } from 'react';
import { Database, FolderHeart, Activity, CheckCircle2, XCircle, BrainCircuit, Box, HardDrive, ChevronDown, FileType } from 'lucide-react';
import { DatasetStatistics } from '../types';
import { useAppContext } from '../context/AppContext';
import { useAPI } from '../hooks/useAPI';

interface DashboardOverviewProps {
  stats: DatasetStatistics;
  isHealthy: boolean;
}

function formatBytes(n: number): string {
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = n;
  let i = 0;
  while (size >= 1024 && i < units.length - 1) { size /= 1024; i++; }
  return `${size.toFixed(1)} ${units[i]}`;
}

export function DashboardOverview({ stats, isHealthy }: DashboardOverviewProps) {
  const { state } = useAppContext();
  const { apiClient, request } = useAPI();
  const labels = Object.entries(stats.by_label || {});
  const tasks = Object.entries(stats.by_task || {});

  const modelsCount = state.trainedModels.length;
  const recentModel = state.trainedModels.length > 0 ? state.trainedModels[state.trainedModels.length - 1] : null;

  const [storageOverview, setStorageOverview] = useState<any | null>(null);
  const [storageLoading, setStorageLoading] = useState(false);
  const [showStorageDetails, setShowStorageDetails] = useState(false);

  useEffect(() => {
    setStorageLoading(true);
    request(() => apiClient.getStorageOverview())
      .then((res: any) => { if (res && res.overview) setStorageOverview(res.overview); })
      .finally(() => setStorageLoading(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Top Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-6 rounded-xl border border-slate-700 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-gray-400 font-medium">System Status</h3>
            <Activity className={`w-5 h-5 ${isHealthy ? 'text-green-400' : 'text-red-400'}`} />
          </div>
          <div className="flex items-center gap-2">
            {isHealthy ? (
              <><CheckCircle2 className="w-6 h-6 text-green-500" /><span className="text-2xl font-bold text-white">Online</span></>
            ) : (
              <><XCircle className="w-6 h-6 text-red-500" /><span className="text-2xl font-bold text-white">Offline</span></>
            )}
          </div>
          <p className="text-sm text-gray-500 mt-2">API Connection</p>
        </div>

        <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-6 rounded-xl border border-slate-700 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-gray-400 font-medium">DB Samples</h3>
            <Database className="w-5 h-5 text-purple-400" />
          </div>
          <p className="text-3xl font-bold text-white">{stats.total_samples}</p>
          <p className="text-sm text-gray-500 mt-2">Ready for training</p>
        </div>

        <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-6 rounded-xl border border-slate-700 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-gray-400 font-medium">Unique Labels</h3>
            <FolderHeart className="w-5 h-5 text-pink-400" />
          </div>
          <p className="text-3xl font-bold text-white">{labels.length}</p>
          <p className="text-sm text-gray-500 mt-2">Categories identified</p>
        </div>

        <div className="bg-gradient-to-br from-blue-900/40 to-slate-900 p-6 rounded-xl border border-blue-500/30 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-blue-300 font-medium">Trained Models</h3>
            <BrainCircuit className="w-5 h-5 text-blue-400" />
          </div>
          <p className="text-3xl font-bold text-white">{modelsCount}</p>
          <p className="text-sm text-blue-400/70 mt-2">Locally cached models</p>
        </div>
      </div>

      {/* Overview Analytics Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Label Distribution */}
        <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 col-span-1">
          <h3 className="text-lg font-semibold text-white mb-4 border-b border-slate-700 pb-2">Distribution by Label</h3>
          {labels.length > 0 ? (
            <div className="space-y-4 max-h-[40vh] overflow-y-scroll">
              {labels.map(([label, count]) => (
                <div key={label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300 font-medium">{label}</span>
                    <span className="text-purple-400">{count}</span>
                  </div>
                  <div className="w-full bg-slate-900 rounded-full h-2">
                    <div
                      className="bg-purple-500 h-2 rounded-full"
                      style={{ width: `${(Number(count) / stats.total_samples) * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500 text-center py-6">No data collected yet. Head to Data Collection to upload samples.</p>
          )}
        </div>

        {/* Global Tasks */}
        <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 col-span-1">
          <h3 className="text-lg font-semibold text-white mb-4 border-b border-slate-700 pb-2">Active Target Tasks</h3>
          {tasks.length > 0 ? (
            <div className="space-y-3">
              {tasks.map(([task, count]) => (
                <div key={task} className="flex items-center justify-between p-3 bg-slate-900/50 rounded-lg border border-slate-700/50">
                  <span className="text-sm text-gray-300">{task.replace(/_/g, ' ')}</span>
                  <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-xs font-semibold">
                    {count}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500 text-center py-6">No tasks initialized.</p>
          )}
        </div>

        {/* Model Readiness & Latest Cache */}
        <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 col-span-1">
          <h3 className="text-lg font-semibold text-white mb-4 border-b border-slate-700 pb-2">Model Readiness</h3>
          {recentModel ? (
            <div className="space-y-4">
              <div className="p-4 bg-slate-900/80 rounded-lg border border-slate-600">
                <span className="text-xs text-gray-400 block mb-1">Latest Trained:</span>
                <strong className="text-white text-sm block mb-3 truncate">{recentModel.name}</strong>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2 bg-slate-800 rounded">
                    <span className="block text-gray-500">Val Acc</span>
                    <strong className="text-cyan-400">{(recentModel.val_accuracy * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="p-2 bg-slate-800 rounded">
                    <span className="block text-gray-500">Loss</span>
                    <strong className="text-yellow-400">{recentModel.val_loss.toFixed(3)}</strong>
                  </div>
                </div>
              </div>

              <div className="p-3 bg-green-900/20 border border-green-500/30 rounded-lg flex gap-3 items-center">
                <Box className="w-5 h-5 text-green-400 flex-shrink-0" />
                <p className="text-xs text-gray-300">Models compiled successfully. Head to <strong className="text-white">Optimization</strong> to proceed with target compression.</p>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 opacity-50">
              <BrainCircuit className="w-12 h-12 text-gray-500 mb-3" />
              <p className="text-sm text-gray-500 text-center">No models trained yet.</p>
            </div>
          )}
        </div>

      </div>

      {/* Storage Overview */}
      <div className="bg-slate-800/50 rounded-xl border border-slate-700 overflow-hidden">
        <button
          onClick={() => setShowStorageDetails((v) => !v)}
          className="w-full flex items-center justify-between p-6 hover:bg-white/5 transition-colors"
        >
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-emerald-400" />
            Backend Storage Overview
          </h3>
          <div className="flex items-center gap-3">
            {storageOverview && (
              <span className="text-sm text-gray-400 font-mono">
                {storageOverview.total_storage_human} used
                {storageOverview.disk_free_human && ` · ${storageOverview.disk_free_human} free on disk`}
              </span>
            )}
            <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${showStorageDetails ? 'rotate-180' : ''}`} />
          </div>
        </button>

        {showStorageDetails && (
          <div className="px-6 pb-6 space-y-5 border-t border-slate-700 pt-5">
            {storageLoading ? (
              <p className="text-sm text-gray-500 text-center py-4">Loading storage report...</p>
            ) : !storageOverview ? (
              <p className="text-sm text-gray-500 text-center py-4">Storage overview unavailable.</p>
            ) : (
              <>
                {/* Summary cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-700">
                    <span className="text-xs text-gray-500 block mb-1">Datasets</span>
                    <span className="text-lg font-bold text-white">{storageOverview.datasets.total_human}</span>
                    <span className="text-xs text-gray-500 block">{storageOverview.datasets.total_samples} samples across {storageOverview.datasets.count} datasets</span>
                  </div>
                  <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-700">
                    <span className="text-xs text-gray-500 block mb-1">Trained Models</span>
                    <span className="text-lg font-bold text-white">{storageOverview.trained_models.total_human}</span>
                    <span className="text-xs text-gray-500 block">{storageOverview.trained_models.count} .keras files</span>
                  </div>
                  <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-700">
                    <span className="text-xs text-gray-500 block mb-1">Optimized Models</span>
                    <span className="text-lg font-bold text-white">{storageOverview.optimized_models.total_human}</span>
                    <span className="text-xs text-gray-500 block">{storageOverview.optimized_models.count} .tflite files</span>
                  </div>
                  <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-700">
                    <span className="text-xs text-gray-500 block mb-1">Training Sessions</span>
                    <span className="text-lg font-bold text-white">{storageOverview.training_sessions.total}</span>
                    <span className="text-xs text-gray-500 block">{storageOverview.training_sessions.archived} archived</span>
                  </div>
                </div>

                {/* Global file type breakdown */}
                {storageOverview.datasets.file_types && Object.keys(storageOverview.datasets.file_types).length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2 flex items-center gap-1.5">
                      <FileType className="w-3.5 h-3.5" /> File Types Across All Datasets
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(storageOverview.datasets.file_types).map(([ext, info]: [string, any]) => (
                        <span key={ext} className="text-xs px-2.5 py-1 rounded-md bg-slate-900/60 border border-slate-700 text-gray-300 font-mono">
                          {ext} <span className="text-gray-500">×{info.count}</span> <span className="text-emerald-400">{formatBytes(info.bytes)}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Per-dataset breakdown table */}
                <div>
                  <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">Per-Dataset Breakdown</h4>
                  {storageOverview.datasets.items.length === 0 ? (
                    <p className="text-sm text-gray-500">No datasets yet.</p>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="text-left text-gray-500 border-b border-slate-700">
                            <th className="pb-2 pr-3">Dataset</th>
                            <th className="pb-2 pr-3">Task</th>
                            <th className="pb-2 pr-3">Samples</th>
                            <th className="pb-2 pr-3">Size</th>
                            <th className="pb-2 pr-3">Avg/Sample</th>
                            <th className="pb-2 pr-3">File Types</th>
                            <th className="pb-2">Split (train/val/test/unassigned)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {storageOverview.datasets.items.map((d: any) => (
                            <tr key={d.id} className="border-b border-slate-800/60">
                              <td className="py-2 pr-3 text-white font-medium">{d.name}</td>
                              <td className="py-2 pr-3 text-gray-400">{d.task?.replace(/_/g, ' ')}</td>
                              <td className="py-2 pr-3 text-gray-300">{d.sample_count}</td>
                              <td className="py-2 pr-3 text-emerald-400 font-mono">{d.size_human}</td>
                              <td className="py-2 pr-3 text-gray-400 font-mono">{formatBytes(d.avg_sample_bytes)}</td>
                              <td className="py-2 pr-3 text-gray-400 font-mono">
                                {Object.entries(d.file_types).map(([ext, info]: [string, any]) => `${ext}×${info.count}`).join(', ') || '—'}
                              </td>
                              <td className="py-2 text-gray-400 font-mono">
                                {d.split_counts.train}/{d.split_counts.val}/{d.split_counts.test}/{d.split_counts.unassigned}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                <p className="text-[10px] text-gray-600">
                  Storage directory: <span className="font-mono">{storageOverview.storage_dir}</span>
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
