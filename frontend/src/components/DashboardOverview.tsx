import { Database, FolderHeart, Activity, CheckCircle2, XCircle, BrainCircuit, Box } from 'lucide-react';
import { DatasetStatistics } from '../types';
import { useAppContext } from '../context/AppContext';

interface DashboardOverviewProps {
  stats: DatasetStatistics;
  isHealthy: boolean;
}

export function DashboardOverview({ stats, isHealthy }: DashboardOverviewProps) {
  const { state } = useAppContext();
  const labels = Object.entries(stats.by_label || {});
  const tasks = Object.entries(stats.by_task || {});

  const modelsCount = state.trainedModels.length;
  const recentModel = state.trainedModels.length > 0 ? state.trainedModels[state.trainedModels.length - 1] : null;

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
            <div className="space-y-4">
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
    </div>
  );
}