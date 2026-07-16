import { useEffect, useState, useMemo } from 'react';
import { Database, FolderHeart, Activity, CheckCircle2, XCircle, BrainCircuit, Box, HardDrive, ChevronDown, FileType, LayoutTemplate, GripVertical, Settings } from 'lucide-react';
import { GridSkeleton } from './Skeleton';
import { DatasetStatistics } from '../types';
import { useAppContext } from '../context/AppContext';
import { useAPI } from '../hooks/useAPI';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, Legend } from 'recharts';

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

// Available widgets IDs
type WidgetID = 'status-row' | 'label-dist' | 'target-tasks' | 'model-readiness' | 'training-chart' | 'storage-details';

const DEFAULT_VIEWS: Record<string, WidgetID[]> = {
  'View 1 (Default)': ['status-row', 'training-chart', 'label-dist', 'target-tasks', 'model-readiness', 'storage-details'],
  'View 2 (Data Heavy)': ['status-row', 'storage-details', 'label-dist'],
  'View 3 (Model Focus)': ['status-row', 'training-chart', 'model-readiness', 'target-tasks'],
  'View 4 (Minimal)': ['status-row', 'model-readiness']
};

export function DashboardOverview({ stats, isHealthy }: DashboardOverviewProps) {
  const { state } = useAppContext();
  const { apiClient, request } = useAPI();
  const labels = Object.entries(stats.by_label || {});
  const tasks = Object.entries(stats.by_task || {});

  const modelsCount = state.trainedModels.length;
  const recentModel = state.trainedModels.length > 0 ? state.trainedModels[state.trainedModels.length - 1] : null;

  const [storageOverview, setStorageOverview] = useState<any | null>(null);
  const [trainingSessions, setTrainingSessions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);

  // View Management State
  const [activeViewName, setActiveViewName] = useLocalStorage<string>('dashboard_active_view', 'View 1 (Default)');
  const [savedViews, setSavedViews] = useLocalStorage<Record<string, WidgetID[]>>('dashboard_saved_views', DEFAULT_VIEWS);
  const activeWidgets = savedViews[activeViewName] || DEFAULT_VIEWS['View 1 (Default)'];

  // Drag and Drop State
  const [draggedWidget, setDraggedWidget] = useState<WidgetID | null>(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      request(() => apiClient.getStorageOverview()),
      request(() => apiClient.listAllSessions(false)) // Fetch past trainings for chart
    ]).then(([storeRes, sessionRes]: any) => {
      if (storeRes && storeRes.overview) setStorageOverview(storeRes.overview);
      if (sessionRes && sessionRes.sessions) setTrainingSessions(sessionRes.sessions.reverse()); // Oldest to newest
    }).finally(() => setLoading(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Prepare chart data from training history
  const chartData = useMemo(() => {
    return trainingSessions.map((session, index) => ({
      name: `Run ${index + 1}`,
      accuracy: session.metrics?.val_accuracy ? Number((session.metrics.val_accuracy * 100).toFixed(2)) : 0,
      loss: session.metrics?.val_loss ? Number(session.metrics.val_loss.toFixed(3)) : 0,
    }));
  }, [trainingSessions]);

  // Handle Drag & Drop Logic
  const handleDragStart = (e: React.DragEvent, id: WidgetID) => {
    setDraggedWidget(id);
    e.dataTransfer.effectAllowed = "move";
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  };

  const handleDrop = (e: React.DragEvent, targetId: WidgetID) => {
    e.preventDefault();
    if (!draggedWidget || draggedWidget === targetId) return;

    const newWidgets = [...activeWidgets];
    const draggedIndex = newWidgets.indexOf(draggedWidget);
    const targetIndex = newWidgets.indexOf(targetId);

    newWidgets.splice(draggedIndex, 1);
    newWidgets.splice(targetIndex, 0, draggedWidget);

    setSavedViews({ ...savedViews, [activeViewName]: newWidgets });
    setDraggedWidget(null);
  };

  const toggleWidget = (id: WidgetID) => {
    let newWidgets = [...activeWidgets];
    if (newWidgets.includes(id)) {
      newWidgets = newWidgets.filter(w => w !== id);
    } else {
      newWidgets.push(id);
    }
    setSavedViews({ ...savedViews, [activeViewName]: newWidgets });
  };

  // --- WIDGET RENDERERS ---

  const renderStatusRow = () => (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
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
      </div>
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-6 rounded-xl border border-slate-700 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-gray-400 font-medium">DB Samples</h3>
          <Database className="w-5 h-5 text-purple-400" />
        </div>
        <p className="text-3xl font-bold text-white">{stats.total_samples}</p>
      </div>
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-6 rounded-xl border border-slate-700 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-gray-400 font-medium">Unique Labels</h3>
          <FolderHeart className="w-5 h-5 text-pink-400" />
        </div>
        <p className="text-3xl font-bold text-white">{labels.length}</p>
      </div>
      <div className="bg-gradient-to-br from-blue-900/40 to-slate-900 p-6 rounded-xl border border-blue-500/30 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-blue-300 font-medium">Trained Models</h3>
          <BrainCircuit className="w-5 h-5 text-blue-400" />
        </div>
        <p className="text-3xl font-bold text-white">{modelsCount}</p>
      </div>
    </div>
  );

  const renderTrainingChart = () => (
    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 h-80 flex flex-col">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2"><BrainCircuit className="w-5 h-5 text-purple-400" /> Training Accuracy History</h3>
      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} tickLine={false} />
            <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} domain={['auto', 'auto']} />
            <RechartsTooltip
              contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
              itemStyle={{ color: '#e2e8f0' }}
            />
            <Area type="monotone" dataKey="accuracy" stroke="#8b5cf6" strokeWidth={3} fillOpacity={1} fill="url(#colorAcc)" />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <div className="flex-1 flex items-center justify-center text-gray-500">No training data available to chart.</div>
      )}
    </div>
  );

  const renderLabelDist = () => (
    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col h-80">
      <h3 className="text-lg font-semibold text-white mb-4 border-b border-slate-700 pb-2">Distribution by Label</h3>
      {labels.length > 0 ? (
        <div className="space-y-4 flex-1 overflow-y-auto custom-scrollbar pr-2">
          {labels.map(([label, count]) => (
            <div key={label}>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-300 font-medium">{label}</span>
                <span className="text-purple-400">{count}</span>
              </div>
              <div className="w-full bg-slate-900 rounded-full h-2">
                <div className="bg-purple-500 h-2 rounded-full" style={{ width: `${(Number(count) / stats.total_samples) * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-sm text-gray-500">No data collected yet.</div>
      )}
    </div>
  );

  const renderTasks = () => (
    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col h-80">
      <h3 className="text-lg font-semibold text-white mb-4 border-b border-slate-700 pb-2">Active Target Tasks</h3>
      {tasks.length > 0 ? (
        <div className="space-y-3 flex-1 overflow-y-auto custom-scrollbar pr-2">
          {tasks.map(([task, count]) => (
            <div key={task} className="flex items-center justify-between p-3 bg-slate-900/50 rounded-lg border border-slate-700/50">
              <span className="text-sm text-gray-300">{task.replace(/_/g, ' ')}</span>
              <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-xs font-semibold">{count}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-sm text-gray-500">No tasks initialized.</div>
      )}
    </div>
  );

  const renderModelReadiness = () => (
    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col h-80">
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
        <div className="flex-1 flex flex-col items-center justify-center opacity-50">
          <BrainCircuit className="w-12 h-12 text-gray-500 mb-3" />
          <p className="text-sm text-gray-500 text-center">No models trained yet.</p>
        </div>
      )}
    </div>
  );

  const renderStorageDetails = () => (
    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 col-span-1 lg:col-span-full">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <HardDrive className="w-5 h-5 text-emerald-400" /> Storage Analytics
        </h3>
        {storageOverview && (
          <span className="text-sm text-gray-400 font-mono">
            {storageOverview.total_storage_human} used
          </span>
        )}
      </div>

      {!storageOverview ? (
        <GridSkeleton count={4} />
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-slate-900/60 rounded-lg border border-slate-700 text-center">
              <span className="text-xs text-gray-500 block mb-1">Datasets</span>
              <span className="text-xl font-bold text-white">{storageOverview.datasets.total_human}</span>
            </div>
            <div className="p-4 bg-slate-900/60 rounded-lg border border-slate-700 text-center">
              <span className="text-xs text-gray-500 block mb-1">Trained Models</span>
              <span className="text-xl font-bold text-white">{storageOverview.trained_models.total_human}</span>
            </div>
            <div className="p-4 bg-slate-900/60 rounded-lg border border-slate-700 text-center">
              <span className="text-xs text-gray-500 block mb-1">Optimized Models</span>
              <span className="text-xl font-bold text-white">{storageOverview.optimized_models.total_human}</span>
            </div>
            <div className="p-4 bg-slate-900/60 rounded-lg border border-slate-700 text-center">
              <span className="text-xs text-gray-500 block mb-1">Training Sessions</span>
              <span className="text-xl font-bold text-white">{storageOverview.training_sessions.total}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const WIDGET_MAP: Record<WidgetID, { component: () => JSX.Element, colSpan: string }> = {
    'status-row': { component: renderStatusRow, colSpan: 'col-span-1 lg:col-span-3' },
    'training-chart': { component: renderTrainingChart, colSpan: 'col-span-1 lg:col-span-3' },
    'label-dist': { component: renderLabelDist, colSpan: 'col-span-1' },
    'target-tasks': { component: renderTasks, colSpan: 'col-span-1' },
    'model-readiness': { component: renderModelReadiness, colSpan: 'col-span-1' },
    'storage-details': { component: renderStorageDetails, colSpan: 'col-span-1 lg:col-span-3' }
  };

  const WIDGET_LABELS: Record<WidgetID, string> = {
    'status-row': 'System Status Cards',
    'training-chart': 'Training History Chart',
    'label-dist': 'Label Distribution',
    'target-tasks': 'Active Tasks',
    'model-readiness': 'Model Readiness',
    'storage-details': 'Storage Details'
  };

  return (
    <div className="space-y-6 animate-fadeIn pb-12">

      {/* View Manager & Edit Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900/80 p-4 rounded-xl border border-slate-700 shadow-md">
        <div className="flex items-center gap-3 overflow-x-auto custom-scrollbar pb-1 md:pb-0">
          <LayoutTemplate className="w-5 h-5 text-gray-400" />
          {Object.keys(savedViews).map((viewName) => (
            <button
              key={viewName}
              onClick={() => setActiveViewName(viewName)}
              className={`px-4 py-1.5 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${activeViewName === viewName
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50'
                  : 'bg-slate-800 text-gray-400 hover:text-gray-200 border border-slate-700'
                }`}
            >
              {viewName}
            </button>
          ))}
        </div>
        <button
          onClick={() => setIsEditMode(!isEditMode)}
          className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${isEditMode ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50' : 'bg-slate-800 text-gray-300 border border-slate-700 hover:bg-slate-700'
            }`}
        >
          <Settings className="w-4 h-4" />
          {isEditMode ? 'Finish Editing' : 'Customize Layout'}
        </button>
      </div>

      {/* Edit Mode Panel */}
      {isEditMode && (
        <div className="bg-blue-900/10 border border-blue-500/30 p-4 rounded-xl animate-slideIn">
          <p className="text-sm text-blue-300 mb-3">Toggle visibility of widgets for <strong>{activeViewName}</strong>. Drag elements below to reorder.</p>
          <div className="flex flex-wrap gap-2">
            {(Object.keys(WIDGET_MAP) as WidgetID[]).map(id => {
              const isActive = activeWidgets.includes(id);
              return (
                <button
                  key={`toggle-${id}`}
                  onClick={() => toggleWidget(id)}
                  className={`px-3 py-1.5 rounded text-xs border transition-colors ${isActive ? 'bg-blue-500 border-blue-400 text-white' : 'bg-slate-800 border-slate-600 text-gray-400 hover:bg-slate-700'}`}
                >
                  {isActive ? '✓ ' : '+ '}{WIDGET_LABELS[id]}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Dynamic Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {activeWidgets.map((widgetId) => {
          const widget = WIDGET_MAP[widgetId];
          if (!widget) return null;

          return (
            <div
              key={widgetId}
              className={`${widget.colSpan} relative group`}
              draggable={isEditMode}
              onDragStart={(e) => handleDragStart(e, widgetId)}
              onDragOver={handleDragOver}
              onDrop={(e) => handleDrop(e, widgetId)}
            >
              {isEditMode && (
                <div className="absolute top-2 right-2 z-10 cursor-move p-2 bg-slate-900/80 backdrop-blur-sm border border-slate-600 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-gray-400 hover:text-white hover:bg-slate-800 shadow-xl">
                  <GripVertical className="w-4 h-4" />
                </div>
              )}
              <div className={`h-full ${isEditMode ? 'border-2 border-dashed border-transparent group-hover:border-blue-500/50 rounded-xl transition-all' : ''}`}>
                {widget.component()}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}