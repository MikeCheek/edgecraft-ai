import { useState, useEffect, useCallback } from 'react';
import { Play, Clock, History, AlertTriangle } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';
import { useAppContext } from '../context/AppContext';
import { TrainingConfig, TinyMLTask, TrainingStatus, DatasetInfo, TrainingMetrics } from '../types';

// Custom SVG Line Chart for live progress
const LineChart = ({ data, dataKey1, dataKey2, color1, color2, title }: { data: TrainingMetrics[], dataKey1: keyof TrainingMetrics, dataKey2: keyof TrainingMetrics, color1: string, color2: string, title: string }) => {
  if (!data || data.length === 0) return null;
  const height = 140;
  const width = 400;
  const padding = 10;
  const usableHeight = height - padding * 2;
  const maxVal = Math.max(
    ...data.map(d => Math.max((d[dataKey1] as number) || 0, (d[dataKey2] as number) || 0)), 1
  );
  const dx = width / Math.max(data.length - 1, 1);
  const points1 = data.map((d, i) => `${i * dx},${height - padding - (((d[dataKey1] as number) || 0) / maxVal) * usableHeight}`).join(' ');
  const points2 = data.map((d, i) => `${i * dx},${height - padding - (((d[dataKey2] as number) || (d[dataKey1] as number) || 0) / maxVal) * usableHeight}`).join(' ');

  return (
    <div className="flex-1 bg-slate-900/80 p-4 rounded-xl border border-slate-700 shadow-inner">
      <h4 className="text-sm font-semibold text-gray-300 mb-3">{title}</h4>
      <div className="relative w-full overflow-hidden" style={{ height: `${height}px` }}>
        <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="overflow-visible">
          <line x1="0" y1={padding} x2={width} y2={padding} stroke="#334155" strokeDasharray="4 4" />
          <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke="#334155" strokeDasharray="4 4" />
          <line x1="0" y1={height - padding} x2={width} y2={height - padding} stroke="#334155" strokeDasharray="4 4" />
          <polyline points={points1} fill="none" stroke={color1} strokeWidth="3" strokeLinecap="round" />
          <polyline points={points2} fill="none" stroke={color2} strokeWidth="3" strokeDasharray="6 6" strokeLinecap="round" />
        </svg>
      </div>
    </div>
  );
}

interface ModelTrainerProps { task: TinyMLTask; onTrainingComplete?: () => void; }

export function ModelTrainer({ task, onTrainingComplete }: ModelTrainerProps) {
  const { state, dispatch } = useAppContext();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [config, setConfig] = useState<TrainingConfig>({ dataset_id: '', epochs: 50, batch_size: 32, learning_rate: 0.001, base_model: 'MobileNetV2', task, validation_split: 0.2 });
  const [trainingId, setTrainingId] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [now, setNow] = useState(Date.now()); // Timer state
  const { request, error, apiClient } = useAPI();

  // Update timer every second for elapsed time tracking
  useEffect(() => {
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    request(() => apiClient.listDatasets(task)).then(res => {
      if (res?.datasets) {
        setDatasets(res.datasets);
        const validExisting = res.datasets.find((d: DatasetInfo) => d.id === config.dataset_id);
        if (!validExisting) setConfig(c => ({ ...c, dataset_id: res.datasets.length > 0 ? res.datasets[0].id : '', task }));
        else setConfig(c => ({ ...c, task }));
      }
    });
  }, [task, request, apiClient]);

  const pollStatus = useCallback(async (id: string) => {
    const raw = await request(() => apiClient.getTrainingStatus(id));
    if (!raw) return;

    const status: TrainingStatus = {
      id: raw.id, status: raw.status, current_epoch: raw.current_epoch, total_epochs: raw.total_epochs,
      progress: raw.progress, created_at: raw.created_at, started_at: raw.started_at, metrics: []
    };

    const mapMetrics = (rawMetrics: any) => (rawMetrics.metrics || rawMetrics).map((m: any) => ({
      epoch: m.epoch, loss: m.loss || 0, accuracy: m.accuracy || 0,
      val_loss: m.val_loss ?? m.valLoss ?? 0, val_accuracy: m.val_accuracy ?? m.valAccuracy ?? 0, timestamp: m.timestamp
    }));

    if (status.status === 'running' || status.status === 'initialized') {
      const metrics = await request(() => apiClient.getTrainingMetrics(id));
      if (metrics) status.metrics = mapMetrics(metrics);
      setTrainingStatus(status);
      dispatch({ type: 'SET_TRAINING', payload: status });
      setTimeout(() => pollStatus(id), 1500);
    } else if (status.status === 'completed') {
      const metrics = await request(() => apiClient.getTrainingMetrics(id));
      if (metrics) status.metrics = mapMetrics(metrics);
      setTrainingStatus(status);
      dispatch({ type: 'SET_TRAINING', payload: status });
      onTrainingComplete?.();
    }
  }, [request, apiClient, dispatch, onTrainingComplete]);

  const handleStart = async () => {
    if (!config.dataset_id) { alert("Please select a dataset first."); return; }

    // NEW: Duplicate Dataset Protection
    const duplicateModel = state.trainedModels.find(m => m.dataset_id === config.dataset_id && m.base_model === config.base_model);
    if (duplicateModel) {
      const confirmRun = window.confirm(`⚠️ A model ("${duplicateModel.name}") has already been trained on this dataset using ${config.base_model}.\n\nDo you want to re-run the training pipeline anyway?`);
      if (!confirmRun) return;
    }

    setIsStarting(true);
    const result = await request(() => apiClient.startTraining(config));
    setIsStarting(false);

    if (result?.training_id) {
      setTrainingId(result.training_id);
      pollStatus(result.training_id);
    }
  };

  const isTraining = trainingStatus?.status === 'running' || trainingStatus?.status === 'initialized';

  // NEW: Time Calculations
  const formatTime = (seconds: number) => `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  const elapsedSec = trainingStatus?.started_at ? Math.max(0, (now - (trainingStatus.started_at * 1000)) / 1000) : 0;
  const timePerEpoch = trainingStatus?.current_epoch && trainingStatus.current_epoch > 0 ? elapsedSec / trainingStatus.current_epoch : 0;
  const leftSec = trainingStatus ? Math.max(0, timePerEpoch * (trainingStatus.total_epochs - trainingStatus.current_epoch)) : 0;

  return (
    <div className="space-y-6">
      {/* Config Form */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Training Dataset</label>
          <select value={config.dataset_id} onChange={(e) => setConfig({ ...config, dataset_id: e.target.value })} disabled={isTraining} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white">
            <option value="">-- Select a Compatible Dataset --</option>
            {datasets.map(d => <option key={d.id} value={d.id}>{d.name} ({d.sample_count} samples)</option>)}
          </select>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm text-gray-300 mb-2">Epochs</label>
            <input type="number" value={config.epochs} onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })} disabled={isTraining} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white" />
          </div>
          <div>
            <label className="block text-sm text-gray-300 mb-2">Batch Size</label>
            <input type="number" value={config.batch_size} onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })} disabled={isTraining} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white" />
          </div>
          <div>
            <label className="block text-sm text-gray-300 mb-2">Learning Rate</label>
            <input type="number" step="0.0001" value={config.learning_rate} onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })} disabled={isTraining} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white" />
          </div>
          <div>
            <label className="block text-sm text-gray-300 mb-2">Base Model</label>
            <select value={config.base_model} onChange={(e) => setConfig({ ...config, base_model: e.target.value })} disabled={isTraining} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white">
              <option>MobileNetV2</option>
              <option>EfficientNet</option>
              <option>Custom3LayerCNN</option>
            </select>
          </div>
        </div>

        <button onClick={handleStart} disabled={isTraining || isStarting} className="w-full py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-lg flex justify-center gap-2 font-bold shadow-lg disabled:opacity-50 transition-all">
          <Play className="w-5 h-5" /> {isStarting ? 'Allocating Resources...' : `Start ${task.replace(/_/g, ' ')} Training`}
        </button>
      </div>

      {/* Live Training Status & Timer */}
      {trainingStatus && (
        <div className="p-4 bg-slate-900/50 rounded-lg border border-blue-500/30 space-y-5 animate-slideIn">
          <div className="flex justify-between items-center bg-slate-800 p-3 rounded-lg border border-slate-700">
            <div className="flex items-center gap-4 text-sm">
              <span className={`px-3 py-1 rounded-full font-semibold capitalize ${trainingStatus.status === 'completed' ? 'bg-green-500/20 text-green-400' : 'bg-blue-500/20 text-blue-400'}`}>
                {trainingStatus.status}
              </span>
              <div className="flex items-center gap-2 text-gray-400">
                <Clock className="w-4 h-4 text-cyan-400" /> Elapsed: <span className="text-white">{formatTime(elapsedSec)}</span>
              </div>
            </div>
            {trainingStatus.status === 'running' && (
              <div className="text-sm text-gray-400">
                ETA: <span className="text-yellow-400 font-mono">{formatTime(leftSec)}</span>
              </div>
            )}
          </div>

          {trainingStatus.total_epochs > 0 && (
            <>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Epoch Progress:</span>
                <span className="text-white font-mono">{trainingStatus.current_epoch} / {trainingStatus.total_epochs}</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div className="bg-blue-500 h-2 rounded-full transition-all duration-300" style={{ width: `${trainingStatus.progress}%` }} />
              </div>
            </>
          )}

          {trainingStatus.metrics?.length > 0 && (
            <div className="space-y-4 animate-fadeIn">
              <div className="flex gap-4">
                <LineChart data={trainingStatus.metrics} dataKey1="accuracy" dataKey2="val_accuracy" color1="#4ade80" color2="#22d3ee" title="Accuracy History" />
                <LineChart data={trainingStatus.metrics} dataKey1="loss" dataKey2="val_loss" color1="#facc15" color2="#fb923c" title="Loss History" />
              </div>
            </div>
          )}
        </div>
      )}

      {/* NEW: Training History */}
      {state.trainedModels.length > 0 && !isTraining && (
        <div className="mt-8 pt-6 border-t border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <History className="w-5 h-5 text-purple-400" /> Training History
          </h3>
          <div className="space-y-3">
            {state.trainedModels.filter(m => m.task === task).map(model => (
              <div key={model.id} className="flex items-center justify-between p-3 bg-slate-800 rounded-lg border border-slate-700">
                <div>
                  <div className="text-white font-medium">{model.name}</div>
                  <div className="text-xs text-gray-400 flex gap-3 mt-1">
                    <span>Base: {model.base_model || 'Unknown'}</span>
                    <span>Acc: {(model.accuracy * 100).toFixed(1)}%</span>
                    <span>Val Acc: {(model.val_accuracy * 100).toFixed(1)}%</span>
                  </div>
                </div>
                <div className={`px-2 py-1 text-xs rounded-full font-medium ${model.optimized ? 'bg-green-500/20 text-green-400' : 'bg-slate-700 text-gray-400'}`}>
                  {model.optimized ? 'Optimized' : 'Raw Model'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}