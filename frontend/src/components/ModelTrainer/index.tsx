import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Play, Square, RefreshCw, TrendingUp, Clock, Award, Timer,
  ShieldCheck, History, ZoomIn, ChevronDown, ChevronUp,
  AlertTriangle, Settings, Activity, Shuffle
} from 'lucide-react';
import { useAPI } from '../../hooks/useAPI';
import { useAppContext } from '../../context/AppContext';
import { TinyMLTask, TrainingStatus } from '../../types';
import { MetricChart, ChartModal } from './Chart';
import { getTaskDefaults, AUDIO_TASKS, AUDIO_MODELS, IMAGE_MODELS, formatDate, formatTime } from './constants';
import PastSessionPopup from './PastSessionPopUp';

interface ModelTrainerProps {
  task: TinyMLTask;
  onTrainingComplete?: () => void;
}

// === Main component ============================================================

export function ModelTrainer({ task, onTrainingComplete }: ModelTrainerProps) {
  const { dispatch } = useAppContext();
  const { request, apiClient, error } = useAPI();
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const defaults = getTaskDefaults(task);
  const [datasetId, setDatasetId] = useState('');
  const [datasets, setDatasets] = useState<{ id: string; name: string; sample_count: number }[]>([]);
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);
  const [baseModel, setBaseModel] = useState(defaults.base_model);

  // ── Input shape (editable) ──────────────────────────────────────────────────
  const [inputShape, setInputShape] = useState<number[]>(defaults.input_shape);

  // Reset shape when task changes
  useEffect(() => {
    setInputShape(getTaskDefaults(task).input_shape);
  }, [task]);

  const updateShapeDim = (index: number, value: string) => {
    const num = parseInt(value, 10);
    if (isNaN(num) || num < 1) return;
    setInputShape(prev => prev.map((v, i) => (i === index ? num : v)));
  };
  // ───────────────────────────────────────────────────────────────────────────

  // Layout expansion state
  const [isConfigExpanded, setIsConfigExpanded] = useState(true);

  // Regularization
  const [dropoutRate, setDropoutRate] = useState(0.5);
  const [l2Reg, setL2Reg] = useState(0.0);
  const [showRegularization, setShowRegularization] = useState(false);

  // Early stopping
  const [earlyStopping, setEarlyStopping] = useState(false);
  const [esPatience, setEsPatience] = useState(5);
  const [esMonitor, setEsMonitor] = useState<'val_loss' | 'val_accuracy'>('val_loss');

  const [trainingId, setTrainingId] = useState<string | null>(null);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  // Advanced options
  const [trainableLayers, setTrainableLayers] = useState(0); // 0 = All
  const [freezeEpochs, setFreezeEpochs] = useState(0);
  const [augmentation, setAugmentation] = useState({
    horizontal_flip: false,
    random_rotation: 0,
    random_crop: false
  });

  // Past trainings
  const [pastSessions, setPastSessions] = useState<any[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [viewingSession, setViewingSession] = useState<any | null>(null);

  // Duplicate alert
  const [duplicateWarning, setDuplicateWarning] = useState(false);

  // Live chart expand/export
  const [expandedLiveChart, setExpandedLiveChart] = useState<'accuracy' | 'loss' | null>(null);

  // Split validation state
  const [splitSummary, setSplitSummary] = useState<{
    train: number; val: number; test: number; unassigned: number
  } | null>(null);
  const [isSplitting, setIsSplitting] = useState(false);

  const fetchDatasets = async () => {
    const raw = await request(() => apiClient.listDatasets(task));
    if (raw && raw.datasets) setDatasets(raw.datasets);
  };

  const fetchPastSessions = useCallback(async () => {
    const raw = await request(() => apiClient.listAllSessions());
    if (raw && raw.sessions) {
      setPastSessions(raw.sessions.filter((s: any) => s.task === task));
    }
  }, [task]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    fetchDatasets();
    fetchPastSessions();
    setBaseModel(getTaskDefaults(task).base_model);
  }, [task]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch split summary whenever dataset selection changes
  useEffect(() => {
    if (!datasetId) { setSplitSummary(null); return; }
    apiClient.getSplitSummary(datasetId)
      .then(res => setSplitSummary(res.summary))
      .catch(() => setSplitSummary(null));
  }, [datasetId]); // eslint-disable-line react-hooks/exhaustive-deps

  const pollStatus = useCallback(async (id: string) => {
    const raw = await request(() => apiClient.getTrainingStatus(id));
    if (!raw) return;
    const s: TrainingStatus = raw.data ?? raw;
    setStatus(s);
    dispatch({ type: 'SET_TRAINING', payload: s });
    if (s.status === 'running' || s.status === 'initialized') {
      pollRef.current = setTimeout(() => pollStatus(id), 5000);
    } else if (s.status === 'completed') {
      fetchPastSessions();
      onTrainingComplete?.();
    }
  }, [request, apiClient, dispatch, onTrainingComplete, fetchPastSessions]);

  useEffect(() => {
    return () => { if (pollRef.current) clearTimeout(pollRef.current); };
  }, []);

  // Check for duplicate trainings
  const checkDuplicate = useCallback(() => {
    if (!datasetId) return false;
    return pastSessions.some(
      s =>
        s.dataset_id === datasetId &&
        s.base_model === baseModel &&
        s.epochs === epochs &&
        s.batch_size === batchSize &&
        Math.abs(s.learning_rate - learningRate) < 1e-9 &&
        (s.status === 'completed' || s.status === 'running')
    );
  }, [pastSessions, datasetId, baseModel, epochs, batchSize, learningRate]);

  const handleStart = async () => {
    if (!datasetId) { alert('Please select a dataset first.'); return; }

    if (!splitReady) {
      alert(
        'This dataset doesn\'t have a complete train/val split yet.\n\n' +
        'Training always uses the precomputed split — assign all samples to ' +
        'train/val/test first (use "Auto Split" above, or assign them manually).'
      );
      return;
    }

    if (checkDuplicate()) {
      setDuplicateWarning(true);
      return;
    }
    await doStartTraining();
  };

  const doStartTraining = async () => {
    setDuplicateWarning(false);
    setIsStarting(true);
    setStatus(null);
    setIsConfigExpanded(false); // Automatically collapse config upon starting

    const raw = await request(() =>
      apiClient.startTraining({
        task,
        dataset_id: datasetId,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        base_model: baseModel,
        input_shape: inputShape,          // ← uses editable state
        early_stopping: earlyStopping,
        early_stopping_patience: esPatience,
        early_stopping_monitor: esMonitor,
        dropout_rate: dropoutRate,
        l2_reg: l2Reg,
        trainable_layers: trainableLayers,
        freeze_epochs: freezeEpochs,
        augmentation: {
          horizontal_flip: augmentation.horizontal_flip,
          random_rotation: augmentation.random_rotation,
          random_crop: augmentation.random_crop
        }
      })
    );
    setIsStarting(false);

    if (raw && raw.training_id) {
      setTrainingId(raw.training_id);
      pollStatus(raw.training_id);
    }
  };

  const handleCancel = async () => {
    if (!trainingId) return;
    await request(() => apiClient.cancelTraining(trainingId));
    if (pollRef.current) clearTimeout(pollRef.current);
    setStatus(prev => prev ? { ...prev, status: 'cancelled' } : null);
  };

  /** Quick auto-split 70/20/10, then refresh split summary */
  const handleQuickAutoSplit = async () => {
    if (!datasetId) return;
    setIsSplitting(true);
    await request(() => apiClient.autoSplitDataset(datasetId, 70, 20, 10));
    const res = await apiClient.getSplitSummary(datasetId);
    setSplitSummary(res.summary);
    setIsSplitting(false);
  };

  const isRunning = status?.status === 'running' || status?.status === 'initialized';
  const availableModels = AUDIO_TASKS.includes(task) ? AUDIO_MODELS : IMAGE_MODELS;
  const latestMetrics = status?.metrics?.length ? status.metrics[status.metrics.length - 1] : null;

  const accuracyData = (status?.metrics ?? []).map(m => ({
    epoch: m.epoch,
    train: parseFloat((m.accuracy * 100).toFixed(2)),
    val: parseFloat((m.val_accuracy * 100).toFixed(2)),
  }));
  const lossData = (status?.metrics ?? []).map(m => ({
    epoch: m.epoch,
    train: parseFloat(m.loss.toFixed(4)),
    val: parseFloat(m.val_loss.toFixed(4)),
  }));

  const elapsed: number = (status as any)?.elapsed_seconds ?? 0;
  const remaining: number = (status as any)?.remaining_seconds ?? 0;

  const statusColor =
    status?.status === 'completed' ? 'text-green-400' :
      status?.status === 'failed' ? 'text-red-400' :
        status?.status === 'cancelled' ? 'text-yellow-400' : 'text-purple-400';

  const barColor =
    status?.status === 'completed' ? 'bg-green-500' :
      status?.status === 'failed' ? 'bg-red-500' :
        'bg-gradient-to-r from-purple-500 to-pink-500';

  // Derived split state
  const hasUnassigned = splitSummary ? splitSummary.unassigned > 0 : false;
  const splitReady = splitSummary
    ? splitSummary.unassigned === 0 && splitSummary.train > 0 && splitSummary.val > 0
    : false;

  // Dimension labels depend on task type
  const isAudio = AUDIO_TASKS.includes(task);
  const dimLabels = isAudio
    ? ['n_mfcc', 'time frames', 'channels']
    : ['width', 'height', 'channels'];

  return (
    <div className="space-y-6">

      {/* -- Configuration Wrapper -- */}
      <div className="bg-slate-800/40 rounded-xl border border-slate-700 overflow-hidden shadow-sm transition-all duration-300">
        <button
          onClick={() => setIsConfigExpanded(!isConfigExpanded)}
          className="w-full px-6 py-4 flex items-center justify-between bg-slate-800/80 hover:bg-slate-700/80 transition-colors"
        >
          <div className="flex items-center gap-3">
            <Settings className="w-5 h-5 text-purple-400" />
            <h3 className="text-lg font-semibold text-white">Model Configuration</h3>
          </div>
          {isConfigExpanded ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
        </button>

        {isConfigExpanded && (
          <div className="p-6 space-y-6 border-t border-slate-700 animate-fadeIn bg-slate-900/30">
            {/* -- Config Grid -- */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

              {/* Dataset selector */}
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-300 mb-1">Dataset</label>
                <select
                  value={datasetId}
                  onChange={e => setDatasetId(e.target.value)}
                  disabled={isRunning}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white disabled:opacity-50"
                >
                  <option value="">-- Select a dataset --</option>
                  {datasets.map(d => (
                    <option key={d.id} value={d.id}>{d.name} ({d.sample_count} samples)</option>
                  ))}
                </select>

                {/* Split health indicator beneath the dataset selector */}
                {datasetId && splitSummary && (
                  <div className="mt-2">
                    {hasUnassigned ? (
                      <div className="flex items-start gap-2 p-3 bg-amber-900/30 border border-amber-500/40 rounded-lg text-amber-300 text-xs">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                        <div className="flex-1">
                          <span className="font-semibold">Dataset not fully split.</span>
                          {' '}{splitSummary.unassigned} sample(s) are unassigned. Training requires all samples
                          to be assigned to train/val/test.
                          <button
                            onClick={handleQuickAutoSplit}
                            disabled={isSplitting || isRunning}
                            className="ml-2 inline-flex items-center gap-1 px-2 py-0.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded-md font-semibold transition"
                          >
                            {isSplitting
                              ? <RefreshCw className="w-3 h-3 animate-spin" />
                              : <Shuffle className="w-3 h-3" />}
                            Auto Split (70/20/10)
                          </button>
                        </div>
                      </div>
                    ) : splitReady ? (
                      <p className="text-xs text-emerald-400 mt-1">
                        ✓ Split ready — train: {splitSummary.train} · val: {splitSummary.val} · test: {splitSummary.test}
                      </p>
                    ) : null}
                  </div>
                )}
              </div>

              {/* Base model */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Base Model</label>
                <select
                  value={baseModel}
                  onChange={e => setBaseModel(e.target.value)}
                  disabled={isRunning}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white disabled:opacity-50"
                >
                  {availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>

              {/* Epochs */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Epochs: {epochs}</label>
                <input
                  type="range" min={5} max={200} step={5} value={epochs}
                  onChange={e => setEpochs(Number(e.target.value))}
                  disabled={isRunning}
                  className="w-full accent-purple-500 disabled:opacity-50"
                />
              </div>

              {/* Batch Size */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Batch Size: {batchSize}</label>
                <input
                  type="range" min={8} max={128} step={8} value={batchSize}
                  onChange={e => setBatchSize(Number(e.target.value))}
                  disabled={isRunning}
                  className="w-full accent-purple-500 disabled:opacity-50"
                />
              </div>

              {/* Learning Rate */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Learning Rate</label>
                <select
                  value={learningRate}
                  onChange={e => setLearningRate(Number(e.target.value))}
                  disabled={isRunning}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white disabled:opacity-50"
                >
                  {[0.01, 0.005, 0.001, 0.0005, 0.0001].map(lr => (
                    <option key={lr} value={lr}>{lr}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* -- Regularization -- */}
            <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700 space-y-3">
              <button
                className="flex items-center justify-between w-full"
                onClick={() => setShowRegularization(v => !v)}
              >
                <span className="text-sm font-medium text-gray-300 flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-cyan-400" />
                  Regularization
                  {(dropoutRate !== 0.5 || l2Reg !== 0) && (
                    <span className="px-1.5 py-0.5 bg-cyan-900/50 border border-cyan-500/40 text-cyan-300 text-xs rounded-full">
                      active
                    </span>
                  )}
                </span>
                {showRegularization
                  ? <ChevronUp className="w-4 h-4 text-gray-400" />
                  : <ChevronDown className="w-4 h-4 text-gray-400" />}
              </button>

              {showRegularization && (
                <div className="grid grid-cols-2 gap-4 pt-1">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      Dropout rate: {dropoutRate.toFixed(2)}
                    </label>
                    <input
                      type="range" min={0} max={0.9} step={0.05} value={dropoutRate}
                      onChange={e => setDropoutRate(Number(e.target.value))}
                      disabled={isRunning}
                      className="w-full accent-cyan-500 disabled:opacity-50"
                    />
                    <p className="text-xs text-gray-500 mt-1">Fraction of neurons dropped during training</p>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      L2 regularization: {l2Reg === 0 ? 'off' : l2Reg.toExponential(1)}
                    </label>
                    <select
                      value={l2Reg}
                      onChange={e => setL2Reg(Number(e.target.value))}
                      disabled={isRunning}
                      className="w-full px-2 py-1.5 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm disabled:opacity-50"
                    >
                      {[0, 0.0001, 0.0005, 0.001, 0.005, 0.01].map(v => (
                        <option key={v} value={v}>{v === 0 ? 'off' : v.toExponential(1)}</option>
                      ))}
                    </select>
                  </div>
                </div>
              )}
            </div>

            {/* -- Early Stopping -- */}
            <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700 space-y-3">
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-3 cursor-pointer select-none">
                  <div
                    onClick={() => !isRunning && setEarlyStopping(v => !v)}
                    className={`relative w-9 h-5 rounded-full transition-colors ${earlyStopping ? 'bg-purple-600' : 'bg-slate-600'} ${isRunning ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}`}
                  >
                    <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${earlyStopping ? 'translate-x-4' : ''}`} />
                  </div>
                  <span className="text-sm font-medium text-gray-300 flex items-center gap-1.5">
                    <ShieldCheck className="w-4 h-4 text-purple-400" />
                    Early Stopping
                  </span>
                </label>
                {earlyStopping && (
                  <span className="text-xs text-gray-500">stops when {esMonitor} stops improving</span>
                )}
              </div>

              {earlyStopping && (
                <div className="grid grid-cols-2 gap-4 pt-1">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">Monitor metric</label>
                    <select
                      value={esMonitor}
                      onChange={e => setEsMonitor(e.target.value as any)}
                      disabled={isRunning}
                      className="w-full px-2 py-1.5 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm disabled:opacity-50"
                    >
                      <option value="val_loss">val_loss (recommended)</option>
                      <option value="val_accuracy">val_accuracy</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      Patience: {esPatience} epochs
                    </label>
                    <input
                      type="range" min={2} max={20} step={1} value={esPatience}
                      onChange={e => setEsPatience(Number(e.target.value))}
                      disabled={isRunning}
                      className="w-full accent-purple-500 disabled:opacity-50 mt-1"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Advanced Regularization & Fine-Tuning */}
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div>
                <label className="block text-sm text-gray-300">Dropout Rate</label>
                <input type="number" step="0.1" value={dropoutRate} onChange={e => setDropoutRate(parseFloat(e.target.value))} className="w-full bg-slate-800 rounded p-2" />
              </div>
              <div>
                <label className="block text-sm text-gray-300">L2 Regularization</label>
                <input type="number" step="0.0001" value={l2Reg} onChange={e => setL2Reg(parseFloat(e.target.value))} className="w-full bg-slate-800 rounded p-2" />
              </div>
              <div>
                <label className="block text-sm text-gray-300">Trainable Layers (0=All)</label>
                <input type="number" value={trainableLayers} onChange={e => setTrainableLayers(parseInt(e.target.value))} className="w-full bg-slate-800 rounded p-2" />
              </div>
              <div>
                <label className="block text-sm text-gray-300">Freeze Encoder Epochs</label>
                <input type="number" value={freezeEpochs} onChange={e => setFreezeEpochs(parseInt(e.target.value))} className="w-full bg-slate-800 rounded p-2" />
              </div>
            </div>

            {/* Augmentation Toggles */}
            <div className="mt-4 p-4 border border-slate-700 rounded-lg">
              <h4 className="text-sm font-bold text-gray-400 mb-2">Data Augmentation</h4>
              <label className="flex items-center gap-2 text-sm text-gray-300">
                <input type="checkbox" checked={augmentation.horizontal_flip} onChange={e => setAugmentation({ ...augmentation, horizontal_flip: e.target.checked })} />
                Horizontal Flip
              </label>
              <label className="flex items-center gap-2 text-sm text-gray-300 mt-2">
                <input type="checkbox" checked={augmentation.random_crop} onChange={e => setAugmentation({ ...augmentation, random_crop: e.target.checked })} />
                Random Crop (Zoom)
              </label>
              <div className="mt-2">
                <label className="block text-sm text-gray-300">Random Rotation (%)</label>
                <input type="number" step="0.1" value={augmentation.random_rotation} onChange={e => setAugmentation({ ...augmentation, random_rotation: parseFloat(e.target.value) })} className="w-full bg-slate-800 rounded p-2" />
              </div>
            </div>

            {/* -- Input Shape -- */}
            <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700 space-y-3">
              <p className="text-sm font-medium text-gray-300">
                Input Shape
                <span className="ml-2 text-xs text-gray-500 font-normal">
                  ({inputShape.join(' × ')}) — must match backend <code className="text-cyan-400">data_processor.py</code>
                </span>
              </p>
              <div className="grid grid-cols-3 gap-3">
                {inputShape.map((val, i) => (
                  <div key={i}>
                    <label className="block text-xs text-gray-400 mb-1">{dimLabels[i] ?? `dim ${i}`}</label>
                    <input
                      type="number"
                      min={1}
                      value={val}
                      onChange={e => updateShapeDim(i, e.target.value)}
                      disabled={isRunning}
                      className="w-full px-3 py-1.5 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm text-center disabled:opacity-50 focus:border-purple-500 focus:outline-none"
                    />
                  </div>
                ))}
              </div>
              <button
                onClick={() => setInputShape(getTaskDefaults(task).input_shape)}
                disabled={isRunning}
                className="text-xs text-gray-500 hover:text-purple-400 disabled:opacity-40 transition"
              >
                ↺ Reset to default ({getTaskDefaults(task).input_shape.join('×')})
              </button>
            </div>

            {/* Task info badge */}
            <div className="px-3 py-2 bg-slate-800/50 rounded-lg border border-slate-700 text-xs text-gray-400 flex gap-4">
              <span>Task: <span className="text-purple-300 font-medium">{task.replace(/_/g, ' ')}</span></span>
              <span>Input: <span className="text-cyan-300 font-medium">{inputShape.join('×')}</span></span>
            </div>

          </div>
        )}
      </div>

      {/* -- Backend error (e.g. split not ready) -- */}
      {error && (
        <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* -- Action Buttons -- */}
      <div className="flex gap-3">
        <button
          onClick={handleStart}
          disabled={isRunning || isStarting || !datasetId}
          className="flex-1 flex items-center justify-center gap-2 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed text-white font-bold rounded-xl shadow-lg transition-all"
        >
          {isStarting ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
          {isStarting ? 'Initializing...' : 'Start Training'}
        </button>

        {isRunning && (
          <button
            onClick={handleCancel}
            className="px-4 py-3 bg-red-600/20 hover:bg-red-600/40 border border-red-500/50 text-red-400 font-semibold rounded-xl transition-all"
          >
            <Square className="w-5 h-5" />
          </button>
        )}

        {/* History button */}
        <button
          onClick={() => setShowHistory(v => !v)}
          className="flex items-center gap-2 px-4 py-3 bg-slate-700 hover:bg-slate-600 border border-slate-600 text-gray-300 font-semibold rounded-xl transition-all"
          title="Past trainings"
        >
          <History className="w-5 h-5" />
          {pastSessions.length > 0 && (
            <span className="text-xs bg-purple-600 text-white rounded-full px-1.5 py-0.5">{pastSessions.length}</span>
          )}
        </button>
      </div>

      {/* -- Duplicate warning -- */}
      {duplicateWarning && (
        <div className="p-4 bg-amber-900/30 border border-amber-500/50 rounded-xl flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-amber-300 font-semibold text-sm">Duplicate training detected</p>
            <p className="text-amber-400/80 text-xs mt-1">
              The same dataset, model, epochs, batch size and learning rate were already used in a previous training. Do you still want to proceed?
            </p>
            <div className="flex gap-2 mt-3">
              <button
                onClick={doStartTraining}
                className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold rounded-lg transition"
              >
                Train anyway
              </button>
              <button
                onClick={() => setDuplicateWarning(false)}
                className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-gray-300 text-xs rounded-lg transition"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* -- Past Sessions Panel -- */}
      {showHistory && (
        <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700 space-y-2 animate-slideIn">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <History className="w-4 h-4 text-purple-400" /> Past Trainings ({pastSessions.length})
          </h3>
          {pastSessions.length === 0 ? (
            <p className="text-gray-500 text-sm text-center py-4">No past trainings for this task.</p>
          ) : (
            <div className="space-y-2 max-h-72 overflow-y-auto custom-scrollbar">
              {pastSessions.map(s => {
                const last = s.metrics?.length ? s.metrics[s.metrics.length - 1] : null;
                const sc =
                  s.status === 'completed' ? 'text-green-400 bg-green-900/20 border-green-500/30' :
                    s.status === 'failed' ? 'text-red-400 bg-red-900/20 border-red-500/30' :
                      s.status === 'cancelled' ? 'text-yellow-400 bg-yellow-900/20 border-yellow-500/30' :
                        'text-purple-400 bg-purple-900/20 border-purple-500/30';
                return (
                  <button
                    key={s.id}
                    onClick={() => setViewingSession(s)}
                    className="w-full text-left p-3 bg-slate-900/50 hover:bg-slate-700/50 rounded-lg border border-slate-700 hover:border-slate-500 transition-all group"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold border ${sc}`}>
                          {s.status?.toUpperCase()}
                        </span>
                        <span className="text-white text-sm font-medium">{s.base_model}</span>
                      </div>
                      <span className="text-xs text-gray-500">{formatDate(s.created_at)}</span>
                    </div>
                    <div className="flex gap-4 mt-1.5 text-xs text-gray-400">
                      <span>Ep {s.current_epoch}/{s.total_epochs}</span>
                      {last && (
                        <>
                          <span className="text-green-400">Acc {(last.accuracy * 100).toFixed(1)}%</span>
                          <span className="text-cyan-400">Val {(last.val_accuracy * 100).toFixed(1)}%</span>
                        </>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* -- Dedicated Progress Card -- */}
      {status && (
        <div className="mt-8 bg-slate-900/80 border border-purple-500/30 rounded-2xl p-6 shadow-2xl relative overflow-hidden animate-slideIn">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 to-pink-500 opacity-80"></div>

          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-purple-400" /> Live Training Progress
            </h3>
            <span className={`px-3 py-1 text-xs font-bold rounded-full ${statusColor} bg-slate-800 border border-slate-700`}>
              {status.status.toUpperCase()}
            </span>
          </div>

          <div className="space-y-6">
            {/* Progress bar */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Epoch {status.current_epoch} / {status.total_epochs}</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden shadow-inner">
                <div
                  className={`h-3 rounded-full transition-all duration-500 ${barColor}`}
                  style={{ width: `${status.progress || 0}%` }}
                />
              </div>

              {(isRunning || status.status === 'completed') && elapsed > 0 && (
                <div className="flex gap-6 mt-3 text-xs text-gray-400">
                  <span className="flex items-center gap-1">
                    <Clock className="w-4 h-4 text-cyan-500" />
                    Elapsed: <span className="text-cyan-300 font-mono ml-1">{formatTime(elapsed)}</span>
                  </span>
                  {isRunning && (
                    <span className="flex items-center gap-1">
                      <Timer className="w-4 h-4 text-amber-500" />
                      Remaining: <span className="text-amber-300 font-mono ml-1">{formatTime(remaining)}</span>
                    </span>
                  )}
                  {status.status === 'completed' && (
                    <span className="flex items-center gap-1">
                      <Timer className="w-4 h-4 text-green-500" />
                      Total: <span className="text-green-300 font-mono ml-1">{formatTime(elapsed)}</span>
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Live metric cards */}
            {latestMetrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
                  <div className="flex items-center gap-1 text-xs text-gray-400 mb-1">
                    <TrendingUp className="w-3 h-3" /> Accuracy
                  </div>
                  <span className="text-2xl font-bold text-green-400">
                    {(latestMetrics.accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
                  <div className="flex items-center gap-1 text-xs text-gray-400 mb-1">
                    <Award className="w-3 h-3" /> Val Accuracy
                  </div>
                  <span className="text-2xl font-bold text-cyan-400">
                    {(latestMetrics.val_accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
                  <div className="text-xs text-gray-400 mb-1">Loss</div>
                  <span className="text-2xl font-bold text-yellow-400">{latestMetrics.loss.toFixed(4)}</span>
                </div>
                <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
                  <div className="flex items-center gap-1 text-xs text-gray-400 mb-1">
                    <Clock className="w-3 h-3" /> Val Loss
                  </div>
                  <span className="text-2xl font-bold text-orange-400">{latestMetrics.val_loss.toFixed(4)}</span>
                </div>
              </div>
            )}

            {/* Live charts with expand button */}
            {accuracyData.length >= 2 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="relative">
                  <button
                    onClick={() => setExpandedLiveChart('accuracy')}
                    className="absolute top-2 right-2 z-10 p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded-lg text-gray-400 hover:text-white transition"
                    title="Expand chart"
                  >
                    <ZoomIn className="w-4 h-4" />
                  </button>
                  <MetricChart data={accuracyData} label="Accuracy (%)" color="#22c55e" valColor="#06b6d4" formatY={v => `${v}%`} />
                </div>
                <div className="relative">
                  <button
                    onClick={() => setExpandedLiveChart('loss')}
                    className="absolute top-2 right-2 z-10 p-1.5 bg-slate-700/80 hover:bg-slate-600 rounded-lg text-gray-400 hover:text-white transition"
                    title="Expand chart"
                  >
                    <ZoomIn className="w-4 h-4" />
                  </button>
                  <MetricChart data={lossData} label="Loss" color="#eab308" valColor="#f97316" />
                </div>
              </div>
            )}

            {status.status === 'failed' && (status as any).error && (
              <div className="p-4 bg-red-900/30 border border-red-500/50 rounded-xl text-red-300 text-sm">
                <strong className="block mb-1">Training Error:</strong>
                {(status as any).error}
              </div>
            )}
          </div>
        </div>
      )}

      {/* -- Live chart modal -- */}
      {expandedLiveChart === 'accuracy' && (
        <ChartModal
          label="Accuracy (%)" color="#22c55e" valColor="#06b6d4"
          data={accuracyData} formatY={v => `${v}%`}
          onClose={() => setExpandedLiveChart(null)}
        />
      )}
      {expandedLiveChart === 'loss' && (
        <ChartModal
          label="Loss" color="#eab308" valColor="#f97316"
          data={lossData}
          onClose={() => setExpandedLiveChart(null)}
        />
      )}

      {/* -- Past session popup -- */}
      {viewingSession && (
        <PastSessionPopup session={viewingSession} onClose={() => setViewingSession(null)} />
      )}
    </div>
  );
}