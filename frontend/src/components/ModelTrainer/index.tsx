// ModelTrainer.tsx
// Full refactored ModelTrainer component with SelectOrCustom dropdown pattern,
// merged Regularization section, compact Augmentation row, and all custom states.

import { useState, useEffect, useCallback, useRef, lazy, Suspense } from 'react';
import {
  Play, Square, RefreshCw, Clock,
  ShieldCheck, History, ChevronDown, ChevronUp,
  AlertTriangle, Settings, Activity, Shuffle, Lightbulb, X
} from 'lucide-react';
import { useAPI } from '../../hooks/useAPI';
import { useToast } from '../../context/ToastContext';
import { useAppContext } from '../../context/AppContext';
import { TinyMLTask, TrainingStatus } from '../../types';
import { TrainingDashboard } from './TrainingDashboard';
import { APPLY_TRAINING_CHANGES_EVENT, ApplyTrainingChangesDetail } from '../../utils/trainingChanges';
import {
  getTaskDefaults, AUDIO_TASKS, AUDIO_MODELS, IMAGE_MODELS, OD_MODELS, MODEL_HINTS, FREEZE_AUTO,
  formatDate,
  BATCH_SIZE_OPTIONS, DROPOUT_OPTIONS, LEARNING_RATE_OPTIONS,
  EPOCHS_OPTIONS,
  ES_PATIENCE_OPTIONS,
  FREEZE_EPOCHS_OPTIONS,
  TRAINABLE_LAYERS_OPTIONS
} from './constants';
import PastSessionPopup from './PastSessionPopUp';
import SelectOrCustom from './SelectOrCustom';
import { TerminalLogPanel } from '../TerminalLogPanel';
import { EvaluationReport } from '../EvaluationReport';
import { useBackendInfo } from '../../hooks/useBackendInfo';

const LLMAdvisor = lazy(() => import('../LLMAdvisor').then(m => ({ default: m.LLMAdvisor })));

// ---------------------------------------------------------------------------
// ModelTrainer
// ---------------------------------------------------------------------------
interface ModelTrainerProps {
  task: TinyMLTask;
  onTrainingComplete?: () => void;
}

export function ModelTrainer({ task, onTrainingComplete }: ModelTrainerProps) {
  const { state, dispatch } = useAppContext();
  const { request, apiClient, error } = useAPI();
  const { toast } = useToast();
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const backendInfo = useBackendInfo();
  const taskDefaults = useCallback((t: TinyMLTask) => {
    const d = getTaskDefaults(t);
    const shape = backendInfo?.default_input_shapes?.[t];
    return shape ? { ...d, input_shape: shape } : d;
  }, [backendInfo]);
  const defaults = taskDefaults(task);
  const isAudio = AUDIO_TASKS.includes(task);
  const isODTask = task === 'OBJECT_DETECTION';

  // --- Core config ---
  const [datasetId, setDatasetId] = useState('');
  const [datasets, setDatasets] = useState<{ id: string; name: string; sample_count: number }[]>([]);
  const [runName, setRunName] = useState('');
  const [epochs, setEpochs] = useState<number>(30);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [baseModel, setBaseModel] = useState(defaults.base_model);
  const [inputShape, setInputShape] = useState<number[]>(defaults.input_shape);

  // --- Compute device (CPU/GPU) ---
  const [device, setDevice] = useState<'auto' | 'cpu' | 'gpu'>('auto');
  const [availableDevices, setAvailableDevices] = useState<{ cpu_available: boolean; gpu_available: boolean; gpus: { name: string }[] } | null>(null);

  useEffect(() => {
    apiClient.getAvailableDevices()
      .then((res) => { if (res.status === 'success') setAvailableDevices(res.devices); })
      .catch(() => { /* devices endpoint unreachable; leave selector enabled */ });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // AI-assisted pre-training configuration suggestion
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [suggestionReasoning, setSuggestionReasoning] = useState<string | null>(null);
  const [suggestionError, setSuggestionError] = useState<string | null>(null);

  useEffect(() => {
    setInputShape(taskDefaults(task).input_shape);
  }, [task, taskDefaults]);

  const updateShapeDim = (index: number, value: string) => {
    const num = parseInt(value, 10);
    if (isNaN(num) || num < 1) return;
    setInputShape((prev) => prev.map((v, i) => (i === index ? num : v)));
  };

  // --- UI toggles ---
  const [isConfigExpanded, setIsConfigExpanded] = useState(true);
  const [showRegularization, setShowRegularization] = useState(false);

  // --- Regularization ---
  const [dropoutRate, setDropoutRate] = useState(0.3);
  const [l2Reg, setL2Reg] = useState(0.0);
  const [trainableLayers, setTrainableLayers] = useState(0);
  const [freezeEpochs, setFreezeEpochs] = useState<number>(FREEZE_AUTO);
  const [classWeighting, setClassWeighting] = useState(false);
  const [seed, setSeed] = useState('');

  // --- Early stopping ---
  const [earlyStopping, setEarlyStopping] = useState(false);
  const [esPatience, setEsPatience] = useState(5);
  const [esMonitor, setEsMonitor] = useState<'val_loss' | 'val_accuracy'>('val_loss');

  // --- Augmentation ---
  const [augmentation, setAugmentation] = useState({
    horizontal_flip: false,
    random_rotation: 0,
    random_crop: false,
    random_brightness: 0,
    random_contrast: 0,
    random_translation: 0,
  });

  // --- Training runtime ---
  const [trainingId, setTrainingId] = useState<string | null>(null);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showArchived, setShowArchived] = useState(false);

  // --- History / sessions ---
  const [pastSessions, setPastSessions] = useState<any[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [viewingSession, setViewingSession] = useState<any | null>(null);
  const [suggestSession, setSuggestSession] = useState<any | null>(null);

  // --- Warnings / split ---
  const [duplicateWarning, setDuplicateWarning] = useState(false);
  const [splitSummary, setSplitSummary] = useState<{
    train: number; val: number; test: number; unassigned: number;
  } | null>(null);
  const [isSplitting, setIsSplitting] = useState(false);

  // --- Chart modal ---

  // --- Fetch helpers ---
  const fetchDatasets = async () => {
    const raw = await request(() => apiClient.listDatasets(task));
    if (raw && raw.datasets) setDatasets(raw.datasets);
  };

  const fetchPastSessions = useCallback(async () => {
    const raw = await request(() => apiClient.listAllSessions(showArchived));
    if (raw && raw.sessions)
      setPastSessions(raw.sessions.filter((s: any) => s.task === task));
  }, [task, showArchived]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleArchiveToggle = async (sessionId: string, currentlyArchived: boolean) => {
    if (currentlyArchived) {
      await request(() => apiClient.unarchiveTraining(sessionId));
    } else {
      await request(() => apiClient.archiveTraining(sessionId));
    }
    fetchPastSessions();
  };

  useEffect(() => {
    fetchDatasets();
    fetchPastSessions();
    setBaseModel(getTaskDefaults(task).base_model);
  }, [task]); // eslint-disable-line react-hooks/exhaustive-deps

  // Reattach to a training job already in progress for this task - without
  // this, navigating to another tab and back (or just reloading the page)
  // would lose track of a running job entirely and show a blank "Start
  // Training" form even though something is actively training server-side.
  useEffect(() => {
    let cancelled = false;
    apiClient.getActiveTraining(task).then((res: any) => {
      if (cancelled) return;
      if (res?.status === 'success' && res.session) {
        setTrainingId(res.session.id);
        pollStatus(res.session.id);
      }
    }).catch(() => { /* best-effort - just don't reattach if this fails */ });
    return () => { cancelled = true; };
  }, [task]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    fetchPastSessions();
  }, [showArchived]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!datasetId) { setSplitSummary(null); return; }
    apiClient
      .getSplitSummary(datasetId)
      .then((res: any) => setSplitSummary(res.summary))
      .catch(() => setSplitSummary(null));
  }, [datasetId]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- Polling ---
  const pollStatus = useCallback(
    async (id: string) => {
      const raw = await request(() => apiClient.getTrainingStatus(id));
      if (!raw) return;
      const s: TrainingStatus = raw.data ?? raw;
      setStatus(s);
      dispatch({ type: 'SET_TRAINING', payload: s });
      if (s.status === 'running' || s.status === 'initialized' || s.status === 'queued') {
        pollRef.current = setTimeout(() => pollStatus(id), s.status === 'queued' ? 3000 : 1000);
      } else if (s.status === 'completed' || s.status === 'cancelled' || s.status === 'failed') {
        fetchPastSessions();
        if (s.status === 'completed') onTrainingComplete?.();
      }
    },
    [request, apiClient, dispatch, onTrainingComplete, fetchPastSessions],
  );

  useEffect(() => {
    return () => { if (pollRef.current) clearTimeout(pollRef.current); };
  }, []);

  // "Apply to configuration" from the training review: load the reviewed
  // run's settings plus the suggested changes into the form, so the next run
  // is exactly "that run, with this fix".
  useEffect(() => {
    const onApply = (e: Event) => {
      const { changes, datasetId: ds, base } = (e as CustomEvent<ApplyTrainingChangesDetail>).detail;
      const cfg = { ...base, ...changes, augmentation: { ...(base?.augmentation ?? {}), ...(changes.augmentation ?? {}) } };
      if (ds) setDatasetId(ds);
      if (cfg.base_model) setBaseModel(cfg.base_model);
      if (Array.isArray(cfg.input_shape)) setInputShape(cfg.input_shape);
      if (cfg.epochs != null) setEpochs(cfg.epochs);
      if (cfg.batch_size != null) setBatchSize(cfg.batch_size);
      if (cfg.learning_rate != null) setLearningRate(cfg.learning_rate);
      if (cfg.dropout_rate != null) setDropoutRate(cfg.dropout_rate);
      if (cfg.l2_reg != null) setL2Reg(cfg.l2_reg);
      if (cfg.trainable_layers != null) setTrainableLayers(cfg.trainable_layers);
      if (cfg.freeze_encoder_epochs != null) setFreezeEpochs(cfg.freeze_encoder_epochs);
      if (cfg.class_weighting != null) setClassWeighting(!!cfg.class_weighting);
      if (cfg.early_stopping != null) setEarlyStopping(!!cfg.early_stopping);
      if (cfg.early_stopping_patience != null) setEsPatience(cfg.early_stopping_patience);
      if (cfg.early_stopping_monitor) setEsMonitor(cfg.early_stopping_monitor);
      setAugmentation((prev) => ({ ...prev, ...cfg.augmentation }));
      if (changes.dropout_rate != null || changes.l2_reg != null || changes.trainable_layers != null
        || changes.freeze_encoder_epochs != null || changes.class_weighting != null) setShowRegularization(true);
      setSuggestSession(null);
      setIsConfigExpanded(true);
      const { augmentation: augChanges, ...scalar } = changes;
      const n = Object.keys(scalar).length + Object.keys(augChanges ?? {}).length;
      toast('success', `Applied ${n} change${n !== 1 ? 's' : ''} to the configuration. Review and start training.`);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    };
    window.addEventListener(APPLY_TRAINING_CHANGES_EVENT, onApply);
    return () => window.removeEventListener(APPLY_TRAINING_CHANGES_EVENT, onApply);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Collapse the configuration whenever a job becomes active (started here,
  // or reattached after navigation / reload) so the live dashboard is in view.
  // Only on the transition, so the user can re-open it while training.
  const wasActiveRef = useRef(false);
  useEffect(() => {
    const active = status?.status === 'running' || status?.status === 'initialized' || status?.status === 'queued';
    if (active && !wasActiveRef.current) setIsConfigExpanded(false);
    wasActiveRef.current = active;
  }, [status?.status]);

  // --- Duplicate check ---
  const checkDuplicate = useCallback(() => {
    if (!datasetId) return false;
    return pastSessions.some(
      (s) =>
        s.dataset_id === datasetId &&
        s.base_model === baseModel &&
        s.epochs === epochs &&
        s.batch_size === batchSize &&
        Math.abs(s.learning_rate - learningRate) < 1e-9 &&
        (s.status === 'completed' || s.status === 'running'),
    );
  }, [pastSessions, datasetId, baseModel, epochs, batchSize, learningRate]);

  // --- Start / stop ---
  const handleStart = async () => {
    if (!datasetId) { toast('warning', 'Please select a dataset first.'); return; }
    if (!splitReady) {
      toast('warning', "This dataset doesn't have a complete train/val split yet. Assign all samples to train/val/test first.");
      return;
    }
    if (checkDuplicate()) { setDuplicateWarning(true); return; }
    await doStartTraining();
  };

  const doStartTraining = async () => {
    setDuplicateWarning(false);
    setIsStarting(true);
    setStatus(null);
    setIsConfigExpanded(false);
    const raw = await request(() =>
      apiClient.startTraining({
        task,
        name: runName.trim() || undefined,
        dataset_id: datasetId,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        base_model: baseModel,
        input_shape: inputShape,
        device,
        early_stopping: earlyStopping,
        early_stopping_patience: esPatience,
        early_stopping_monitor: esMonitor,
        dropout_rate: dropoutRate,
        l2_reg: l2Reg,
        trainable_layers: trainableLayers,
        // NOTE: was "freeze_epochs" - the backend's TrainingRequest field is
        // "freeze_encoder_epochs"; the mismatched name meant this always
        // silently fell back to the default (0), disabling the freeze-then-
        // fine-tune two-phase training path whenever it was set in the UI.
        freeze_encoder_epochs: freezeEpochs === FREEZE_AUTO ? null : freezeEpochs,
        class_weighting: classWeighting,
        seed: seed.trim() === '' ? null : Number(seed),
        augmentation: isAudio || isODTask ? {} : augmentation,
      }),
    );
    setIsStarting(false);
    if (raw && raw.training_id) {
      setTrainingId(raw.training_id);
      if (raw.queue_position > 0) toast('info', `Queued behind ${raw.queue_position} other job(s).`);
      pollStatus(raw.training_id);
    }
  };

  const handleCancel = async () => {
    if (!trainingId) return;
    // NOTE: previously this cleared the poll loop and immediately overwrote
    // the status to "cancelled" locally - but the backend only actually
    // stops at the next epoch boundary, so the UI could claim "cancelled"
    // while training was still running, and would never learn the real
    // final outcome since polling had already been killed. Just request
    // the cancellation and let the existing poll loop discover the real
    // status once the backend settles.
    setIsCancelling(true);
    await request(() => apiClient.cancelTraining(trainingId));
    setIsCancelling(false);
  };

  const handleQuickAutoSplit = async () => {
    if (!datasetId) return;
    setIsSplitting(true);
    await request(() => apiClient.autoSplitDataset(datasetId, 70, 20, 10));
    const res = await apiClient.getSplitSummary(datasetId);
    setSplitSummary(res.summary);
    setIsSplitting(false);
  };

  const handleSuggestConfig = async () => {
    if (!datasetId) return;
    setIsSuggesting(true);
    setSuggestionError(null);
    setSuggestionReasoning(null);
    try {
      // Provider (and, for Ollama, its model) come from AppContext - set
      // automatically when the backend .env only exposes one provider, or
      // picked by the user in the Global Config panel when both OpenRouter
      // and Ollama are available (see App.tsx's "AI Studio Assistant"
      // section). Previously this was hardcoded to 'openrouter' with
      // state.llmModel, so selecting Ollama here had no effect at all.
      const modelName = state.llmProvider === 'ollama'
        ? (state.llmConfig?.ollama_model || 'phi3')
        : state.llmModel;
      const result = await request(() =>
        apiClient.getTrainingRecommendation({
          task,
          dataset_id: datasetId,
          target_board: state.currentBoard ?? 'ESP32_S3_N16R8',
          provider: state.llmProvider,
          model_name: modelName,
        })
      );
      const rec = result?.recommendation;
      if (rec) {
        if (rec.base_model) setBaseModel(rec.base_model);
        if (Array.isArray(rec.input_shape) && rec.input_shape.length === 3) setInputShape(rec.input_shape);
        if (typeof rec.batch_size === 'number') setBatchSize(rec.batch_size);
        if (typeof rec.epochs === 'number') setEpochs(rec.epochs);
        if (typeof rec.learning_rate === 'number') setLearningRate(rec.learning_rate);
        if (typeof rec.dropout_rate === 'number') setDropoutRate(rec.dropout_rate);
        if (rec.augmentation && typeof rec.augmentation === 'object') {
          setAugmentation((prev) => ({ ...prev, ...rec.augmentation }));
        }
        setSuggestionReasoning(rec.reasoning ?? null);
      } else {
        // request() swallows the backend's real error into its own `error`
        // state and returns null here - previously this branch always
        // showed a generic "No recommendation returned." regardless of the
        // actual cause (bad API key, network error, invalid model, etc).
        setSuggestionError(error ?? 'No recommendation returned.');
      }
    } catch (e: any) {
      setSuggestionError(e?.message ?? 'Failed to get suggestion.');
    } finally {
      setIsSuggesting(false);
    }
  };

  // --- Derived values ---
  const isRunning = status?.status === 'running' || status?.status === 'initialized' || status?.status === 'queued';
  const availableModels = isODTask ? OD_MODELS : AUDIO_TASKS.includes(task) ? AUDIO_MODELS : IMAGE_MODELS;
  const statusColor =
    status?.status === 'completed'
      ? 'text-green-400'
      : status?.status === 'failed'
        ? 'text-red-400'
        : status?.status === 'cancelled'
          ? 'text-yellow-400'
          : 'text-purple-400';
  const hasUnassigned = splitSummary ? splitSummary.unassigned > 0 : false;
  const splitReady = splitSummary
    ? splitSummary.unassigned === 0 && splitSummary.train > 0 && splitSummary.val > 0
    : false;
  const dimLabels = isAudio
    ? ['n_mfcc', 'time frames', 'channels']
    : ['height', 'width', 'channels'];
  const metricLabel = isODTask ? 'F1' : 'Accuracy';

  // Consistent select class
  const selectCls =
    'w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm disabled:opacity-50 focus:border-purple-500 focus:outline-none';

  // =========================================================================
  // Render
  // =========================================================================
  return (
    <div className="space-y-6">

      {/* --- Configuration Wrapper --- */}
      <div className="bg-slate-800/40 rounded-xl border border-slate-700 overflow-hidden shadow-sm">
        <button
          onClick={() => setIsConfigExpanded(!isConfigExpanded)}
          className="w-full px-6 py-4 flex items-center justify-between bg-slate-800/80 hover:bg-slate-700/80 transition-colors"
        >
          <div className="flex items-center gap-3">
            <Settings className="w-5 h-5 text-purple-400" />
            <h3 className="text-lg font-semibold text-white">Model Configuration</h3>
            {!isConfigExpanded && (
              <span className="hidden sm:inline text-xs text-gray-400 font-mono truncate">
                {isRunning && status
                  ? `${status.base_model ?? baseModel} · ${status.total_epochs} ep · batch ${status.batch_size ?? batchSize} · lr ${status.learning_rate ?? learningRate}`
                  : `${baseModel} · ${epochs} ep · batch ${batchSize} · lr ${learningRate}`}
                {isRunning && <span className="ml-2 text-purple-300 font-sans">(locked while training)</span>}
              </span>
            )}
          </div>
          {isConfigExpanded
            ? <ChevronUp className="w-5 h-5 text-gray-400" />
            : <ChevronDown className="w-5 h-5 text-gray-400" />}
        </button>

        {isConfigExpanded && (
          <div className="p-6 space-y-6 border-t border-slate-700 bg-slate-900/30">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

              {/* Dataset */}
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-300 mb-1">Dataset</label>
                <select
                  value={datasetId}
                  onChange={(e) => setDatasetId(e.target.value)}
                  disabled={isRunning}
                  className={selectCls}
                >
                  <option value="">-- Select a dataset --</option>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.sample_count} samples)
                    </option>
                  ))}
                </select>
                {datasetId && splitSummary && (
                  <div className="mt-2">
                    {hasUnassigned ? (
                      <div className="flex items-start gap-2 p-3 bg-amber-900/30 border border-amber-500/40 rounded-lg text-amber-300 text-xs">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                        <div className="flex-1">
                          <span className="font-semibold">Dataset not fully split.</span>{' '}
                          {splitSummary.unassigned} sample(s) are unassigned.
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
                        ✓ Split ready — train: {splitSummary.train} • val: {splitSummary.val} • test: {splitSummary.test}
                      </p>
                    ) : null}
                  </div>
                )}
              </div>

              {/* AI-assisted config suggestion */}
              <div className="md:col-span-2">
                <button
                  onClick={handleSuggestConfig}
                  disabled={!datasetId || isSuggesting || isRunning}
                  className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-semibold bg-gradient-to-r from-fuchsia-600 to-purple-600 hover:from-fuchsia-500 hover:to-purple-500 disabled:from-slate-800 disabled:to-slate-800 disabled:text-gray-500 text-white transition-all shadow-md"
                >
                  {isSuggesting ? (
                    <><RefreshCw className="w-4 h-4 animate-spin" /> Analyzing dataset &amp; hardware target...</>
                  ) : (
                    <>✨ Suggest Optimal Config for {state.currentBoard?.replace(/_/g, ' ') ?? 'ESP32-S3'} via {state.llmProvider === 'ollama' ? 'Ollama' : 'OpenRouter'}</>
                  )}
                </button>
                {suggestionError && (
                  <p className="text-xs text-red-400 mt-1.5">{suggestionError}</p>
                )}
                {suggestionReasoning && (
                  <div className="mt-2 p-3 bg-fuchsia-900/10 border border-fuchsia-500/20 rounded-lg text-xs text-fuchsia-200/90">
                    <span className="font-semibold text-fuchsia-300">Why this config: </span>
                    {suggestionReasoning}
                  </div>
                )}
              </div>

              {/* Run name */}
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-300 mb-1">Run name <span className="text-gray-500 font-normal">(optional)</span></label>
                <input value={runName} onChange={(e) => setRunName(e.target.value)} disabled={isRunning} maxLength={80}
                  placeholder={`${baseModel} on ${datasets.find((d) => d.id === datasetId)?.name ?? 'dataset'}`}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm disabled:opacity-50 focus:border-purple-500 focus:outline-none" />
              </div>

              {/* Base Model */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Base Model</label>
                <select
                  value={baseModel}
                  onChange={(e) => setBaseModel(e.target.value)}
                  disabled={isRunning}
                  className={selectCls}
                >
                  {availableModels.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
                {MODEL_HINTS[baseModel] && <p className="text-[11px] text-gray-500 mt-1">{MODEL_HINTS[baseModel]}</p>}
              </div>

              {/* Compute Device */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Compute Device
                  {availableDevices && !availableDevices.gpu_available && (
                    <span className="ml-1.5 text-[10px] font-normal text-gray-500 normal-case">(no GPU detected)</span>
                  )}
                </label>
                <select
                  value={device}
                  onChange={(e) => setDevice(e.target.value as 'auto' | 'cpu' | 'gpu')}
                  disabled={isRunning}
                  className={selectCls}
                >
                  <option value="auto">Auto (prefer GPU if available)</option>
                  <option value="cpu">CPU only</option>
                  <option value="gpu" disabled={!!availableDevices && !availableDevices.gpu_available}>
                    GPU (CUDA) only{availableDevices && !availableDevices.gpu_available ? ' - unavailable' : ''}
                  </option>
                </select>
                <p className="text-[11px] text-gray-500 mt-1">
                  Small models often train just as fast (or faster) on CPU once you factor in
                  host↔GPU transfer overhead. Force CPU here to skip that overhead.
                </p>
              </div>

              {/* Epochs — SelectOrCustom */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Epochs</label>
                <SelectOrCustom
                  options={EPOCHS_OPTIONS}
                  value={epochs}
                  onChange={(v) => setEpochs(Number(v))}
                  disabled={isRunning}
                  min={1}
                  max={10000}
                  step={1}
                />
              </div>

              {/* Batch Size — SelectOrCustom */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Batch Size</label>
                <SelectOrCustom
                  options={BATCH_SIZE_OPTIONS}
                  value={batchSize}
                  onChange={(v) => setBatchSize(Number(v))}
                  disabled={isRunning}
                  min={1}
                  max={512}
                  step={1}
                />
              </div>

              {/* Learning Rate — SelectOrCustom */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Learning Rate</label>
                <SelectOrCustom
                  options={LEARNING_RATE_OPTIONS}
                  value={learningRate}
                  onChange={(v) => setLearningRate(Number(v))}
                  disabled={isRunning}
                  min={0.000001}
                  max={1}
                  step={0.0001}
                />
              </div>

            </div>

            {/* --- Regularization (collapsible) --- */}
            <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700 space-y-3">
              <button
                className="flex items-center justify-between w-full"
                onClick={() => setShowRegularization((v) => !v)}
              >
                <span className="text-sm font-medium text-gray-300 flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-cyan-400" />
                  Regularization
                  {(dropoutRate !== 0.3 || l2Reg !== 0 || trainableLayers !== 0 || freezeEpochs !== FREEZE_AUTO || classWeighting || seed !== '') && (
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

                  {/* Dropout Rate */}
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">Dropout Rate</label>
                    <SelectOrCustom
                      options={DROPOUT_OPTIONS}
                      value={dropoutRate}
                      onChange={(v) => setDropoutRate(Number(v))}
                      disabled={isRunning}
                      min={0}
                      max={0.99}
                      step={0.05}
                    />
                    <p className="text-xs text-gray-500 mt-1">Fraction of neurons dropped during training</p>
                  </div>

                  {/* L2 Regularization */}
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">L2 Regularization</label>
                    <select
                      value={l2Reg}
                      onChange={(e) => setL2Reg(Number(e.target.value))}
                      disabled={isRunning}
                      className="w-full px-2 py-1.5 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm disabled:opacity-50 focus:border-purple-500 focus:outline-none"
                    >
                      {[0, 0.0001, 0.0005, 0.001, 0.005, 0.01].map((v) => (
                        <option key={v} value={v}>
                          {v === 0 ? 'off' : v.toExponential(1)}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Trainable Layers */}
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      Fine-tune last N backbone layers
                      <span className="text-gray-500 ml-1" style={{ fontSize: '0.72rem' }}>(0 = whole backbone)</span>
                    </label>
                    <SelectOrCustom
                      options={TRAINABLE_LAYERS_OPTIONS}
                      value={trainableLayers}
                      onChange={(v) => setTrainableLayers(Number(v))}
                      disabled={isRunning}
                      min={0}
                      max={200}
                      step={1}
                    />
                  </div>

                  {/* Freeze Epochs */}
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      Head warm-up epochs
                      <span className="text-gray-500 ml-1" style={{ fontSize: '0.72rem' }}>(backbone frozen first)</span>
                    </label>
                    <SelectOrCustom
                      options={FREEZE_EPOCHS_OPTIONS}
                      value={freezeEpochs}
                      onChange={(v) => setFreezeEpochs(Number(v))}
                      disabled={isRunning}
                      min={0}
                      max={500}
                      step={1}
                    />
                  </div>

                  {/* Class weighting */}
                  <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer select-none">
                    <input type="checkbox" checked={classWeighting} onChange={(e) => setClassWeighting(e.target.checked)}
                      disabled={isRunning || isODTask} className="accent-purple-500" />
                    Balance classes (weight loss by inverse frequency)
                  </label>

                  {/* Seed */}
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">Random seed <span className="text-gray-500">(reproducible runs)</span></label>
                    <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} disabled={isRunning} placeholder="random"
                      className="w-full px-2 py-1.5 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm disabled:opacity-50 focus:border-purple-500 focus:outline-none" />
                  </div>

                </div>
              )}
            </div>

            {/* --- Early Stopping --- */}
            <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700 space-y-3">
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-3 cursor-pointer select-none">
                  <div
                    onClick={() => !isRunning && setEarlyStopping((v) => !v)}
                    className={`relative w-9 h-5 rounded-full transition-colors ${earlyStopping ? 'bg-purple-600' : 'bg-slate-600'
                      } ${isRunning ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}`}
                  >
                    <span
                      className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${earlyStopping ? 'translate-x-4' : ''
                        }`}
                    />
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
                      onChange={(e) => setEsMonitor(e.target.value as any)}
                      disabled={isRunning}
                      className="w-full px-2 py-1.5 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm disabled:opacity-50 focus:border-purple-500 focus:outline-none"
                    >
                      <option value="val_loss">val_loss (recommended)</option>
                      <option value="val_accuracy">val_accuracy</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1">
                      Patience (epochs)
                    </label>
                    <SelectOrCustom
                      options={ES_PATIENCE_OPTIONS}
                      value={esPatience}
                      onChange={(v) => setEsPatience(Number(v))}
                      disabled={isRunning}
                      min={1}
                      max={100}
                      step={1}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* --- Data Augmentation (image classification only) --- */}
            {!isAudio && !isODTask && (
            <div className="p-3 bg-slate-800/50 rounded-xl border border-slate-700">
              <p className="text-sm font-medium text-gray-300 flex items-center gap-2 mb-2">
                <Shuffle className="w-4 h-4 text-purple-400" />
                Data Augmentation
              </p>
              <div className="flex flex-wrap items-center gap-4">
                {/* Horizontal Flip */}
                <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={augmentation.horizontal_flip}
                    onChange={(e) =>
                      setAugmentation({ ...augmentation, horizontal_flip: e.target.checked })
                    }
                    disabled={isRunning}
                    className="accent-purple-500"
                  />
                  Horizontal Flip
                </label>

                {/* Random Crop */}
                <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={augmentation.random_crop}
                    onChange={(e) =>
                      setAugmentation({ ...augmentation, random_crop: e.target.checked })
                    }
                    disabled={isRunning}
                    className="accent-purple-500"
                  />
                  Random Crop
                </label>

                {/* Rotation inline */}
                <div className="flex items-center gap-2">
                  <label className="text-sm text-gray-300 whitespace-nowrap">Rotation (× 360°):</label>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={1}
                    value={augmentation.random_rotation}
                    onChange={(e) =>
                      setAugmentation({
                        ...augmentation,
                        random_rotation: parseFloat(e.target.value) || 0,
                      })
                    }
                    disabled={isRunning}
                    className="w-20 px-2 py-1 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm disabled:opacity-50 focus:border-purple-500 focus:outline-none"
                  />
                </div>

                {([
                  ['random_brightness', 'Brightness'],
                  ['random_contrast', 'Contrast'],
                  ['random_translation', 'Shift'],
                ] as const).map(([key, label]) => (
                  <div key={key} className="flex items-center gap-2">
                    <label className="text-sm text-gray-300 whitespace-nowrap">{label}:</label>
                    <input type="number" step={0.05} min={0} max={0.5} value={augmentation[key]}
                      onChange={(e) => setAugmentation({ ...augmentation, [key]: parseFloat(e.target.value) || 0 })}
                      disabled={isRunning}
                      className="w-20 px-2 py-1 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm disabled:opacity-50 focus:border-purple-500 focus:outline-none" />
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-gray-500 mt-2">Applied on the fly during training only - never baked into the exported model.</p>
            </div>
            )}

            {/* --- Input Shape --- */}
            <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700 space-y-3">
              <p className="text-sm font-medium text-gray-300">
                Input Shape
                <span className="ml-2 text-xs text-gray-500 font-normal">
                  ({inputShape.join(' × ')})
                  {isAudio && ' - time frames follow from the clip length; only n_mfcc is usually worth changing'}
                  {isODTask && ' - height/width must be multiples of 8'}
                </span>
              </p>
              <div className="grid grid-cols-3 gap-3">
                {inputShape.map((val, i) => (
                  <div key={i}>
                    <label className="block text-xs text-gray-400 mb-1">
                      {dimLabels[i] ?? `dim ${i}`}
                    </label>
                    <input
                      type="number"
                      min={1}
                      value={val}
                      onChange={(e) => updateShapeDim(i, e.target.value)}
                      disabled={isRunning}
                      className="w-full px-3 py-1.5 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm text-center disabled:opacity-50 focus:border-purple-500 focus:outline-none"
                    />
                  </div>
                ))}
              </div>
              <button
                onClick={() => setInputShape(defaults.input_shape)}
                disabled={isRunning}
                className="text-xs text-gray-500 hover:text-purple-400 disabled:opacity-40 transition"
              >
                ↺ Reset to default ({defaults.input_shape.join('×')})
              </button>
            </div>

            {/* --- Task info badge --- */}
            <div className="px-3 py-2 bg-slate-800/50 rounded-lg border border-slate-700 text-xs text-gray-400 flex gap-4">
              <span>
                Task:{' '}
                <span className="text-purple-300 font-medium">{task.replace(/_/g, ' ')}</span>
              </span>
              <span>
                Input:{' '}
                <span className="text-cyan-300 font-medium">{inputShape.join('×')}</span>
              </span>
              {isODTask && (
                <span>
                  Output:{' '}
                  <span className="text-emerald-300 font-medium">object centroid grid ({Math.floor(inputShape[0] / 8)}×{Math.floor(inputShape[1] / 8)})</span>
                </span>
              )}
            </div>

          </div>
        )}
      </div>

      {/* --- Error --- */}
      {error && (
        <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* --- Action Buttons --- */}
      <div className="flex gap-3">
        <button
          onClick={handleStart}
          disabled={isRunning || isStarting || !datasetId}
          className="flex-1 flex items-center justify-center gap-2 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed text-white font-bold rounded-xl shadow-lg transition-all"
        >
          {isStarting
            ? <RefreshCw className="w-5 h-5 animate-spin" />
            : <Play className="w-5 h-5" />}
          {isStarting ? 'Submitting...' : 'Start Training'}
        </button>

        {isRunning && (
          <button
            onClick={handleCancel}
            disabled={isCancelling}
            title="Cancel training (takes effect at the end of the current epoch)"
            className="px-4 py-3 bg-red-600/20 hover:bg-red-600/40 disabled:opacity-50 border border-red-500/50 text-red-400 font-semibold rounded-xl transition-all"
          >
            {isCancelling ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Square className="w-5 h-5" />}
          </button>
        )}

        <button
          onClick={() => setShowHistory((v) => !v)}
          className="flex items-center gap-2 px-4 py-3 bg-slate-700 hover:bg-slate-600 border border-slate-600 text-gray-300 font-semibold rounded-xl transition-all"
          title="Past trainings"
        >
          <History className="w-5 h-5" />
          {pastSessions.length > 0 && (
            <span className="text-xs bg-purple-600 text-white rounded-full px-1.5 py-0.5">
              {pastSessions.length}
            </span>
          )}
        </button>
      </div>

      {/* --- Duplicate Warning --- */}
      {duplicateWarning && (
        <div className="p-4 bg-amber-900/30 border border-amber-500/50 rounded-xl flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-amber-300 font-semibold text-sm">Duplicate training detected</p>
            <p className="text-amber-400/80 text-xs mt-1">
              The same dataset, model, epochs, batch size and learning rate were already used in a
              previous training. Do you still want to proceed?
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

      {/* --- Past Sessions Panel --- */}
      {showHistory && (
        <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700 space-y-2 animate-fadeIn">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
              <History className="w-4 h-4 text-purple-400" />
              Past Trainings ({pastSessions.length})
            </h3>
            <label className="flex items-center gap-1.5 text-xs text-gray-400 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showArchived}
                onChange={(e) => setShowArchived(e.target.checked)}
                className="rounded border-slate-600 bg-slate-800 text-purple-500 focus:ring-purple-500 focus:ring-offset-0"
              />
              Show archived
            </label>
          </div>
          {pastSessions.length === 0 ? (
            <p className="text-gray-500 text-sm text-center py-4">
              No past trainings for this task.
            </p>
          ) : (
            <div className="space-y-2 max-h-72 overflow-y-auto custom-scrollbar">
              {pastSessions.map((s) => {
                const last = s.metrics?.length ? s.metrics[s.metrics.length - 1] : null;
                const isSessionRunning = s.status === 'running' || s.status === 'initialized' || s.status === 'queued';
                const sc =
                  s.status === 'completed'
                    ? 'text-green-400 bg-green-900/20 border-green-500/30'
                    : s.status === 'failed'
                      ? 'text-red-400 bg-red-900/20 border-red-500/30'
                      : s.status === 'cancelled'
                        ? 'text-yellow-400 bg-yellow-900/20 border-yellow-500/30'
                        : 'text-purple-400 bg-purple-900/20 border-purple-500/30';
                return (
                  <div
                    key={s.id}
                    className={`w-full text-left p-3 bg-slate-900/50 hover:bg-slate-700/50 rounded-lg border transition-all group ${s.archived ? 'border-slate-800 opacity-60' : 'border-slate-700 hover:border-slate-500'}`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <button onClick={() => setViewingSession(s)} className="flex-1 min-w-0 text-left">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className={`px-2 py-0.5 rounded-full text-xs font-semibold border ${sc}`}>
                            {s.status?.toUpperCase()}
                          </span>
                          <span className="text-white text-sm font-medium">{s.name || s.base_model}</span>
                          {s.name && <span className="text-gray-500 text-xs">{s.base_model}</span>}
                          {s.archived && (
                            <span className="text-[10px] text-gray-500 border border-slate-700 rounded px-1.5 py-0.5">archived</span>
                          )}
                        </div>
                        <div className="flex gap-4 mt-1.5 text-xs text-gray-400">
                          <span>Ep {s.current_epoch}/{s.total_epochs}</span>
                          {last && (
                            <>
                              <span className="text-green-400">{s.task === 'OBJECT_DETECTION' ? 'F1' : 'Acc'} {(last.accuracy * 100).toFixed(1)}%</span>
                              <span className="text-cyan-400">Val {(last.val_accuracy * 100).toFixed(1)}%</span>
                              {s.evaluation && (
                                <span className="text-emerald-300">Test {(((s.task === 'OBJECT_DETECTION' ? s.evaluation.f1 : s.evaluation.accuracy) ?? 0) * 100).toFixed(1)}%</span>
                              )}
                            </>
                          )}
                        </div>
                      </button>
                      <div className="flex items-center gap-1 shrink-0">
                        <span className="text-xs text-gray-500 mr-1 hidden sm:inline">{formatDate(s.created_at)}</span>
                        {isSessionRunning && (
                          <button
                            onClick={async (e) => { e.stopPropagation(); await request(() => apiClient.cancelTraining(s.id)); fetchPastSessions(); }}
                            title="Cancel this training"
                            className="p-1.5 rounded-lg bg-red-600/10 hover:bg-red-600/30 text-red-400 transition-colors"
                          >
                            <Square className="w-3.5 h-3.5" />
                          </button>
                        )}
                        {!isSessionRunning && (
                          <>
                            <button
                              onClick={(e) => { e.stopPropagation(); setSuggestSession(s); }}
                              title="AI Suggestions for this training"
                              className="p-1.5 rounded-lg bg-purple-600/10 hover:bg-purple-600/30 text-purple-400 hover:text-purple-300 transition-colors"
                            >
                              <Lightbulb className="w-3.5 h-3.5" />
                            </button>
                            <button
                              onClick={(e) => { e.stopPropagation(); handleArchiveToggle(s.id, !!s.archived); }}
                              title={s.archived ? 'Unarchive' : 'Archive (hide from history)'}
                              className="p-1.5 rounded-lg bg-slate-700/50 hover:bg-slate-600 text-gray-400 hover:text-white transition-colors"
                            >
                              {s.archived ? <History className="w-3.5 h-3.5" /> : <ShieldCheck className="w-3.5 h-3.5" />}
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* --- Live Training Progress --- */}
      {status && (
        <div className="mt-8 bg-slate-900/80 border border-purple-500/30 rounded-2xl p-6 shadow-2xl relative overflow-hidden animate-fadeIn">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 to-pink-500 opacity-80" />
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-purple-400" />
              Live Training Progress
            </h3>
            <div className="flex items-center gap-2">
              {status.device_used && (
                <span className="px-2.5 py-1 text-[10px] font-bold rounded-full bg-slate-800 border border-slate-700 text-cyan-400 uppercase tracking-wide">
                  {status.device_used === 'gpu' ? '⚡ GPU' : '🖥 CPU'}
                </span>
              )}
              <span
                className={`px-3 py-1 text-xs font-bold rounded-full ${statusColor} bg-slate-800 border border-slate-700`}
              >
                {status.status.toUpperCase()}
              </span>
            </div>
          </div>

          <div className="space-y-6">
            <TrainingDashboard status={status} isRunning={isRunning} isOD={isODTask} metricLabel={metricLabel} />

            {/* Queue */}
            {status.status === 'queued' && (
              <div className="p-3 bg-slate-800 border border-slate-700 rounded-xl text-sm text-gray-300 flex items-center gap-2">
                <Clock className="w-4 h-4 text-cyan-400" />
                Waiting for the ML worker{typeof status.queue_position === 'number' && status.queue_position >= 0 ? ` - ${status.queue_position} job(s) ahead` : ''}.
                Jobs run one at a time so they don't fight over the GPU.
              </div>
            )}

            {/* Held-out evaluation */}
            {status.evaluation && (
              <div className="p-4 bg-slate-800/60 border border-slate-700 rounded-xl">
                <EvaluationReport metrics={status.evaluation} detection={isODTask}
                  title={`Held-out evaluation (${status.evaluation.split} split)`} />
              </div>
            )}

            {/* Training error */}
            {status.status === 'failed' && status.error && (
              <div className="p-4 bg-red-900/30 border border-red-500/50 rounded-xl text-red-300 text-sm">
                <strong className="block mb-1">Training Error:</strong>
                {status.error}
              </div>
            )}

            {/* Live console output */}
            <TerminalLogPanel
              jobId={trainingId}
              title="Training Console"
              defaultOpen={status.status === 'failed'}
            />
          </div>
        </div>
      )}

      {/* --- Past Session Detail Popup --- */}
      {viewingSession && (
        <PastSessionPopup
          session={viewingSession}
          onClose={() => setViewingSession(null)}
        />
      )}

      {/* --- AI Suggest Modal --- */}
      {suggestSession && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm" onClick={() => setSuggestSession(null)}>
          <div
            className="w-full max-w-2xl max-h-[85vh] overflow-y-auto bg-slate-900 rounded-2xl border border-purple-500/30 shadow-2xl animate-slideIn"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700 sticky top-0 bg-slate-900 z-10">
              <div>
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <Lightbulb className="w-5 h-5 text-yellow-400" /> AI Suggestions
                </h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  {suggestSession.base_model} — {suggestSession.total_epochs} epochs
                </p>
              </div>
              <button onClick={() => setSuggestSession(null)} className="p-2 text-gray-400 hover:text-white hover:bg-slate-700 rounded-lg transition" aria-label="Close">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6">
              <Suspense fallback={<div className="text-gray-400 text-sm">Loading advisor...</div>}>
                <LLMAdvisor trainingId={suggestSession.id} status={suggestSession.status} datasetId={suggestSession.dataset_id} pastSessions={pastSessions} />
              </Suspense>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}