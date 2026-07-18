import { useState, useEffect, useRef, Suspense, lazy } from 'react';
import { Routes, Route, NavLink, useNavigate } from 'react-router-dom';
import { useAppContext } from './context/AppContext';
import { useHealthCheck } from './hooks';
import { TinyMLTask, TargetBoard } from './types';
import {
  BarChart3,
  Code2,
  Zap,
  LayoutDashboard,
  Database,
  BrainCircuit,
  Cpu,
  Settings2,
  Activity,
  Lightbulb,
  ChevronDown,
  Check,
  CpuIcon,
  GitBranch,
  HardDrive,
  Server,
  Cloud,
  AlertTriangle,
  Menu,
  X,
} from 'lucide-react';
import { useAPI } from './hooks/useAPI';
import { useLocalStorage } from './hooks/useLocalStorage';
import { GridSkeleton } from './components/Skeleton';
import { ErrorBoundary } from './components/ErrorBoundary';

// Route-level code splitting
const DashboardOverview = lazy(() => import('./components/DashboardOverview').then(m => ({ default: m.DashboardOverview })));
const DatasetManager = lazy(() => import('./components/DatasetManager').then(m => ({ default: m.DatasetManager })));
const ModelTrainer = lazy(() => import('./components/ModelTrainer').then(m => ({ default: m.ModelTrainer })));
const OptimizationStudio = lazy(() => import('./components/OptimizationStudio').then(m => ({ default: m.OptimizationStudio })));
const ModelTree = lazy(() => import('./components/ModelTree').then(m => ({ default: m.ModelTree })));
const DeploymentPanel = lazy(() => import('./components/DeploymentPanel').then(m => ({ default: m.DeploymentPanel })));
const LLMAdvisor = lazy(() => import('./components/LLMAdvisor').then(m => ({ default: m.LLMAdvisor })));

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { path: '/collect', label: 'Data Collection', icon: Database },
  { path: '/train', label: 'Model Training', icon: BrainCircuit },
  { path: '/optimize', label: 'Optimization', icon: Cpu },
  { path: '/models', label: 'Models', icon: GitBranch },
  { path: '/deploy', label: 'Deployment', icon: Code2 },
];

export default function App() {
  const { state, dispatch } = useAppContext();
  const isHealthy = useHealthCheck();
  const { request, apiClient } = useAPI();
  const navigate = useNavigate();

  const [selectedTask, setSelectedTask] = useLocalStorage<TinyMLTask>('ec_task', 'IMAGE_CLASSIFICATION');
  const [selectedBoard, setSelectedBoard] = useLocalStorage<TargetBoard>('ec_board', 'ESP32_S3_N16R8');

  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [storageOverview, setStorageOverview] = useState<any | null>(null);

  const openRouterModels = [
    { id: 'openrouter/free', label: 'OpenRouter Free', specs: 'Automatic selection' },
    { id: 'google/gemini-2.0-flash-lite-preview-02-05:free', label: 'Gemini Flash Lite (Free)', specs: 'Google - Fast & Accurate' },
    { id: 'meta-llama/llama-3.1-8b-instruct:free', label: 'Llama 3.1 8B (Free)', specs: 'Meta - Open Source Core' },
    { id: 'qwen/qwen-2.5-7b-instruct:free', label: 'Qwen 2.5 7B (Free)', specs: 'Alibaba - Strong coding/logic' },
  ];

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsConfigOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Close sidebar on route change (mobile)
  useEffect(() => {
    setSidebarOpen(false);
  }, [state]);

  const fetchStatsAndModels = async () => {
    const rawStats = await request(() => apiClient.getDatasetStats());
    if (rawStats) {
      dispatch({ type: 'UPDATE_DATASET_STATS', payload: { total_samples: rawStats.total_samples || 0, by_task: rawStats.by_task || {}, by_label: rawStats.by_label || {} } });
    }
    const rawModels = await request(() => apiClient.listModels());
    if (rawModels && rawModels.models) {
      dispatch({ type: 'SET_MODELS', payload: rawModels.models });
    }
    const rawStorage = await request(() => apiClient.getStorageOverview());
    if (rawStorage && rawStorage.overview) {
      setStorageOverview(rawStorage.overview);
    }
  };

  const fetchLLMConfig = async () => {
    const rawConfig = await request(() => apiClient.getLLMConfig());
    if (!rawConfig) return;
    const config = rawConfig.config || rawConfig;
    dispatch({ type: 'SET_LLM_CONFIG', payload: config });

    const { openrouter_available, ollama_available } = config;
    if (ollama_available && !openrouter_available && state.llmProvider !== 'ollama') {
      dispatch({ type: 'SET_LLM_PROVIDER', payload: 'ollama' });
    } else if (openrouter_available && !ollama_available && state.llmProvider !== 'openrouter') {
      dispatch({ type: 'SET_LLM_PROVIDER', payload: 'openrouter' });
    }
  };

  useEffect(() => {
    if (isHealthy) {
      fetchStatsAndModels();
      fetchLLMConfig();
    }
  }, [isHealthy]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleTaskChange = (task: TinyMLTask) => {
    setSelectedTask(task);
    dispatch({ type: 'SET_TASK', payload: task });
  };

  const handleBoardChange = (board: TargetBoard) => {
    setSelectedBoard(board);
    dispatch({ type: 'SET_BOARD', payload: board });
  };

  const taskOptions: { id: TinyMLTask; label: string; desc: string }[] = [
    { id: 'IMAGE_CLASSIFICATION', label: 'Image Classification', desc: 'Categorize whole images' },
    { id: 'OBJECT_DETECTION', label: 'Object Detection', desc: 'Locate and classify items' },
    { id: 'VISUAL_WAKE_WORDS', label: 'Visual Wake Words', desc: 'Binary presence detector (96x96)' },
    { id: 'KEYWORD_SPOTTING', label: 'Keyword Spotting', desc: 'Detect spoken wake words' },
    { id: 'AUDIO_CLASSIFICATION', label: 'Audio Classification', desc: 'Identify continuous audio streams' },
  ];

  const boardOptions: { id: TargetBoard; label: string; specs: string }[] = [
    { id: 'ESP32_S3_N16R8', label: 'ESP32-S3 (N16R8)', specs: 'Xtensa LX7, 16MB Flash, 8MB PSRAM' },
    { id: 'ESP32_CAM', label: 'ESP32-CAM (AI-Thinker)', specs: 'Xtensa LX6, OV2640 camera, ~4MB Flash+PSRAM' },
    { id: 'RASPBERRY_PI_PICO_2_W', label: 'Raspberry Pi Pico 2 W', specs: 'RP2350, 520KB SRAM, Wireless' },
    { id: 'ARDUINO_NANO_33_BLE', label: 'Arduino Nano 33 BLE', specs: 'nRF52840, 256KB RAM, IMU' },
  ];

  const currentTaskLabel = taskOptions.find(t => t.id === selectedTask)?.label || selectedTask;
  const currentBoardLabel = boardOptions.find(b => b.id === selectedBoard)?.label || selectedBoard;

  return (
    <div className="flex h-screen bg-slate-950 text-gray-100 overflow-hidden">
      {/* Skip to content link */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-[100] focus:px-4 focus:py-2 focus:bg-purple-600 focus:text-white focus:rounded-lg"
      >
        Skip to main content
      </a>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar Navigation */}
      <aside className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-slate-900 border-r border-slate-800 flex flex-col transform transition-transform duration-200 ease-in-out
        lg:relative lg:translate-x-0
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="h-16 flex items-center justify-between px-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-sm">EC</span>
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">EdgeCraft AI</h1>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden p-1 text-gray-400 hover:text-white"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto" aria-label="Main navigation">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.end}
              aria-label={item.label}
              className={({ isActive }) => `w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 font-medium ${isActive ? 'bg-purple-600/20 text-purple-400 border border-purple-500/30' : 'text-gray-400 hover:bg-slate-800/50 hover:text-gray-200'}`}
            >
              {({ isActive }) => (
                <>
                  <item.icon className={`w-5 h-5 ${isActive ? 'text-purple-400' : 'text-gray-500'}`} /> {item.label}
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </aside>

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-slate-900/50 backdrop-blur-md border-b border-slate-800 flex items-center justify-between px-4 sm:px-8 z-40">
          <div className="flex items-center gap-3">
            {/* Mobile menu button */}
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden p-2 text-gray-400 hover:text-white hover:bg-slate-800 rounded-lg"
              aria-label="Open sidebar"
            >
              <Menu className="w-5 h-5" />
            </button>

            {/* Health Status Indicator */}
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 border border-slate-700 rounded-full">
              <span className="text-xs text-gray-400 font-medium flex items-center gap-1">
                <Activity className="w-3 h-3 text-gray-500" /> Backend API
              </span>
              <div className="flex items-center gap-1.5 ml-1">
                <span className="relative flex h-2 w-2">
                  {isHealthy && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>}
                  <span className={`relative inline-flex rounded-full h-2 w-2 ${isHealthy ? 'bg-green-500' : 'bg-red-500'}`}></span>
                </span>
                <span className={`text-[10px] font-bold uppercase tracking-wider ${isHealthy ? 'text-green-400' : 'text-red-400'}`}>
                  {isHealthy ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>

            {/* Quick Storage Indicator */}
            {storageOverview && (
              <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 border border-slate-700 rounded-full" title="Storage Used">
                <HardDrive className="w-3 h-3 text-emerald-400" />
                <span className="text-xs text-gray-300 font-mono">{storageOverview.total_storage_human}</span>
              </div>
            )}
          </div>

          {/* Global Configuration Menu */}
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setIsConfigOpen(!isConfigOpen)}
              aria-expanded={isConfigOpen}
              aria-haspopup="true"
              aria-label="Global configuration settings"
              className={`flex items-center gap-3 bg-slate-900 border px-4 py-1.5 rounded-xl text-sm transition-all duration-200 shadow-md ${isConfigOpen
                ? 'border-purple-500 ring-2 ring-purple-500/10 text-white'
                : 'border-slate-700 hover:border-slate-600 text-gray-300 hover:text-white'
                }`}
            >
              <Settings2 className={`w-4 h-4 ${isConfigOpen ? 'text-purple-400 animate-spin-slow' : 'text-gray-400'}`} />
              <div className="hidden sm:flex items-center gap-2 divide-x divide-slate-700 text-xs">
                <span className="text-gray-400 font-medium">Global Config:</span>
                <span className="pl-2 font-semibold text-purple-400">{currentTaskLabel}</span>
                <span className="pl-2 font-semibold text-cyan-400">{currentBoardLabel}</span>
              </div>
              <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform duration-200 ${isConfigOpen ? 'rotate-180' : ''}`} />
            </button>

            {isConfigOpen && (
              <div className="absolute right-0 mt-2 w-[480px] bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl p-5 flex flex-col gap-5 animate-slideIn z-50 backdrop-blur-xl bg-slate-900/95">
                <div>
                  <div className="flex items-center gap-2 mb-3 px-1">
                    <BarChart3 className="w-4 h-4 text-purple-400" />
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Active Studio Pipeline</h4>
                  </div>
                  <div className="grid grid-cols-1 gap-1.5 max-h-[180px] overflow-y-auto custom-scrollbar pr-1">
                    {taskOptions.map((task) => {
                      const isSelected = selectedTask === task.id;
                      return (
                        <button
                          key={task.id}
                          onClick={() => { handleTaskChange(task.id); }}
                          className={`w-full text-left p-2.5 rounded-xl border text-xs transition-all flex items-center justify-between group ${isSelected
                            ? 'bg-purple-600/10 border-purple-500/40 text-purple-300'
                            : 'bg-slate-950/40 border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200'
                            }`}
                        >
                          <div className="flex flex-col gap-0.5">
                            <span className={`font-semibold ${isSelected ? 'text-white' : 'text-slate-300 group-hover:text-white'}`}>
                              {task.label}
                            </span>
                            <span className="text-[10px] text-slate-500 group-hover:text-slate-400">{task.desc}</span>
                          </div>
                          {isSelected && <Check className="w-4 h-4 text-purple-400 shrink-0 ml-2" />}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="h-px bg-slate-800" />

                <div>
                  <div className="flex items-center gap-2 mb-3 px-1">
                    <Zap className="w-4 h-4 text-cyan-400" />
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Target Cross-Compilation Board</h4>
                  </div>
                  <div className="grid grid-cols-1 gap-1.5">
                    {boardOptions.map((board) => {
                      const isSelected = selectedBoard === board.id;
                      return (
                        <button
                          key={board.id}
                          onClick={() => { handleBoardChange(board.id); }}
                          className={`w-full text-left p-2.5 rounded-xl border text-xs transition-all flex items-center justify-between group ${isSelected
                            ? 'bg-cyan-600/10 border-cyan-500/40 text-cyan-300'
                            : 'bg-slate-950/40 border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200'
                            }`}
                        >
                          <div className="flex flex-col gap-0.5">
                            <span className={`font-semibold ${isSelected ? 'text-white' : 'text-slate-300 group-hover:text-white'}`}>
                              {board.label}
                            </span>
                            <span className="text-[10px] text-slate-500 group-hover:text-slate-400 font-mono">{board.specs}</span>
                          </div>
                          {isSelected && <Check className="w-4 h-4 text-cyan-400 shrink-0 ml-2" />}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="h-px bg-slate-800" />

                <div>
                  <div className="flex items-center gap-2 mb-3 px-1">
                    <Lightbulb className="w-4 h-4 text-yellow-400" />
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">AI Studio Assistant</h4>
                  </div>

                  {state.llmConfig && state.llmConfig.openrouter_available && state.llmConfig.ollama_available ? (
                    <div className="grid grid-cols-2 gap-1.5 mb-2">
                      <button
                        onClick={() => dispatch({ type: 'SET_LLM_PROVIDER', payload: 'openrouter' })}
                        className={`flex items-center justify-center gap-1.5 p-2 rounded-xl border text-xs font-semibold transition-all ${state.llmProvider === 'openrouter'
                          ? 'bg-yellow-600/10 border-yellow-500/40 text-yellow-300'
                          : 'bg-slate-950/40 border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200'
                          }`}
                      >
                        <Cloud className="w-3.5 h-3.5" /> OpenRouter
                      </button>
                      <button
                        onClick={() => dispatch({ type: 'SET_LLM_PROVIDER', payload: 'ollama' })}
                        className={`flex items-center justify-center gap-1.5 p-2 rounded-xl border text-xs font-semibold transition-all ${state.llmProvider === 'ollama'
                          ? 'bg-yellow-600/10 border-yellow-500/40 text-yellow-300'
                          : 'bg-slate-950/40 border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200'
                          }`}
                      >
                        <Server className="w-3.5 h-3.5" /> Ollama (local)
                      </button>
                    </div>
                  ) : state.llmConfig ? (
                    <div className="mb-2 px-2.5 py-1.5 rounded-xl border border-slate-800 bg-slate-950/40 text-[10px] text-slate-500 flex items-center gap-1.5">
                      {state.llmProvider === 'ollama' ? <Server className="w-3 h-3" /> : <Cloud className="w-3 h-3" />}
                      Using {state.llmProvider === 'ollama' ? 'local Ollama' : 'OpenRouter'} (only provider configured in backend .env)
                    </div>
                  ) : null}

                  {state.llmProvider === 'ollama' ? (
                    <div className="p-2.5 rounded-xl border border-slate-800 bg-slate-950/40 text-xs text-slate-400">
                      Model: <span className="text-yellow-300 font-mono">{state.llmConfig?.ollama_model || 'phi3'}</span>
                      <div className="text-[10px] text-slate-500 mt-0.5">Set via OLLAMA_MODEL in the backend .env.</div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 gap-1.5">
                      {openRouterModels.map((model) => {
                        const isSelected = state.llmModel === model.id;
                        return (
                          <button
                            key={model.id}
                            onClick={() => dispatch({ type: 'SET_LLM_MODEL', payload: model.id })}
                            className={`w-full text-left p-2.5 rounded-xl border text-xs transition-all flex items-center justify-between group ${isSelected
                              ? 'bg-yellow-600/10 border-yellow-500/40 text-yellow-300'
                              : 'bg-slate-950/40 border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200'
                              }`}
                          >
                            <div className="flex flex-col gap-0.5">
                              <span className={`font-semibold ${isSelected ? 'text-white' : 'text-slate-300 group-hover:text-white'}`}>
                                {model.label}
                              </span>
                              <span className="text-[10px] text-slate-500 group-hover:text-slate-400 font-mono">{model.specs}</span>
                            </div>
                            {isSelected && <Check className="w-4 h-4 text-yellow-400 shrink-0 ml-2" />}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>

                <div className="bg-slate-950/60 p-2 rounded-xl border border-slate-800 flex items-center gap-2 text-[10px] text-slate-500">
                  <CpuIcon size={12} className="text-slate-600" />
                  <span>Modifying variables will dynamically recalibrate processing pipelines.</span>
                </div>
              </div>
            )}
          </div>
        </header>

        <main id="main-content" className="flex-1 overflow-y-auto p-4 sm:p-8 custom-scrollbar">
          <h1 className="sr-only">EdgeCraft AI Studio</h1>
          <div className="max-w-6xl mx-auto animate-slideIn">
            <Suspense fallback={<GridSkeleton count={4} />}>
              <ErrorBoundary>
                <Routes>
                  <Route path="/" element={
                    <DashboardOverview stats={state.datasetStats} isHealthy={isHealthy} />
                  } />

                  <Route path="/collect" element={
                    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 p-8 shadow-xl">
                      <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3 border-b border-slate-700 pb-4"><Database className="w-6 h-6 text-purple-400" /> Dataset Manager</h2>
                      <DatasetManager task={selectedTask} onDatasetChanged={fetchStatsAndModels} />
                    </div>
                  } />

                  <Route path="/train" element={
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 transition-all duration-500">
                      <div className={`transition-all duration-500 ${state.currentTraining?.status === 'completed' ? 'lg:col-span-2' : 'lg:col-span-3'}`}>
                        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 p-8 shadow-xl">
                          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3 border-b border-slate-700 pb-4"><BrainCircuit className="w-6 h-6 text-purple-400" /> Neural Network Training</h2>
                          <ModelTrainer task={selectedTask} onTrainingComplete={fetchStatsAndModels} />
                        </div>
                      </div>

                      {state.currentTraining?.status === 'completed' && (
                        <div className="space-y-6 animate-slideIn">
                          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-purple-500/30 p-6 shadow-xl h-full relative overflow-hidden">
                            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 to-pink-500"></div>
                            <h3 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                              <Lightbulb className="w-5 h-5 text-yellow-400" /> AI Suggestions & Review
                            </h3>
                            <p className="text-sm text-gray-400 mb-6 pb-4 border-b border-slate-700">Based on your specific training parameters and final validation metrics.</p>
                            <LLMAdvisor trainingId={state.currentTraining?.id} status={state.currentTraining?.status} />
                          </div>
                        </div>
                      )}
                    </div>
                  } />

                  <Route path="/optimize" element={
                    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 p-8 shadow-xl">
                      <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3 border-b border-slate-700 pb-4"><Cpu className="w-6 h-6 text-cyan-400" /> TinyML Quantization Studio</h2>
                      <OptimizationStudio models={state.trainedModels} />
                    </div>
                  } />

                  <Route path="/models" element={
                    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 p-8 shadow-xl">
                      <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3 border-b border-slate-700 pb-4"><GitBranch className="w-6 h-6 text-emerald-400" /> Models Explorer</h2>
                      <p className="text-sm text-gray-400 mb-6 -mt-3">
                        Every dataset you've trained on, the models trained from it, and every optimized variant generated
                        from each model. Click a model to open it in Optimization, or a completed optimization to open it in Deployment.
                      </p>
                      <ModelTree
                        onSelectModel={(m) => navigate(`/optimize?model=${m.training_id}`)}
                        onSelectOptimization={(opt) => navigate(`/deploy?optimization=${opt.id}`)}
                      />
                    </div>
                  } />

                  <Route path="/deploy" element={
                    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 p-8 shadow-xl">
                      <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3 border-b border-slate-700 pb-4"><Code2 className="w-6 h-6 text-pink-400" /> Deployment</h2>
                      <DeploymentPanel board={selectedBoard} />
                    </div>
                  } />

                  <Route path="*" element={
                    <div className="flex flex-col items-center justify-center py-20 text-center">
                      <AlertTriangle className="w-16 h-16 text-yellow-400 mb-6" />
                      <h2 className="text-3xl font-bold text-white mb-2">Page Not Found</h2>
                      <p className="text-gray-400 mb-6">The page you're looking for doesn't exist.</p>
                      <NavLink
                        to="/"
                        className="px-6 py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-xl font-medium transition-colors"
                      >
                        Back to Dashboard
                      </NavLink>
                    </div>
                  } />
                </Routes>
              </ErrorBoundary>
            </Suspense>
          </div>
        </main>
      </div>
    </div>
  );
}
