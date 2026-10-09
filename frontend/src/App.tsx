import { useState, useEffect, Suspense, lazy, useCallback } from 'react';
import { Routes, Route, NavLink, useNavigate, useLocation } from 'react-router-dom';
import {
  AlertTriangle, BrainCircuit, Check, Code2, Cpu, Database, FlaskConical, GitBranch, HardDrive,
  LayoutDashboard, Lightbulb, Menu, Settings, X, Activity, KeyRound,
} from 'lucide-react';
import { useAppContext } from './context/AppContext';
import { useHealthCheck } from './hooks';
import { useAPI } from './hooks/useAPI';
import { apiFetch } from './config';
import { TinyMLTask, TargetBoard } from './types';
import { GridSkeleton } from './components/Skeleton';
import { ErrorBoundary } from './components/ErrorBoundary';
import { JobsIndicator } from './components/JobsIndicator';
import { BOARD_OPTIONS, TASK_OPTIONS } from './constants/catalog';

// Route-level code splitting
const DashboardOverview = lazy(() => import('./components/DashboardOverview').then(m => ({ default: m.DashboardOverview })));
const DatasetManager = lazy(() => import('./components/DatasetManager').then(m => ({ default: m.DatasetManager })));
const ModelTrainer = lazy(() => import('./components/ModelTrainer').then(m => ({ default: m.ModelTrainer })));
const OptimizationStudio = lazy(() => import('./components/OptimizationStudio').then(m => ({ default: m.OptimizationStudio })));
const ModelTree = lazy(() => import('./components/ModelTree').then(m => ({ default: m.ModelTree })));
const DeploymentPanel = lazy(() => import('./components/DeploymentPanel').then(m => ({ default: m.DeploymentPanel })));
const LLMAdvisor = lazy(() => import('./components/LLMAdvisor').then(m => ({ default: m.LLMAdvisor })));
const SettingsPage = lazy(() => import('./components/SettingsPage').then(m => ({ default: m.SettingsPage })));
const ExperimentsPage = lazy(() => import('./components/ExperimentsPage').then(m => ({ default: m.ExperimentsPage })));

interface NavItem {
  path: string;
  label: string;
  icon: typeof LayoutDashboard;
  end?: boolean;
  step?: number;
  description: string;
}

// The four pipeline stages, in order, then the supporting pages.
const WORKFLOW: NavItem[] = [
  { path: '/collect', label: 'Data', icon: Database, step: 1, description: 'Import, label and split samples' },
  { path: '/train', label: 'Train', icon: BrainCircuit, step: 2, description: 'Train a model with live metrics' },
  { path: '/optimize', label: 'Optimize', icon: Cpu, step: 3, description: 'Quantize, prune, compare and test variants' },
  { path: '/deploy', label: 'Deploy', icon: Code2, step: 4, description: 'Generate a ready-to-flash Arduino project' },
];
const LIBRARY: NavItem[] = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard, end: true, description: 'Overview of your workspace' },
  { path: '/models', label: 'Models', icon: GitBranch, description: 'Dataset → model → variant lineage' },
  { path: '/experiments', label: 'Experiments', icon: FlaskConical, description: 'Compare training runs side by side' },
];
const SYSTEM: NavItem[] = [
  { path: '/settings', label: 'Settings', icon: Settings, description: 'Connection, AI assistant and defaults' },
];
const ALL_PAGES = [...WORKFLOW, ...LIBRARY, ...SYSTEM];

const panel = 'bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 p-6 sm:p-8 shadow-xl';

export default function App() {
  const { state, dispatch } = useAppContext();
  const isHealthy = useHealthCheck();
  const { request, apiClient } = useAPI();
  const navigate = useNavigate();
  const location = useLocation();

  const selectedTask: TinyMLTask = state.currentTask ?? 'IMAGE_CLASSIFICATION';
  const selectedBoard: TargetBoard = state.currentBoard ?? 'ESP32_S3_N16R8';

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [storageHuman, setStorageHuman] = useState<string | null>(null);
  const [backendVersion, setBackendVersion] = useState<string | null>(null);
  const [authProblem, setAuthProblem] = useState(false);

  useEffect(() => { setSidebarOpen(false); }, [location.pathname]);

  const fetchStatsAndModels = useCallback(async () => {
    const rawStats = await request(() => apiClient.getDatasetStats());
    if (rawStats) {
      dispatch({ type: 'UPDATE_DATASET_STATS', payload: { total_samples: rawStats.total_samples || 0, by_task: rawStats.by_task || {}, by_label: rawStats.by_label || {} } });
    }
    const rawModels = await request(() => apiClient.listModels());
    if (rawModels && rawModels.models) dispatch({ type: 'SET_MODELS', payload: rawModels.models });
    const rawStorage = await request(() => apiClient.getStorageOverview());
    if (rawStorage && rawStorage.overview) setStorageHuman(rawStorage.overview.total_storage_human ?? null);
  }, [request, apiClient, dispatch]);

  const fetchLLMConfig = useCallback(async () => {
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
  }, [request, apiClient, dispatch, state.llmProvider]);

  useEffect(() => {
    if (!isHealthy) return;
    apiFetch('/info').then(async (r) => {
      setAuthProblem(r.status === 401);
      if (r.ok) setBackendVersion((await r.json()).version ?? null);
    }).catch(() => { });
    fetchStatsAndModels();
    fetchLLMConfig();
  }, [isHealthy]); // eslint-disable-line react-hooks/exhaustive-deps

  const stageDone: Record<string, boolean> = {
    '/collect': state.datasetStats.total_samples > 0,
    '/train': state.trainedModels.length > 0,
    '/optimize': state.trainedModels.some((m) => m.optimized),
  };
  const currentPage = ALL_PAGES.find((p) => (p.end ? location.pathname === p.path : location.pathname.startsWith(p.path)));

  const navLink = (item: NavItem) => (
    <NavLink
      key={item.path}
      to={item.path}
      end={item.end}
      title={item.description}
      className={({ isActive }) => `w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-150 text-sm font-medium ${isActive ? 'bg-purple-600/20 text-purple-300 border border-purple-500/30' : 'text-gray-400 hover:bg-slate-800/60 hover:text-gray-200 border border-transparent'}`}
    >
      {({ isActive }) => (
        <>
          {item.step ? (
            <span className={`w-6 h-6 rounded-full grid place-items-center text-[11px] font-bold shrink-0 ${stageDone[item.path] ? 'bg-emerald-500/20 text-emerald-300' : isActive ? 'bg-purple-500 text-white' : 'bg-slate-800 text-slate-400'}`}>
              {stageDone[item.path] && !isActive ? <Check className="w-3.5 h-3.5" /> : item.step}
            </span>
          ) : (
            <item.icon className={`w-5 h-5 shrink-0 ${isActive ? 'text-purple-400' : 'text-gray-500'}`} />
          )}
          <span className="truncate">{item.label}</span>
          {item.step && <item.icon className={`w-4 h-4 ml-auto shrink-0 ${isActive ? 'text-purple-400' : 'text-gray-600'}`} />}
        </>
      )}
    </NavLink>
  );

  const selectCls = 'bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-gray-200 focus:border-purple-500 focus:outline-none';

  return (
    <div className="flex h-screen bg-slate-950 text-gray-100 overflow-hidden">
      <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-[100] focus:px-4 focus:py-2 focus:bg-purple-600 focus:text-white focus:rounded-lg">
        Skip to main content
      </a>

      {sidebarOpen && <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSidebarOpen(false)} />}

      {/* Sidebar */}
      <aside className={`fixed inset-y-0 left-0 z-50 w-64 bg-slate-900 border-r border-slate-800 flex flex-col transform transition-transform duration-200 lg:relative lg:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="h-16 flex items-center justify-between px-5 border-b border-slate-800">
          <NavLink to="/" className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-sm">EC</span>
            </div>
            <span className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">EdgeCraft AI</span>
          </NavLink>
          <button onClick={() => setSidebarOpen(false)} className="lg:hidden p-1 text-gray-400 hover:text-white" aria-label="Close sidebar">
            <X className="w-5 h-5" />
          </button>
        </div>
        <nav className="flex-1 px-3 py-5 space-y-6 overflow-y-auto" aria-label="Main navigation">
          <div className="space-y-1">
            <p className="px-3 mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">Pipeline</p>
            {WORKFLOW.map(navLink)}
          </div>
          <div className="space-y-1">
            <p className="px-3 mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">Workspace</p>
            {LIBRARY.map(navLink)}
          </div>
          <div className="space-y-1">{SYSTEM.map(navLink)}</div>
        </nav>
        <div className="px-5 py-4 border-t border-slate-800 text-[11px] text-slate-500 space-y-1.5">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-green-500' : 'bg-red-500'}`} />
            Backend {isHealthy ? 'online' : 'offline'}{backendVersion ? ` · v${backendVersion}` : ''}
          </div>
          {storageHuman && <div className="flex items-center gap-2"><HardDrive className="w-3 h-3" /> {storageHuman} stored</div>}
          <div className="flex items-center gap-2">
            <Lightbulb className="w-3 h-3" /> AI: {state.llmProvider === 'ollama' ? `Ollama (${state.llmConfig?.ollama_model ?? 'local'})` : state.llmModel}
          </div>
        </div>
      </aside>

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-slate-900/50 backdrop-blur-md border-b border-slate-800 flex items-center gap-3 px-4 sm:px-6 z-30">
          <button onClick={() => setSidebarOpen(true)} className="lg:hidden p-2 text-gray-400 hover:text-white hover:bg-slate-800 rounded-lg" aria-label="Open sidebar">
            <Menu className="w-5 h-5" />
          </button>
          <div className="min-w-0">
            <h1 className="text-base sm:text-lg font-semibold text-white truncate flex items-center gap-2">
              {currentPage?.step && <span className="text-xs text-purple-400 font-bold">Step {currentPage.step}</span>}
              {currentPage?.label ?? 'EdgeCraft AI'}
            </h1>
            <p className="hidden sm:block text-[11px] text-slate-500 truncate">{currentPage?.description}</p>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <JobsIndicator enabled={isHealthy && !authProblem} />
            {!isHealthy && (
              <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-500/10 border border-red-500/30 text-red-300 text-xs">
                <Activity className="w-3 h-3" /> Backend offline
              </span>
            )}
            <label className="hidden sm:flex items-center gap-1.5 text-[11px] text-slate-500">
              Task
              <select value={selectedTask} onChange={(e) => dispatch({ type: 'SET_TASK', payload: e.target.value as TinyMLTask })} className={selectCls} aria-label="Task">
                {TASK_OPTIONS.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
              </select>
            </label>
            <label className="hidden sm:flex items-center gap-1.5 text-[11px] text-slate-500">
              Board
              <select value={selectedBoard} onChange={(e) => dispatch({ type: 'SET_BOARD', payload: e.target.value as TargetBoard })} className={selectCls} aria-label="Target board">
                {BOARD_OPTIONS.map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}
              </select>
            </label>
          </div>
        </header>

        {authProblem && (
          <div className="px-6 py-2.5 bg-amber-500/10 border-b border-amber-500/30 text-amber-200 text-sm flex items-center gap-2">
            <KeyRound className="w-4 h-4" /> The backend requires an API key.
            <button onClick={() => navigate('/settings')} className="underline font-semibold">Open Settings</button>
          </div>
        )}

        <main id="main-content" className="flex-1 overflow-y-auto p-4 sm:p-8 custom-scrollbar">
          <div className="max-w-6xl mx-auto animate-slideIn">
            <Suspense fallback={<GridSkeleton count={4} />}>
              <ErrorBoundary>
                <Routes>
                  <Route path="/" element={<DashboardOverview stats={state.datasetStats} isHealthy={isHealthy} />} />

                  <Route path="/collect" element={
                    <div className={panel}>
                      <DatasetManager task={selectedTask} onDatasetChanged={fetchStatsAndModels} />
                    </div>
                  } />

                  <Route path="/train" element={
                    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                      <div className={state.currentTraining?.status === 'completed' ? 'xl:col-span-2' : 'xl:col-span-3'}>
                        <div className={panel}>
                          <ModelTrainer task={selectedTask} onTrainingComplete={fetchStatsAndModels} />
                        </div>
                      </div>
                      {state.currentTraining?.status === 'completed' && (
                        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-purple-500/30 p-6 shadow-xl h-fit">
                          <h3 className="text-lg font-bold text-white mb-1 flex items-center gap-2">
                            <Lightbulb className="w-5 h-5 text-yellow-400" /> AI review
                          </h3>
                          <p className="text-xs text-gray-400 mb-4 pb-3 border-b border-slate-700">Suggestions based on this run's settings and metrics.</p>
                          <LLMAdvisor trainingId={state.currentTraining?.id} status={state.currentTraining?.status} />
                        </div>
                      )}
                    </div>
                  } />

                  <Route path="/optimize" element={<OptimizationStudio models={state.trainedModels} />} />

                  <Route path="/models" element={
                    <div className={panel}>
                      <p className="text-sm text-gray-400 mb-6">
                        Every dataset, the models trained on it, and the optimized variants of each model. Click a model to optimize it, or a variant to deploy it.
                      </p>
                      <ModelTree
                        onSelectModel={(m) => navigate(`/optimize?model=${m.training_id}`)}
                        onSelectOptimization={(opt) => navigate(`/deploy?optimization=${opt.id}`)}
                      />
                    </div>
                  } />

                  <Route path="/experiments" element={<ExperimentsPage />} />

                  <Route path="/deploy" element={
                    <div className={panel}>
                      <DeploymentPanel board={selectedBoard} />
                    </div>
                  } />

                  <Route path="/settings" element={<SettingsPage />} />

                  <Route path="*" element={
                    <div className="flex flex-col items-center justify-center py-20 text-center">
                      <AlertTriangle className="w-16 h-16 text-yellow-400 mb-6" />
                      <h2 className="text-3xl font-bold text-white mb-2">Page Not Found</h2>
                      <p className="text-gray-400 mb-6">The page you're looking for doesn't exist.</p>
                      <NavLink to="/" className="px-6 py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-xl font-medium transition-colors">
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
