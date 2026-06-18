import { useState, useEffect, useRef } from 'react';
import { DataCollector, ModelTrainer, OptimizationStudio, BoardAdvisor, LLMAdvisor, DashboardOverview, DatasetManager } from './components';
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
  CpuIcon
} from 'lucide-react';
import { useAPI } from './hooks/useAPI';

export default function App() {
  const { state, dispatch } = useAppContext();
  const isHealthy = useHealthCheck();
  const { request, apiClient } = useAPI();

  const [selectedTask, setSelectedTask] = useState<TinyMLTask>('IMAGE_CLASSIFICATION');
  const [selectedBoard, setSelectedBoard] = useState<TargetBoard>('ESP32_S3_N16R8');
  const [activeTab, setActiveTab] = useState<'dashboard' | 'collect' | 'train' | 'optimize' | 'deploy'>('dashboard');

  // Local UI State for the custom Global Config dropdown
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const openRouterModels = [
    { id: 'openrouter/free', label: 'OpenRouter Free', specs: 'Automatic selection' },
    { id: 'google/gemini-2.0-flash-lite-preview-02-05:free', label: 'Gemini Flash Lite (Free)', specs: 'Google - Fast & Accurate' },
    { id: 'meta-llama/llama-3.1-8b-instruct:free', label: 'Llama 3.1 8B (Free)', specs: 'Meta - Open Source Core' },
    { id: 'qwen/qwen-2.5-7b-instruct:free', label: 'Qwen 2.5 7B (Free)', specs: 'Alibaba - Strong coding/logic' },
  ];

  // Close dropdown if user clicks outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsConfigOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const fetchStatsAndModels = async () => {
    const rawStats = await request(() => apiClient.getDatasetStats());
    if (rawStats) {
      dispatch({ type: 'UPDATE_DATASET_STATS', payload: { total_samples: rawStats.total_samples || 0, by_task: rawStats.by_task || {}, by_label: rawStats.by_label || {} } });
    }
    const rawModels = await request(() => apiClient.listModels());
    if (rawModels && rawModels.models) {
      dispatch({ type: 'SET_MODELS', payload: rawModels.models });
    }
  };

  useEffect(() => { if (isHealthy) fetchStatsAndModels(); }, [isHealthy]);

  const handleTaskChange = (task: TinyMLTask) => {
    setSelectedTask(task);
    dispatch({ type: 'SET_TASK', payload: task });
  };

  const handleBoardChange = (board: TargetBoard) => {
    setSelectedBoard(board);
    dispatch({ type: 'SET_BOARD', payload: board });
  };

  // Human readable lookups for modern item formatting
  const taskOptions: { id: TinyMLTask; label: string; desc: string }[] = [
    { id: 'IMAGE_CLASSIFICATION', label: 'Image Classification', desc: 'Categorize whole images' },
    { id: 'OBJECT_DETECTION', label: 'Object Detection', desc: 'Locate and classify items' },
    { id: 'VISUAL_WAKE_WORDS', label: 'Visual Wake Words', desc: 'Binary presence detector (96x96)' },
    { id: 'KEYWORD_SPOTTING', label: 'Keyword Spotting', desc: 'Detect spoken wake words' },
    { id: 'AUDIO_CLASSIFICATION', label: 'Audio Classification', desc: 'Identify continuous audio streams' },
  ];

  const boardOptions: { id: TargetBoard; label: string; specs: string }[] = [
    { id: 'ESP32_S3_N16R8', label: 'ESP32-S3 (N16R8)', specs: 'Xtensa LX7, 16MB Flash, 8MB PSRAM' },
    { id: 'RASPBERRY_PI_PICO_2_W', label: 'Raspberry Pi Pico 2 W', specs: 'RP2350, 520KB SRAM, Wireless' },
    { id: 'ARDUINO_NANO_33_BLE', label: 'Arduino Nano 33 BLE', specs: 'nRF52840, 256KB RAM, IMU' },
  ];

  const currentTaskLabel = taskOptions.find(t => t.id === selectedTask)?.label || selectedTask;
  const currentBoardLabel = boardOptions.find(b => b.id === selectedBoard)?.label || selectedBoard;

  return (
    <div className="flex h-screen bg-slate-950 text-gray-100 overflow-hidden">
      {/* Sidebar Navigation */}
      <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col">
        <div className="h-16 flex items-center px-6 border-b border-slate-800">
          <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg flex items-center justify-center mr-3 shadow-lg">
            <span className="text-white font-bold text-sm">EC</span>
          </div>
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">EdgeCraft AI</h1>
        </div>
        <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
          {[
            { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
            { id: 'collect', label: 'Data Collection', icon: Database },
            { id: 'train', label: 'Model Training', icon: BrainCircuit },
            { id: 'optimize', label: 'Optimization', icon: Cpu },
            { id: 'deploy', label: 'Deployment', icon: Code2 },
          ].map((item) => (
            <button key={item.id} onClick={() => setActiveTab(item.id as any)} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 font-medium ${activeTab === item.id ? 'bg-purple-600/20 text-purple-400 border border-purple-500/30' : 'text-gray-400 hover:bg-slate-800/50 hover:text-gray-200'}`}>
              <item.icon className={`w-5 h-5 ${activeTab === item.id ? 'text-purple-400' : 'text-gray-500'}`} /> {item.label}
            </button>
          ))}
        </nav>
      </aside>

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header containing custom Global Config */}
        <header className="h-16 bg-slate-900/50 backdrop-blur-md border-b border-slate-800 flex items-center justify-between px-8 z-50">

          {/* Health Status Indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 border border-slate-700 rounded-full mr-4">
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

          {/* Redesigned Premium Global Configuration Menu */}
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setIsConfigOpen(!isConfigOpen)}
              className={`flex items-center gap-3 bg-slate-900 border px-4 py-1.5 rounded-xl text-sm transition-all duration-200 shadow-md ${isConfigOpen
                ? 'border-purple-500 ring-2 ring-purple-500/10 text-white'
                : 'border-slate-700 hover:border-slate-600 text-gray-300 hover:text-white'
                }`}
            >
              <Settings2 className={`w-4 h-4 ${isConfigOpen ? 'text-purple-400 animate-spin-slow' : 'text-gray-400'}`} />
              <div className="flex items-center gap-2 divide-x divide-slate-700 text-xs">
                <span className="text-gray-400 font-medium">Global Config:</span>
                <span className="pl-2 font-semibold text-purple-400">{currentTaskLabel}</span>
                <span className="pl-2 font-semibold text-cyan-400">{currentBoardLabel}</span>
              </div>
              <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform duration-200 ${isConfigOpen ? 'rotate-180' : ''}`} />
            </button>

            {/* Floating Config Dropdown Panel */}
            {isConfigOpen && (
              <div className="absolute right-0 mt-2 w-[480px] bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl p-5 flex flex-col gap-5 animate-slideIn z-50 backdrop-blur-xl bg-slate-900/95">

                {/* Section 1: ML Pipeline Task Selection */}
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

                {/* Divider Line */}
                <div className="h-px bg-slate-800" />

                {/* Section 2: Target Hardware Device Selection */}
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

                {/* Divider Line */}
                <div className="h-px bg-slate-800" />

                {/* Section 3: AI Assistant Config */}
                <div>
                  <div className="flex items-center gap-2 mb-3 px-1">
                    <Lightbulb className="w-4 h-4 text-yellow-400" />
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">AI Studio Assistant (OpenRouter)</h4>
                  </div>
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
                </div>

                {/* Dropdown Micro Footer info */}
                <div className="bg-slate-950/60 p-2 rounded-xl border border-slate-800 flex items-center gap-2 text-[10px] text-slate-500">
                  <CpuIcon size={12} className="text-slate-600" />
                  <span>Modifying variables will dynamically recalibrate processing pipelines.</span>
                </div>

              </div>
            )}
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-8 custom-scrollbar">
          <div className="max-w-6xl mx-auto animate-slideIn">
            {activeTab === 'dashboard' && <DashboardOverview stats={state.datasetStats} isHealthy={isHealthy} />}

            {activeTab === 'collect' && (
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 p-8 shadow-xl">
                <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3 border-b border-slate-700 pb-4"><Database className="w-6 h-6 text-purple-400" /> Dataset Manager</h2>
                <DatasetManager task={selectedTask} onDatasetChanged={fetchStatsAndModels} />
              </div>
            )}

            {activeTab === 'train' && (
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
                      <LLMAdvisor
                        trainingId={state.currentTraining?.id}
                        metrics={state.currentTraining?.metrics}
                        status={state.currentTraining?.status}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'optimize' && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 p-8 shadow-xl">
                  <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3 border-b border-slate-700 pb-4"><Cpu className="w-6 h-6 text-cyan-400" /> TinyML Quantization Studio</h2>
                  <OptimizationStudio models={state.trainedModels} />
                </div>
                <div className="space-y-6">
                  <div className="bg-slate-800/50 rounded-xl border border-slate-700 p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">Hardware Target</h3>
                    <BoardAdvisor optimizationId={state.currentOptimization?.id} board={selectedBoard} />
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'deploy' && (
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 p-8 shadow-xl max-w-3xl">
                <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3 border-b border-slate-700 pb-4"><Code2 className="w-6 h-6 text-pink-400" /> Export C-Array</h2>
                <div className="p-6 bg-slate-900/80 rounded-lg border border-pink-500/20 text-center">
                  <button disabled={!state.currentOptimization} className="px-8 py-3 bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 disabled:from-slate-700 text-white font-bold rounded-lg transition-all shadow-lg">
                    {state.currentOptimization ? 'Generate edgecraft_model.h' : 'Complete optimization first'}
                  </button>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}