import { useState, useEffect } from 'react';
import { DataCollector, ModelTrainer, OptimizationStudio, BoardAdvisor, LLMAdvisor, DashboardOverview, DatasetManager } from './components';
import { useAppContext } from './context/AppContext';
import { useHealthCheck } from './hooks';
import { TinyMLTask, TargetBoard } from './types';
import { BarChart3, Code2, Zap, LayoutDashboard, Database, BrainCircuit, Cpu, Settings2, Activity, Lightbulb } from 'lucide-react';
import { useAPI } from './hooks/useAPI';

export default function App() {
  const { state, dispatch } = useAppContext();
  const isHealthy = useHealthCheck();
  const { request, apiClient } = useAPI();

  const [selectedTask, setSelectedTask] = useState<TinyMLTask>('IMAGE_CLASSIFICATION');
  const [selectedBoard, setSelectedBoard] = useState<TargetBoard>('ESP32_S3_N16R8');
  const [activeTab, setActiveTab] = useState<'dashboard' | 'collect' | 'train' | 'optimize' | 'deploy'>('dashboard');

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

  const handleTaskSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const task = e.target.value as TinyMLTask;
    setSelectedTask(task);
    dispatch({ type: 'SET_TASK', payload: task });
  };

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
        <header className="h-16 bg-slate-900/50 backdrop-blur-md border-b border-slate-800 flex items-center justify-between px-8 z-10">

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

          <div className="flex items-center gap-6 flex-1 justify-end">
            <div className="flex items-center gap-2 text-sm"><Settings2 className="w-4 h-4 text-gray-500" /><span className="text-gray-400">Global Config:</span></div>
            <div className="flex items-center gap-2 bg-slate-800/50 border border-slate-700 rounded-lg px-3 py-1.5">
              <BarChart3 className="w-4 h-4 text-purple-400" />
              <select value={selectedTask} onChange={handleTaskSelect} className="bg-transparent text-sm text-white focus:outline-none cursor-pointer">
                <option value="IMAGE_CLASSIFICATION">Image Classification</option>
                <option value="OBJECT_DETECTION">Object Detection</option>
                <option value="VISUAL_WAKE_WORDS">Visual Wake Words</option>
                <option value="KEYWORD_SPOTTING">Keyword Spotting</option>
                <option value="AUDIO_CLASSIFICATION">Audio Classification</option>
              </select>
            </div>
            <div className="flex items-center gap-2 bg-slate-800/50 border border-slate-700 rounded-lg px-3 py-1.5">
              <Zap className="w-4 h-4 text-cyan-400" />
              <select value={selectedBoard} onChange={(e) => { setSelectedBoard(e.target.value as TargetBoard); dispatch({ type: 'SET_BOARD', payload: e.target.value as TargetBoard }); }} className="bg-transparent text-sm text-white focus:outline-none cursor-pointer">
                <option value="ESP32_S3_N16R8">ESP32-S3 (N16R8)</option>
                <option value="RASPBERRY_PI_PICO_2_W">Raspberry Pi Pico 2 W</option>
                <option value="ARDUINO_NANO_33_BLE">Arduino Nano 33 BLE</option>
              </select>
            </div>
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
                {/* Dynamically shift columns so config dominates the screen until training finishes */}
                <div className={`transition-all duration-500 ${state.currentTraining?.status === 'completed' ? 'lg:col-span-2' : 'lg:col-span-3'}`}>
                  <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 p-8 shadow-xl">
                    <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3 border-b border-slate-700 pb-4"><BrainCircuit className="w-6 h-6 text-purple-400" /> Neural Network Training</h2>
                    <ModelTrainer task={selectedTask} onTrainingComplete={fetchStatsAndModels} />
                  </div>
                </div>

                {/* Only reveal the LLM Card when training hits 'completed' */}
                {state.currentTraining?.status === 'completed' && (
                  <div className="space-y-6 animate-slideIn">
                    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-purple-500/30 p-6 shadow-xl h-full relative overflow-hidden">
                      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 to-pink-500"></div>
                      <h3 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                        <Lightbulb className="w-5 h-5 text-yellow-400" /> AI Suggestions & Review
                      </h3>
                      <p className="text-sm text-gray-400 mb-6 pb-4 border-b border-slate-700">Based on your specific training parameters and final validation metrics, our AI provides actionable insights for deployment or further parameter tuning.</p>
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