import { useState, useEffect } from 'react';
import {
  DataCollector,
  ModelTrainer,
  OptimizationStudio,
  BoardAdvisor,
  LLMAdvisor,
} from './components';
import { useAppContext } from './context/AppContext';
import { useHealthCheck } from './hooks';
import { TinyMLTask, TargetBoard, TrainingStatus } from './types';
import { BarChart3, Code2, Zap } from 'lucide-react';

export default function App() {
  const { state, dispatch } = useAppContext();
  const isHealthy = useHealthCheck();

  const [selectedTask, setSelectedTask] = useState<TinyMLTask>('IMAGE_CLASSIFICATION');
  const [selectedBoard, setSelectedBoard] = useState<TargetBoard>('ESP32_S3_N16R8');
  const [trainingMetrics, setTrainingMetrics] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'collect' | 'train' | 'optimize' | 'deploy'>('collect');

  const handleTaskSelect = (task: TinyMLTask) => {
    setSelectedTask(task);
    dispatch({ type: 'SET_TASK', payload: task });
  };

  const handleBoardSelect = (board: TargetBoard) => {
    setSelectedBoard(board);
    dispatch({ type: 'SET_BOARD', payload: board });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-black bg-opacity-40 backdrop-blur-md border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-400 to-pink-500 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">EC</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">EdgeCraft AI</h1>
                <p className="text-sm text-purple-200">Local TinyML Studio</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="px-4 py-2 rounded-lg bg-purple-500/20 border border-purple-500/50">
                <p className="text-sm text-purple-100">
                  Backend:{' '}
                  <span className={isHealthy ? 'text-green-400 font-semibold' : 'text-red-400 font-semibold'}>
                    {isHealthy ? 'Connected' : 'Offline'}
                  </span>
                </p>
              </div>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex gap-2 border-b border-purple-500/20">
            {[
              { id: 'collect', label: 'Data Collection', icon: '📊' },
              { id: 'train', label: 'Training', icon: '🚀' },
              { id: 'optimize', label: 'Optimization', icon: '⚡' },
              { id: 'deploy', label: 'Deployment', icon: '📦' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`px-4 py-3 font-medium transition ${activeTab === tab.id
                    ? 'text-white border-b-2 border-purple-400'
                    : 'text-gray-400 hover:text-gray-200'
                  }`}
              >
                {tab.icon} {tab.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Task & Board Selection */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Task Selection */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-purple-500/30 p-6">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-purple-400" />
              Select Task
            </h2>
            <div className="space-y-2">
              {['IMAGE_CLASSIFICATION', 'OBJECT_DETECTION', 'VISUAL_WAKE_WORDS', 'KEYWORD_SPOTTING', 'AUDIO_CLASSIFICATION'].map((task) => (
                <button
                  key={task}
                  onClick={() => handleTaskSelect(task as TinyMLTask)}
                  className={`w-full px-4 py-3 rounded-lg text-left transition border ${selectedTask === task
                      ? 'bg-purple-600 border-purple-400 text-white'
                      : 'bg-purple-500/20 hover:bg-purple-500/40 text-purple-100 border-purple-500/30'
                    }`}
                >
                  {task.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
          </div>

          {/* Board Selection */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-cyan-500/30 p-6">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-cyan-400" />
              Target Board
            </h2>
            <div className="space-y-2">
              {['ESP32_S3_N16R8', 'RASPBERRY_PI_PICO_2_W', 'ARDUINO_NANO_33_BLE'].map((board) => (
                <button
                  key={board}
                  onClick={() => handleBoardSelect(board as TargetBoard)}
                  className={`w-full px-4 py-3 rounded-lg text-left transition border ${selectedBoard === board
                      ? 'bg-cyan-600 border-cyan-400 text-white'
                      : 'bg-cyan-500/20 hover:bg-cyan-500/40 text-cyan-100 border-cyan-500/30'
                    }`}
                >
                  {board.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main Content Area - Tab Based */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Panel */}
          <div className="lg:col-span-2">
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-purple-500/30 p-6">
              {activeTab === 'collect' && (
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">📊 Collect Training Data</h3>
                  <DataCollector task={selectedTask} />
                </div>
              )}

              {activeTab === 'train' && (
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">🚀 Train Model</h3>
                  <ModelTrainer task={selectedTask} />
                </div>
              )}

              {activeTab === 'optimize' && (
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">⚡ Optimize Model</h3>
                  <OptimizationStudio trainingId={state.currentTraining?.id} />
                </div>
              )}

              {activeTab === 'deploy' && (
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Code2 className="w-5 h-5" />
                    Export & Deploy
                  </h3>
                  <div className="space-y-4">
                    <div className="p-4 bg-slate-900/50 rounded-lg border border-pink-500/20">
                      <p className="text-sm text-gray-300 mb-3">
                        Export your optimized model as C-array and deployment files
                      </p>
                      <button
                        disabled={!state.currentOptimization}
                        className="w-full px-4 py-2 bg-pink-600 hover:bg-pink-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition"
                      >
                        Export C-Array
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Board Recommendations */}
            {activeTab === 'optimize' && (
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-cyan-500/30 p-6">
                <h3 className="text-lg font-semibold text-white mb-4">📍 Board Compatibility</h3>
                <BoardAdvisor
                  optimizationId={state.currentOptimization?.id}
                  board={selectedBoard}
                />
              </div>
            )}

            {/* LLM Suggestions */}
            {(activeTab === 'train' || activeTab === 'optimize') && (
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-purple-500/30 p-6">
                <h3 className="text-lg font-semibold text-white mb-4">💡 AI Suggestions</h3>
                <LLMAdvisor
                  trainingId={state.currentTraining?.id}
                  metrics={trainingMetrics}
                  optimizationId={state.currentOptimization?.id}
                  board={selectedBoard}
                />
              </div>
            )}

            {/* Model Info */}
            {state.currentTraining && (
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-blue-500/30 p-6">
                <h3 className="text-lg font-semibold text-white mb-4">📈 Training Status</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Task:</span>
                    <span className="text-white font-semibold">{selectedTask}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Status:</span>
                    <span className="text-blue-300 font-semibold">{state.currentTraining.status}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Progress:</span>
                    <span className="text-white font-semibold">
                      {state.currentTraining.currentEpoch}/{state.currentTraining.totalEpochs}
                    </span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2 mt-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all"
                      style={{ width: `${state.currentTraining.progress}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
