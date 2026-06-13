import { useState } from 'react';
import { Play, Square } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';
import { TrainingConfig, TinyMLTask } from '../types';

interface ModelTrainerProps {
  task: TinyMLTask;
  onTrainingStart?: (trainingId: string) => void;
}

export function ModelTrainer({ task, onTrainingStart }: ModelTrainerProps) {
  const [config, setConfig] = useState<TrainingConfig>({
    epochs: 50,
    batchSize: 32,
    learningRate: 0.001,
    baseModel: 'MobileNetV2',
    task,
    validationSplit: 0.2,
  });

  const [isTraining, setIsTraining] = useState(false);
  const { request, error } = useAPI();

  const handleStart = async () => {
    setIsTraining(true);
    // Implementation will be connected in Phase 4
    setIsTraining(false);
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Epochs</label>
          <input
            type="number"
            value={config.epochs}
            onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Batch Size</label>
          <input
            type="number"
            value={config.batchSize}
            onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Learning Rate</label>
        <input
          type="number"
          step="0.0001"
          value={config.learningRate}
          onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
          className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Base Model</label>
        <select
          value={config.baseModel}
          onChange={(e) => setConfig({ ...config, baseModel: e.target.value })}
          className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
        >
          <option>MobileNetV2</option>
          <option>EfficientNet</option>
          <option>Custom3LayerCNN</option>
        </select>
      </div>

      {error && <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200 text-sm">{error}</div>}

      <button
        onClick={handleStart}
        disabled={isTraining}
        className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg transition flex items-center justify-center gap-2"
      >
        {isTraining ? (
          <>
            <Square className="w-4 h-4" /> Training...
          </>
        ) : (
          <>
            <Play className="w-4 h-4" /> Start Training
          </>
        )}
      </button>
    </div>
  );
}
