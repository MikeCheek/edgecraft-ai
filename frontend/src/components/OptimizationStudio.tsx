import { useState } from 'react';
import { Download, Copy, AlertCircle } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';
import { QuantizationMethod } from '../types';

interface OptimizationStudioProps {
  trainingId?: string;
  onOptimizationComplete?: () => void;
}

export function OptimizationStudio({ trainingId, onOptimizationComplete }: OptimizationStudioProps) {
  const [method, setMethod] = useState<QuantizationMethod>('INT8_QUANTIZATION');
  const [sparsityLevel, setSparsityLevel] = useState(0.5);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [cArray, setCArray] = useState<string | null>(null);
  const { request, error } = useAPI();

  const handleOptimize = async () => {
    if (!trainingId) {
      alert('Please select a trained model first');
      return;
    }

    setIsOptimizing(true);
    // Implementation will be connected in Phase 4
    setIsOptimizing(false);
  };

  const handleCopy = () => {
    if (cArray) {
      navigator.clipboard.writeText(cArray);
      alert('C-array copied to clipboard!');
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Quantization Method</label>
        <select
          value={method}
          onChange={(e) => setMethod(e.target.value as QuantizationMethod)}
          className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-orange-500"
        >
          <option value="INT8_QUANTIZATION">INT8 Quantization (Small Size)</option>
          <option value="FLOAT16_QUANTIZATION">FLOAT16 Quantization (Balanced)</option>
          <option value="PRUNING">Pruning (Sparse)</option>
        </select>
      </div>

      {method === 'PRUNING' && (
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Sparsity Level: {(sparsityLevel * 100).toFixed(0)}%</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={sparsityLevel}
            onChange={(e) => setSparsityLevel(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      )}

      <div className="p-3 bg-blue-900/30 border border-blue-500/50 rounded-lg flex gap-2">
        <AlertCircle className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-blue-200">
          Quantization reduces model size by ~{method === 'INT8_QUANTIZATION' ? '75' : method === 'FLOAT16_QUANTIZATION' ? '50' : '35'}% with minimal accuracy loss.
        </div>
      </div>

      {error && <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200 text-sm">{error}</div>}

      <button
        onClick={handleOptimize}
        disabled={isOptimizing || !trainingId}
        className="w-full px-4 py-2 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition flex items-center justify-center gap-2"
      >
        <Download className="w-4 h-4" />
        {isOptimizing ? 'Optimizing...' : 'Quantize Model'}
      </button>

      {cArray && (
        <div className="space-y-2">
          <div className="p-3 bg-slate-900/50 rounded-lg border border-green-500/30">
            <p className="text-xs text-gray-400 mb-2">C-Array Preview (click to expand):</p>
            <pre className="text-xs text-green-400 overflow-auto max-h-32">{cArray.slice(0, 200)}...</pre>
          </div>
          <button
            onClick={handleCopy}
            className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition flex items-center justify-center gap-2"
          >
            <Copy className="w-4 h-4" />
            Copy C-Array
          </button>
        </div>
      )}
    </div>
  );
}
