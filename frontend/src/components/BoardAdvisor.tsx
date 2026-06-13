import { AlertCircle, CheckCircle, TrendingUp } from 'lucide-react';
import { BoardRecommendation, TargetBoard } from '../types';
import { useState } from 'react';
import { useAPI } from '../hooks/useAPI';

interface BoardAdvisorProps {
  optimizationId?: string;
  board?: TargetBoard;
  onEvaluate?: () => void;
}

export function BoardAdvisor({ optimizationId, board, onEvaluate }: BoardAdvisorProps) {
  const [recommendation, setRecommendation] = useState<BoardRecommendation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { request, error } = useAPI();

  const handleEvaluate = async () => {
    if (!optimizationId || !board) {
      alert('Please complete optimization and select a board');
      return;
    }

    setIsLoading(true);
    // Implementation will be connected in Phase 4
    setIsLoading(false);
  };

  return (
    <div className="space-y-4">
      {recommendation ? (
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-slate-900/50 border border-green-500/30">
            <div className="flex items-start gap-3 mb-4">
              {recommendation.deploymentFeasible ? (
                <CheckCircle className="w-5 h-5 text-green-400 mt-0.5" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
              )}
              <div>
                <p className="font-semibold text-white">{recommendation.boardName}</p>
                <p className="text-sm text-gray-400">
                  {recommendation.deploymentFeasible
                    ? 'Model is compatible with this board'
                    : 'Model may exceed board constraints'}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-xs text-gray-400 mb-1">RAM Usage</p>
                <div className="bg-slate-700 rounded-full h-2 mb-1">
                  <div
                    className={`h-2 rounded-full ${recommendation.ramPercentage > 80 ? 'bg-red-500' : recommendation.ramPercentage > 60 ? 'bg-yellow-500' : 'bg-green-500'}`}
                    style={{ width: `${Math.min(recommendation.ramPercentage, 100)}%` }}
                  ></div>
                </div>
                <p className="text-sm text-white">
                  {recommendation.ramUsageKb} KB / {recommendation.ramUsageKb / (recommendation.ramPercentage / 100)} KB
                </p>
              </div>

              <div>
                <p className="text-xs text-gray-400 mb-1">Flash Usage</p>
                <div className="bg-slate-700 rounded-full h-2 mb-1">
                  <div
                    className={`h-2 rounded-full ${recommendation.flashPercentage > 80 ? 'bg-red-500' : recommendation.flashPercentage > 60 ? 'bg-yellow-500' : 'bg-green-500'}`}
                    style={{ width: `${Math.min(recommendation.flashPercentage, 100)}%` }}
                  ></div>
                </div>
                <p className="text-sm text-white">{recommendation.flashPercentage.toFixed(1)}% used</p>
              </div>
            </div>

            <div className="mb-4">
              <p className="text-xs text-gray-400 mb-2 flex items-center gap-1">
                <TrendingUp className="w-3 h-3" /> Estimated Inference
              </p>
              <p className="text-lg font-semibold text-white">{recommendation.estimatedInferenceMs}ms</p>
            </div>

            {recommendation.warnings.length > 0 && (
              <div className="mb-4 p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                <p className="text-xs font-semibold text-red-300 mb-1">Warnings:</p>
                {recommendation.warnings.map((w, i) => (
                  <p key={i} className="text-xs text-red-200">
                    • {w}
                  </p>
                ))}
              </div>
            )}

            {recommendation.suggestions.length > 0 && (
              <div className="p-3 bg-blue-900/20 border border-blue-500/30 rounded-lg">
                <p className="text-xs font-semibold text-blue-300 mb-1">Suggestions:</p>
                {recommendation.suggestions.map((s, i) => (
                  <p key={i} className="text-xs text-blue-200">
                    • {s}
                  </p>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700">
          <p className="text-sm text-gray-400 mb-3">Select a board to evaluate model compatibility</p>
          <button
            onClick={handleEvaluate}
            disabled={!optimizationId || !board || isLoading}
            className="w-full px-4 py-2 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition"
          >
            {isLoading ? 'Evaluating...' : 'Evaluate for Board'}
          </button>
        </div>
      )}

      {error && <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200 text-sm">{error}</div>}
    </div>
  );
}
