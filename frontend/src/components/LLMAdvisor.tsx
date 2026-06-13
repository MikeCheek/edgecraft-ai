import { Lightbulb, TrendingUp, AlertCircle } from 'lucide-react';
import { LLMSuggestion } from '../types';
import { useState } from 'react';
import { useAPI } from '../hooks/useAPI';

interface LLMAdvisorProps {
  trainingId?: string;
  metrics?: any;
  optimizationId?: string;
  board?: string;
}

export function LLMAdvisor({ trainingId, metrics, optimizationId, board }: LLMAdvisorProps) {
  const [suggestions, setSuggestions] = useState<LLMSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { request, error } = useAPI();

  const handleGetSuggestions = async () => {
    if (!trainingId || !metrics) {
      alert('Complete training first to get optimization suggestions');
      return;
    }

    setIsLoading(true);
    // Implementation will be connected in Phase 4
    setIsLoading(false);
  };

  const handleGetDeploymentAdvice = async () => {
    if (!optimizationId || !board) {
      alert('Complete optimization and select a board for deployment advice');
      return;
    }

    setIsLoading(true);
    // Implementation will be connected in Phase 4
    setIsLoading(false);
  };

  return (
    <div className="space-y-4">
      {suggestions.length > 0 ? (
        <div className="space-y-3">
          {suggestions.map((s, i) => (
            <div key={i} className="p-4 rounded-lg bg-gradient-to-r from-purple-900/30 to-pink-900/30 border border-purple-500/30">
              <div className="flex gap-3">
                <div className="flex-shrink-0">
                  {s.estimatedImprovement.includes('Production') ? (
                    <Lightbulb className="w-5 h-5 text-yellow-400 mt-0.5" />
                  ) : (
                    <TrendingUp className="w-5 h-5 text-green-400 mt-0.5" />
                  )}
                </div>
                <div className="flex-grow">
                  <p className="font-semibold text-white mb-1">{s.suggestion}</p>
                  <p className="text-sm text-gray-300 mb-2">{s.reasoning}</p>

                  {Object.keys(s.parametersToAdjust).length > 0 && (
                    <div className="mb-2 p-2 bg-slate-900/50 rounded border border-slate-700">
                      <p className="text-xs text-gray-400 mb-1">Recommended Parameters:</p>
                      <div className="text-xs text-gray-300 space-y-1">
                        {Object.entries(s.parametersToAdjust).map(([key, value]) => (
                          <p key={key}>
                            <span className="text-purple-300">{key}:</span> {String(value)}
                          </p>
                        ))}
                      </div>
                    </div>
                  )}

                  <p className="text-xs text-green-300 flex items-center gap-1">
                    <TrendingUp className="w-3 h-3" />
                    Expected: {s.estimatedImprovement}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700 space-y-3">
          <p className="text-sm text-gray-400 flex items-center gap-2">
            <Lightbulb className="w-4 h-4 text-yellow-400" />
            AI-Powered Training Suggestions
          </p>

          {trainingId && metrics ? (
            <button
              onClick={handleGetSuggestions}
              disabled={isLoading}
              className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition text-sm"
            >
              {isLoading ? 'Analyzing...' : 'Get Training Suggestions'}
            </button>
          ) : (
            <p className="text-xs text-gray-500">Complete training to receive optimization suggestions</p>
          )}

          {optimizationId && board && (
            <button
              onClick={handleGetDeploymentAdvice}
              disabled={isLoading}
              className="w-full px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition text-sm"
            >
              {isLoading ? 'Analyzing...' : 'Get Deployment Advice'}
            </button>
          )}
        </div>
      )}

      {error && <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200 text-sm flex gap-2">
        <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
        <span>{error}</span>
      </div>}
    </div>
  );
}
