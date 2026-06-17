import { useState } from 'react';
import { Cpu, AlertTriangle, CheckCircle2, RefreshCw } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';
import { BoardRecommendation } from '../types';

interface BoardAdvisorProps {
  optimizationId?: string;
  board?: string;
}

export function BoardAdvisor({ optimizationId, board }: BoardAdvisorProps) {
  const [recommendation, setRecommendation] = useState<BoardRecommendation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { request, error, apiClient } = useAPI();

  const handleEvaluate = async () => {
    if (!optimizationId || !board) {
      alert("Please complete an optimization session and select a board first.");
      return;
    }

    setIsLoading(true);
    const result = await request(() => apiClient.evaluateBoard(optimizationId, board));
    setIsLoading(false);

    // Ensure the backend returned the 'recommendation' object safely
    if (result && result.recommendation) {
      setRecommendation(result.recommendation);
    }
  };

  return (
    <div className="space-y-4">
      {/* Action Button */}
      <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700">
        <p className="text-sm text-gray-400 mb-3">
          Simulate the deployment of your optimized model onto the physical hardware to check for memory constraints.
        </p>
        <button
          onClick={handleEvaluate}
          disabled={!optimizationId || !board || isLoading}
          className="w-full px-4 py-2 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition text-sm flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <><RefreshCw className="w-4 h-4 animate-spin" /> Simulating...</>
          ) : (
            <><Cpu className="w-4 h-4" /> Evaluate for {board ? board.replace(/_/g, ' ') : 'Board'}</>
          )}
        </button>
      </div>

      {/* Error State */}
      {error && (
        <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200 text-sm">
          {error}
        </div>
      )}

      {/* Results View */}
      {recommendation && (
        <div className="space-y-4 animate-slideIn">
          <div className="grid grid-cols-2 gap-3 text-sm">

            {/* RAM Usage Block */}
            <div className="p-3 bg-slate-800 rounded-lg border border-slate-600">
              <span className="text-gray-400 block mb-1">RAM Usage</span>
              <div className="flex justify-between items-end mb-1">
                <span className="text-white font-bold">{recommendation.ram_usage_kb} KB</span>
                <span className={recommendation.ram_percentage > 80 ? 'text-red-400' : 'text-green-400'}>
                  {recommendation.ram_percentage?.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-900 rounded-full h-1.5">
                <div
                  className={`h-1.5 rounded-full ${recommendation.ram_percentage > 80 ? 'bg-red-500' : 'bg-green-500'}`}
                  style={{ width: `${Math.min(recommendation.ram_percentage, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* Flash Usage Block */}
            <div className="p-3 bg-slate-800 rounded-lg border border-slate-600">
              <span className="text-gray-400 block mb-1">Flash Usage</span>
              <div className="flex justify-between items-end mb-1">
                <span className="text-white font-bold">{recommendation.flash_usage_kb} KB</span>
                <span className={recommendation.flash_percentage > 80 ? 'text-red-400' : 'text-green-400'}>
                  {recommendation.flash_percentage.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-900 rounded-full h-1.5">
                <div
                  className={`h-1.5 rounded-full ${recommendation.flash_percentage > 80 ? 'bg-red-500' : 'bg-green-500'}`}
                  style={{ width: `${Math.min(recommendation.flash_percentage, 100)}%` }}
                ></div>
              </div>
            </div>
          </div>

          {/* Warnings (if any) */}
          {recommendation.warnings && recommendation.warnings.length > 0 && (
            <div className="p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
              <h4 className="text-xs font-semibold text-yellow-400 mb-2 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" /> Warnings
              </h4>
              <ul className="text-xs text-yellow-200/80 space-y-1 list-disc pl-4">
                {recommendation.warnings.map((warn, i) => (
                  <li key={i}>{warn}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Suggestions */}
          {recommendation.suggestions && recommendation.suggestions.length > 0 && (
            <div className="p-3 bg-green-900/20 border border-green-500/30 rounded-lg">
              <h4 className="text-xs font-semibold text-green-400 mb-2 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> Hardware Suggestions
              </h4>
              <ul className="text-xs text-green-200/80 space-y-1 list-disc pl-4">
                {recommendation.suggestions.map((sug, i) => (
                  <li key={i}>{sug}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}