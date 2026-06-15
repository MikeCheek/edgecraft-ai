import { Lightbulb, TrendingUp, AlertCircle, RefreshCw } from 'lucide-react';
import { LLMSuggestion } from '../types';
import { useState, useEffect, useRef } from 'react';
import { useAPI } from '../hooks/useAPI';

interface LLMAdvisorProps {
  trainingId?: string;
  metrics?: any;
  status?: string;
  optimizationId?: string;
  board?: string;
}

export function LLMAdvisor({ trainingId, metrics, status, optimizationId, board }: LLMAdvisorProps) {
  const [suggestions, setSuggestions] = useState<LLMSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { request, error, apiClient } = useAPI();
  const lastFetchedTrainingId = useRef<string | null>(null);

  useEffect(() => {
    const autoFetch = async () => {
      if (status === 'completed' && trainingId && metrics && metrics.length > 0 && trainingId !== lastFetchedTrainingId.current) {
        lastFetchedTrainingId.current = trainingId;
        setIsLoading(true);

        const finalMetrics = metrics[metrics.length - 1];

        const raw = await request(() => apiClient.getLLMSuggestions(trainingId, finalMetrics));
        setIsLoading(false);

        if (raw) {
          const list = Array.isArray(raw) ? raw : raw.suggestions || [raw];

          setSuggestions(list.map((s: any) => ({
            suggestion: s.suggestion || 'Suggestion available',
            reasoning: s.reasoning || '',
            // Fallback mapper to catch Python mismatches between LocalLLMAdvisor and LLMAdvisor
            parameters_to_adjust: s.parameters_to_adjust || s.parameters || {},
            estimated_improvement: s.estimated_improvement || s.expected_improvement || ''
          })));
        }
      }
    };
    autoFetch();
  }, [trainingId, metrics, status, request, apiClient]);

  return (
    <div className="space-y-4">
      {isLoading ? (
        <div className="text-sm text-gray-400 flex items-center justify-center gap-2 py-6">
          <RefreshCw className="w-4 h-4 animate-spin" /> Generating AI Analysis...
        </div>
      ) : suggestions.length > 0 ? (
        <div className="space-y-3 animate-fadeIn">
          {suggestions.map((s, i) => (
            <div key={i} className="p-4 rounded-lg bg-gradient-to-r from-purple-900/30 to-pink-900/30 border border-purple-500/30">
              <div className="flex gap-3">
                <Lightbulb className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-white mb-1">{s.suggestion}</p>
                  <p className="text-sm text-gray-300 mb-2">{s.reasoning}</p>

                  {/* SAFE RENDERING: Ensure parameters_to_adjust exists before accessing keys */}
                  {s.parameters_to_adjust && Object.keys(s.parameters_to_adjust).length > 0 && (
                    <div className="mb-2 p-2 bg-slate-900/50 rounded text-xs text-gray-300">
                      {Object.entries(s.parameters_to_adjust).map(([key, value]) => (
                        <p key={key}><span className="text-purple-300">{key}:</span> {String(value)}</p>
                      ))}
                    </div>
                  )}

                  {s.estimated_improvement && (
                    <p className="text-xs text-green-300 flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" /> Expected: {s.estimated_improvement}
                    </p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700">
          <p className="text-sm text-gray-400 flex items-center gap-2">
            <Lightbulb className="w-4 h-4 text-yellow-400" /> Complete training to auto-generate AI suggestions.
          </p>
        </div>
      )}
      {error && (
        <div className="text-red-400 text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4" /> {error}
        </div>
      )}
    </div>
  );
}