import { Lightbulb, TrendingUp, RefreshCw } from 'lucide-react';
import { LLMSuggestion } from '../types';
import { useState } from 'react';
import { useAPI } from '../hooks/useAPI';
import { useAppContext } from '../context/AppContext';

interface LLMAdvisorProps {
  trainingId?: string;
  metrics?: any;
  status?: string;
}

export function LLMAdvisor({ trainingId, status }: LLMAdvisorProps) {
  const { state } = useAppContext();
  const [suggestions, setSuggestions] = useState<LLMSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const { request, apiClient } = useAPI();

  const handleGetSuggestions = async () => {
    if (!trainingId) return;

    setIsLoading(true);
    // Fires using the globally selected model from AppContext
    const result = await request(() =>
      apiClient.getLLMSuggestions(trainingId, 'openrouter', state.llmModel)
    );
    setIsLoading(false);

    if (result && result.suggestions) {
      setSuggestions(result.suggestions);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Action Execution Button */}
      <button
        onClick={handleGetSuggestions}
        disabled={isLoading || !trainingId || status !== 'completed'}
        className="w-full py-3 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:from-slate-800 disabled:to-slate-800 disabled:text-gray-500 text-white rounded-xl font-medium transition-all shadow-lg flex items-center justify-center gap-2"
      >
        {isLoading ? (
          <><RefreshCw className="w-5 h-5 animate-spin" /> Analyzing Metrics...</>
        ) : (
          <><Lightbulb className="w-5 h-5" /> Generate Insights via OpenRouter</>
        )}
      </button>

      {/* Results Feed */}
      <div className="flex-1 overflow-y-auto pr-2 space-y-4 custom-scrollbar mt-2">
        {suggestions.length > 0 ? (
          suggestions.map((sug, idx) => (
            <div key={idx} className="bg-slate-800/80 rounded-xl p-4 border border-slate-700 hover:border-purple-500/50 transition-colors group">
              <div className="flex items-start gap-3 mb-2">
                <div className="w-8 h-8 rounded-full bg-purple-900/50 flex items-center justify-center border border-purple-500/30 flex-shrink-0">
                  <Lightbulb className="w-4 h-4 text-purple-400" />
                </div>
                <div>
                  <h4 className="text-white font-medium text-sm">{sug.suggestion}</h4>
                  <p className="text-gray-400 text-xs mt-1">{sug.reasoning}</p>
                </div>
              </div>

              <div className="ml-11 space-y-2">
                {sug.parameters_to_adjust && Object.entries(sug.parameters_to_adjust).length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(sug.parameters_to_adjust).map(([k, v]) => (
                      <span key={k} className="px-2 py-1 bg-slate-900 rounded-md text-[10px] text-gray-300 font-mono border border-slate-700">
                        {k}: <span className="text-cyan-400">{String(v)}</span>
                      </span>
                    ))}
                  </div>
                )}

                <div className="flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-900/10 py-1.5 px-3 rounded-lg w-fit border border-emerald-500/20">
                  <TrendingUp className="w-3.5 h-3.5" />
                  {sug.estimated_improvement}
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 text-gray-500">
            <Lightbulb className="w-12 h-12 mb-3 opacity-20" />
            <p className="text-sm">Run AI analysis to get optimization suggestions based on your model's performance.</p>
          </div>
        )}
      </div>
    </div>
  );
}