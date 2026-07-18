import { Lightbulb, TrendingUp, RefreshCw, AlertTriangle } from 'lucide-react';
import { LLMSuggestion } from '../types';
import { useState } from 'react';
import { useAPI } from '../hooks/useAPI';
import { useAppContext } from '../context/AppContext';

interface LLMAdvisorProps {
  trainingId?: string;
  metrics?: any;
  status?: string;
  datasetId?: string;
  pastSessions?: any[];
}

export function LLMAdvisor({ trainingId, status, datasetId, pastSessions }: LLMAdvisorProps) {
  const { state } = useAppContext();
  const [suggestions, setSuggestions] = useState<LLMSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [includePast, setIncludePast] = useState(false);

  const { request, apiClient, error } = useAPI();

  const handleGetSuggestions = async () => {
    if (!trainingId) return;

    setIsLoading(true);
    const modelName = state.llmProvider === 'ollama'
      ? (state.llmConfig?.ollama_model || 'phi3')
      : state.llmModel;

    const pastData = includePast && pastSessions?.length
      ? pastSessions
          .filter((s: any) => s.id !== trainingId && s.status === 'completed' && s.dataset_id === datasetId)
          .map((s: any) => ({
            training_id: s.id,
            base_model: s.base_model,
            epochs: s.total_epochs,
            batch_size: s.batch_size,
            learning_rate: s.learning_rate,
            dropout_rate: s.dropout_rate,
            l2_reg: s.l2_reg,
            input_shape: s.input_shape,
            final_metrics: s.metrics?.length ? s.metrics[s.metrics.length - 1] : null,
          }))
      : undefined;

    const result = await request(() =>
      apiClient.getLLMSuggestions(trainingId, state.llmProvider, modelName, pastData)
    );
    setIsLoading(false);

    if (result && result.suggestions) {
      setSuggestions(result.suggestions);
    }
  };

  const completedPastCount = pastSessions?.filter(
    (s: any) => s.id !== trainingId && s.status === 'completed' && s.dataset_id === datasetId
  ).length ?? 0;

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Include past training history checkbox */}
      {completedPastCount > 0 && (
        <label className="flex items-center gap-2.5 cursor-pointer group">
          <input
            type="checkbox"
            checked={includePast}
            onChange={(e) => setIncludePast(e.target.checked)}
            className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-purple-500 focus:ring-purple-500/50 focus:ring-offset-0 cursor-pointer"
          />
          <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
            Include past training history
            <span className="text-gray-500 ml-1">({completedPastCount} previous run{completedPastCount !== 1 ? 's' : ''} on this dataset)</span>
          </span>
        </label>
      )}

      {/* Action Execution Button */}
      <button
        onClick={handleGetSuggestions}
        disabled={isLoading || !trainingId || status !== 'completed'}
        className="w-full py-3 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:from-slate-800 disabled:to-slate-800 disabled:text-gray-500 text-white rounded-xl font-medium transition-all shadow-lg flex items-center justify-center gap-2"
      >
        {isLoading ? (
          <><RefreshCw className="w-5 h-5 animate-spin" /> Analyzing Metrics...</>
        ) : (
          <><Lightbulb className="w-5 h-5" /> Generate Insights via {state.llmProvider === 'ollama' ? 'Ollama' : 'OpenRouter'}</>
        )}
      </button>

      {error && (
        <div className="flex items-start gap-2 p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200 text-sm">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Results Feed */}
      <div className="flex-1 overflow-y-auto pr-2 space-y-4 custom-scrollbar mt-2">
        {suggestions.length > 0 ? (
          <>
            {/* Quality Score Badge */}
            {suggestions[0].quality_score != null && (
              <div className={`flex items-center gap-3 p-4 rounded-xl border ${
                suggestions[0].quality_score >= 90 ? 'bg-emerald-900/20 border-emerald-500/40' :
                suggestions[0].quality_score >= 70 ? 'bg-cyan-900/20 border-cyan-500/40' :
                suggestions[0].quality_score >= 50 ? 'bg-yellow-900/20 border-yellow-500/40' :
                'bg-red-900/20 border-red-500/40'
              }`}>
                <div className={`w-14 h-14 rounded-xl flex items-center justify-center text-2xl font-bold ${
                  suggestions[0].quality_score >= 90 ? 'bg-emerald-900/50 text-emerald-400' :
                  suggestions[0].quality_score >= 70 ? 'bg-cyan-900/50 text-cyan-400' :
                  suggestions[0].quality_score >= 50 ? 'bg-yellow-900/50 text-yellow-400' :
                  'bg-red-900/50 text-red-400'
                }`}>
                  {suggestions[0].quality_score}
                </div>
                <div>
                  <div className="text-white font-semibold text-sm">Training Quality Score</div>
                  <div className="text-gray-400 text-xs mt-0.5">
                    {suggestions[0].quality_score >= 90 ? 'Excellent — model is well-optimized for edge deployment' :
                     suggestions[0].quality_score >= 70 ? 'Good — minor improvements possible' :
                     suggestions[0].quality_score >= 50 ? 'Needs work — several areas for improvement' :
                     'Poor — significant issues detected, review suggestions below'}
                  </div>
                </div>
              </div>
            )}

            {suggestions.map((sug, idx) => (
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
          ))}
          </>
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
