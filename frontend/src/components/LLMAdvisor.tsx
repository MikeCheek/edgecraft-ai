import { Lightbulb, TrendingUp, RefreshCw, Cpu, Key } from 'lucide-react';
import { LLMSuggestion } from '../types';
import { useState } from 'react';
import { useAPI } from '../hooks/useAPI';

interface LLMAdvisorProps {
  trainingId?: string;
  metrics?: any;
  status?: string;
}

const OLLAMA_MODELS = ['llama3', 'phi3', 'mistral', 'gemma:2b', 'neural-chat'];
const OPENROUTER_MODELS = ['openrouter/free'];

export function LLMAdvisor({ trainingId, status }: LLMAdvisorProps) {
  const [suggestions, setSuggestions] = useState<LLMSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Provider Toggle Configuration state
  const [provider, setProvider] = useState<'ollama' | 'openrouter'>('ollama');

  // Model Selectors state variables
  const [llmModel, setLlmModel] = useState('llama3');
  const [openRouterModel, setOpenRouterModel] = useState('google/gemini-flash-1.5');
  const [openRouterKey, setOpenRouterKey] = useState('');

  const { request, apiClient } = useAPI();

  const handleGetSuggestions = async () => {
    if (!trainingId) return;

    if (provider === 'openrouter' && !openRouterKey) {
      alert("Please enter your OpenRouter API key.");
      return;
    }

    setIsLoading(true);
    const activeModel = provider === 'ollama' ? llmModel : openRouterModel;

    const result = await request(() =>
      apiClient.getLLMSuggestions(
        trainingId,
        provider,
        activeModel,
        provider === 'openrouter' ? openRouterKey : undefined
      )
    );

    setIsLoading(false);

    if (result && result.suggestions) {
      setSuggestions(result.suggestions);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Selector Container */}
      <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700 flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">

          {/* Local / Cloud Toggle button segment */}
          <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700">
            <button
              onClick={() => setProvider('ollama')}
              className={`px-3 py-1 text-sm rounded-md transition-all whitespace-nowrap ${provider === 'ollama' ? 'bg-purple-600 text-white shadow-md' : 'text-gray-400 hover:text-white'
                }`}
            >
              Local (Ollama)
            </button>
            <button
              onClick={() => setProvider('openrouter')}
              className={`px-3 py-1 text-sm rounded-md transition-all whitespace-nowrap ${provider === 'openrouter' ? 'bg-blue-600 text-white shadow-md' : 'text-gray-400 hover:text-white'
                }`}
            >
              Cloud (OpenRouter)
            </button>
          </div>

          {/* Model selection dropdown menu list */}
          <div className="flex items-center gap-2 w-full sm:flex-1">
            <Cpu className="w-4 h-4 text-gray-400 flex-shrink-0" />
            <select
              value={provider === 'ollama' ? llmModel : openRouterModel}
              onChange={(e) => {
                if (provider === 'ollama') setLlmModel(e.target.value);
                else setOpenRouterModel(e.target.value);
              }}
              className="bg-slate-800 text-white text-sm rounded-lg px-3 py-1.5 border border-slate-700 w-full outline-none focus:border-purple-500 transition-colors cursor-pointer"
            >
              {provider === 'ollama'
                ? OLLAMA_MODELS.map(m => <option key={m} value={m}>{m}</option>)
                : OPENROUTER_MODELS.map(m => <option key={m} value={m}>{m}</option>)
              }
            </select>
          </div>
        </div>

        {/* Dynamic API Token Row for Cloud Integration */}
        {provider === 'openrouter' && (
          <div className="flex items-center gap-2 animate-fadeIn">
            <Key className="w-4 h-4 text-blue-400 flex-shrink-0" />
            <input
              type="password"
              placeholder="Enter OpenRouter API Key (sk-or-v1-...)"
              value={openRouterKey}
              onChange={(e) => setOpenRouterKey(e.target.value)}
              className="bg-slate-800 text-white text-sm rounded-lg px-3 py-1.5 border border-slate-700 flex-1 outline-none focus:border-blue-500 transition-colors"
            />
          </div>
        )}
      </div>

      {/* Action Execution Button Control */}
      <button
        onClick={handleGetSuggestions}
        disabled={isLoading || !trainingId || status !== 'completed'}
        className="w-full py-3 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:from-slate-800 disabled:to-slate-800 disabled:text-gray-500 text-white rounded-xl font-medium transition-all shadow-lg flex items-center justify-center gap-2"
      >
        {isLoading ? (
          <><RefreshCw className="w-5 h-5 animate-spin" /> Analyzing with AI...</>
        ) : (
          <><Lightbulb className="w-5 h-5" /> Generate AI Insights</>
        )}
      </button>

      {/* Results Box Feed Rendering */}
      <div className="flex-1 overflow-y-auto pr-2 space-y-4 custom-scrollbar mt-4">
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
            <p className="text-sm">Run AI analysis to get optimization suggestions for your model architecture and parameters.</p>
          </div>
        )}
      </div>
    </div>
  );
}