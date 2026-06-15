import { Lightbulb, TrendingUp, AlertCircle, RefreshCw, Cpu, ChevronDown } from 'lucide-react';
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

const OLLAMA_MODELS = ['phi3', 'mistral', 'gemma:2b', 'llama3:8b', 'neural-chat'];

export function LLMAdvisor({ trainingId, metrics, status, optimizationId, board }: LLMAdvisorProps) {
  const [suggestions, setSuggestions] = useState<LLMSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [useLocalLLM, setUseLocalLLM] = useState(false);
  const [llmAvailable, setLlmAvailable] = useState<boolean | null>(null);
  const [llmModel, setLlmModel] = useState('phi3');
  const [selectedModel, setSelectedModel] = useState('phi3');
  const { request, error, apiClient } = useAPI();
  const lastFetchedTrainingId = useRef<string | null>(null);

  // Check if Ollama is running on mount
  useEffect(() => {
    const checkLLM = async () => {
      const raw = await request(() => apiClient.getLLMStatus());
      if (raw?.llm) {
        setLlmAvailable(raw.llm.available ?? false);
        if (raw.llm.model) setLlmModel(raw.llm.model);
      }
    };
    checkLLM();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-fetch suggestions when training completes
  useEffect(() => {
    const autoFetch = async () => {
      if (
        status === 'completed' &&
        trainingId &&
        metrics &&
        metrics.length > 0 &&
        trainingId !== lastFetchedTrainingId.current
      ) {
        lastFetchedTrainingId.current = trainingId;
        await fetchSuggestions();
      }
    };
    autoFetch();
  }, [trainingId, metrics, status]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchSuggestions = async () => {
    if (!trainingId || !metrics?.length) return;
    setIsLoading(true);
    setSuggestions([]);

    const finalMetrics = metrics[metrics.length - 1];
    const raw = await request(() =>
      apiClient.getLLMSuggestions(trainingId, finalMetrics, useLocalLLM)
    );
    setIsLoading(false);

    if (raw) {
      const list = Array.isArray(raw) ? raw : raw.suggestions || [raw];
      setSuggestions(
        list.map((s: any) => ({
          suggestion: s.suggestion || 'Suggestion available',
          reasoning: s.reasoning || '',
          parameters_to_adjust: s.parameters_to_adjust || s.parameters || {},
          estimated_improvement: s.estimated_improvement || s.expected_improvement || '',
        }))
      );
    }
  };

  const LLMStatusBadge = () => {
    if (llmAvailable === null) return null;
    return llmAvailable ? (
      <span className="flex items-center gap-1 text-xs text-green-400 bg-green-900/30 border border-green-600/40 rounded-full px-2 py-0.5">
        <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
        Ollama online · {llmModel}
      </span>
    ) : (
      <span className="flex items-center gap-1 text-xs text-gray-500 bg-slate-800 border border-slate-600 rounded-full px-2 py-0.5">
        <span className="w-1.5 h-1.5 rounded-full bg-gray-500" />
        Ollama offline
      </span>
    );
  };

  return (
    <div className="space-y-4">
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-3">
        <LLMStatusBadge />

        {/* Real LLM toggle */}
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <div
            onClick={() => setUseLocalLLM(v => !v)}
            className={`relative w-9 h-5 rounded-full transition-colors ${useLocalLLM && llmAvailable
                ? 'bg-purple-600'
                : 'bg-slate-600 opacity-60'
              } ${!llmAvailable ? 'cursor-not-allowed' : 'cursor-pointer'}`}
          >
            <span
              className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${useLocalLLM && llmAvailable ? 'translate-x-4' : ''
                }`}
            />
          </div>
          <span className="text-xs text-gray-400 flex items-center gap-1">
            <Cpu className="w-3 h-3" />
            Use local LLM
            {!llmAvailable && <span className="text-gray-600">(install Ollama)</span>}
          </span>
        </label>

        {/* Model selector (only when local LLM enabled) */}
        {useLocalLLM && llmAvailable && (
          <div className="relative">
            <select
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
              className="appearance-none text-xs bg-slate-800 border border-slate-600 rounded-lg pl-2 pr-6 py-1 text-gray-300 focus:outline-none focus:border-purple-500"
            >
              {OLLAMA_MODELS.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <ChevronDown className="absolute right-1 top-1.5 w-3 h-3 text-gray-500 pointer-events-none" />
          </div>
        )}

        {/* Manual re-fetch */}
        {status === 'completed' && trainingId && (
          <button
            onClick={fetchSuggestions}
            disabled={isLoading}
            className="ml-auto flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300 disabled:opacity-40 transition-colors"
          >
            <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        )}
      </div>

      {/* Install hint when local LLM toggled but Ollama is offline */}
      {useLocalLLM && !llmAvailable && (
        <div className="p-3 rounded-lg bg-slate-900/50 border border-amber-700/40 text-xs text-amber-300 space-y-1">
          <p className="font-semibold">Ollama is not running</p>
          <p>Install it from <span className="text-amber-200">ollama.com</span>, then pull a model:</p>
          <code className="block bg-slate-800 rounded p-1.5 mt-1 text-gray-300 font-mono">
            ollama pull phi3
          </code>
          <p className="text-gray-400 mt-1">Then restart the backend and refresh this page.</p>
        </div>
      )}

      {/* Loading */}
      {isLoading ? (
        <div className="text-sm text-gray-400 flex items-center justify-center gap-2 py-6">
          <RefreshCw className="w-4 h-4 animate-spin" />
          {useLocalLLM ? 'Querying local LLM…' : 'Generating AI Analysis…'}
        </div>
      ) : suggestions.length > 0 ? (
        <div className="space-y-3 animate-fadeIn">
          {useLocalLLM && llmAvailable && (
            <p className="text-xs text-purple-400 flex items-center gap-1">
              <Cpu className="w-3 h-3" /> Generated by local LLM ({llmModel})
            </p>
          )}
          {suggestions.map((s, i) => (
            <div
              key={i}
              className="p-4 rounded-lg bg-gradient-to-r from-purple-900/30 to-pink-900/30 border border-purple-500/30"
            >
              <div className="flex gap-3">
                <Lightbulb className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-white mb-1">{s.suggestion}</p>
                  <p className="text-sm text-gray-300 mb-2">{s.reasoning}</p>

                  {s.parameters_to_adjust && Object.keys(s.parameters_to_adjust).length > 0 && (
                    <div className="mb-2 p-2 bg-slate-900/50 rounded text-xs text-gray-300">
                      {Object.entries(s.parameters_to_adjust).map(([key, value]) => (
                        <p key={key}>
                          <span className="text-purple-300">{key}:</span> {String(value)}
                        </p>
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
            <Lightbulb className="w-4 h-4 text-yellow-400" />
            Complete training to auto-generate AI suggestions.
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