import { useState, useEffect, useCallback } from 'react';
import {
  Database, BrainCircuit, Cpu, ChevronRight, ChevronDown,
  CheckCircle2, XCircle, Loader, Archive, RefreshCw,
} from 'lucide-react';
import { TreeSkeleton } from './Skeleton';
import { formatBytes } from '../utils/format';
import { useAPI } from '../hooks/useAPI';

interface OptimizationNode {
  id: string;
  method: string;
  status: string;
  error?: string | null;
  created_at?: number;
  original_size_bytes: number;
  optimized_size_bytes: number;
  compression_ratio: number;
  comparison_summary?: { deltas: { accuracy_delta: number; speedup_factor: number; size_reduction_pct: number } } | null;
}

interface ModelNode {
  id: string;
  training_id: string;
  name: string;
  task: string;
  base_model?: string;
  device_used?: string;
  accuracy: number;
  val_accuracy: number;
  size_bytes: number;
  created_at?: number;
  archived: boolean;
  optimizations: OptimizationNode[];
}

interface DatasetNode {
  id: string;
  name: string;
  task: string;
  sample_count: number;
  models: ModelNode[];
}

interface ModelTreeData {
  datasets: DatasetNode[];
  unassigned_models: ModelNode[];
}

interface ModelTreeProps {
  /** Called when the user clicks a trained model row. */
  onSelectModel?: (model: ModelNode, dataset: DatasetNode | null) => void;
  /** Called when the user clicks an optimized variant row. */
  onSelectOptimization?: (opt: OptimizationNode, model: ModelNode) => void;
  selectedModelId?: string | null;
  selectedOptimizationId?: string | null;
  /** Restrict the tree to a single dataset (used by the OptimizationStudio picker). */
  datasetIdFilter?: string | null;
  compact?: boolean;
}

function statusIcon(status: string) {
  if (status === 'completed') return <CheckCircle2 size={12} className="text-emerald-400" />;
  if (status === 'failed') return <XCircle size={12} className="text-red-400" />;
  if (status === 'running') return <Loader size={12} className="text-cyan-400 animate-spin" />;
  return <Loader size={12} className="text-slate-500" />;
}

export function ModelTree({
  onSelectModel, onSelectOptimization, selectedModelId, selectedOptimizationId,
  datasetIdFilter, compact = false,
}: ModelTreeProps) {
  const { apiClient } = useAPI();
  const [tree, setTree] = useState<ModelTreeData | null>(null);
  const [loading, setLoading] = useState(false);
  const [expandedDatasets, setExpandedDatasets] = useState<Set<string>>(new Set());
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());

  const fetchTree = useCallback(async () => {
    setLoading(true);
    try {
      const res = await apiClient.getModelTree();
      if (res.status === 'success') {
        setTree(res.tree);
        // Auto-expand the first dataset (or the filtered one) so the tree
        // isn't just a wall of collapsed rows on first load.
        const first = datasetIdFilter
          ? res.tree.datasets.find((d: DatasetNode) => d.id === datasetIdFilter)
          : res.tree.datasets[0];
        if (first) setExpandedDatasets(new Set([first.id]));
      }
    } finally {
      setLoading(false);
    }
  }, [apiClient, datasetIdFilter]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { fetchTree(); }, [fetchTree]);

  const toggleDataset = (id: string) => {
    setExpandedDatasets((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };
  const toggleModel = (id: string) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  if (loading && !tree) {
    return <TreeSkeleton />;
  }
  if (!tree) {
    return <p className="text-sm text-gray-500 text-center py-6">Could not load models.</p>;
  }

  const datasets = datasetIdFilter ? tree.datasets.filter((d) => d.id === datasetIdFilter) : tree.datasets;
  const totalModels = datasets.reduce((n, d) => n + d.models.length, 0) + (datasetIdFilter ? 0 : tree.unassigned_models.length);

  if (totalModels === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <BrainCircuit className="w-10 h-10 mx-auto mb-2 opacity-30" />
        <p className="text-sm">No trained models yet{datasetIdFilter ? ' for this dataset' : ''}.</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-gray-500">{totalModels} model{totalModels !== 1 ? 's' : ''}</span>
        <button
          onClick={fetchTree}
          className="p-1 rounded hover:bg-slate-700 text-gray-500 hover:text-gray-300 transition-colors"
          title="Refresh"
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {datasets.map((ds) => {
        const dsOpen = expandedDatasets.has(ds.id);
        if (ds.models.length === 0) return null;
        return (
          <div key={ds.id} className="rounded-xl border border-slate-700 bg-slate-900/40 overflow-hidden">
            <button
              onClick={() => toggleDataset(ds.id)}
              className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-slate-800/60 transition-colors text-left"
            >
              {dsOpen ? <ChevronDown size={14} className="text-slate-500 shrink-0" /> : <ChevronRight size={14} className="text-slate-500 shrink-0" />}
              <Database size={14} className="text-purple-400 shrink-0" />
              <span className="text-sm font-semibold text-white truncate">{ds.name}</span>
              <span className="text-[10px] text-slate-500 shrink-0">{ds.task?.replace(/_/g, ' ')}</span>
              <span className="ml-auto text-[10px] text-slate-500 shrink-0">{ds.models.length} model{ds.models.length !== 1 ? 's' : ''}</span>
            </button>

            {dsOpen && (
              <div className="px-2 pb-2 space-y-1.5">
                {ds.models.map((m) => {
                  const modelOpen = expandedModels.has(m.id);
                  const isSelectedModel = selectedModelId === m.id;
                  return (
                    <div key={m.id} className={`rounded-lg border ${isSelectedModel ? 'border-violet-500/60 bg-violet-500/5' : 'border-slate-800 bg-slate-950/40'}`}>
                      <div className="flex items-center gap-2 px-3 py-2">
                        {m.optimizations.length > 0 ? (
                          <button onClick={() => toggleModel(m.id)} className="shrink-0">
                            {modelOpen ? <ChevronDown size={12} className="text-slate-500" /> : <ChevronRight size={12} className="text-slate-500" />}
                          </button>
                        ) : (
                          <span className="w-3 shrink-0" />
                        )}
                        <BrainCircuit size={13} className="text-cyan-400 shrink-0" />
                        <button
                          onClick={() => onSelectModel?.(m, ds)}
                          className="flex-1 min-w-0 text-left"
                        >
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-xs font-medium text-white truncate">{m.name}</span>
                            {m.archived && <span title="Archived"><Archive size={10} className="text-slate-600 shrink-0" /></span>}
                          </div>
                          {!compact && (
                            <div className="flex gap-3 text-[10px] text-gray-500 mt-0.5">
                              <span>Acc {(m.accuracy * 100).toFixed(1)}%</span>
                              <span>Val {(m.val_accuracy * 100).toFixed(1)}%</span>
                              <span>{formatBytes(m.size_bytes)}</span>
                              {m.device_used && <span className="uppercase">{m.device_used}</span>}
                            </div>
                          )}
                        </button>
                        {m.optimizations.length > 0 && (
                          <span className="text-[10px] text-slate-500 shrink-0">{m.optimizations.length} opt.</span>
                        )}
                      </div>

                      {modelOpen && m.optimizations.length > 0 && (
                        <div className="pl-8 pr-3 pb-2 space-y-1">
                          {m.optimizations.map((opt) => {
                            const isSelectedOpt = selectedOptimizationId === opt.id;
                            return (
                              <button
                                key={opt.id}
                                onClick={() => onSelectOptimization?.(opt, m)}
                                disabled={opt.status !== 'completed'}
                                className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md border text-left transition-colors ${isSelectedOpt
                                    ? 'border-emerald-500/60 bg-emerald-500/10'
                                    : 'border-slate-800 bg-slate-900/60 hover:bg-slate-800/60'
                                  } ${opt.status !== 'completed' ? 'opacity-60 cursor-not-allowed' : ''}`}
                              >
                                <Cpu size={11} className="text-amber-400 shrink-0" />
                                {statusIcon(opt.status)}
                                <span className="text-[11px] text-slate-300 font-mono">{opt.method}</span>
                                {opt.status === 'completed' && (
                                  <span className="ml-auto text-[10px] text-slate-500 font-mono">
                                    {formatBytes(opt.optimized_size_bytes)}
                                    {opt.comparison_summary && (
                                      <span className="text-cyan-400 ml-1.5">
                                        {opt.comparison_summary.deltas.speedup_factor}x
                                      </span>
                                    )}
                                  </span>
                                )}
                                {opt.status === 'failed' && (
                                  <span className="ml-auto text-[10px] text-red-400 truncate max-w-[140px]">{opt.error}</span>
                                )}
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}

      {!datasetIdFilter && tree.unassigned_models.length > 0 && (
        <div className="rounded-xl border border-amber-700/40 bg-amber-900/10 p-3">
          <p className="text-[11px] text-amber-400 mb-1">Models with no linked dataset (dataset deleted):</p>
          {tree.unassigned_models.map((m) => (
            <button key={m.id} onClick={() => onSelectModel?.(m, null)} className="block text-xs text-slate-300 hover:text-white py-0.5">
              {m.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export type { ModelNode, OptimizationNode, DatasetNode, ModelTreeData };
