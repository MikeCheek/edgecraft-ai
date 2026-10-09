// Post-training review. The backend computes a deterministic analysis of the
// whole run (score, findings with evidence, validated recommendations) and,
// on request, asks the configured LLM to turn it into a prioritised plan.
// AI failures fall back to the rule-based recommendations, never to an error.

import {
  Lightbulb, RefreshCw, AlertTriangle, CheckCircle2, Info, XCircle, Sparkles, Wand2, ChevronDown, ChevronRight,
  Database, Cpu, SlidersHorizontal,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useAPI } from '../hooks/useAPI';
import { useAppContext } from '../context/AppContext';
import { ReviewSeverity, ReviewSuggestion, TrainingChanges, TrainingReview } from '../types';
import { applyTrainingChanges, mergeSuggestionChanges } from '../utils/trainingChanges';

interface LLMAdvisorProps {
  trainingId?: string;
  status?: string;
  datasetId?: string;
  /** @deprecated past runs are read by the backend */
  pastSessions?: any[];
}

const SEVERITY: Record<ReviewSeverity, { icon: typeof Info; cls: string; label: string }> = {
  critical: { icon: XCircle, cls: 'text-red-300 bg-red-500/10 border-red-500/30', label: 'Problem' },
  warning: { icon: AlertTriangle, cls: 'text-amber-300 bg-amber-500/10 border-amber-500/30', label: 'Warning' },
  info: { icon: Info, cls: 'text-sky-300 bg-sky-500/10 border-sky-500/30', label: 'Info' },
  good: { icon: CheckCircle2, cls: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30', label: 'Good' },
};

const PRIORITY_CLS = {
  high: 'bg-red-500/15 text-red-300 border-red-500/30',
  medium: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  low: 'bg-slate-700/60 text-gray-300 border-slate-600',
};

const CATEGORY_ICON = { training: SlidersHorizontal, data: Database, deployment: Cpu };

const PARAM_LABEL: Record<string, string> = {
  base_model: 'model', input_shape: 'input', epochs: 'epochs', batch_size: 'batch', learning_rate: 'lr',
  dropout_rate: 'dropout', l2_reg: 'L2', early_stopping: 'early stop', early_stopping_patience: 'patience',
  early_stopping_monitor: 'monitor', freeze_encoder_epochs: 'frozen epochs', trainable_layers: 'trainable layers',
  class_weighting: 'class weights', augmentation: 'augmentation',
};

const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

function fmtValue(v: unknown): string {
  if (v == null) return '—';
  if (Array.isArray(v)) return v.join('×');
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  if (typeof v === 'number') return Math.abs(v) > 0 && Math.abs(v) < 0.01 ? v.toExponential(0) : String(+v.toFixed(4));
  return String(v);
}

function scoreTone(score: number) {
  if (score >= 85) return { ring: 'border-emerald-500/50 bg-emerald-900/20', text: 'text-emerald-300' };
  if (score >= 70) return { ring: 'border-cyan-500/50 bg-cyan-900/20', text: 'text-cyan-300' };
  if (score >= 50) return { ring: 'border-amber-500/50 bg-amber-900/20', text: 'text-amber-300' };
  return { ring: 'border-red-500/50 bg-red-900/20', text: 'text-red-300' };
}

function ChangeChips({ changes, base }: { changes: TrainingChanges; base: Record<string, any> }) {
  const entries = Object.entries(changes);
  if (!entries.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {entries.flatMap(([k, v]) => {
        if (k === 'augmentation' && v && typeof v === 'object') {
          return Object.entries(v).map(([ak, av]) => (
            <span key={`aug-${ak}`} className="px-2 py-0.5 bg-slate-900 rounded-md text-[10px] text-gray-300 font-mono border border-slate-700">
              {ak}: <span className="text-cyan-300">{fmtValue(av)}</span>
            </span>
          ));
        }
        return [(
          <span key={k} className="px-2 py-0.5 bg-slate-900 rounded-md text-[10px] text-gray-300 font-mono border border-slate-700">
            {PARAM_LABEL[k] ?? k}: <span className="text-gray-500">{fmtValue(base?.[k])}</span> → <span className="text-cyan-300">{fmtValue(v)}</span>
          </span>
        )];
      })}
    </div>
  );
}

function SuggestionCard({ sug, base, onApply }: { sug: ReviewSuggestion; base: Record<string, any>; onApply?: (c: TrainingChanges) => void }) {
  const Cat = CATEGORY_ICON[sug.category] ?? SlidersHorizontal;
  const hasChanges = Object.keys(sug.changes || {}).length > 0;
  return (
    <li className="bg-slate-800/80 rounded-xl p-3.5 border border-slate-700">
      <div className="flex items-start gap-2.5">
        <Cat className="w-4 h-4 text-purple-300 mt-0.5 shrink-0" aria-label={sug.category} />
        <div className="min-w-0 flex-1 space-y-1.5">
          <div className="flex flex-wrap items-center gap-1.5">
            <h4 className="text-white font-medium text-sm mr-1">{sug.title}</h4>
            <span className={`px-1.5 py-0.5 rounded border text-[9px] uppercase tracking-wide ${PRIORITY_CLS[sug.priority]}`}>{sug.priority}</span>
          </div>
          {sug.reasoning && <p className="text-gray-400 text-xs leading-relaxed">{sug.reasoning}</p>}
          <ChangeChips changes={sug.changes} base={base} />
          {sug.rejected_changes?.length ? (
            <p className="text-[10px] text-gray-500">Ignored: {sug.rejected_changes.join('; ')}</p>
          ) : null}
          <div className="flex flex-wrap items-center justify-between gap-2 pt-0.5">
            {sug.expected_effect && <span className="text-[11px] text-emerald-300/90">→ {sug.expected_effect}</span>}
            {hasChanges && onApply && (
              <button onClick={() => onApply(sug.changes)}
                className="ml-auto flex items-center gap-1 px-2.5 py-1 text-[11px] rounded-lg bg-purple-600/80 hover:bg-purple-500 text-white transition">
                <Wand2 className="w-3 h-3" /> Apply to configuration
              </button>
            )}
          </div>
        </div>
      </div>
    </li>
  );
}

export function LLMAdvisor({ trainingId, status, datasetId }: LLMAdvisorProps) {
  const { state } = useAppContext();
  const { request, apiClient, error } = useAPI();
  const [review, setReview] = useState<TrainingReview | null>(null);
  const [loading, setLoading] = useState<'rules' | 'ai' | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [showRules, setShowRules] = useState(false);
  const [showBreakdown, setShowBreakdown] = useState(false);
  const [showPast, setShowPast] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const reviewable = status === 'completed' || status === 'cancelled';
  const cfg = state.llmConfig;
  const provider = state.llmProvider;
  const aiAvailable = provider === 'ollama' ? !!cfg?.ollama_available : cfg ? !!cfg.openrouter_available : true;
  const modelName = provider === 'ollama' ? (cfg?.ollama_model || 'phi3') : state.llmModel;
  const providerLabel = provider === 'ollama' ? 'Ollama' : 'OpenRouter';

  const load = useCallback(async (withAi: boolean) => {
    if (!trainingId) return;
    setLoading(withAi ? 'ai' : 'rules');
    setElapsed(0);
    if (timer.current) clearInterval(timer.current);
    const t0 = Date.now();
    timer.current = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 1000);
    const res = await request(() => apiClient.getTrainingReview(trainingId, {
      provider: withAi ? provider : null,
      modelName: withAi ? modelName : undefined,
      board: state.currentBoard,
      useCache: !withAi,
    }));
    if (timer.current) clearInterval(timer.current);
    setLoading(null);
    if (res?.review) setReview(res.review);
  }, [trainingId, provider, modelName, state.currentBoard]); // eslint-disable-line react-hooks/exhaustive-deps

  // Deterministic review (or the stored AI review) as soon as the run is done.
  useEffect(() => {
    setReview(null);
    if (trainingId && reviewable) load(false);
  }, [trainingId, reviewable]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => () => { if (timer.current) clearInterval(timer.current); }, []);

  const base = review?.facts?.config ?? {};
  const applyChanges = (changes: TrainingChanges) => {
    applyTrainingChanges({ changes, datasetId: datasetId ?? review?.facts?.run?.dataset_id, base });
  };

  const allChanges = useMemo(() => {
    const changes = mergeSuggestionChanges(review?.suggestions ?? []);
    const { augmentation, ...rest } = changes;
    return { changes, count: Object.keys(rest).length + Object.keys(augmentation ?? {}).length };
  }, [review]);

  if (!reviewable) {
    return <p className="text-sm text-gray-400">The review is available once the run has finished.</p>;
  }

  const tone = review ? scoreTone(review.score) : null;
  const t = review?.facts?.training;
  const ho = review?.facts?.held_out;
  const ds = review?.facts?.dataset;
  const dep = review?.facts?.deployment;
  const pastRuns: any[] = review?.facts?.past_runs ?? [];
  const aiDone = review?.suggestions_source === 'ai';

  return (
    <div className="flex flex-col space-y-4">
      {/* AI action */}
      <div className="space-y-2">
        <button
          onClick={() => load(true)}
          disabled={!!loading || !trainingId || !aiAvailable}
          className="w-full py-2.5 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:from-slate-800 disabled:to-slate-800 disabled:text-gray-500 text-white rounded-xl text-sm font-medium transition-all shadow-lg flex items-center justify-center gap-2"
        >
          {loading === 'ai' ? (
            <><RefreshCw className="w-4 h-4 animate-spin" /> Asking {providerLabel}… {elapsed}s</>
          ) : (
            <><Sparkles className="w-4 h-4" /> {aiDone ? 'Regenerate' : 'Get'} AI suggestions ({providerLabel})</>
          )}
        </button>
        {loading === 'ai' && (
          <p className="text-[11px] text-gray-500 text-center">Free models can take a minute or two; slow or failed attempts are retried automatically.</p>
        )}
        {!aiAvailable && (
          <p className="text-[11px] text-gray-500 text-center">
            No AI provider configured ({provider === 'ollama' ? 'set OLLAMA_ENABLED=true' : 'set OPENROUTER_API_KEY'} in backend/.env). The analysis below is computed locally.
          </p>
        )}
      </div>

      {error && !review && (
        <div className="flex items-start gap-2 p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200 text-sm">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {loading === 'rules' && !review && (
        <div className="flex items-center gap-2 text-sm text-gray-400"><RefreshCw className="w-4 h-4 animate-spin" /> Analysing the run…</div>
      )}

      {review && tone && (
        <>
          {review.ai_error && (
            <div className="flex items-start gap-2 p-3 bg-amber-900/20 border border-amber-500/40 rounded-lg text-amber-200 text-xs">
              <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
              <div className="min-w-0">
                <p className="font-medium">AI suggestions unavailable; showing the rule-based recommendations.</p>
                <p className="text-amber-200/70 mt-0.5 break-words">{review.ai_error}</p>
                <button onClick={() => load(true)} disabled={!!loading} className="mt-1.5 underline hover:text-white">Try again</button>
              </div>
            </div>
          )}

          {/* Score */}
          <div className={`p-3.5 rounded-xl border ${tone.ring}`}>
            <button className="w-full flex items-center gap-3 text-left" onClick={() => setShowBreakdown(v => !v)} aria-expanded={showBreakdown}>
              <div className={`w-14 h-14 rounded-xl bg-slate-900/60 flex flex-col items-center justify-center ${tone.text}`}>
                <span className="text-2xl font-bold leading-none">{review.score}</span>
                <span className="text-[9px] text-gray-500">/ 100</span>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-white font-semibold text-sm">Training quality: {review.score_label}</div>
                <div className="text-gray-400 text-[11px] mt-0.5">Computed from the run's metrics, data and target board. Same run, same score.</div>
              </div>
              {showBreakdown ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
            </button>
            {showBreakdown && (
              <ul className="mt-3 space-y-1.5">
                {review.score_breakdown.map((p) => (
                  <li key={p.name} className="text-[11px]">
                    <div className="flex justify-between text-gray-300"><span>{p.name}</span><span className="font-mono">{p.points}/{p.max}</span></div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden my-0.5">
                      <div className="h-full rounded-full bg-[#3987e5]" style={{ width: `${(p.points / p.max) * 100}%` }} />
                    </div>
                    <div className="text-gray-500">{p.detail}</div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Summary */}
          <p className="text-sm text-gray-200 leading-relaxed">{review.ai_summary || review.summary}</p>

          {/* Key numbers */}
          {t && (
            <dl className="grid grid-cols-2 gap-2 text-[11px]">
              {([
                ['Final val', pct(t.final?.val_accuracy)],
                ['Best val', `${pct(t.best_val_accuracy?.value)} @ ep ${t.best_val_accuracy?.epoch}`],
                [ho ? `${ho.split} (${ho.num_samples})` : 'Held-out', pct(ho?.score)],
                ['Train−val gap', `${t.train_val_gap_pts ?? '—'} pts`],
                ['Samples / class', ds?.min_per_class != null ? `${ds.min_per_class}–${ds.median_per_class} (min–median)` : '—'],
                ['INT8 size', dep?.est_int8_model_kb ? `~${Math.round(dep.est_int8_model_kb)} KB${dep.int8_flash_usage_pct != null ? ` · ${dep.int8_flash_usage_pct}% flash` : ''}` : '—'],
              ] as [string, string][]).map(([k, v]) => (
                <div key={k} className="px-2.5 py-1.5 rounded-lg bg-slate-800/70 border border-slate-700/70 min-w-0">
                  <dt className="text-gray-500 truncate">{k}</dt>
                  <dd className="text-gray-100 font-mono truncate" title={v}>{v}</dd>
                </div>
              ))}
            </dl>
          )}

          {/* Findings */}
          <section>
            <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wide mb-2">What the run shows</h4>
            <ul className="space-y-1.5">
              {review.findings.map((f, i) => {
                const sev = SEVERITY[f.severity];
                const Icon = sev.icon;
                return (
                  <li key={i} className={`flex items-start gap-2 px-2.5 py-2 rounded-lg border text-xs ${sev.cls}`}>
                    <Icon className="w-3.5 h-3.5 shrink-0 mt-0.5" aria-label={sev.label} />
                    <div className="min-w-0">
                      <span className="font-medium text-gray-100">{f.title}. </span>
                      <span className="text-gray-300">{f.evidence}</span>
                    </div>
                  </li>
                );
              })}
            </ul>
          </section>

          {/* Suggestions */}
          <section>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wide">What to try next</h4>
              <span className="text-[10px] text-gray-500 flex items-center gap-1">
                {aiDone ? <><Sparkles className="w-3 h-3" /> AI · {review.ai_model}{review.ai_seconds != null ? ` · ${review.ai_seconds}s` : ''}</> : <><Lightbulb className="w-3 h-3" /> rule-based</>}
              </span>
            </div>
            {allChanges.count > 0 && (
              <button
                onClick={() => applyChanges(allChanges.changes)}
                className="w-full mb-2.5 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg bg-purple-600 hover:bg-purple-500 text-white transition"
                title={`Load this run's settings with every suggested change into the training form (${allChanges.count} parameter${allChanges.count !== 1 ? 's' : ''})`}
              >
                <Wand2 className="w-3.5 h-3.5" /> Apply all suggested values ({allChanges.count})
              </button>
            )}
            {review.suggestions.length ? (
              <ul className="space-y-2.5">
                {review.suggestions.map((s, i) => <SuggestionCard key={i} sug={s} base={base} onApply={applyChanges} />)}
              </ul>
            ) : (
              <p className="text-xs text-gray-400">No changes recommended: this run looks healthy.</p>
            )}
            {aiDone && review.rule_suggestions.length > 0 && (
              <div className="mt-3">
                <button onClick={() => setShowRules(v => !v)} className="text-[11px] text-gray-400 hover:text-white flex items-center gap-1">
                  {showRules ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                  Rule-based checks ({review.rule_suggestions.length})
                </button>
                {showRules && (
                  <ul className="space-y-2.5 mt-2">
                    {review.rule_suggestions.map((s, i) => <SuggestionCard key={i} sug={s} base={base} onApply={applyChanges} />)}
                  </ul>
                )}
              </div>
            )}
          </section>

          {/* Earlier runs */}
          {pastRuns.length > 0 && (
            <section>
              <button onClick={() => setShowPast(v => !v)} className="text-xs font-semibold text-gray-300 uppercase tracking-wide flex items-center gap-1">
                {showPast ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                Earlier runs on this dataset ({pastRuns.length})
                {review.comparison?.delta_vs_best_previous_pts != null && (
                  <span className={`ml-2 normal-case font-mono ${review.comparison.delta_vs_best_previous_pts >= 0 ? 'text-emerald-300' : 'text-red-300'}`}>
                    {review.comparison.delta_vs_best_previous_pts >= 0 ? '+' : ''}{review.comparison.delta_vs_best_previous_pts} pts vs best
                  </span>
                )}
              </button>
              {showPast && (
                <div className="mt-2 overflow-x-auto">
                  <table className="w-full text-[10px] text-gray-300">
                    <thead className="text-gray-500">
                      <tr><th className="text-left font-normal pr-2">Model</th><th className="text-left font-normal pr-2">Input</th><th className="font-normal pr-2">Ep</th><th className="font-normal pr-2">LR</th><th className="font-normal">Best val</th></tr>
                    </thead>
                    <tbody>
                      {pastRuns.slice().reverse().map((r, i) => (
                        <tr key={i} className="border-t border-slate-800">
                          <td className="pr-2 py-1 truncate max-w-[7rem]" title={r.name || r.base_model}>{r.base_model}</td>
                          <td className="pr-2 font-mono">{r.input_shape?.join('×')}</td>
                          <td className="pr-2 text-center font-mono">{r.epochs_run}</td>
                          <td className="pr-2 text-center font-mono">{fmtValue(r.learning_rate)}</td>
                          <td className="text-center font-mono">{pct(r.best_val_accuracy)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          )}
        </>
      )}
    </div>
  );
}
