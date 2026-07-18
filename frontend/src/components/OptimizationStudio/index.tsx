// OptimizationStudio/index.tsx
// Optimization Studio — compare original (.keras) vs optimized (.tflite) model
// inference side-by-side with real server-side inference.

import React, { useState, useCallback, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Camera, Mic, Upload, Link, ChevronDown, Zap, History, X,
  Settings, Play, CheckCircle, AlertCircle, Loader, TrendingUp,
} from "lucide-react";
import { AudioUploadTab, AudioUrlTab, AudioMicTab } from "./AudioTabs";
import ComparisonPanel from "./ComparisonPanel";
import { ImageUploadTab, ImageUrlTab, ImageCameraTab } from "./ImageTabs";
import { TerminalLogPanel } from "../TerminalLogPanel";
import {
  TrainedModel,
  ImageTab,
  AudioTab,
  runInference,
  VersionMode,
  ResultCard,
  InferenceResult,
  InferenceInput,
} from "./utility";
import { API_BASE } from "../../hooks/useAPI";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface OptimizationStudioProps {
  /** ModelMetadata[] passed from App.tsx via AppContext. */
  models: TrainedModel[];
}

type OptimizationMethod =
  | "INT8_QUANTIZATION"
  | "FLOAT16_QUANTIZATION"
  | "DYNAMIC_QUANTIZATION"
  | "PRUNING"
  | "WEIGHT_CLUSTERING"
  | "TRANSFER_LEARNING";

interface OptimizationOption {
  method: OptimizationMethod;
  label: string;
  desc: string;
  color: string;
}

const OPTIMIZATION_OPTIONS: OptimizationOption[] = [
  {
    method: "INT8_QUANTIZATION",
    label: "INT8 Quantization",
    desc: "Converts weights & activations to 8-bit integers. ~4× size reduction, fastest on MCU.",
    color: "from-emerald-600 to-teal-600",
  },
  {
    method: "FLOAT16_QUANTIZATION",
    label: "Float16 Quantization",
    desc: "Half-precision weights. ~2× size reduction, minimal accuracy loss.",
    color: "from-blue-600 to-cyan-600",
  },
  {
    method: "DYNAMIC_QUANTIZATION",
    label: "Dynamic Quantization",
    desc: "Quantizes weights only at runtime. Good balance between speed and accuracy.",
    color: "from-violet-600 to-purple-600",
  },
  {
    method: "PRUNING",
    label: "Weight Pruning",
    desc: "Removes low-magnitude weights. Configurable sparsity level.",
    color: "from-orange-600 to-amber-600",
  },
  {
    method: "WEIGHT_CLUSTERING",
    label: "Weight Clustering",
    desc: "Groups weights into clusters, reducing unique values. Pairs well with quantization.",
    color: "from-pink-600 to-rose-600",
  },
  {
    method: "TRANSFER_LEARNING",
    label: "Transfer Learning Re-export",
    desc: "Re-heads the model onto a frozen MobileNetV2 backbone and re-exports as INT8. Useful when the original architecture is too large to quantize directly.",
    color: "from-cyan-600 to-sky-600",
  },
];

type OptimizationRunStatus = "idle" | "running" | "completed" | "failed";

interface OptimizationRun {
  method: OptimizationMethod;
  status: OptimizationRunStatus;
  optimizationId: string | null;
  error: string | null;
}

interface SamplePrediction {
  predicted_label: string;
  confidence: number;
  correct: boolean;
}

interface SampleResult {
  sample_id: string;
  filename: string | null;
  true_label: string;
  original: SamplePrediction;
  optimized: SamplePrediction;
}

interface ComparisonMetrics {
  test_split_used: string;
  num_samples_evaluated: number;
  original: { accuracy: number; loss: number; avg_inference_ms: number; size_bytes: number };
  optimized: { accuracy: number; avg_inference_ms: number; size_bytes: number };
  deltas: { accuracy_delta: number; speedup_factor: number; size_reduction_pct: number };
  sample_results?: SampleResult[];
  error?: string;
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
const OptimizationStudio: React.FC<OptimizationStudioProps> = ({ models }) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const modelParam = searchParams.get("model"); // a training_id, e.g. from the Models Explorer tree

  // --- Dataset filter (previously there was no way to narrow the model
  // picker down to a single dataset, so with more than a handful of models
  // it became a long, hard-to-scan flat list). ---
  const datasetOptions = Array.from(
    new Map(
      models
        .filter((m) => m.dataset_id)
        .map((m) => [m.dataset_id as string, m.dataset_name || (m.dataset_id as string)])
    ).entries()
  );
  const [datasetFilter, setDatasetFilter] = useState<string>("");
  const filteredModels = datasetFilter ? models.filter((m) => m.dataset_id === datasetFilter) : models;

  // --- Model selection ---
  const initialModelId = modelParam
    ? models.find((m) => m.training_id === modelParam)?.id ?? models[0]?.id ?? ""
    : models[0]?.id ?? "";
  const [selectedModelId, setSelectedModelId] = useState<string>(initialModelId);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const selectedModel = models.find((m) => m.id === selectedModelId) ?? null;

  // If the model list loads asynchronously after mount (common - it's
  // fetched once at the App level), make sure a `?model=` deep link still
  // resolves once the data actually arrives instead of only on first render.
  useEffect(() => {
    if (!modelParam || selectedModelId) return;
    const match = models.find((m) => m.training_id === modelParam);
    if (match) setSelectedModelId(match.id);
  }, [modelParam, models, selectedModelId]);

  // Keep the dataset filter in sync with a deep-linked model so it doesn't
  // appear to "disappear" if a different dataset filter was left selected.
  useEffect(() => {
    if (selectedModel?.dataset_id && datasetFilter && selectedModel.dataset_id !== datasetFilter) {
      setDatasetFilter("");
    }
  }, [selectedModel]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- Optimization session selection (for .tflite) ---
  const [optimizationId, setOptimizationId] = useState<string | null>(null);

  // --- Optimization panel state ---
  const [showOptimizationPanel, setShowOptimizationPanel] = useState(true);
  const [sparsityLevel, setSparsityLevel] = useState(0.5);
  const [optimizationRuns, setOptimizationRuns] = useState<Record<OptimizationMethod, OptimizationRun>>(
    {} as Record<OptimizationMethod, OptimizationRun>
  );
  // Tracks the most recently triggered optimization job so the live
  // console panel below the options grid always follows the latest run.
  const [activeLogJobId, setActiveLogJobId] = useState<string | null>(null);

  // Reattach to an optimization job already in progress for the selected
  // model - without this, navigating away and back (or reloading) loses
  // track of a running optimization entirely.
  useEffect(() => {
    if (!selectedModel?.training_id) return;
    let cancelled = false;
    fetch(`${API_BASE}/optimization/active?training_id=${selectedModel.training_id}`)
      .then((r) => r.json())
      .then((res) => {
        if (cancelled || res?.status !== "success" || !res.session) return;
        const session = res.session;
        setActiveLogJobId(session.id);
        setOptimizationRuns((prev) => ({
          ...prev,
          [session.frontend_method ?? session.method]: {
            method: session.frontend_method ?? session.method,
            status: "running",
            optimizationId: session.id,
            error: null,
          },
        }));
      })
      .catch(() => { /* best-effort */ });
    return () => { cancelled = true; };
  }, [selectedModel?.training_id]);

  // --- Version mode ---
  const [versionMode, setVersionMode] = useState<VersionMode>("both");

  // --- Tab state ---
  const [imageTab, setImageTab] = useState<ImageTab>("upload");
  const [audioTab, setAudioTab] = useState<AudioTab>("upload");

  // --- Inference state ---
  const [originalResult, setOriginalResult] = useState<InferenceResult | null>(null);
  const [optimizedResult, setOptimizedResult] = useState<InferenceResult | null>(null);
  const [originalLoading, setOriginalLoading] = useState(false);
  const [optimizedLoading, setOptimizedLoading] = useState(false);
  const [originalError, setOriginalError] = useState<string | null>(null);
  const [optimizedError, setOptimizedError] = useState<string | null>(null);

  // --- History panel ---
  const [showHistory, setShowHistory] = useState(false);
  const [historyEntries, setHistoryEntries] = useState<any[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // --- Test-set comparison (real accuracy/latency/size, original vs optimized) ---
  const [comparison, setComparison] = useState<ComparisonMetrics | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [showSampleGallery, setShowSampleGallery] = useState(false);
  const [galleryFilter, setGalleryFilter] = useState<"all" | "mismatches">("all");

  // Pull the real test-set comparison as soon as an optimization is selected.
  useEffect(() => {
    if (!optimizationId) {
      setComparison(null);
      return;
    }
    setComparisonLoading(true);
    fetch(`${API_BASE}/optimization/result/${optimizationId}`)
      .then((r) => r.json())
      .then((j) => {
        if (j?.status === "success" && j?.result?.comparison) {
          setComparison(j.result.comparison);
        } else {
          setComparison(null);
        }
      })
      .catch(() => setComparison(null))
      .finally(() => setComparisonLoading(false));
  }, [optimizationId]);

  const fetchHistory = async () => {
    setHistoryLoading(true);
    try {
      const r = await fetch(`${API_BASE}/inference/history?limit=50`);
      const json = await r.json();
      if (json.status === "success") setHistoryEntries(json.entries ?? []);
    } catch {
      // silently ignore
    } finally {
      setHistoryLoading(false);
    }
  };

  // Reset results when model/mode changes
  useEffect(() => {
    setOriginalResult(null);
    setOptimizedResult(null);
    setOriginalError(null);
    setOptimizedError(null);
  }, [selectedModelId, versionMode]);

  // --- Run optimization via backend API ---
  const handleRunOptimization = useCallback(
    async (method: OptimizationMethod) => {
      if (!selectedModel) return;

      // Use training_id — this is what the backend uses to find the .keras file
      const trainingId = selectedModel.training_id;
      if (!trainingId) {
        setOptimizationRuns((prev) => ({
          ...prev,
          [method]: { method, status: "failed", optimizationId: null, error: "Model has no training_id." },
        }));
        return;
      }

      setOptimizationRuns((prev) => ({
        ...prev,
        [method]: { method, status: "running", optimizationId: null, error: null },
      }));

      try {
        const resp = await fetch(`${API_BASE}/optimization/quantize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            training_id: trainingId,
            method,
            sparsity_level: sparsityLevel,
            representative_dataset_size: 100,
          }),
        });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err?.detail ?? `HTTP ${resp.status}`);
        }

        const json = await resp.json();
        const oid: string = json.optimization_id;
        setActiveLogJobId(oid);

        // Poll for completion
        setOptimizationRuns((prev) => ({
          ...prev,
          [method]: { method, status: "running", optimizationId: oid, error: null },
        }));

        const poll = async () => {
          for (let i = 0; i < 120; i++) {
            await new Promise((r) => setTimeout(r, 3000));
            let sj: any;
            try {
              const sr = await fetch(`${API_BASE}/optimization/status/${oid}`);
              sj = await sr.json();
            } catch {
              continue; // transient network hiccup while polling - just retry
            }

            // NOTE: the status endpoint returns a FLAT object
            // ({status, error, metrics, comparison, ...}) - there is no
            // nested `.data` field. Reading `sj?.data?.error` here always
            // returned undefined, so a real failure (e.g. a Python
            // traceback from a crashed optimization) was silently replaced
            // with a generic "Optimization failed" string, and a broken
            // catch block below then swallowed even that, eventually
            // showing a misleading "Timed out" message instead of the
            // actual cause.
            const s = sj?.status;
            if (s === "completed") {
              setOptimizationRuns((prev) => ({
                ...prev,
                [method]: { method, status: "completed", optimizationId: oid, error: null },
              }));
              setOptimizationId(oid);
              return;
            }
            if (s === "failed") {
              setOptimizationRuns((prev) => ({
                ...prev,
                [method]: { method, status: "failed", optimizationId: oid, error: sj?.error || "Optimization failed (no error detail returned)." },
              }));
              return;
            }
          }
          setOptimizationRuns((prev) => ({
            ...prev,
            [method]: { method, status: "failed", optimizationId: oid, error: "Timed out waiting for optimization." },
          }));
        };

        poll();
      } catch (e: any) {
        setOptimizationRuns((prev) => ({
          ...prev,
          [method]: { method, status: "failed", optimizationId: null, error: e.message },
        }));
      }
    },
    [selectedModel, sparsityLevel]
  );

  // --- Inference trigger ---
  const handleInference = useCallback(
    async (input: InferenceInput) => {
      if (!selectedModel) return;

      // Always use training_id for model resolution — fall back to id if missing
      const tid = selectedModel.training_id || selectedModel.id;
      const runOriginal = versionMode === "original" || versionMode === "both";
      const runOptimized = versionMode === "optimized" || versionMode === "both";

      if (runOriginal) {
        setOriginalLoading(true);
        setOriginalError(null);
        setOriginalResult(null);
      }
      if (runOptimized) {
        setOptimizedLoading(true);
        setOptimizedError(null);
        setOptimizedResult(null);
      }

      const tasks: Promise<void>[] = [];

      if (runOriginal) {
        tasks.push(
          runInference(tid, null, input)
            .then((r) => setOriginalResult(r))
            .catch((e: Error) => setOriginalError(e.message))
            .finally(() => setOriginalLoading(false))
        );
      }

      if (runOptimized) {
        if (!optimizationId && !selectedModel.optimized) {
          setOptimizedError(
            "No optimized model found. Run an optimization above, then it will be selected automatically."
          );
          setOptimizedLoading(false);
        } else {
          const oid = optimizationId ?? null;
          tasks.push(
            runInference(tid, oid, input)
              .then((r) => setOptimizedResult(r))
              .catch((e: Error) => setOptimizedError(e.message))
              .finally(() => setOptimizedLoading(false))
          );
        }
      }

      await Promise.all(tasks);
    },
    [selectedModel, versionMode, optimizationId]
  );

  // --- Style helpers ---
  const tabBase =
    "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 cursor-pointer select-none";
  const tabActive = "bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-md";
  const tabInactive = "text-slate-400 hover:text-white hover:bg-white/10";

  const versionModes: { value: VersionMode; label: string }[] = [
    { value: "original", label: "Original (.keras)" },
    { value: "optimized", label: "Optimized (.tflite)" },
    { value: "both", label: "Both" },
  ];

  // --- Results panel ---
  const renderResults = () => {
    if (versionMode === "both") {
      return (
        <ComparisonPanel
          originalResult={originalResult}
          optimizedResult={optimizedResult}
          originalLoading={originalLoading}
          optimizedLoading={optimizedLoading}
          originalError={originalError}
          optimizedError={optimizedError}
        />
      );
    }
    const result = versionMode === "original" ? originalResult : optimizedResult;
    const loading = versionMode === "original" ? originalLoading : optimizedLoading;
    const error = versionMode === "original" ? originalError : optimizedError;
    const title = versionMode === "original" ? "Original Model (.keras)" : "Optimized Model (.tflite)";

    return (
      <div className="mt-4">
        <ResultCard title={title} result={result} loading={loading} error={error} />
      </div>
    );
  };

  // --- Input panel ---
  const renderInputPanel = () => {
    if (!selectedModel) return null;

    const modelType = selectedModel.manualType ?? selectedModel.type ?? "image";

    if (modelType === "image") {
      const imageTabs: { value: ImageTab; label: string; icon: React.ReactNode }[] = [
        { value: "upload", label: "Upload", icon: <Upload size={13} /> },
        { value: "url", label: "URL", icon: <Link size={13} /> },
        { value: "camera", label: "Camera", icon: <Camera size={13} /> },
      ];

      return (
        <div className="space-y-4">
          <div className="flex gap-1 p-1 bg-white/5 rounded-xl w-fit">
            {imageTabs.map((t) => (
              <button
                key={t.value}
                onClick={() => setImageTab(t.value)}
                className={`${tabBase} ${imageTab === t.value ? tabActive : tabInactive}`}
              >
                {t.icon} {t.label}
              </button>
            ))}
          </div>
          {imageTab === "upload" && (
            <ImageUploadTab onData={(blob) => handleInference({ kind: "file", blob })} />
          )}
          {imageTab === "url" && (
            <ImageUrlTab onData={(url) => handleInference({ kind: "url", url })} />
          )}
          {imageTab === "camera" && (
            <ImageCameraTab onData={(blob) => handleInference({ kind: "file", blob })} />
          )}
        </div>
      );
    }

    if (modelType === "audio") {
      const audioTabs: { value: AudioTab; label: string; icon: React.ReactNode }[] = [
        { value: "upload", label: "Upload", icon: <Upload size={13} /> },
        { value: "url", label: "URL", icon: <Link size={13} /> },
        { value: "microphone", label: "Microphone", icon: <Mic size={13} /> },
      ];

      return (
        <div className="space-y-4">
          <div className="flex gap-1 p-1 bg-white/5 rounded-xl w-fit">
            {audioTabs.map((t) => (
              <button
                key={t.value}
                onClick={() => setAudioTab(t.value)}
                className={`${tabBase} ${audioTab === t.value ? tabActive : tabInactive}`}
              >
                {t.icon} {t.label}
              </button>
            ))}
          </div>
          {audioTab === "upload" && (
            <AudioUploadTab onData={(blob) => handleInference({ kind: "file", blob })} />
          )}
          {audioTab === "url" && (
            <AudioUrlTab onData={(url) => handleInference({ kind: "url", url })} />
          )}
          {audioTab === "microphone" && (
            <AudioMicTab onData={(blob) => handleInference({ kind: "file", blob })} />
          )}
        </div>
      );
    }

    // Tabular / text fallback
    return (
      <div className="space-y-3">
        <textarea
          rows={4}
          placeholder={
            modelType === "text"
              ? "Enter text to classify…"
              : "Enter comma-separated feature values…"
          }
          className="w-full bg-white/5 border border-white/15 rounded-xl px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
          onKeyDown={(e) => {
            if (e.key === "Enter" && e.ctrlKey) {
              handleInference({ kind: "text", text: (e.target as HTMLTextAreaElement).value });
            }
          }}
        />
        <button
          onClick={(e) => {
            const ta = (e.currentTarget as HTMLElement).previousSibling as HTMLTextAreaElement;
            handleInference({ kind: "text", text: ta.value });
          }}
          className="w-full py-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white text-sm font-semibold rounded-lg transition-all"
        >
          Run Inference
        </button>
        <p className="text-xs text-slate-500">Tip: Ctrl + Enter to run</p>
      </div>
    );
  };

  // --- Metrics summary helper ---
  const renderMetricsSummary = (m: TrainedModel) => {
    const acc = (m as any).val_accuracy ?? (m as any).accuracy;
    const loss = (m as any).val_loss ?? (m as any).loss;
    const sizeKb = (m as any).size_bytes ? Math.round((m as any).size_bytes / 1024) : null;
    const parts: string[] = [];
    if (acc != null) parts.push(`acc ${(acc * 100).toFixed(1)}%`);
    if (loss != null) parts.push(`loss ${loss.toFixed(3)}`);
    if (sizeKb != null) parts.push(`${sizeKb} KB`);
    return parts.length > 0 ? parts.join(" • ") : null;
  };

  // --- Main render ---
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 pt-8">
      <div className="w-full max-w-2xl rounded-2xl shadow-xl border border-white/10">

        {/* Header */}
        <div className="px-6 py-5 border-b border-white/10">
          <button
            onClick={() => {
              setShowHistory((v) => !v);
              if (!showHistory) fetchHistory();
            }}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition-colors"
          >
            <History size={14} />
            {showHistory ? "Hide history" : "Inference history"}
          </button>
        </div>
        <p className="text-xs text-slate-400 mt-0.5">
          Real inference — original .keras vs quantized .tflite
        </p>
      </div>

      {/* Inference history panel */}
      {showHistory && (
        <div className="border-b border-white/10 bg-slate-950/60 px-6 py-4 max-h-64 overflow-y-auto">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">
              Recent runs
            </p>
            <button onClick={() => setShowHistory(false)} className="text-slate-500 hover:text-white">
              <X size={14} />
            </button>
          </div>
          {historyLoading ? (
            <p className="text-sm text-slate-500">Loading…</p>
          ) : historyEntries.length === 0 ? (
            <p className="text-sm text-slate-500 italic">No inference runs yet.</p>
          ) : (
            <div className="space-y-2">
              {historyEntries.map((e, i) => (
                <div
                  key={e.inference_id ?? i}
                  className="flex items-center justify-between text-xs rounded-lg bg-white/5 px-3 py-2"
                >
                  <span className="text-white font-medium capitalize">{e.top_class}</span>
                  <span className="text-slate-400">
                    {(e.confidence * 100).toFixed(1)}% • {e.inference_time_ms}ms •{" "}
                    <span className="font-mono uppercase text-[10px] text-slate-500">
                      {e.model_kind}
                    </span>
                  </span>
                  <span className="text-slate-600">
                    {new Date(e.timestamp * 1000).toLocaleTimeString()}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="p-6 space-y-6">

        {/* --- Dataset filter --- */}
        {datasetOptions.length > 1 && (
          <div className="space-y-1.5">
            <label className="text-xs font-semibold uppercase tracking-widest text-slate-400">
              Filter by Dataset
            </label>
            <select
              value={datasetFilter}
              onChange={(e) => {
                setDatasetFilter(e.target.value);
                // The currently selected model may not belong to the
                // newly chosen dataset - clear it so the picker doesn't
                // silently keep showing a model outside the filter.
                const stillValid = models.find(
                  (m) => m.id === selectedModelId && (!e.target.value || m.dataset_id === e.target.value)
                );
                if (!stillValid) setSelectedModelId("");
              }}
              className="w-full bg-white/5 border border-white/15 rounded-xl px-4 py-2.5 text-sm text-white focus:border-violet-500 focus:outline-none"
            >
              <option value="">All datasets</option>
              {datasetOptions.map(([id, name]) => (
                <option key={id} value={id}>{name}</option>
              ))}
            </select>
          </div>
        )}

        {/* --- Model selector --- */}
        <div className="space-y-1.5">
          <label className="text-xs font-semibold uppercase tracking-widest text-slate-400">
            Active Model
          </label>
          <div className="relative">
            <button
              onClick={() => setDropdownOpen((o) => !o)}
              className="w-full flex items-center justify-between bg-white/5 hover:bg-white/8 border border-white/15 rounded-xl px-4 py-3 text-sm text-white transition-colors"
            >
              <div className="flex items-center gap-3 min-w-0">
                {selectedModel ? (
                  <>
                    <span
                      className={`shrink-0 text-xs font-bold px-2 py-0.5 rounded-md ${selectedModel.type === "image"
                        ? "bg-blue-500/20 text-blue-300"
                        : selectedModel.type === "audio"
                          ? "bg-purple-500/20 text-purple-300"
                          : selectedModel.type === "text"
                            ? "bg-green-500/20 text-green-300"
                            : "bg-orange-500/20 text-orange-300"
                        }`}
                    >
                      {(selectedModel.type ?? "?").toUpperCase()}
                    </span>
                    <span className="font-medium truncate">{selectedModel.name}</span>
                    {/* Metrics summary inline */}
                    {renderMetricsSummary(selectedModel) && (
                      <span className="text-[10px] text-slate-400 font-mono bg-white/5 px-1.5 py-0.5 rounded shrink-0">
                        {renderMetricsSummary(selectedModel)}
                      </span>
                    )}
                    {selectedModel.optimized && (
                      <span className="text-[10px] text-violet-400 font-mono bg-violet-500/10 px-1.5 py-0.5 rounded shrink-0">
                        has .tflite
                      </span>
                    )}
                  </>
                ) : (
                  <span className="text-slate-500">Select a model…</span>
                )}
              </div>
              <ChevronDown
                size={16}
                className={`shrink-0 text-slate-400 transition-transform duration-200 ${dropdownOpen ? "rotate-180" : ""}`}
              />
            </button>

            {dropdownOpen && (
              <div className="absolute z-50 top-full mt-1 w-full bg-slate-800 border border-white/15 rounded-xl shadow-2xl overflow-hidden">
                {filteredModels.length === 0 && (
                  <p className="text-sm text-slate-500 px-4 py-3 italic">
                    {datasetFilter ? "No trained models for this dataset." : "No trained models yet."}
                  </p>
                )}
                {filteredModels.map((m) => {
                  const metrics = renderMetricsSummary(m);
                  return (
                    <button
                      key={m.id}
                      onClick={() => {
                        setSelectedModelId(m.id);
                        setOptimizationId(null);
                        setDropdownOpen(false);
                        setSearchParams(m.training_id ? { model: m.training_id } : {});
                      }}
                      className={`w-full flex items-center gap-3 px-4 py-3 text-sm hover:bg-white/10 transition-colors text-left ${m.id === selectedModelId ? "bg-indigo-600/20" : ""
                        }`}
                    >
                      <span
                        className={`shrink-0 text-xs font-bold px-2 py-0.5 rounded-md ${m.type === "image"
                          ? "bg-blue-500/20 text-blue-300"
                          : m.type === "audio"
                            ? "bg-purple-500/20 text-purple-300"
                            : m.type === "text"
                              ? "bg-green-500/20 text-green-300"
                              : "bg-orange-500/20 text-orange-300"
                          }`}
                      >
                        {(m.type ?? "?").toUpperCase()}
                      </span>
                      <div className="flex-1 min-w-0">
                        <span className="text-white font-medium block truncate">{m.name}</span>
                        <div className="flex items-center gap-1.5">
                          {metrics && (
                            <span className="text-[10px] text-slate-400 font-mono">{metrics}</span>
                          )}
                          {m.dataset_name && (
                            <span className="text-[10px] text-slate-500 truncate">· {m.dataset_name}</span>
                          )}
                        </div>
                      </div>
                      {m.optimized && (
                        <span className="shrink-0 text-[10px] text-violet-400 font-mono bg-violet-500/10 px-1.5 py-0.5 rounded">
                          .tflite ready
                        </span>
                      )}
                      {m.id === selectedModelId && !m.optimized && (
                        <span className="shrink-0 text-indigo-400 text-xs">? active</span>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* --- Optimization Options Panel --- */}
        {selectedModel && (
          <div className="space-y-3">
            <button
              onClick={() => setShowOptimizationPanel((v) => !v)}
              className="flex items-center gap-2 w-full text-left"
            >
              <Settings size={14} className="text-violet-400" />
              <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
                Optimization Options
              </span>
              <ChevronDown
                size={13}
                className={`text-slate-500 transition-transform duration-200 ml-auto ${showOptimizationPanel ? "rotate-180" : ""
                  }`}
              />
            </button>

            {showOptimizationPanel && (
              <div className="space-y-3 rounded-xl border border-white/10 bg-white/3 p-4">
                {/* Sparsity level (only relevant for Pruning) */}
                <div className="flex items-center gap-3 mb-1">
                  <label className="text-xs text-slate-400 shrink-0">
                    Sparsity (pruning): <span className="text-white font-mono">{(sparsityLevel * 100).toFixed(0)}%</span>
                  </label>
                  <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={sparsityLevel}
                    onChange={(e) => setSparsityLevel(parseFloat(e.target.value))}
                    className="flex-1 accent-violet-500"
                  />
                </div>

                {/* One card per optimization method */}
                <div className="grid grid-cols-1 gap-2">
                  {OPTIMIZATION_OPTIONS.map((opt) => {
                    const run = optimizationRuns[opt.method];
                    const status = run?.status ?? "idle";
                    const oid = run?.optimizationId;

                    return (
                      <div
                        key={opt.method}
                        className="flex items-start gap-3 rounded-lg bg-white/5 border border-white/10 px-3 py-2.5"
                      >
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-semibold text-white">{opt.label}</p>
                          <p className="text-[11px] text-slate-400 mt-0.5">{opt.desc}</p>
                          {status === "completed" && oid && (
                            <div className="mt-1.5 flex items-center gap-2">
                              <CheckCircle size={11} className="text-emerald-400 shrink-0" />
                              <span className="text-[10px] font-mono text-slate-300">
                                id: {oid.slice(0, 8)}…
                              </span>
                              <button
                                onClick={() => setOptimizationId(oid)}
                                className={`text-[10px] px-1.5 py-0.5 rounded font-semibold transition-colors ${optimizationId === oid
                                  ? "bg-violet-500/30 text-violet-300"
                                  : "bg-white/10 text-slate-300 hover:bg-violet-500/20 hover:text-violet-300"
                                  }`}
                              >
                                {optimizationId === oid ? "✓ Selected" : "Use for inference"}
                              </button>
                            </div>
                          )}
                          {status === "failed" && run.error && (
                            <div className="mt-1 flex items-center gap-1.5">
                              <AlertCircle size={11} className="text-red-400 shrink-0" />
                              <span className="text-[10px] text-red-300">{run.error}</span>
                            </div>
                          )}
                        </div>
                        <button
                          onClick={() => handleRunOptimization(opt.method)}
                          disabled={status === "running"}
                          className={`shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${status === "running"
                            ? "bg-white/10 text-slate-500 cursor-not-allowed"
                            : status === "completed"
                              ? "bg-emerald-600/20 text-emerald-300 hover:bg-emerald-600/30 border border-emerald-500/30"
                              : `bg-gradient-to-r ${opt.color} text-white hover:opacity-90 shadow-sm`
                            }`}
                        >
                          {status === "running" ? (
                            <>
                              <Loader size={11} className="animate-spin" />
                              Running…
                            </>
                          ) : status === "completed" ? (
                            <>
                              <CheckCircle size={11} />
                              Re-run
                            </>
                          ) : (
                            <>
                              <Play size={11} />
                              Run
                            </>
                          )}
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* --- Live console output for the most recently triggered job --- */}
        <TerminalLogPanel jobId={activeLogJobId} title="Optimization Console" />

        {/* --- Test-Set Comparison: real accuracy/latency/size, original vs optimized --- */}
        {optimizationId && (
          <div className="space-y-3 rounded-xl border border-white/10 bg-white/3 p-4">
            <div className="flex items-center gap-2">
              <TrendingUp size={14} className="text-emerald-400" />
              <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
                Test-Set Comparison (Original vs Optimized)
              </span>
            </div>

            {comparisonLoading ? (
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <Loader size={13} className="animate-spin" /> Loading comparison…
              </div>
            ) : !comparison ? (
              <p className="text-xs text-slate-500 italic">
                No comparison data yet for this optimization.
              </p>
            ) : comparison.error ? (
              <div className="flex items-center gap-1.5 text-xs text-amber-300">
                <AlertCircle size={12} /> {comparison.error}
              </div>
            ) : (
              <>
                <p className="text-[11px] text-slate-500">
                  Evaluated on {comparison.num_samples_evaluated} samples from the{" "}
                  <span className="text-slate-300 font-semibold">{comparison.test_split_used}</span> split.
                </p>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="rounded-lg bg-white/5 border border-white/10 p-3">
                    <p className="text-[11px] uppercase tracking-wide text-slate-500 mb-2">Original (.keras)</p>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between"><span className="text-slate-400">Accuracy</span><span className="text-white font-mono">{(comparison.original.accuracy * 100).toFixed(1)}%</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Loss</span><span className="text-white font-mono">{comparison.original.loss.toFixed(4)}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Inference</span><span className="text-white font-mono">{comparison.original.avg_inference_ms.toFixed(3)} ms</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Size</span><span className="text-white font-mono">{(comparison.original.size_bytes / 1024).toFixed(1)} KB</span></div>
                    </div>
                  </div>
                  <div className="rounded-lg bg-emerald-500/5 border border-emerald-500/20 p-3">
                    <p className="text-[11px] uppercase tracking-wide text-emerald-400 mb-2">Optimized (.tflite)</p>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between"><span className="text-slate-400">Accuracy</span><span className="text-white font-mono">{(comparison.optimized.accuracy * 100).toFixed(1)}%</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Loss</span><span className="text-slate-500 font-mono">n/a</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Inference</span><span className="text-white font-mono">{comparison.optimized.avg_inference_ms.toFixed(3)} ms</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Size</span><span className="text-white font-mono">{(comparison.optimized.size_bytes / 1024).toFixed(1)} KB</span></div>
                    </div>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2 pt-1">
                  <span className={`text-[11px] px-2 py-1 rounded-md font-mono ${comparison.deltas.accuracy_delta >= 0 ? "bg-emerald-500/10 text-emerald-300" : "bg-amber-500/10 text-amber-300"}`}>
                    Δ accuracy: {(comparison.deltas.accuracy_delta * 100).toFixed(2)} pp
                  </span>
                  <span className="text-[11px] px-2 py-1 rounded-md font-mono bg-sky-500/10 text-sky-300">
                    {comparison.deltas.speedup_factor}x faster
                  </span>
                  <span className="text-[11px] px-2 py-1 rounded-md font-mono bg-violet-500/10 text-violet-300">
                    -{comparison.deltas.size_reduction_pct}% size
                  </span>
                </div>

                {/* Per-sample test set predictions gallery */}
                {comparison.sample_results && comparison.sample_results.length > 0 && (
                  <div className="pt-2 border-t border-white/10 mt-2">
                    <button
                      onClick={() => setShowSampleGallery((v) => !v)}
                      className="flex items-center gap-2 w-full text-left"
                    >
                      <span className="text-xs font-semibold text-slate-300">
                        Test Set Predictions ({comparison.sample_results.length} samples)
                      </span>
                      <ChevronDown
                        size={13}
                        className={`text-slate-500 transition-transform duration-200 ml-auto ${showSampleGallery ? "rotate-180" : ""}`}
                      />
                    </button>

                    {showSampleGallery && (
                      <div className="mt-2 space-y-2">
                        <div className="flex gap-1.5">
                          {(["all", "mismatches"] as const).map((f) => (
                            <button
                              key={f}
                              onClick={() => setGalleryFilter(f)}
                              className={`text-[10px] px-2 py-1 rounded-md font-semibold transition-colors ${galleryFilter === f
                                ? "bg-violet-600/30 text-violet-300 border border-violet-500/40"
                                : "bg-white/5 text-slate-400 border border-white/10 hover:bg-white/10"
                                }`}
                            >
                              {f === "all" ? "All samples" : "Only mismatches"}
                            </button>
                          ))}
                        </div>

                        <div className="grid grid-cols-3 sm:grid-cols-4 gap-2 max-h-96 overflow-y-auto pr-1">
                          {comparison.sample_results
                            .filter((s) =>
                              galleryFilter === "all"
                                ? true
                                : !s.original.correct || !s.optimized.correct
                            )
                            .map((s) => (
                              <div
                                key={s.sample_id}
                                className="rounded-lg overflow-hidden border border-white/10 bg-white/5"
                              >
                                <img
                                  src={`${API_BASE}/datasets/image/${s.sample_id}`}
                                  alt={s.filename ?? s.sample_id}
                                  loading="lazy"
                                  className="w-full aspect-square object-cover bg-black/30"
                                />
                                <div className="p-1.5 space-y-0.5">
                                  <p className="text-[9px] text-slate-500 truncate">true: {s.true_label}</p>
                                  <p className={`text-[9px] font-mono truncate ${s.original.correct ? "text-emerald-400" : "text-red-400"}`}>
                                    {s.original.correct ? "✓" : "✗"} orig: {s.original.predicted_label} ({(s.original.confidence * 100).toFixed(0)}%)
                                  </p>
                                  <p className={`text-[9px] font-mono truncate ${s.optimized.correct ? "text-emerald-400" : "text-red-400"}`}>
                                    {s.optimized.correct ? "✓" : "✗"} opt: {s.optimized.predicted_label} ({(s.optimized.confidence * 100).toFixed(0)}%)
                                  </p>
                                </div>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* --- Optimization ID input (for .tflite slot) --- */}
        {(versionMode === "optimized" || versionMode === "both") && (
          <div className="space-y-1.5">
            <label className="text-xs font-semibold uppercase tracking-widest text-slate-400">
              Optimization ID{" "}
              <span className="normal-case font-normal text-slate-500">
                (auto-filled after running optimization above)
              </span>
            </label>
            <input
              type="text"
              value={optimizationId ?? ""}
              onChange={(e) => setOptimizationId(e.target.value.trim() || null)}
              placeholder="e.g. a1b2c3d4-…"
              className="w-full bg-white/5 border border-white/15 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 font-mono"
            />
          </div>
        )}

        {/* --- Version mode toggle --- */}
        <div className="space-y-1.5">
          <label className="text-xs font-semibold uppercase tracking-widest text-slate-400">
            Test Version
          </label>
          <div className="flex gap-2">
            {versionModes.map((v) => (
              <button
                key={v.value}
                onClick={() => setVersionMode(v.value)}
                className={`flex-1 py-2 rounded-xl text-sm font-semibold transition-all duration-200 ${versionMode === v.value
                  ? "bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-lg shadow-indigo-500/20"
                  : "bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white border border-white/10"
                  }`}
              >
                {v.label}
              </button>
            ))}
          </div>
        </div>

        <div className="border-t border-white/10" />

        {/* --- Input panel --- */}
        <div>
          <label className="text-xs font-semibold uppercase tracking-widest text-slate-400 block mb-3">
            Input
          </label>
          {selectedModel ? (
            renderInputPanel()
          ) : (
            <p className="text-sm text-slate-500 italic">Please select a model to continue.</p>
          )}
        </div>

        {/* --- Results --- */}
        {selectedModel && (
          <div>
            <label className="text-xs font-semibold uppercase tracking-widest text-slate-400">
              Results
            </label>
            {renderResults()}
          </div>
        )}
      </div>
    </div >
  );
};

export { OptimizationStudio };
