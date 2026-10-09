// Optimization Studio
// 1. pick a trained model, 2. create optimized variants (queued on the backend),
// 3. inspect each variant's honest evaluation (float32 TFLite baseline vs
// optimized, per-class metrics, TensorFlow Lite Micro check), 4. try any
// variant live in the playground.

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  AlertCircle, Camera, CheckCircle, ChevronDown, Cpu, History, Link, Loader, Mic, Play,
  Rocket, Trash2, TrendingUp, Upload, X, Clock,
} from "lucide-react";
import { AudioMicTab, AudioUploadTab, AudioUrlTab } from "./AudioTabs";
import ComparisonPanel from "./ComparisonPanel";
import { ImageCameraTab, ImageUploadTab, ImageUrlTab } from "./ImageTabs";
import { TerminalLogPanel } from "../TerminalLogPanel";
import { EvaluationReport, EvaluationMetrics } from "../EvaluationReport";
import {
  AudioTab, ImageTab, InferenceInput, InferenceResult, ResultCard, TrainedModel, VersionMode, runInference,
} from "./utility";
import { API_BASE } from "../../hooks/useAPI";
import { apiFetch, withAuthQuery } from "../../config";
import { formatBytes } from "../../utils/format";
import { useToast } from "../../context/ToastContext";

interface OptimizationStudioProps {
  models: (TrainedModel & { task?: string; base_model?: string; val_accuracy?: number; size_bytes?: number })[];
}

type OptimizationMethod =
  | "INT8_QUANTIZATION"
  | "FLOAT16_QUANTIZATION"
  | "DYNAMIC_QUANTIZATION"
  | "PRUNING"
  | "WEIGHT_CLUSTERING";

const METHODS: { method: OptimizationMethod; label: string; desc: string }[] = [
  {
    method: "INT8_QUANTIZATION",
    label: "Full INT8",
    desc: "Weights and activations in 8-bit, calibrated on real training samples. Smallest and fastest on MCUs - the default choice.",
  },
  {
    method: "DYNAMIC_QUANTIZATION",
    label: "Dynamic range",
    desc: "INT8 weights, float activations. ~4× smaller, no calibration. Runs on CPUs; most MCU kernels prefer full INT8.",
  },
  {
    method: "FLOAT16_QUANTIZATION",
    label: "Float16",
    desc: "Half-precision weights, ~2× smaller with negligible accuracy change. Useful for GPU / desktop targets.",
  },
  {
    method: "PRUNING",
    label: "Pruning + fine-tune",
    desc: "Zeroes the smallest weights, fine-tunes with the masks enforced, then quantizes. The gain shows in the compressed (gzip) size.",
  },
  {
    method: "WEIGHT_CLUSTERING",
    label: "Clustering + fine-tune",
    desc: "Snaps each layer's weights to a few shared values, then quantizes. Pairs with flash compression.",
  },
];

interface Variant {
  id: string;
  training_id: string;
  method: string;
  frontend_method?: string;
  status: "queued" | "pending" | "running" | "completed" | "failed" | "cancelled";
  error?: string | null;
  created_at: number;
  original_size_bytes?: number;
  optimized_size_bytes?: number;
  metrics?: {
    compressed_size_bytes?: number;
    baseline_compressed_size_bytes?: number;
    note?: string;
    tflm?: { available?: boolean; supported?: boolean; arena_bytes?: number; unsupported_ops?: string[]; error?: string; max_abs_diff?: number };
    ops?: string[];
    sparsity_actual?: number;
  };
  comparison?: Comparison | null;
  queue_position?: number;
}

interface SideMetrics {
  accuracy: number;
  avg_inference_ms: number;
  size_bytes: number;
  metrics?: EvaluationMetrics;
}

interface SampleResult {
  sample_id: string;
  filename: string | null;
  true_label: string;
  original: { predicted_label?: string; confidence?: number; correct?: boolean; detections?: unknown[] };
  optimized: { predicted_label?: string; confidence?: number; correct?: boolean; detections?: unknown[] };
}

interface Comparison {
  test_split_used: string;
  num_samples_evaluated: number;
  metric_name?: string;
  original: SideMetrics;
  optimized: SideMetrics;
  deltas: { accuracy_delta: number; speedup_factor: number; size_reduction_pct: number };
  timing_note?: string;
  sample_results?: SampleResult[];
  error?: string;
}

const ACTIVE = new Set(["queued", "pending", "running"]);
const labelFor = (m: string) => METHODS.find((x) => x.method === m)?.label ?? m.replace(/_/g, " ").toLowerCase();

const card = "rounded-2xl border border-slate-700 bg-slate-900/60 p-5";
const sectionTitle = "text-xs font-semibold uppercase tracking-widest text-slate-400";

const OptimizationStudio: React.FC<OptimizationStudioProps> = ({ models }) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const modelParam = searchParams.get("model");
  const mountedRef = useRef(true);
  useEffect(() => () => { mountedRef.current = false; }, []);

  // ---------------- model selection ----------------
  const datasetOptions = useMemo(() => Array.from(
    new Map(models.filter((m) => m.dataset_id).map((m) => [m.dataset_id as string, m.dataset_name || (m.dataset_id as string)])).entries()
  ), [models]);
  const [datasetFilter, setDatasetFilter] = useState("");
  const filteredModels = datasetFilter ? models.filter((m) => m.dataset_id === datasetFilter) : models;
  const [selectedModelId, setSelectedModelId] = useState<string>(
    (modelParam && models.find((m) => m.training_id === modelParam)?.id) || models[0]?.id || ""
  );
  useEffect(() => {
    if (!modelParam) return;
    const match = models.find((m) => m.training_id === modelParam);
    if (match && match.id !== selectedModelId) setSelectedModelId(match.id);
  }, [modelParam, models]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!selectedModelId && models[0]) setSelectedModelId(models[0].id);
  }, [models, selectedModelId]);
  const selectedModel = models.find((m) => m.id === selectedModelId) ?? null;
  const isDetection = selectedModel?.task === "OBJECT_DETECTION";

  // ---------------- variants ----------------
  const [variants, setVariants] = useState<Variant[]>([]);
  const [selectedVariantId, setSelectedVariantId] = useState<string | null>(searchParams.get("optimization"));
  const [activeLogJobId, setActiveLogJobId] = useState<string | null>(null);

  // Last seen status per variant, to announce jobs that just finished.
  const prevStatusRef = useRef<Map<string, string>>(new Map());

  const loadVariants = useCallback(async () => {
    if (!selectedModel?.training_id) { setVariants([]); return; }
    try {
      const r = await apiFetch(`/optimization/history`);
      const j = await r.json();
      if (!mountedRef.current || j?.status !== "success") return;
      const mine: Variant[] = (j.sessions ?? []).filter((s: Variant) => s.training_id === selectedModel.training_id);
      mine.forEach((v) => {
        const prev = prevStatusRef.current.get(v.id);
        if (!prev || !ACTIVE.has(prev)) return;
        if (v.status === "completed") {
          toast("success", `${labelFor(v.frontend_method ?? v.method)} finished`);
          setSelectedVariantId(v.id);
        } else if (v.status === "failed") {
          toast("error", `${labelFor(v.frontend_method ?? v.method)} failed: ${v.error ?? "see console"}`);
        }
      });
      prevStatusRef.current = new Map(mine.map((v) => [v.id, v.status]));
      setVariants(mine);
      const running = mine.find((v) => ACTIVE.has(v.status));
      if (running) setActiveLogJobId((cur) => cur ?? running.id);
    } catch { /* backend unreachable - keep the current list */ }
  }, [selectedModel?.training_id, toast]);

  useEffect(() => { loadVariants(); }, [loadVariants]);

  // Poll while anything for this model is queued or running (no fixed
  // timeout: large INT8 + evaluation runs can legitimately take a while).
  const hasActive = variants.some((v) => ACTIVE.has(v.status));
  useEffect(() => {
    if (!hasActive) return;
    const t = setInterval(loadVariants, 2500);
    return () => clearInterval(t);
  }, [hasActive, loadVariants]);

  // ---------------- create variant ----------------
  const [method, setMethod] = useState<OptimizationMethod>("INT8_QUANTIZATION");
  const [sparsity, setSparsity] = useState(0.5);
  const [fineTuneEpochs, setFineTuneEpochs] = useState(2);
  const [followUpQuant, setFollowUpQuant] = useState<"int8" | "dynamic">("int8");
  const [numClusters, setNumClusters] = useState(16);
  const [submitting, setSubmitting] = useState(false);

  const runOptimization = async () => {
    if (!selectedModel?.training_id) return;
    setSubmitting(true);
    try {
      const resp = await apiFetch(`/optimization/quantize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          training_id: selectedModel.training_id,
          method,
          sparsity_level: sparsity,
          fine_tune_epochs: fineTuneEpochs,
          quantization: followUpQuant,
          num_clusters: numClusters,
        }),
      });
      const json = await resp.json().catch(() => ({}));
      if (!resp.ok || json.status !== "success") throw new Error(json?.detail ?? json?.message ?? `HTTP ${resp.status}`);
      setActiveLogJobId(json.optimization_id);
      toast("info", json.queue_position > 0 ? `Queued (position ${json.queue_position + 1})` : "Optimization started");
      await loadVariants();
    } catch (e) {
      toast("error", (e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const cancelVariant = async (id: string) => {
    await apiFetch(`/optimization/cancel/${id}`, { method: "POST" });
    loadVariants();
  };
  const deleteVariant = async (id: string) => {
    const r = await apiFetch(`/optimization/session/${id}`, { method: "DELETE" });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      toast("error", j?.detail ?? "Could not delete");
      return;
    }
    if (selectedVariantId === id) setSelectedVariantId(null);
    loadVariants();
  };

  // ---------------- variant report ----------------
  const selectedVariant = variants.find((v) => v.id === selectedVariantId) ?? null;
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [reportSide, setReportSide] = useState<"optimized" | "original">("optimized");
  const [showGallery, setShowGallery] = useState(false);
  const [galleryFilter, setGalleryFilter] = useState<"all" | "mismatches">("mismatches");

  useEffect(() => {
    if (!selectedVariant || selectedVariant.status !== "completed") { setComparison(null); return; }
    let cancelled = false;
    setComparisonLoading(true);
    apiFetch(`/optimization/result/${selectedVariant.id}`)
      .then((r) => r.json())
      .then((j) => { if (!cancelled) setComparison(j?.result?.comparison ?? null); })
      .catch(() => { if (!cancelled) setComparison(null); })
      .finally(() => { if (!cancelled) setComparisonLoading(false); });
    return () => { cancelled = true; };
  }, [selectedVariant?.id, selectedVariant?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------- playground ----------------
  const [versionMode, setVersionMode] = useState<VersionMode>("both");
  const [imageTab, setImageTab] = useState<ImageTab>("upload");
  const [audioTab, setAudioTab] = useState<AudioTab>("upload");
  const [originalResult, setOriginalResult] = useState<InferenceResult | null>(null);
  const [optimizedResult, setOptimizedResult] = useState<InferenceResult | null>(null);
  const [originalLoading, setOriginalLoading] = useState(false);
  const [optimizedLoading, setOptimizedLoading] = useState(false);
  const [originalError, setOriginalError] = useState<string | null>(null);
  const [optimizedError, setOptimizedError] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [historyEntries, setHistoryEntries] = useState<Record<string, unknown>[]>([]);

  useEffect(() => {
    setOriginalResult(null); setOptimizedResult(null); setOriginalError(null); setOptimizedError(null);
  }, [selectedModelId, versionMode, selectedVariantId]);

  const playgroundVariant = selectedVariant?.status === "completed" ? selectedVariant : null;

  const handleInference = useCallback(async (input: InferenceInput) => {
    if (!selectedModel) return;
    const tid = selectedModel.training_id;
    const runOriginal = versionMode !== "optimized";
    const runOptimized = versionMode !== "original";
    const jobs: Promise<void>[] = [];
    if (runOriginal) {
      setOriginalLoading(true); setOriginalError(null); setOriginalResult(null);
      jobs.push(runInference(tid, null, input).then(setOriginalResult)
        .catch((e: Error) => setOriginalError(e.message)).finally(() => setOriginalLoading(false)));
    }
    if (runOptimized) {
      if (!playgroundVariant) {
        setOptimizedError("Select a completed variant above to test the optimized model.");
      } else {
        setOptimizedLoading(true); setOptimizedError(null); setOptimizedResult(null);
        jobs.push(runInference(tid, playgroundVariant.id, input).then(setOptimizedResult)
          .catch((e: Error) => setOptimizedError(e.message)).finally(() => setOptimizedLoading(false)));
      }
    }
    await Promise.all(jobs);
  }, [selectedModel, versionMode, playgroundVariant]);

  const fetchHistory = async () => {
    try {
      const r = await apiFetch(`/inference/history?limit=50`);
      const j = await r.json();
      if (j.status === "success") setHistoryEntries(j.entries ?? []);
    } catch { /* ignore */ }
  };

  // ---------------- render helpers ----------------
  const tabBase = "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer select-none";
  const tabActive = "bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-md";
  const tabInactive = "text-slate-400 hover:text-white hover:bg-white/10";

  const renderInput = () => {
    if (!selectedModel) return null;
    if ((selectedModel.type ?? "image") === "audio") {
      const tabs: { value: AudioTab; label: string; icon: React.ReactNode }[] = [
        { value: "upload", label: "Upload", icon: <Upload size={13} /> },
        { value: "url", label: "URL", icon: <Link size={13} /> },
        { value: "microphone", label: "Microphone", icon: <Mic size={13} /> },
      ];
      return (
        <div className="space-y-4">
          <div className="flex gap-1 p-1 bg-white/5 rounded-xl w-fit" role="tablist">
            {tabs.map((t) => (
              <button key={t.value} role="tab" aria-selected={audioTab === t.value} onClick={() => setAudioTab(t.value)}
                className={`${tabBase} ${audioTab === t.value ? tabActive : tabInactive}`}>{t.icon} {t.label}</button>
            ))}
          </div>
          {audioTab === "upload" && <AudioUploadTab onData={(blob) => handleInference({ kind: "file", blob })} />}
          {audioTab === "url" && <AudioUrlTab onData={(url) => handleInference({ kind: "url", url })} />}
          {audioTab === "microphone" && <AudioMicTab onData={(blob) => handleInference({ kind: "file", blob })} />}
        </div>
      );
    }
    const tabs: { value: ImageTab; label: string; icon: React.ReactNode }[] = [
      { value: "upload", label: "Upload", icon: <Upload size={13} /> },
      { value: "url", label: "URL", icon: <Link size={13} /> },
      { value: "camera", label: "Camera", icon: <Camera size={13} /> },
    ];
    return (
      <div className="space-y-4">
        <div className="flex gap-1 p-1 bg-white/5 rounded-xl w-fit" role="tablist">
          {tabs.map((t) => (
            <button key={t.value} role="tab" aria-selected={imageTab === t.value} onClick={() => setImageTab(t.value)}
              className={`${tabBase} ${imageTab === t.value ? tabActive : tabInactive}`}>{t.icon} {t.label}</button>
          ))}
        </div>
        {imageTab === "upload" && <ImageUploadTab onData={(blob) => handleInference({ kind: "file", blob })} />}
        {imageTab === "url" && <ImageUrlTab onData={(url) => handleInference({ kind: "url", url })} />}
        {imageTab === "camera" && <ImageCameraTab onData={(blob) => handleInference({ kind: "file", blob })} />}
      </div>
    );
  };

  const statusBadge = (v: Variant) => {
    const map: Record<string, string> = {
      completed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
      failed: "bg-red-500/15 text-red-300 border-red-500/30",
      cancelled: "bg-yellow-500/15 text-yellow-300 border-yellow-500/30",
      running: "bg-violet-500/15 text-violet-300 border-violet-500/30",
      queued: "bg-slate-500/15 text-slate-300 border-slate-500/30",
      pending: "bg-slate-500/15 text-slate-300 border-slate-500/30",
    };
    return (
      <span className={`text-[10px] px-1.5 py-0.5 rounded border font-semibold uppercase ${map[v.status] ?? ""}`}>
        {v.status === "running" && <Loader size={9} className="inline animate-spin mr-1" />}
        {v.status}
      </span>
    );
  };

  const tflmBadge = (v: Variant) => {
    const t = v.metrics?.tflm;
    if (!t) return null;
    if (t.unsupported_ops?.length || (t.available && !t.supported)) {
      return <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-300" title={t.error ?? t.unsupported_ops?.join(", ")}>TFLM ✗</span>;
    }
    if (t.available && t.supported) {
      return <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-300" title="Verified with the TensorFlow Lite Micro interpreter">TFLM ✓ {formatBytes(t.arena_bytes)}</span>;
    }
    return null;
  };

  if (models.length === 0) {
    return (
      <div className={`${card} text-center py-12`}>
        <Cpu className="w-10 h-10 text-slate-600 mx-auto mb-3" />
        <p className="text-slate-300 font-medium">No trained models yet</p>
        <p className="text-sm text-slate-500 mt-1">Train a model first, then come back to optimize it.</p>
        <button onClick={() => navigate("/train")} className="mt-4 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-sm font-semibold">
          Go to training
        </button>
      </div>
    );
  }

  const metricName = comparison?.metric_name === "f1" ? "F1" : "Accuracy";

  return (
    <div className="space-y-6">
      {/* ---------------- 1. Model ---------------- */}
      <div className={card}>
        <div className="grid gap-3 md:grid-cols-[1fr_2fr]">
          {datasetOptions.length > 1 && (
            <label className="block">
              <span className={sectionTitle}>Dataset</span>
              <select value={datasetFilter} onChange={(e) => setDatasetFilter(e.target.value)}
                className="mt-1.5 w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white">
                <option value="">All datasets</option>
                {datasetOptions.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
              </select>
            </label>
          )}
          <label className={`block ${datasetOptions.length > 1 ? "" : "md:col-span-2"}`}>
            <span className={sectionTitle}>Trained model</span>
            <select value={selectedModelId}
              onChange={(e) => {
                setSelectedModelId(e.target.value);
                setSelectedVariantId(null);
                const m = models.find((x) => x.id === e.target.value);
                setSearchParams(m?.training_id ? { model: m.training_id } : {});
              }}
              className="mt-1.5 w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white">
              {filteredModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}{m.dataset_name ? ` · ${m.dataset_name}` : ""}
                  {m.val_accuracy != null ? ` · val ${(m.val_accuracy * 100).toFixed(1)}%` : ""}
                  {m.size_bytes ? ` · ${formatBytes(m.size_bytes)}` : ""}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {selectedModel && (
        <div className="grid gap-6 lg:grid-cols-2">
          {/* ---------------- 2. Create variant ---------------- */}
          <div className={`${card} space-y-4`}>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2"><Cpu size={16} className="text-cyan-400" /> Create an optimized variant</h3>
            <div className="grid gap-2" role="radiogroup" aria-label="Optimization method">
              {METHODS.map((m) => (
                <label key={m.method}
                  className={`flex gap-3 rounded-xl border p-3 cursor-pointer transition-colors ${method === m.method ? "border-violet-500/60 bg-violet-500/10" : "border-slate-700 hover:border-slate-500"}`}>
                  <input type="radio" name="opt-method" className="mt-1 accent-violet-500" checked={method === m.method} onChange={() => setMethod(m.method)} />
                  <span>
                    <span className="block text-sm font-semibold text-white">{m.label}</span>
                    <span className="block text-[11px] text-slate-400 mt-0.5">{m.desc}</span>
                  </span>
                </label>
              ))}
            </div>

            {(method === "PRUNING" || method === "WEIGHT_CLUSTERING") && (
              <div className="grid grid-cols-2 gap-3 rounded-xl bg-slate-800/50 border border-slate-700 p-3 text-xs">
                {method === "PRUNING" ? (
                  <label className="col-span-2 flex items-center gap-3 text-slate-300">
                    Sparsity <span className="font-mono text-white w-10">{Math.round(sparsity * 100)}%</span>
                    <input type="range" min={0.1} max={0.9} step={0.05} value={sparsity}
                      onChange={(e) => setSparsity(parseFloat(e.target.value))} className="flex-1 accent-violet-500" />
                  </label>
                ) : (
                  <label className="col-span-2 flex items-center gap-3 text-slate-300">
                    Clusters per layer
                    <select value={numClusters} onChange={(e) => setNumClusters(Number(e.target.value))}
                      className="bg-slate-900 border border-slate-600 rounded px-2 py-1 text-white">
                      {[4, 8, 16, 32, 64].map((n) => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </label>
                )}
                <label className="flex flex-col gap-1 text-slate-400">
                  Fine-tune epochs
                  <input type="number" min={0} max={50} value={fineTuneEpochs}
                    onChange={(e) => setFineTuneEpochs(Math.max(0, Number(e.target.value) || 0))}
                    className="bg-slate-900 border border-slate-600 rounded px-2 py-1 text-white" />
                </label>
                <label className="flex flex-col gap-1 text-slate-400">
                  Then quantize to
                  <select value={followUpQuant} onChange={(e) => setFollowUpQuant(e.target.value as "int8" | "dynamic")}
                    className="bg-slate-900 border border-slate-600 rounded px-2 py-1 text-white">
                    <option value="int8">Full INT8 (MCU)</option>
                    <option value="dynamic">Dynamic range</option>
                  </select>
                </label>
              </div>
            )}

            <button onClick={runOptimization} disabled={submitting}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-violet-600 hover:opacity-90 disabled:opacity-50 text-white text-sm font-semibold">
              {submitting ? <Loader size={14} className="animate-spin" /> : <Play size={14} />} Run {labelFor(method)}
            </button>
            <TerminalLogPanel jobId={activeLogJobId} title="Optimization console" />
          </div>

          {/* ---------------- 3. Variants ---------------- */}
          <div className={`${card} space-y-3`}>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2"><History size={16} className="text-violet-400" /> Variants of this model</h3>
            {variants.length === 0 ? (
              <p className="text-sm text-slate-500 italic">No variants yet - run an optimization.</p>
            ) : (
              <ul className="space-y-2 max-h-[30rem] overflow-y-auto pr-1">
                {variants.map((v) => {
                  const d = v.comparison?.deltas;
                  const selected = v.id === selectedVariantId;
                  return (
                    <li key={v.id}>
                      <div className={`rounded-xl border p-3 transition-colors ${selected ? "border-violet-500/60 bg-violet-500/10" : "border-slate-700 hover:border-slate-500"}`}>
                        <button className="w-full text-left" onClick={() => { setSelectedVariantId(v.id); setActiveLogJobId(v.id); }}>
                          <div className="flex items-center gap-2 flex-wrap">
                            {statusBadge(v)}
                            <span className="text-sm font-semibold text-white">{labelFor(v.frontend_method ?? v.method)}</span>
                            {tflmBadge(v)}
                            <span className="ml-auto text-[10px] text-slate-500">{new Date(v.created_at * 1000).toLocaleString()}</span>
                          </div>
                          {v.status === "completed" && (
                            <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-[11px] font-mono text-slate-400">
                              <span>{formatBytes(v.original_size_bytes)} → <span className="text-white">{formatBytes(v.optimized_size_bytes)}</span></span>
                              {v.metrics?.compressed_size_bytes && <span>gzip {formatBytes(v.metrics.compressed_size_bytes)}</span>}
                              {d && <span className={d.accuracy_delta >= -0.005 ? "text-emerald-300" : "text-amber-300"}>Δ {(d.accuracy_delta * 100).toFixed(1)} pp</span>}
                            </div>
                          )}
                          {v.status === "queued" && <p className="mt-1 text-[11px] text-slate-400 flex items-center gap-1"><Clock size={11} /> Waiting for the ML worker…</p>}
                          {v.status === "failed" && v.error && <p className="mt-1 text-[11px] text-red-300 line-clamp-2">{v.error}</p>}
                        </button>
                        <div className="mt-2 flex gap-2">
                          {v.status === "completed" && (
                            <button onClick={() => navigate(`/deploy?optimization=${v.id}`)}
                              className="text-[11px] px-2 py-1 rounded-md bg-pink-600/20 text-pink-300 hover:bg-pink-600/30 flex items-center gap-1">
                              <Rocket size={11} /> Deploy
                            </button>
                          )}
                          {v.status === "queued" && (
                            <button onClick={() => cancelVariant(v.id)} className="text-[11px] px-2 py-1 rounded-md bg-white/5 text-slate-300 hover:bg-white/10 flex items-center gap-1">
                              <X size={11} /> Cancel
                            </button>
                          )}
                          {!ACTIVE.has(v.status) && (
                            <button onClick={() => deleteVariant(v.id)} aria-label="Delete variant"
                              className="text-[11px] px-2 py-1 rounded-md bg-white/5 text-slate-400 hover:text-red-300 hover:bg-red-500/10 flex items-center gap-1">
                              <Trash2 size={11} /> Delete
                            </button>
                          )}
                        </div>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      )}

      {/* ---------------- 4. Variant report ---------------- */}
      {selectedVariant && selectedVariant.status === "completed" && (
        <div className={`${card} space-y-4`}>
          <div className="flex items-center gap-2 flex-wrap">
            <TrendingUp size={16} className="text-emerald-400" />
            <h3 className="text-sm font-semibold text-white">Report: {labelFor(selectedVariant.frontend_method ?? selectedVariant.method)}</h3>
            {tflmBadge(selectedVariant)}
            <button onClick={() => navigate(`/deploy?optimization=${selectedVariant.id}`)}
              className="ml-auto text-xs px-3 py-1.5 rounded-lg bg-pink-600 hover:bg-pink-500 text-white font-semibold flex items-center gap-1.5">
              <Rocket size={13} /> Deploy this variant
            </button>
          </div>
          {selectedVariant.metrics?.note && <p className="text-xs text-slate-400">{selectedVariant.metrics.note}</p>}
          {selectedVariant.metrics?.tflm?.unsupported_ops?.length ? (
            <div className="text-xs text-red-300 flex items-start gap-1.5"><AlertCircle size={13} className="mt-0.5" />
              TensorFlow Lite Micro does not implement: {selectedVariant.metrics.tflm.unsupported_ops.join(", ")}. This model can't run on a microcontroller.
            </div>
          ) : null}

          {comparisonLoading ? (
            <div className="flex items-center gap-2 text-sm text-slate-400"><Loader size={13} className="animate-spin" /> Loading evaluation…</div>
          ) : !comparison ? (
            <p className="text-xs text-slate-500 italic">No evaluation data for this variant.</p>
          ) : comparison.error ? (
            <div className="flex items-center gap-1.5 text-xs text-amber-300"><AlertCircle size={12} /> {comparison.error}</div>
          ) : (
            <>
              <div className="grid gap-3 sm:grid-cols-3">
                {[
                  { name: metricName, a: `${(comparison.original.accuracy * 100).toFixed(1)}%`, b: `${(comparison.optimized.accuracy * 100).toFixed(1)}%`, delta: `${(comparison.deltas.accuracy_delta * 100).toFixed(2)} pp`, good: comparison.deltas.accuracy_delta >= -0.005 },
                  { name: "Size", a: formatBytes(comparison.original.size_bytes), b: formatBytes(comparison.optimized.size_bytes), delta: `-${comparison.deltas.size_reduction_pct}%`, good: true },
                  { name: "Latency (host)", a: `${comparison.original.avg_inference_ms.toFixed(2)} ms`, b: `${comparison.optimized.avg_inference_ms.toFixed(2)} ms`, delta: `${comparison.deltas.speedup_factor}×`, good: comparison.deltas.speedup_factor >= 1 },
                ].map((t) => (
                  <div key={t.name} className="rounded-xl bg-white/5 border border-white/10 p-3">
                    <div className="text-[10px] uppercase tracking-wide text-slate-500">{t.name}</div>
                    <div className="mt-1 text-sm font-mono text-slate-300">{t.a} → <span className="text-white font-semibold">{t.b}</span></div>
                    <div className={`text-xs font-mono mt-0.5 ${t.good ? "text-emerald-300" : "text-amber-300"}`}>{t.delta}</div>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-slate-500">
                Baseline = float32 TFLite of the same weights. {comparison.num_samples_evaluated} samples from the {comparison.test_split_used} split.
                {comparison.timing_note ? ` ${comparison.timing_note}` : ""}
              </p>

              <div className="flex gap-1 p-1 bg-white/5 rounded-xl w-fit">
                {(["optimized", "original"] as const).map((side) => (
                  <button key={side} onClick={() => setReportSide(side)} className={`${tabBase} ${reportSide === side ? tabActive : tabInactive}`}>
                    {side === "optimized" ? "Optimized" : "Float32 baseline"}
                  </button>
                ))}
              </div>
              <EvaluationReport metrics={comparison[reportSide].metrics} detection={isDetection} />

              {comparison.sample_results && comparison.sample_results.length > 0 && !isDetection && (selectedModel?.type ?? "image") === "image" && (
                <div className="pt-2 border-t border-white/10">
                  <button onClick={() => setShowGallery((v) => !v)} className="flex items-center gap-2 w-full text-left">
                    <span className="text-xs font-semibold text-slate-300">Per-sample predictions ({comparison.sample_results.length})</span>
                    <ChevronDown size={13} className={`text-slate-500 ml-auto transition-transform ${showGallery ? "rotate-180" : ""}`} />
                  </button>
                  {showGallery && (
                    <div className="mt-2 space-y-2">
                      <div className="flex gap-1.5">
                        {(["mismatches", "all"] as const).map((f) => (
                          <button key={f} onClick={() => setGalleryFilter(f)}
                            className={`text-[10px] px-2 py-1 rounded-md font-semibold ${galleryFilter === f ? "bg-violet-600/30 text-violet-300" : "bg-white/5 text-slate-400"}`}>
                            {f === "all" ? "All samples" : "Errors only"}
                          </button>
                        ))}
                      </div>
                      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 max-h-96 overflow-y-auto pr-1">
                        {comparison.sample_results
                          .filter((s) => galleryFilter === "all" || !s.original.correct || !s.optimized.correct)
                          .map((s) => (
                            <div key={s.sample_id} className="rounded-lg overflow-hidden border border-white/10 bg-white/5">
                              <img src={withAuthQuery(`${API_BASE}/datasets/image/${s.sample_id}`)} alt={s.filename ?? s.sample_id}
                                loading="lazy" className="w-full aspect-square object-cover bg-black/30" />
                              <div className="p-1.5 space-y-0.5 text-[9px] font-mono">
                                <p className="text-slate-500 truncate">true: {s.true_label}</p>
                                <p className={`truncate ${s.original.correct ? "text-emerald-400" : "text-red-400"}`}>fp32: {s.original.predicted_label}</p>
                                <p className={`truncate ${s.optimized.correct ? "text-emerald-400" : "text-red-400"}`}>opt: {s.optimized.predicted_label}</p>
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

      {/* ---------------- 5. Playground ---------------- */}
      {selectedModel && (
        <div className={`${card} space-y-4`}>
          <div className="flex items-center gap-2 flex-wrap">
            <CheckCircle size={16} className="text-indigo-400" />
            <h3 className="text-sm font-semibold text-white">Try it</h3>
            <span className="text-xs text-slate-500">
              {playgroundVariant ? `optimized = ${labelFor(playgroundVariant.frontend_method ?? playgroundVariant.method)}` : "select a completed variant to compare"}
            </span>
            <button onClick={() => { setShowHistory((v) => !v); if (!showHistory) fetchHistory(); }}
              className="ml-auto flex items-center gap-1.5 text-xs text-slate-400 hover:text-white">
              <History size={13} /> {showHistory ? "Hide history" : "History"}
            </button>
          </div>

          {showHistory && (
            <div className="rounded-xl bg-slate-950/60 border border-white/10 p-3 max-h-56 overflow-y-auto space-y-1.5">
              {historyEntries.length === 0 ? <p className="text-xs text-slate-500 italic">No inference runs yet.</p> : historyEntries.map((e, i) => (
                <div key={String(e.inference_id ?? i)} className="flex justify-between text-xs bg-white/5 rounded px-2 py-1">
                  <span className="text-white">{String(e.top_class)}</span>
                  <span className="text-slate-400 font-mono">{((e.confidence as number) * 100).toFixed(1)}% · {String(e.inference_time_ms)} ms · {String(e.model_kind)}</span>
                </div>
              ))}
            </div>
          )}

          <div className="flex gap-2">
            {([{ value: "original", label: "Original" }, { value: "optimized", label: "Optimized" }, { value: "both", label: "Side by side" }] as { value: VersionMode; label: string }[]).map((v) => (
              <button key={v.value} onClick={() => setVersionMode(v.value)}
                className={`flex-1 py-2 rounded-xl text-sm font-semibold transition-all ${versionMode === v.value ? "bg-gradient-to-r from-indigo-600 to-violet-600 text-white" : "bg-white/5 text-slate-400 hover:bg-white/10 border border-white/10"}`}>
                {v.label}
              </button>
            ))}
          </div>

          {renderInput()}

          {versionMode === "both" ? (
            <ComparisonPanel originalResult={originalResult} optimizedResult={optimizedResult}
              originalLoading={originalLoading} optimizedLoading={optimizedLoading}
              originalError={originalError} optimizedError={optimizedError} />
          ) : (
            <ResultCard title={versionMode === "original" ? "Original (.keras)" : "Optimized (.tflite)"}
              result={versionMode === "original" ? originalResult : optimizedResult}
              loading={versionMode === "original" ? originalLoading : optimizedLoading}
              error={versionMode === "original" ? originalError : optimizedError} />
          )}
        </div>
      )}
    </div>
  );
};

export { OptimizationStudio };
export default OptimizationStudio;
