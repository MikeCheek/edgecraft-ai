// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

import React from "react";
import { Zap } from "lucide-react";

export interface TrainedModel {
  id: string;
  name: string;
  training_id: string;
  type?: "image" | "audio" | "tabular" | "text" | null;
  labels?: string[] | null;
  optimized?: boolean;
  dataset_id?: string | null;
  dataset_name?: string | null;
  // optimizedModelUrl kept for backward compat but we now use optimization_id from
  // the OptimizationStudio state instead of this URL field.
  originalModelUrl?: string;
  optimizedModelUrl?: string;
  manualLabels?: string[];
  manualType?: "image" | "audio" | "tabular" | "text";
}

export interface TopKPrediction {
  class_name: string;
  confidence: number;
}

export interface InferenceResult {
  /** Primary predicted class */
  label: string;
  confidence: number;
  /** Wall-clock time for the model's forward pass only (ms) */
  latencyMs: number;
  /** Total round-trip time including preprocessing + upload (ms) */
  totalMs: number;
  /** Top-K alternative predictions */
  topK: TopKPrediction[];
  /** 'keras' or 'tflite' */
  modelKind?: string;
}

export type VersionMode = "original" | "optimized" | "both";
export type ImageTab = "upload" | "url" | "camera";
export type AudioTab = "upload" | "url" | "microphone";

// ---------------------------------------------------------------------------
// Structured input type — replaces the overloaded `Blob | string` approach
// ---------------------------------------------------------------------------
export type InferenceInput =
  | { kind: "file"; blob: Blob }
  | { kind: "url"; url: string }
  | { kind: "text"; text: string };

// ---------------------------------------------------------------------------
// Real inference function — calls the backend
// ---------------------------------------------------------------------------

const API_BASE = "http://localhost:8000/api";

/**
 * Run real server-side inference via the EdgeCraft backend.
 *
 * @param trainingId     - The training session whose .keras model is used.
 * @param optimizationId - When set, uses the .tflite produced by that optimization session.
 * @param input          - Structured input: file blob, remote URL, or raw text.
 */
export async function runInference(
  trainingId: string,
  optimizationId: string | null,
  input: InferenceInput
): Promise<InferenceResult> {
  const wallStart = performance.now();

  const form = new FormData();
  form.append("training_id", trainingId);
  if (optimizationId) form.append("optimization_id", optimizationId);
  form.append("top_k", "5");

  if (input.kind === "file") {
    form.append("file", input.blob, "input");
  } else if (input.kind === "url") {
    form.append("input_url", input.url);
  } else {
    // text / tabular — send as a plain-text blob
    form.append("file", new Blob([input.text], { type: "text/plain" }), "input.txt");
  }

  const resp = await fetch(`${API_BASE}/inference/run`, {
    method: "POST",
    body: form,
  });

  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body?.detail ?? `Inference request failed (HTTP ${resp.status})`);
  }

  const json = await resp.json();
  const r = json.result;

  const wallMs = performance.now() - wallStart;

  return {
    label: r.top_class,
    confidence: r.confidence,
    latencyMs: Math.round(r.inference_time_ms),
    totalMs: Math.round(wallMs),
    topK: r.top_k_results ?? [],
    modelKind: r.model_kind,
  };
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

export function confidenceColor(confidence: number): string {
  if (confidence >= 0.75) return "text-green-400";
  if (confidence >= 0.5) return "text-yellow-400";
  return "text-red-400";
}
export function confidenceBorder(confidence: number): string {
  if (confidence >= 0.75) return "border-green-500/40";
  if (confidence >= 0.5) return "border-yellow-500/40";
  return "border-red-500/40";
}
export function confidenceBg(confidence: number): string {
  if (confidence >= 0.75) return "bg-green-500/10";
  if (confidence >= 0.5) return "bg-yellow-500/10";
  return "bg-red-500/10";
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Spinner shown during inference */
export const Spinner: React.FC = () => (
  <div className="flex items-center justify-center py-6">
    <div className="w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
  </div>
);

/** Single inference result card */
export const ResultCard: React.FC<{
  title: string;
  result: InferenceResult | null;
  loading: boolean;
  error: string | null;
}> = ({ title, result, loading, error }) => (
  <div
    className={`flex-1 rounded-xl border p-4 ${result
      ? `${confidenceBorder(result.confidence)} ${confidenceBg(result.confidence)}`
      : "border-white/10 bg-white/5"
      }`}
  >
    <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">
      {title}
    </p>

    {loading && <Spinner />}

    {error && !loading && (
      <div className="rounded-lg bg-red-500/20 border border-red-500/40 p-3 text-sm text-red-300">
        {error}
      </div>
    )}

    {result && !loading && !error && (
      <div className="space-y-2">
        {/* Top prediction */}
        <p className={`text-2xl font-bold capitalize ${confidenceColor(result.confidence)}`}>
          {result.label}
        </p>

        {/* Confidence bar */}
        <div className="w-full bg-white/10 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-500 ${result.confidence >= 0.75
              ? "bg-green-400"
              : result.confidence >= 0.5
                ? "bg-yellow-400"
                : "bg-red-400"
              }`}
            style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
          />
        </div>

        {/* Timing + confidence row */}
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span>{(result.confidence * 100).toFixed(1)}% confidence</span>
          <span className="flex items-center gap-1">
            <Zap size={11} className="text-violet-400" />
            {result.latencyMs} ms inference • {result.totalMs} ms total
            {result.modelKind && (
              <span className="ml-1 px-1.5 py-0.5 rounded bg-white/10 text-[10px] font-mono uppercase">
                {result.modelKind}
              </span>
            )}
          </span>
        </div>

        {/* Top-K breakdown */}
        {result.topK && result.topK.length > 1 && (
          <div className="mt-3 space-y-1">
            <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">Top predictions</p>
            {result.topK.map((p, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="w-24 truncate text-slate-300 capitalize">{p.class_name}</span>
                <div className="flex-1 bg-white/10 rounded-full h-1.5">
                  <div
                    className="h-1.5 rounded-full bg-indigo-400/70"
                    style={{ width: `${(p.confidence * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="text-slate-400 w-10 text-right">
                  {(p.confidence * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    )}

    {!result && !loading && !error && (
      <p className="text-sm text-slate-500 italic">No results yet.</p>
    )}
  </div>
);
