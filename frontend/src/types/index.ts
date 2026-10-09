// Task Types
export type TinyMLTask =
  | 'IMAGE_CLASSIFICATION'
  | 'OBJECT_DETECTION'
  | 'VISUAL_WAKE_WORDS'
  | 'KEYWORD_SPOTTING'
  | 'AUDIO_CLASSIFICATION'

export type TargetBoard =
  | 'ESP32_S3_N16R8'
  | 'ESP32_CAM'
  | 'RASPBERRY_PI_PICO_2_W'
  | 'ARDUINO_NANO_33_BLE'

export type QuantizationMethod =
  | 'INT8_QUANTIZATION'
  | 'FLOAT16_QUANTIZATION'
  | 'PRUNING'
  | 'WEIGHT_CLUSTERING'
  | 'DYNAMIC_QUANTIZATION'

export type DatasetSplit = 'train' | 'val' | 'test' | 'unassigned'

// ── Object Detection Annotations ──────────────────────────────────────
export interface BoundingBox {
  class_name: string
  cx: number  // normalized center x (0-1)
  cy: number  // normalized center y (0-1)
  w: number   // normalized width (0-1)
  h: number   // normalized height (0-1)
  confidence?: number
}

export type AnnotationFormat = 'yolo' | 'voc' | 'coco' | 'csv'

export interface AnnotationSummary {
  has_annotations: boolean
  annotated_count: number
  total_count: number
  format: AnnotationFormat | null
  classes: { name: string; count: number }[]
  total_bboxes: number
}

// Which LLM backend to use for AI-assisted suggestions. Mirrors the
// `provider` field accepted by /api/optimization/llm-suggest and
// /api/training/recommend.
export type LLMProvider = 'openrouter' | 'ollama'

// Server-side, .env-driven view of which providers are actually usable
// (see GET /api/optimization/llm-config). Drives whether the frontend
// auto-selects a provider or lets the user choose between them.
export interface LLMProviderConfig {
  openrouter_available: boolean
  ollama_available: boolean
  ollama_model: string
  ollama_host: string
}

// NEW: dataset-wide image size / aspect-ratio / storage stats, computed by
// the backend from per-sample width/height/size_bytes captured at ingest.
export interface DatasetImageStats {
  total_samples: number
  samples_with_dimensions: number
  formats: Record<string, number>
  total_size_bytes: number
  avg_size_bytes?: number | null
  width?: { min: number; max: number; avg: number } | null
  height?: { min: number; max: number; avg: number } | null
  aspect_ratio?: { min: number; max: number; avg: number } | null
  most_common_resolutions: { resolution: string; count: number }[]
  uniform_dimensions?: boolean | null
}

export interface DatasetInfo {
  id: string
  name: string
  task: TinyMLTask
  sample_count: number
  created_at: number
  description?: string
  metadata?: {
    image_stats?: DatasetImageStats
    image_stats_computed_at?: number
    annotation_format?: AnnotationFormat
    annotation_classes?: string[]
    [key: string]: any
  }
  annotation_format?: AnnotationFormat | null
  annotation_classes?: string[]
}

export interface DatasetSample {
  id: string
  dataset_id: string
  label: string
  task: TinyMLTask
  filename: string
  timestamp: number
  split?: DatasetSplit
  size_bytes?: number
  width?: number | null
  height?: number | null
  updated_at?: number
  annotations?: BoundingBox[]
}

export interface TrainingConfig {
  dataset_id: string
  epochs: number
  batch_size: number
  learning_rate: number
  base_model: string
  task: TinyMLTask
  input_shape?: number[]
}

export interface ModelMetadata {
  test_accuracy?: number | null
  params?: number
  input_shape?: number[]
  id: string
  name: string
  training_id: string
  task: TinyMLTask
  dataset_id?: string
  dataset_name?: string
  base_model?: string
  created_at: number
  accuracy: number
  loss: number
  val_accuracy: number
  val_loss: number
  optimized: boolean
  size_bytes: number
  download_url?: string
  type: 'image' | 'audio' | 'tabular' | 'text'
  labels: string[]
}

export interface TrainingMetrics {
  epoch: number
  loss: number
  accuracy: number
  val_loss: number
  val_accuracy: number
  timestamp: number
  // Live-dashboard extras (absent on sessions trained before they existed)
  val_precision?: number
  val_recall?: number
  val_f1?: number
  val_confidence?: number
  val_ece?: number
  learning_rate?: number | null
  phase?: string
  time_ms?: number
  samples_per_sec?: number | null
  weight_norm?: number
  update_ratio?: number | null
  memory_mb?: number | null
  gpu_memory_mb?: number | null
}

export interface TrainingLiveProgress {
  epoch: number
  batch: number
  batches: number
  loss: number
  accuracy?: number | null
  phase?: string
  epoch_elapsed: number
  updated_at: number
}

export interface TrainingRunInfo {
  num_train: number
  num_val: number
  num_classes: number
  labels: string[]
  steps_per_epoch: number
  params_total: number
  params_trainable: number
  train_class_counts?: Record<string, number> | null
}

export interface TrainingLiveEval {
  epoch: number
  labels?: string[]
  confusion_matrix?: number[][]
  per_class?: { label?: string; precision: number; recall: number; f1: number; support?: number; tp?: number; fp?: number; fn?: number }[]
  num_samples?: number
}

export type JobStatus = 'queued' | 'initialized' | 'running' | 'completed' | 'failed' | 'cancelled'

/** Held-out evaluation stored on a finished session (see EvaluationReport). */
export interface SessionEvaluation {
  split: string
  num_samples: number
  accuracy?: number
  macro_f1?: number
  precision?: number
  recall?: number
  f1?: number
  labels?: string[]
  per_class?: { label?: string; precision: number; recall: number; f1: number; support?: number; tp?: number; fp?: number; fn?: number }[]
  confusion_matrix?: number[][]
}

export interface TrainingStatus {
  id: string
  name?: string
  task?: TinyMLTask
  status: JobStatus
  current_epoch: number
  total_epochs: number
  progress: number
  created_at: number
  started_at?: number
  completed_at?: number
  elapsed_seconds?: number
  remaining_seconds?: number
  device_used?: 'cpu' | 'gpu' | null
  error?: string
  queue_position?: number
  evaluation?: SessionEvaluation | null
  metrics: TrainingMetrics[]
  live?: TrainingLiveProgress | null
  batch_history?: { step: number; loss: number; accuracy?: number }[]
  live_eval?: TrainingLiveEval | null
  run_info?: TrainingRunInfo | null
  base_model?: string
  batch_size?: number
  learning_rate?: number
  input_shape?: number[]
  early_stopping?: boolean
  early_stopping_patience?: number
  early_stopping_monitor?: string
  freeze_encoder_epochs?: number
  dataset_id?: string
}

export interface OptimizationComparisonSide {
  accuracy: number
  loss?: number
  avg_inference_ms: number
  size_bytes: number
}

export interface OptimizationComparison {
  test_split_used: string
  num_samples_evaluated: number
  original: OptimizationComparisonSide
  optimized: OptimizationComparisonSide
  deltas: {
    accuracy_delta: number
    speedup_factor: number
    size_reduction_pct: number
  }
  error?: string
}

export interface OptimizationResult {
  id: string
  original_size_bytes: number
  optimized_size_bytes: number
  compression_ratio: number
  method: QuantizationMethod
  status: 'initialized' | 'running' | 'completed' | 'failed'
  comparison?: OptimizationComparison
  c_array?: string
  cpp_wrapper?: string
  download_url?: string
}

export interface BoardRecommendation {
  board: TargetBoard
  board_name: string
  ram_usage_kb: number
  flash_usage_kb: number
  ram_percentage: number
  flash_percentage: number
  ram_estimation_method?: string
  measured_inference_ms_on_host?: number
  estimated_inference_ms_on_device?: number
  estimation_note?: string
  warnings: string[]
  suggestions: string[]
  estimated_inference_ms: number
  deployment_feasible: boolean
}

export type ReviewSeverity = 'critical' | 'warning' | 'info' | 'good'

export interface TrainingChanges {
  base_model?: string
  input_shape?: number[]
  epochs?: number
  batch_size?: number
  learning_rate?: number
  dropout_rate?: number
  l2_reg?: number
  early_stopping?: boolean
  early_stopping_patience?: number
  early_stopping_monitor?: 'val_loss' | 'val_accuracy'
  freeze_encoder_epochs?: number
  trainable_layers?: number
  class_weighting?: boolean
  augmentation?: Record<string, boolean | number>
}

export interface ReviewSuggestion {
  title: string
  priority: 'high' | 'medium' | 'low'
  category: 'training' | 'data' | 'deployment'
  reasoning: string
  changes: TrainingChanges
  rejected_changes?: string[] | null
  expected_effect: string
  source: 'ai' | 'rules'
}

export interface TrainingReview {
  version: number
  training_id: string
  score: number
  score_label: string
  score_breakdown: { name: string; points: number; max: number; detail: string }[]
  summary: string
  ai_summary?: string
  findings: { severity: ReviewSeverity; title: string; evidence: string }[]
  suggestions: ReviewSuggestion[]
  rule_suggestions: ReviewSuggestion[]
  suggestions_source: 'ai' | 'rules'
  ai_error?: string
  ai_provider?: string
  ai_model?: string
  ai_seconds?: number
  generated_at: number
  board?: string | null
  comparison?: {
    runs: number
    best_previous: Record<string, any>
    this_run_best_val_accuracy: number
    delta_vs_best_previous_pts: number | null
  } | null
  facts: Record<string, any>
}

export interface DatasetStatistics {
  total_samples: number
  by_task: Record<string, number>
  by_label: Record<string, number>
}

export interface ApiResponse<T> {
  status: 'success' | 'error'
  data?: T
  message?: string
  error?: string
}

export interface InferenceResult {
  class_name: string
  confidence: number
  inference_time_ms: number
}

export interface TreeItem {
  path: string
  file_count: number
  split: string
  label: string
  ignore: boolean
  files?: string[]
}

export interface PastTrainingSession {
  id: string
  name?: string
  dataset_id: string
  base_model: string
  task: TinyMLTask
  status: JobStatus
  evaluation?: SessionEvaluation | null
  input_shape?: number[]
  seed?: number | null
  augmentation?: Record<string, unknown>
  archived?: boolean
  freeze_encoder_epochs?: number
  trainable_layers?: number
  current_epoch: number
  total_epochs: number
  batch_size: number
  learning_rate: number
  dropout_rate?: number
  l2_reg?: number
  early_stopping?: boolean
  early_stopping_patience?: number
  created_at: number
  error?: string
  metrics: TrainingMetrics[]
}
