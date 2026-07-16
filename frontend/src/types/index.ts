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
  | 'TRANSFER_LEARNING'

export type DatasetSplit = 'train' | 'val' | 'test' | 'unassigned'

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
  // NEW: user-editable free text description, plus a small metadata bag
  // (currently just a cached `image_stats`) surfaced to the LLM advisor.
  description?: string
  metadata?: {
    image_stats?: DatasetImageStats
    image_stats_computed_at?: number
    [key: string]: any
  }
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
}

export interface TrainingStatus {
  id: string
  status: 'initialized' | 'running' | 'completed' | 'failed' | 'cancelled'
  current_epoch: number
  total_epochs: number
  progress: number
  created_at: number
  started_at?: number
  metrics: TrainingMetrics[]
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

export interface LLMSuggestion {
  suggestion: string
  reasoning: string
  parameters_to_adjust: Record<string, any>
  estimated_improvement: string
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
  dataset_id: string
  base_model: string
  task: TinyMLTask
  status: 'initialized' | 'running' | 'completed' | 'failed' | 'cancelled'
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
