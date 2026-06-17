// Task Types
export type TinyMLTask =
  | 'IMAGE_CLASSIFICATION'
  | 'OBJECT_DETECTION'
  | 'VISUAL_WAKE_WORDS'
  | 'KEYWORD_SPOTTING'
  | 'AUDIO_CLASSIFICATION'

export type TargetBoard =
  | 'ESP32_S3_N16R8'
  | 'RASPBERRY_PI_PICO_2_W'
  | 'ARDUINO_NANO_33_BLE'

export type QuantizationMethod =
  | 'INT8_QUANTIZATION'
  | 'FLOAT16_QUANTIZATION'
  | 'PRUNING'
  | 'WEIGHT_CLUSTERING'
  | 'DYNAMIC_QUANTIZATION'

export type DatasetSplit = 'train' | 'val' | 'test' | 'unassigned'

export interface DatasetInfo {
  id: string
  name: string
  task: TinyMLTask
  sample_count: number
  created_at: number
}

export interface DatasetSample {
  id: string
  dataset_id: string
  label: string
  task: TinyMLTask
  filename: string
  timestamp: number
  split?: DatasetSplit
}

export interface TrainingConfig {
  dataset_id: string
  epochs: number
  batch_size: number
  learning_rate: number
  base_model: string
  task: TinyMLTask
  validation_split: number
  input_shape?: number[]
}

export interface ModelMetadata {
  id: string
  name: string
  training_id: string
  task: TinyMLTask
  dataset_id?: string
  base_model?: string
  created_at: number
  accuracy: number
  loss: number
  val_accuracy: number
  val_loss: number
  optimized: boolean
  size_bytes: number
  download_url?: string
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

export interface OptimizationResult {
  id: string
  original_size_bytes: number
  optimized_size_bytes: number
  compression_ratio: number
  method: QuantizationMethod
  status: 'initialized' | 'running' | 'completed' | 'failed'
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
