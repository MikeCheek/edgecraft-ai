// Task Types
export type TinyMLTask =
  | 'IMAGE_CLASSIFICATION'
  | 'OBJECT_DETECTION'
  | 'VISUAL_WAKE_WORDS'
  | 'KEYWORD_SPOTTING'
  | 'AUDIO_CLASSIFICATION'

// Board Types
export type TargetBoard =
  | 'ESP32_S3_N16R8'
  | 'RASPBERRY_PI_PICO_2_W'
  | 'ARDUINO_NANO_33_BLE'

// Quantization Methods
export type QuantizationMethod =
  | 'INT8_QUANTIZATION'
  | 'FLOAT16_QUANTIZATION'
  | 'PRUNING'

// Dataset Sample
export interface DatasetSample {
  id: string
  label: string
  task: TinyMLTask
  filename: string
  timestamp: number
}

// Training Configuration
export interface TrainingConfig {
  epochs: number
  batchSize: number
  learningRate: number
  baseModel: string
  task: TinyMLTask
  validationSplit: number
}

// Optimization Configuration
export interface OptimizationConfig {
  method: QuantizationMethod
  sparsityLevel: number
  representativeDatasetSize: number
}

// Board Information
export interface BoardInfo {
  board: TargetBoard
  name: string
  ramKb: number
  flashKb: number
  dspEnabled: boolean
  features: string[]
  recommendedModels: string[]
}

// Model Metadata
export interface ModelMetadata {
  id: string
  name: string
  task: TinyMLTask
  createdAt: number
  inputShape: number[]
  outputShape: number[]
  accuracy: number
  loss: number
  optimized: boolean
  sizeBytes: number
}

// Training Metrics
export interface TrainingMetrics {
  epoch: number
  loss: number
  accuracy: number
  valLoss: number
  valAccuracy: number
  timestamp: number
}

// Training Status
export interface TrainingStatus {
  id: string
  status: 'initialized' | 'running' | 'completed' | 'failed' | 'cancelled'
  currentEpoch: number
  totalEpochs: number
  progress: number
  createdAt: number
  startedAt?: number
  metrics: TrainingMetrics[]
}

// Optimization Result
export interface OptimizationResult {
  id: string
  originalSizeBytes: number
  optimizedSizeBytes: number
  compressionRatio: number
  method: QuantizationMethod
  status: 'initialized' | 'running' | 'completed' | 'failed'
  cArray?: string
  cppWrapper?: string
}

// Board Recommendation
export interface BoardRecommendation {
  board: TargetBoard
  boardName: string
  ramUsageKb: number
  flashUsageKb: number
  ramPercentage: number
  flashPercentage: number
  warnings: string[]
  suggestions: string[]
  estimatedInferenceMs: number
  deploymentFeasible: boolean
}

// LLM Suggestion
export interface LLMSuggestion {
  suggestion: string
  reasoning: string
  parametersToAdjust: Record<string, any>
  estimatedImprovement: string
}

// Dataset Statistics
export interface DatasetStatistics {
  totalSamples: number
  byTask: Record<string, number>
  byLabel: Record<string, number>
}

// API Response Wrapper
export interface ApiResponse<T> {
  status: 'success' | 'error'
  data?: T
  message?: string
  error?: string
}
