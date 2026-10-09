import { TinyMLTask } from '../../types'

// --- Input Size Defaults ---
// Fallbacks only: the backend's GET /api/info is the source of truth
// (app/services/preprocessing.py) and overrides these at runtime.

export const INPUT_SIZES = {
  // Image tasks — [width, height, channels]
  // NOTE: 96x96 (not 224x224) is the default because this studio targets
  // microcontroller-class boards (ESP32-S3, ESP32-CAM). 224x224 inputs
  // produce activation tensors far too large for typical MCU RAM budgets;
  // 96x96 matches what TinyML references (e.g. Visual Wake Words) commonly
  // use and keeps the resulting tensor arena in a deployable range.
  IMAGE_CLASSIFICATION:  [96, 96, 3] as number[],
  OBJECT_DETECTION:      [96, 96, 3] as number[],
  VISUAL_WAKE_WORDS:     [96,  96,  1] as number[],   // grayscale

  // Audio tasks — [n_mfcc, time_frames, 1]
  // time_frames = 1 + (samples - 512) / 320  (1 s clip -> 49, 2 s -> 99)
  KEYWORD_SPOTTING:      [40, 49, 1] as number[],
  AUDIO_CLASSIFICATION:  [40, 99, 1] as number[],
} satisfies Record<TinyMLTask, number[]>

// ----------------------------------------

export function getTaskDefaults (task: TinyMLTask): {
  input_shape: number[]
  base_model: string
} {
  switch (task) {
    case 'VISUAL_WAKE_WORDS':
      return { input_shape: INPUT_SIZES.VISUAL_WAKE_WORDS, base_model: 'MobileNetV3Small' }
    case 'KEYWORD_SPOTTING':
      return { input_shape: INPUT_SIZES.KEYWORD_SPOTTING, base_model: 'DS_CNN' }
    case 'AUDIO_CLASSIFICATION':
      return { input_shape: INPUT_SIZES.AUDIO_CLASSIFICATION, base_model: 'DS_CNN' }
    case 'OBJECT_DETECTION':
      return { input_shape: INPUT_SIZES.OBJECT_DETECTION, base_model: 'FOMO_MobileNetV2' }
    case 'IMAGE_CLASSIFICATION':
    default:
      // MobileNetV3Small at 96x96 is a far more realistic edge default than
      // the previous MobileNetV2 @ 224x224 (which alone can exceed 8MB as
      // float32 - unusable on any of the supported boards without heavy
      // optimization first). Use the "Suggest Optimal Config" button for a
      // dataset- and board-aware recommendation instead of relying purely
      // on this static default.
      return { input_shape: INPUT_SIZES.IMAGE_CLASSIFICATION, base_model: 'MobileNetV3Small' }
  }
}

export const IMAGE_MODELS = [
  'MobileNetV2',
  'MobileNetV3Small',
  'MobileNetV1_0.25',
  'EfficientNet',
  'ResNet50V2',
  'Custom3LayerCNN'
]
export const OD_MODELS = ['FOMO_MobileNetV2', 'FOMO_Tiny']
export const AUDIO_MODELS = ['DS_CNN', 'MFCC_CNN', 'AudioGRU', 'AudioLSTM']

/** One-line hints shown under the model picker. */
export const MODEL_HINTS: Record<string, string> = {
  MobileNetV3Small: 'Pretrained, ~0.9M params - good default for ESP32-S3.',
  'MobileNetV1_0.25': 'Pretrained, ~0.2M params - smallest pretrained option.',
  MobileNetV2: 'Pretrained, ~2.3M params - heavier, fine with PSRAM.',
  EfficientNet: 'Pretrained, ~4M params - usually too big for MCUs.',
  ResNet50V2: 'Pretrained, ~23M params - desktop baseline only.',
  Custom3LayerCNN: 'Tiny from-scratch CNN for Nano 33 BLE / Pico class boards.',
  DS_CNN: 'Depthwise-separable CNN - the standard keyword-spotting model.',
  MFCC_CNN: 'Plain CNN over MFCCs.',
  AudioGRU: 'Recurrent, unrolled for TFLite Micro.',
  AudioLSTM: 'Recurrent, unrolled for TFLite Micro.',
  FOMO_MobileNetV2: 'Centroid detector on a pretrained MobileNetV2-0.35 trunk (output grid = input / 8).',
  FOMO_Tiny: 'From-scratch centroid detector for very small MCUs.',
}
export const AUDIO_TASKS: TinyMLTask[] = [
  'KEYWORD_SPOTTING',
  'AUDIO_CLASSIFICATION'
]

export interface SelectOption {
  label: string;
  value: number | string;
}

export const BATCH_SIZE_OPTIONS: SelectOption[] = [
  { label: '8 (Micro-RAM friendly)', value: 8 },
  { label: '16 (Recommended)', value: 16 },
  { label: '32 (Standard)', value: 32 },
  { label: '64', value: 64 },
  { label: 'Custom...', value: 'custom' }
];

export const LEARNING_RATE_OPTIONS: SelectOption[] = [
  { label: '0.01 (Aggressive)', value: 0.01 },
  { label: '0.001 (Recommended Default)', value: 0.001 },
  { label: '0.0005 (Fine-tuning)', value: 0.0005 },
  { label: '0.0001 (Slow & Safe)', value: 0.0001 },
  { label: 'Custom...', value: 'custom' }
];

export const DROPOUT_OPTIONS: SelectOption[] = [
  { label: 'None (0.0)', value: 0 },
  { label: '0.1 (Light)', value: 0.1 },
  { label: '0.3 (Balanced Regularization)', value: 0.3 },
  { label: '0.5 (Heavy Protection)', value: 0.5 },
  { label: 'Custom...', value: 'custom' }
];

export const EPOCHS_OPTIONS: SelectOption[] = [
  { label: '10 (Quick)', value: 10 },
  { label: '30 (Default)', value: 30 },
  { label: '50', value: 50 },
  { label: '100', value: 100 },
  { label: '150', value: 150 },
  { label: '200 (Long)', value: 200 },
  { label: 'Custom...', value: 'custom' },
];

export const ES_PATIENCE_OPTIONS: SelectOption[] = [
  { label: '3 (Aggressive)', value: 3 },
  { label: '5 (Default)', value: 5 },
  { label: '10', value: 10 },
  { label: '15', value: 15 },
  { label: '20 (Patient)', value: 20 },
  { label: 'Custom...', value: 'custom' },
];

export const TRAINABLE_LAYERS_OPTIONS: SelectOption[] = [
  { label: 'Whole backbone (0)', value: 0 },
  { label: '1', value: 1 },
  { label: '2', value: 2 },
  { label: '4', value: 4 },
  { label: '8', value: 8 },
  { label: 'Custom...', value: 'custom' },
];

/** -1 = let the backend pick (≈ a third of the epochs on pretrained backbones). */
export const FREEZE_AUTO = -1

export const FREEZE_EPOCHS_OPTIONS: SelectOption[] = [
  { label: 'Auto (recommended)', value: FREEZE_AUTO },
  { label: 'None (0)', value: 0 },
  { label: '5', value: 5 },
  { label: '10', value: 10 },
  { label: '20', value: 20 },
  { label: 'Custom...', value: 'custom' },
];

export function formatTime (secs: number): string {
  if (!isFinite(secs) || secs < 0) return '--:--'
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  const s = Math.floor(secs % 60)
  if (h > 0)
    return `${h}h ${String(m).padStart(2, '0')}m ${String(s).padStart(2, '0')}s`
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export function formatDate (ts: number): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString()
}
