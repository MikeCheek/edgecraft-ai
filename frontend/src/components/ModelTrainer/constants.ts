import { TinyMLTask } from '../../types'

// ─── Input Size Configuration ────────────────────────────────────────────────
// Edit these values to change the input dimensions sent to the backend.
// Keep them in sync with IMAGE_SHAPES / AUDIO_PARAMS in data_processor.py.

export const INPUT_SIZES = {
  // Image tasks — [width, height, channels]
  IMAGE_CLASSIFICATION:  [224, 224, 3] as number[],
  OBJECT_DETECTION:      [224, 224, 3] as number[],
  VISUAL_WAKE_WORDS:     [96,  96,  1] as number[],   // grayscale

  // Audio tasks — [n_mfcc, time_frames, 1]
  // time_frames ≈ ceil(sample_rate * duration / hop_length)  (default hop = 512)
  KEYWORD_SPOTTING:      [40, 101, 1] as number[],
  AUDIO_CLASSIFICATION:  [64, 101, 1] as number[],
} satisfies Record<TinyMLTask, number[]>

// ─────────────────────────────────────────────────────────────────────────────

export function getTaskDefaults (task: TinyMLTask): {
  input_shape: number[]
  base_model: string
} {
  switch (task) {
    case 'VISUAL_WAKE_WORDS':
      return { input_shape: INPUT_SIZES.VISUAL_WAKE_WORDS, base_model: 'MobileNetV2' }
    case 'KEYWORD_SPOTTING':
      return { input_shape: INPUT_SIZES.KEYWORD_SPOTTING, base_model: 'MFCC_CNN' }
    case 'AUDIO_CLASSIFICATION':
      return { input_shape: INPUT_SIZES.AUDIO_CLASSIFICATION, base_model: 'MFCC_CNN' }
    case 'OBJECT_DETECTION':
      return { input_shape: INPUT_SIZES.OBJECT_DETECTION, base_model: 'MobileNetV2' }
    case 'IMAGE_CLASSIFICATION':
    default:
      return { input_shape: INPUT_SIZES.IMAGE_CLASSIFICATION, base_model: 'MobileNetV2' }
  }
}

export const IMAGE_MODELS = [
  'MobileNetV2',
  'EfficientNet',
  'ResNet50V2',
  'MobileNetV3Small',
  'Custom3LayerCNN'
]
export const AUDIO_MODELS = ['MFCC_CNN', 'WaveNet', 'AudioLSTM', 'AudioGRU']
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
  { label: '10', value: 10 },
  { label: '25 (Quick)', value: 25 },
  { label: '50', value: 50 },
  { label: '100 (Default)', value: 100 },
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
  { label: 'All layers (0)', value: 0 },
  { label: '1', value: 1 },
  { label: '2', value: 2 },
  { label: '4', value: 4 },
  { label: '8', value: 8 },
  { label: 'Custom...', value: 'custom' },
];

export const FREEZE_EPOCHS_OPTIONS: SelectOption[] = [
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
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '00')}`
}

export function formatDate (ts: number): string {
  if (!ts) return '–'
  return new Date(ts * 1000).toLocaleString()
}