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