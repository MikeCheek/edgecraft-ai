import { TargetBoard, TinyMLTask } from '../types'

export const TASK_OPTIONS: { id: TinyMLTask; label: string; desc: string }[] = [
  { id: 'IMAGE_CLASSIFICATION', label: 'Image Classification', desc: 'Categorize whole images' },
  { id: 'OBJECT_DETECTION', label: 'Object Detection', desc: 'Locate objects (centroids, FOMO-style)' },
  { id: 'VISUAL_WAKE_WORDS', label: 'Visual Wake Words', desc: 'Person / object presence detector' },
  { id: 'KEYWORD_SPOTTING', label: 'Keyword Spotting', desc: 'Detect spoken keywords (1 s clips)' },
  { id: 'AUDIO_CLASSIFICATION', label: 'Audio Classification', desc: 'Classify sounds (2 s clips)' },
]

export const BOARD_OPTIONS: { id: TargetBoard; label: string; specs: string }[] = [
  { id: 'ESP32_S3_N16R8', label: 'ESP32-S3 (N16R8)', specs: 'LX7, 16 MB flash, 8 MB PSRAM' },
  { id: 'ESP32_CAM', label: 'ESP32-CAM', specs: 'LX6, OV2640, 4 MB PSRAM' },
  { id: 'RASPBERRY_PI_PICO_2_W', label: 'Pico 2 W', specs: 'RP2350, 520 KB SRAM' },
  { id: 'ARDUINO_NANO_33_BLE', label: 'Nano 33 BLE', specs: 'nRF52840, 256 KB RAM' },
]

export const OPENROUTER_MODELS = [
  { id: 'openrouter/free', label: 'OpenRouter Free (auto)' },
  { id: 'meta-llama/llama-3.3-70b-instruct:free', label: 'Llama 3.3 70B (free)' },
  { id: 'qwen/qwen-2.5-72b-instruct:free', label: 'Qwen 2.5 72B (free)' },
  { id: 'mistralai/mistral-small-3.1-24b-instruct:free', label: 'Mistral Small 3.1 (free)' },
]

export const taskLabel = (t?: string) => TASK_OPTIONS.find(o => o.id === t)?.label ?? (t ?? '').replace(/_/g, ' ')
export const boardLabel = (b?: string) => BOARD_OPTIONS.find(o => o.id === b)?.label ?? (b ?? '').replace(/_/g, ' ')
