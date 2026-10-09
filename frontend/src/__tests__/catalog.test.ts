import { describe, expect, it } from 'vitest'
import { boardLabel, taskLabel, TASK_OPTIONS } from '../constants/catalog'
import { getTaskDefaults, OD_MODELS, AUDIO_MODELS } from '../components/ModelTrainer/constants'

describe('catalog', () => {
  it('labels tasks and boards', () => {
    expect(taskLabel('KEYWORD_SPOTTING')).toBe('Keyword Spotting')
    expect(boardLabel('ESP32_CAM')).toBe('ESP32-CAM')
    expect(TASK_OPTIONS).toHaveLength(5)
  })

  it('task defaults use models the backend knows', () => {
    expect(OD_MODELS).toContain(getTaskDefaults('OBJECT_DETECTION').base_model)
    expect(AUDIO_MODELS).toContain(getTaskDefaults('KEYWORD_SPOTTING').base_model)
    expect(getTaskDefaults('KEYWORD_SPOTTING').input_shape).toEqual([40, 49, 1])
  })
})
