import { describe, expect, it } from 'vitest'
import { mergeSuggestionChanges } from '../utils/trainingChanges'

describe('mergeSuggestionChanges', () => {
  it('lets higher-priority suggestions win and merges augmentation', () => {
    const merged = mergeSuggestionChanges([
      { priority: 'low', changes: { dropout_rate: 0.4, epochs: 80, augmentation: { random_rotation: 0.2 } } },
      { priority: 'high', changes: { dropout_rate: 0.5, augmentation: { horizontal_flip: true, random_rotation: 0.1 } } },
      { priority: 'medium', changes: { early_stopping: true, epochs: 60 } },
      { priority: 'high', changes: {} },
    ])
    expect(merged).toEqual({
      dropout_rate: 0.5,
      epochs: 60,
      early_stopping: true,
      augmentation: { horizontal_flip: true, random_rotation: 0.1 },
    })
  })

  it('returns an empty object when nothing changes', () => {
    expect(mergeSuggestionChanges([{ priority: 'high', changes: {} }, {}])).toEqual({})
  })
})
