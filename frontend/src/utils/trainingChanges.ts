// "Apply to configuration" from the training review -> the ModelTrainer form.
// Kept outside the (lazily loaded) LLMAdvisor so the trainer can listen
// without pulling the advisor into its bundle.
import { TrainingChanges } from '../types'

export const APPLY_TRAINING_CHANGES_EVENT = 'edgecraft:apply-training-changes'

export interface ApplyTrainingChangesDetail {
  changes: TrainingChanges
  datasetId?: string
  /** The reviewed run's full configuration (changes are relative to it). */
  base: Record<string, any>
}

export function applyTrainingChanges (detail: ApplyTrainingChangesDetail) {
  window.dispatchEvent(new CustomEvent(APPLY_TRAINING_CHANGES_EVENT, { detail }))
}

const PRIORITY_RANK: Record<string, number> = { high: 0, medium: 1, low: 2 }

/**
 * Merge the parameter changes of several suggestions into one set. When two
 * suggestions set the same parameter, the higher-priority one wins (ties: the
 * earlier one); augmentation options are merged key by key the same way.
 */
export function mergeSuggestionChanges (
  suggestions: { priority?: string; changes?: TrainingChanges }[]
): TrainingChanges {
  const ordered = suggestions
    .map((s, i) => ({ s, i }))
    .sort((a, b) => (PRIORITY_RANK[a.s.priority ?? 'medium'] ?? 1) - (PRIORITY_RANK[b.s.priority ?? 'medium'] ?? 1) || a.i - b.i)
  const out: Record<string, unknown> = {}
  for (const { s } of ordered) {
    for (const [k, v] of Object.entries(s.changes ?? {})) {
      if (v === undefined || v === null) continue
      if (k === 'augmentation' && typeof v === 'object' && !Array.isArray(v)) {
        out.augmentation = { ...v, ...((out.augmentation as Record<string, boolean | number>) ?? {}) }
      } else if (!(k in out)) {
        out[k] = v
      }
    }
  }
  return out as TrainingChanges
}
