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
