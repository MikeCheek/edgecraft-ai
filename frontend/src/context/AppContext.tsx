// AppContext.tsx
// Full React context with localStorage persistence for TinyML app state.
// Hydrates task, board, and LLM model selections from localStorage on init.
// Dispatch wrapper persists relevant actions as side-effects (pure reducer stays clean).

import { createContext, useContext, useReducer, useCallback, ReactNode } from 'react';
import {
  TrainingStatus,
  OptimizationResult,
  DatasetStatistics,
  ModelMetadata,
  TinyMLTask,
  TargetBoard,
} from '../types';

// --- Storage Keys ---
const STORAGE_KEYS = {
  TASK: 'ec_task',
  BOARD: 'ec_board',
  LLM_MODEL: 'ec_llm',
} as const;

// --- localStorage Helpers ---

/**
 * Safely loads and parses a value from localStorage.
 * Returns the fallback if the key is missing or JSON parsing fails.
 */
function load<T>(key: string, fallback: T): T {
  try {
    const item = localStorage.getItem(key);
    return item ? (JSON.parse(item) as T) : fallback;
  } catch {
    return fallback;
  }
}

/**
 * Safely serialises and saves a value to localStorage.
 * Fails silently on quota-exceeded or serialisation errors.
 */
function save(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota exceeded — fail silently */
  }
}

// --- State Shape ---

interface AppState {
  /** The currently selected TinyML task (image, audio, motion, etc.) */
  currentTask?: TinyMLTask;

  /** The currently selected deployment target board */
  currentBoard?: TargetBoard;

  /** Live training status polled from the backend */
  currentTraining?: TrainingStatus;

  /** Latest optimisation result (quantisation, pruning, etc.) */
  currentOptimization?: OptimizationResult;

  /** All trained model metadata entries fetched from the backend */
  trainedModels: ModelMetadata[];

  /** Aggregated dataset statistics */
  datasetStats: DatasetStatistics;

  /** Global loading flag used by async operations */
  isLoading: boolean;

  /** Human-readable error message; undefined means no active error */
  error?: string;

  /** OpenRouter / LLM model identifier string */
  llmModel: string;
}

// --- Action Union ---

type Action =
  | { type: 'SET_TASK'; payload: TinyMLTask }
  | { type: 'SET_BOARD'; payload: TargetBoard }
  | { type: 'SET_TRAINING'; payload: TrainingStatus }
  | { type: 'SET_OPTIMIZATION'; payload: OptimizationResult }
  | { type: 'SET_MODELS'; payload: ModelMetadata[] }
  | { type: 'UPDATE_DATASET_STATS'; payload: DatasetStatistics }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload?: string }
  | { type: 'SET_LLM_MODEL'; payload: string }
  | { type: 'RESET' };

// --- Initial State ---

/** Static defaults; never references localStorage directly. */
const BASE_INITIAL_STATE: AppState = {
  trainedModels: [],
  datasetStats: { total_samples: 0, by_task: {}, by_label: {} },
  isLoading: false,
  llmModel: 'openrouter/free',
};

/**
 * Builds the initial state by merging localStorage-persisted values
 * on top of the static defaults. Called once on mount (and again on RESET).
 */
function getInitialState(): AppState {
  return {
    ...BASE_INITIAL_STATE,
    currentTask: load<TinyMLTask | undefined>(STORAGE_KEYS.TASK, undefined),
    currentBoard: load<TargetBoard | undefined>(STORAGE_KEYS.BOARD, undefined),
    llmModel: load<string>(STORAGE_KEYS.LLM_MODEL, BASE_INITIAL_STATE.llmModel),
  };
}

// --- Reducer ---

/**
 * Pure state reducer — no side-effects.
 * All localStorage writes happen in the wrapped dispatch below.
 */
function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_TASK':
      return { ...state, currentTask: action.payload };

    case 'SET_BOARD':
      return { ...state, currentBoard: action.payload };

    case 'SET_TRAINING':
      return { ...state, currentTraining: action.payload };

    case 'SET_OPTIMIZATION':
      return { ...state, currentOptimization: action.payload };

    case 'SET_MODELS':
      return { ...state, trainedModels: action.payload };

    case 'UPDATE_DATASET_STATS':
      return { ...state, datasetStats: action.payload };

    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };

    case 'SET_ERROR':
      return { ...state, error: action.payload };

    case 'SET_LLM_MODEL':
      return { ...state, llmModel: action.payload };

    case 'RESET':
      // Re-hydrate from (now-cleared) localStorage — dispatch wrapper
      // removes the keys before baseDispatch reaches here.
      return getInitialState();

    default:
      return state;
  }
}

// --- Context Type ---

interface AppContextType {
  state: AppState;
  /**
   * Wrapped dispatch: persists relevant actions to localStorage
   * before forwarding every action to the reducer.
   */
  dispatch: (action: Action) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

// --- Provider ---

/**
 * AppProvider
 *
 * Wrap your component tree with this provider to give all descendants
 * access to global app state and a stable dispatch function.
 *
 * localStorage persistence is handled transparently:
 *  - SET_TASK      ? persists currentTask
 *  - SET_BOARD     ? persists currentBoard
 *  - SET_LLM_MODEL ? persists llmModel
 *  - RESET         ? clears all persisted keys then re-hydrates
 */
export function AppProvider({ children }: { children: ReactNode }) {
  // Pass getInitialState as the lazy initialiser (3rd arg) so it only
  // runs once and avoids repeated localStorage reads on every render.
  const [state, baseDispatch] = useReducer(appReducer, undefined, getInitialState);

  /**
   * Wrapped dispatch.
   *
   * Side-effects (localStorage) are intentionally kept here rather than
   * inside the reducer to maintain reducer purity and testability.
   * useCallback ensures the reference is stable across renders.
   */
  const dispatch = useCallback((action: Action): void => {
    switch (action.type) {
      case 'SET_TASK':
        save(STORAGE_KEYS.TASK, action.payload);
        break;

      case 'SET_BOARD':
        save(STORAGE_KEYS.BOARD, action.payload);
        break;

      case 'SET_LLM_MODEL':
        save(STORAGE_KEYS.LLM_MODEL, action.payload);
        break;

      case 'RESET':
        // Clear all persisted keys so getInitialState() returns clean defaults.
        Object.values(STORAGE_KEYS).forEach((key) => localStorage.removeItem(key));
        break;

      // All other actions carry no persistent data — pass through directly.
      default:
        break;
    }

    baseDispatch(action);
  }, []); // baseDispatch is stable; no dependencies needed

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

// --- Consumer Hook ---

/**
 * useAppContext
 * * Convenience hook for consuming AppContext.
 * Throws a descriptive error if called outside of <AppProvider>.
 * * @example
 * const { state, dispatch } = useAppContext();
 */
export function useAppContext(): AppContextType {
  const context = useContext(AppContext);

  if (!context) {
    throw new Error(
      'useAppContext must be used within an <AppProvider>. ' + 'Ensure your component tree is wrapped with <AppProvider>.'
    );
  }

  return context;
}
