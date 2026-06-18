import { createContext, useContext, useReducer, ReactNode } from 'react';
import { TrainingStatus, OptimizationResult, DatasetStatistics, ModelMetadata, TinyMLTask, TargetBoard } from '../types';

interface AppState {
  currentTask?: TinyMLTask;
  currentBoard?: TargetBoard;
  currentTraining?: TrainingStatus;
  currentOptimization?: OptimizationResult;
  trainedModels: ModelMetadata[];
  datasetStats: DatasetStatistics;
  isLoading: boolean;
  error?: string;
  llmModel: string;
}

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
  | { type: 'RESET' }

const initialState: AppState = { trainedModels: [], datasetStats: { total_samples: 0, by_task: {}, by_label: {} }, isLoading: false, llmModel: 'openrouter/free' };

function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_TASK': return { ...state, currentTask: action.payload };
    case 'SET_BOARD': return { ...state, currentBoard: action.payload };
    case 'SET_TRAINING': return { ...state, currentTraining: action.payload };
    case 'SET_OPTIMIZATION': return { ...state, currentOptimization: action.payload };
    case 'SET_MODELS': return { ...state, trainedModels: action.payload };
    case 'UPDATE_DATASET_STATS': return { ...state, datasetStats: action.payload };
    case 'SET_LOADING': return { ...state, isLoading: action.payload };
    case 'SET_ERROR': return { ...state, error: action.payload };
    case 'SET_LLM_MODEL': return { ...state, llmModel: action.payload };
    case 'RESET': return initialState;
    default: return state;
  }
}

interface AppContextType { state: AppState; dispatch: (action: Action) => void; }
const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  return <AppContext.Provider value={{ state, dispatch }}>{children}</AppContext.Provider>;
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) throw new Error('useAppContext must be used within AppProvider');
  return context;
}