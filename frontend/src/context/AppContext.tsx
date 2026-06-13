import { createContext, useContext, useReducer, ReactNode } from 'react';
import {
  TrainingStatus,
  OptimizationResult,
  DatasetStatistics,
  ModelMetadata,
  TinyMLTask,
  TargetBoard,
} from '../types';

interface AppState {
  currentTask?: TinyMLTask;
  currentBoard?: TargetBoard;
  currentTraining?: TrainingStatus;
  currentOptimization?: OptimizationResult;
  trainedModels: ModelMetadata[];
  datasetStats: DatasetStatistics;
  isLoading: boolean;
  error?: string;
}

type Action =
  | { type: 'SET_TASK'; payload: TinyMLTask }
  | { type: 'SET_BOARD'; payload: TargetBoard }
  | { type: 'SET_TRAINING'; payload: TrainingStatus }
  | { type: 'SET_OPTIMIZATION'; payload: OptimizationResult }
  | { type: 'ADD_MODEL'; payload: ModelMetadata }
  | { type: 'UPDATE_DATASET_STATS'; payload: DatasetStatistics }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload?: string }
  | { type: 'RESET' };

const initialState: AppState = {
  trainedModels: [],
  datasetStats: { totalSamples: 0, byTask: {}, byLabel: {} },
  isLoading: false,
};

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
    case 'ADD_MODEL':
      return {
        ...state,
        trainedModels: [...state.trainedModels, action.payload],
      };
    case 'UPDATE_DATASET_STATS':
      return { ...state, datasetStats: action.payload };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'RESET':
      return initialState;
    default:
      return state;
  }
}

interface AppContextType {
  state: AppState;
  dispatch: (action: Action) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return context;
}
