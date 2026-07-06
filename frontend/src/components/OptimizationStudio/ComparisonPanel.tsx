import { InferenceResult, ResultCard } from "./utility";

/** Side-by-side comparison with VS divider */
const ComparisonPanel: React.FC<{
  originalResult: InferenceResult | null; optimizedResult: InferenceResult | null; originalLoading: boolean; optimizedLoading: boolean;
  originalError: string | null; optimizedError: string | null;
}> = ({
  originalResult,
  optimizedResult, originalLoading, optimizedLoading,
  originalError,
  optimizedError,
}) => (<div className="flex items-stretch gap-0 mt-4">    <ResultCard
  title="Original" result={originalResult}
  loading={originalLoading}
  error={originalError} />
  {/* VS divider */}    <div className="flex flex-col items-center justify-center px-3 select-none">      <div className="flex-1 w-px bg-white/10" />      <span className="my-2 text-xs font-black text-slate-400 bg-slate-800 rounded-full border border-white/10 px-2 py-0.5">
    VS      </span>      <div className="flex-1 w-px bg-white/10" />
  </div>

  <ResultCard title="Optimized"
    result={optimizedResult} loading={optimizedLoading}
    error={optimizedError}
  />  </div>
  );

export default ComparisonPanel
