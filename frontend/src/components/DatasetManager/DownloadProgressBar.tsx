import { Loader2, Check, AlertTriangle, XCircle } from 'lucide-react';
import React, { useEffect, useState } from 'react'


export interface DownloadProgress {
  downloaded: number;
  total: number;
  startTime: number;
  downloadId?: string;
  phase: 'downloading' | 'processing' | 'complete' | 'error' | 'cancelled';
  message?: string;
  count?: number;
}

function DownloadProgressBar({ progress, onCancel }: { progress: DownloadProgress; onCancel: () => void }) {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (progress.phase === 'downloading' || progress.phase === 'processing') {
      const interval = setInterval(() => setNow(Date.now()), 500);
      return () => clearInterval(interval);
    }
  }, [progress.phase]);

  const elapsed = (now - progress.startTime) / 1000;
  const pct = progress.total > 0 ? Math.min((progress.downloaded / progress.total) * 100, 100) : null;
  const speed = elapsed > 0 ? progress.downloaded / elapsed : 0;
  const remaining = speed > 0 && progress.total > 0 ? (progress.total - progress.downloaded) / speed : 0;

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
    return `${(bytes / 1073741824).toFixed(2)} GB`;
  };

  const formatTime = (secs: number) => {
    if (!isFinite(secs) || secs < 0) return '--:--';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const isActive = progress.phase === 'downloading' || progress.phase === 'processing';

  return (
    <div className="mt-3 p-3 bg-slate-950/80 rounded-lg border border-slate-700 space-y-2">
      {/* Status header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-300 flex items-center gap-1.5">
          {isActive && <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-400" />}
          {progress.phase === 'downloading' && 'Downloading...'}
          {progress.phase === 'processing' && (progress.message || 'Processing archive...')}
          {progress.phase === 'complete' && <><Check className="w-3.5 h-3.5 text-emerald-400" /> Complete — {progress.count} samples imported</>}
          {progress.phase === 'error' && <><AlertTriangle className="w-3.5 h-3.5 text-red-400" /> {progress.message || 'Download failed: ' + progress.message}</>}
          {progress.phase === 'cancelled' && <><XCircle className="w-3.5 h-3.5 text-amber-400" /> Cancelled</>}
        </span>
        {isActive && (
          <button onClick={onCancel}
            className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-red-400 hover:text-red-300 bg-red-900/30 hover:bg-red-900/50 border border-red-500/30 rounded-md transition">
            <XCircle className="w-3 h-3" /> Cancel
          </button>
        )}
      </div>

      {/* Progress bar */}
      {(progress.phase === 'downloading' || progress.phase === 'processing') && (
        <>
          <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-300 ${progress.phase === 'processing' ? 'bg-amber-500 animate-pulse' : 'bg-indigo-500'
              } ${pct === null ? 'animate-pulse opacity-60' : ''}`} style={{ width: pct !== null ? `${pct}%` : '100%' }} />
          </div>

          {/* Stats row */}
          {progress.phase === 'downloading' && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] text-gray-400">
              <div>
                <span className="text-gray-500">Downloaded:</span>{' '}
                <span className="text-slate-200 font-medium">{formatBytes(progress.downloaded)}</span>
                {progress.total > 0 && <span> / {formatBytes(progress.total)}</span>}
              </div>
              <div>
                <span className="text-gray-500">Speed:</span>{' '}
                <span className="text-slate-200 font-medium">{formatBytes(speed)}/s</span>
              </div>
              <div>
                <span className="text-gray-500">Elapsed:</span>{' '}
                <span className="text-slate-200 font-medium">{formatTime(elapsed)}</span>
              </div>
              <div>
                <span className="text-gray-500">ETA:</span>{' '}
                <span className="text-slate-200 font-medium">{formatTime(remaining)}</span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default DownloadProgressBar