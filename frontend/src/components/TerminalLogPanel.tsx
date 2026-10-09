import { useState, useEffect, useRef, useCallback } from 'react';
import { Terminal, ChevronDown, Trash2, Circle } from 'lucide-react';
import { WS_BASE, withAuthQuery } from '../config';
import { useWebSocket } from '../hooks/useWebSocket';

interface LogEntry {
  seq?: number;
  ts: number;
  level: 'info' | 'warning' | 'error';
  message: string;
}

interface TerminalLogPanelProps {
  /** training_id or optimization_id - the job whose console output to stream. */
  jobId: string | null | undefined;
  /** Label shown in the panel header, e.g. "Training Console" or "Optimization Console". */
  title?: string;
  /** Start expanded rather than collapsed. */
  defaultOpen?: boolean;
}


function levelColor(level: LogEntry['level']): string {
  switch (level) {
    case 'error': return 'text-red-400';
    case 'warning': return 'text-amber-300';
    default: return 'text-slate-300';
  }
}

export function TerminalLogPanel({ jobId, title = 'Live Console', defaultOpen = false }: TerminalLogPanelProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);

  const wsUrl = jobId ? withAuthQuery(`${WS_BASE}/ws/logs/${jobId}`) : null;
  // Highest sequence number already shown: the server replays the whole
  // buffer on every (re)connect, so anything at or below it is a duplicate.
  const lastSeqRef = useRef(0);

  const handleMessage = useCallback((raw: string) => {
    let entry: LogEntry;
    try {
      entry = JSON.parse(raw);
    } catch {
      return; // ignore malformed frames
    }
    if (typeof entry.seq === 'number') {
      if (entry.seq <= lastSeqRef.current) return;
      lastSeqRef.current = entry.seq;
    }
    setLogs((prev) => (prev.length > 3000 ? [...prev.slice(-2000), entry] : [...prev, entry]));
  }, []);

  // Reset logs when job changes
  useEffect(() => {
    setLogs([]);
    lastSeqRef.current = 0;
  }, [jobId]);

  const { status } = useWebSocket(wsUrl, handleMessage);
  const connected = status === 'connected';

  // Clearing keeps lastSeqRef, so a reconnect doesn't bring cleared lines back.
  const clearLogs = useCallback(() => setLogs([]), []);

  // Auto-scroll to bottom on new lines
  useEffect(() => {
    if (open && autoScrollRef.current) {
      bottomRef.current?.scrollIntoView({ block: 'end' });
    }
  }, [logs, open]);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    autoScrollRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  if (!jobId) return null;

  return (
    <div className="rounded-xl border border-slate-700 bg-black/40 overflow-hidden">
      <div
        role="button"
        tabIndex={0}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen((v) => !v); } }}
        className="w-full flex items-center justify-between px-4 py-2.5 bg-slate-900/60 hover:bg-slate-900 transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-2">
          <Terminal size={14} className="text-emerald-400" />
          <span className="text-xs font-semibold text-slate-300">{title}</span>
          <Circle size={7} className={connected ? 'text-emerald-400 fill-emerald-400' : 'text-slate-600 fill-slate-600'} />
          <span className="text-[10px] text-slate-500">
            {status === 'connected' ? 'live' : status === 'reconnecting' ? 'reconnecting...' : status === 'failed' ? 'failed' : 'disconnected'}
          </span>
          {logs.length > 0 && (
            <span className="text-[10px] text-slate-600">({logs.length} lines)</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {logs.length > 0 && (
            <button
              onClick={(e) => { e.stopPropagation(); clearLogs(); }}
              className="p-1 rounded hover:bg-slate-700 text-slate-500 hover:text-slate-300 transition-colors"
              title="Clear log view"
              aria-label="Clear log view"
            >
              <Trash2 size={12} />
            </button>
          )}
          <ChevronDown size={14} className={`text-slate-500 transition-transform ${open ? 'rotate-180' : ''}`} />
        </div>
      </div>

      {open && (
        <div
          onScroll={handleScroll}
          className="max-h-72 overflow-y-auto px-4 py-3 font-mono text-[11px] leading-relaxed bg-black/60"
        >
          {logs.length === 0 ? (
            <p className="text-slate-600 italic">Waiting for output...</p>
          ) : (
            logs.map((entry, i) => (
              <div key={`${entry.ts}-${i}`} className={`whitespace-pre-wrap break-words ${levelColor(entry.level)}`}>
                <span className="text-slate-600">{new Date(entry.ts * 1000).toLocaleTimeString()} </span>
                {entry.message}
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
