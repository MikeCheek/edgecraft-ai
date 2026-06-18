import React, { useState } from 'react';
import { X, ZoomIn } from 'lucide-react';
import { MetricChart, ChartModal } from './Chart';
import { formatDate } from './constants';


interface PastSessionPopupProps {
  session: any;
  onClose: () => void;
}

function PastSessionPopup({ session, onClose }: PastSessionPopupProps) {
  const [expandedChart, setExpandedChart] = useState<'accuracy' | 'loss' | null>(null);

  const accuracyData = (session.metrics ?? []).map((m: any) => ({
    epoch: m.epoch,
    train: parseFloat((m.accuracy * 100).toFixed(2)),
    val: parseFloat((m.val_accuracy * 100).toFixed(2)),
  }));
  const lossData = (session.metrics ?? []).map((m: any) => ({
    epoch: m.epoch,
    train: parseFloat(m.loss.toFixed(4)),
    val: parseFloat(m.val_loss.toFixed(4)),
  }));

  const last = session.metrics?.length ? session.metrics[session.metrics.length - 1] : null;

  const statusColor =
    session.status === 'completed' ? 'text-green-400' :
      session.status === 'failed' ? 'text-red-400' :
        session.status === 'cancelled' ? 'text-yellow-400' : 'text-purple-400';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <div
        className="w-full max-w-3xl max-h-[90vh] overflow-y-auto bg-slate-900 rounded-2xl border border-slate-700 shadow-2xl animate-slideIn"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700 sticky top-0 bg-slate-900 z-10">
          <div>
            <h3 className="text-lg font-bold text-white">
              {session.base_model} – {session.task?.replace(/_/g, ' ')}
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">{formatDate(session.created_at)}</p>
          </div>
          <button onClick={onClose} className="p-2 text-gray-400 hover:text-white hover:bg-slate-700 rounded-lg transition">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-5">
          {/* Info grid — validation_split removed */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
            {[
              ['Status', <span className={`font-semibold ${statusColor}`}>{session.status?.toUpperCase()}</span>],
              ['Epochs', `${session.current_epoch ?? 0} / ${session.total_epochs}`],
              ['Batch Size', session.batch_size],
              ['Learning Rate', session.learning_rate],
              ['Dropout', session.dropout_rate ?? '–'],
              ['L2 Reg', session.l2_reg ?? '–'],
              ['Early Stop', session.early_stopping ? `Yes (pat. ${session.early_stopping_patience})` : 'No'],
            ].map(([k, v], i) => (
              <div key={i} className="p-3 bg-slate-800 rounded-lg border border-slate-700">
                <div className="text-gray-400 mb-1">{k}</div>
                <div className="text-white font-medium">{v}</div>
              </div>
            ))}
          </div>

          {/* Final metrics */}
          {last && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3 bg-slate-800 rounded-lg border border-slate-700">
                <div className="text-xs text-gray-400 mb-1">Train Acc</div>
                <div className="text-lg font-bold text-green-400">{(last.accuracy * 100).toFixed(1)}%</div>
              </div>
              <div className="p-3 bg-slate-800 rounded-lg border border-slate-700">
                <div className="text-xs text-gray-400 mb-1">Val Acc</div>
                <div className="text-lg font-bold text-cyan-400">{(last.val_accuracy * 100).toFixed(1)}%</div>
              </div>
              <div className="p-3 bg-slate-800 rounded-lg border border-slate-700">
                <div className="text-xs text-gray-400 mb-1">Train Loss</div>
                <div className="text-lg font-bold text-yellow-400">{last.loss.toFixed(4)}</div>
              </div>
              <div className="p-3 bg-slate-800 rounded-lg border border-slate-700">
                <div className="text-xs text-gray-400 mb-1">Val Loss</div>
                <div className="text-lg font-bold text-orange-400">{last.val_loss.toFixed(4)}</div>
              </div>
            </div>
          )}

          {/* Charts */}
          {accuracyData.length >= 2 && (
            <div className="space-y-3">
              <div className="relative">
                <button
                  onClick={() => setExpandedChart('accuracy')}
                  className="absolute top-2 right-2 z-10 p-1.5 bg-slate-700 hover:bg-slate-600 rounded-lg text-gray-400 hover:text-white transition"
                  title="Expand"
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
                <MetricChart data={accuracyData} label="Accuracy (%)" color="#22c55e" valColor="#06b6d4" formatY={v => `${v}%`} />
              </div>
              <div className="relative">
                <button
                  onClick={() => setExpandedChart('loss')}
                  className="absolute top-2 right-2 z-10 p-1.5 bg-slate-700 hover:bg-slate-600 rounded-lg text-gray-400 hover:text-white transition"
                  title="Expand"
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
                <MetricChart data={lossData} label="Loss" color="#eab308" valColor="#f97316" />
              </div>
            </div>
          )}

          {session.status === 'failed' && session.error && (
            <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-300 text-sm">
              Error: {session.error}
            </div>
          )}
        </div>
      </div>

      {/* Nested chart modal */}
      {expandedChart === 'accuracy' && (
        <ChartModal
          label="Accuracy (%)"
          color="#22c55e"
          valColor="#06b6d4"
          data={accuracyData}
          formatY={v => `${v}%`}
          onClose={() => setExpandedChart(null)}
        />
      )}
      {expandedChart === 'loss' && (
        <ChartModal
          label="Loss"
          color="#eab308"
          valColor="#f97316"
          data={lossData}
          onClose={() => setExpandedChart(null)}
        />
      )}
    </div>
  );
}

export default PastSessionPopup;