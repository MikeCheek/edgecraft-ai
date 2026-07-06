import { Info, RefreshCw, Save, ImageIcon, HardDrive, Ratio } from 'lucide-react';
import React, { useCallback, useEffect, useState } from 'react';
import { useAPI } from '../../hooks/useAPI';
import { DatasetImageStats } from '../../types';

interface DatasetInfoPanelProps {
  datasetId: string;
  initialDescription?: string;
  onChanged: () => void;
}

const formatBytes = (bytes?: number | null) => {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let val = bytes;
  let i = 0;
  while (val >= 1024 && i < units.length - 1) { val /= 1024; i++; }
  return `${val.toFixed(1)} ${units[i]}`;
};

function DatasetInfoPanel({ datasetId, initialDescription, onChanged }: DatasetInfoPanelProps) {
  const { request, apiClient } = useAPI();

  const [description, setDescription] = useState(initialDescription ?? '');
  const [savedDescription, setSavedDescription] = useState(initialDescription ?? '');
  const [isSaving, setIsSaving] = useState(false);

  const [stats, setStats] = useState<DatasetImageStats | null>(null);
  const [isLoadingStats, setIsLoadingStats] = useState(false);
  const [statsError, setStatsError] = useState<string | null>(null);

  useEffect(() => {
    setDescription(initialDescription ?? '');
    setSavedDescription(initialDescription ?? '');
  }, [initialDescription]);

  const loadStats = useCallback(async () => {
    setIsLoadingStats(true);
    setStatsError(null);
    try {
      // NEW: requires `getDatasetImageStats` on apiClient, calling
      // GET /api/datasets/{dataset_id}/image_stats
      const res = await request(() => apiClient.getDatasetImageStats(datasetId)) as any;
      if (res?.status === 'success') {
        setStats(res.stats);
      } else {
        setStatsError(res?.message || 'Could not compute image stats.');
      }
    } catch (e: any) {
      setStatsError(e?.message || 'Could not compute image stats.');
    } finally {
      setIsLoadingStats(false);
    }
  }, [datasetId, request, apiClient]);

  useEffect(() => { loadStats(); }, [loadStats]);

  const handleSaveDescription = async () => {
    setIsSaving(true);
    try {
      // NEW: requires `updateDatasetMetadata` on apiClient, calling
      // PATCH /api/datasets/{dataset_id}/metadata with { description }
      const res = await request(() => apiClient.updateDatasetMetadata(datasetId, { description })) as any;
      if (res?.status === 'success') {
        setSavedDescription(description);
        onChanged();
      }
    } finally {
      setIsSaving(false);
    }
  };

  const hasUnsavedChanges = description !== savedDescription;

  return (
    <div className="space-y-4">
      {/* Description */}
      <div className="space-y-1.5">
        <p className="text-xs text-gray-400 font-medium flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5" /> Description
        </p>
        <textarea
          value={description}
          onChange={e => setDescription(e.target.value)}
          placeholder="What's in this dataset, where it came from, known quirks... this gets fed to the LLM advisor as extra context."
          rows={3}
          className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
        />
        <div className="flex justify-end">
          <button
            onClick={handleSaveDescription}
            disabled={isSaving || !hasUnsavedChanges}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-700 hover:bg-purple-600 disabled:opacity-40 text-white text-xs font-semibold rounded-lg transition"
          >
            {isSaving ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
            {hasUnsavedChanges ? 'Save Description' : 'Saved'}
          </button>
        </div>
      </div>

      {/* Image / storage stats */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <p className="text-xs text-gray-400 font-medium flex items-center gap-1.5">
            <ImageIcon className="w-3.5 h-3.5" /> Data Stats
          </p>
          <button
            onClick={loadStats}
            disabled={isLoadingStats}
            className="flex items-center gap-1 text-[11px] text-gray-400 hover:text-white transition"
          >
            <RefreshCw className={`w-3 h-3 ${isLoadingStats ? 'animate-spin' : ''}`} /> Refresh
          </button>
        </div>

        {isLoadingStats ? (
          <div className="flex items-center justify-center py-4 gap-2 text-gray-400 text-xs">
            <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Computing...
          </div>
        ) : statsError ? (
          <p className="text-xs text-red-400 py-2">{statsError}</p>
        ) : !stats ? (
          <p className="text-xs text-gray-500 py-2">No stats yet.</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <div className="p-2.5 bg-slate-900/50 rounded-lg border border-slate-700">
              <div className="flex items-center gap-1 text-gray-500 text-[10px] uppercase tracking-wide">
                <HardDrive className="w-3 h-3" /> Total Size
              </div>
              <div className="text-sm font-semibold text-white mt-0.5">{formatBytes(stats.total_size_bytes)}</div>
              <div className="text-[10px] text-gray-500">avg {formatBytes(stats.avg_size_bytes)}/file</div>
            </div>

            <div className="p-2.5 bg-slate-900/50 rounded-lg border border-slate-700">
              <div className="flex items-center gap-1 text-gray-500 text-[10px] uppercase tracking-wide">
                <ImageIcon className="w-3 h-3" /> Dimensions
              </div>
              {stats.width && stats.height ? (
                <>
                  <div className="text-sm font-semibold text-white mt-0.5">
                    {Math.round(stats.width.avg)}x{Math.round(stats.height.avg)} avg
                  </div>
                  <div className="text-[10px] text-gray-500">
                    {stats.width.min}-{stats.width.max} x {stats.height.min}-{stats.height.max}
                  </div>
                </>
              ) : (
                <div className="text-sm text-gray-500 mt-0.5">Not available</div>
              )}
            </div>

            <div className="p-2.5 bg-slate-900/50 rounded-lg border border-slate-700">
              <div className="flex items-center gap-1 text-gray-500 text-[10px] uppercase tracking-wide">
                <Ratio className="w-3 h-3" /> Aspect Ratio
              </div>
              {stats.aspect_ratio ? (
                <>
                  <div className="text-sm font-semibold text-white mt-0.5">{stats.aspect_ratio.avg.toFixed(2)} avg</div>
                  <div className="text-[10px] text-gray-500">
                    {stats.uniform_dimensions ? 'uniform' : `range ${stats.aspect_ratio.min.toFixed(2)}-${stats.aspect_ratio.max.toFixed(2)}`}
                  </div>
                </>
              ) : (
                <div className="text-sm text-gray-500 mt-0.5">Not available</div>
              )}
            </div>

            <div className="p-2.5 bg-slate-900/50 rounded-lg border border-slate-700">
              <div className="text-gray-500 text-[10px] uppercase tracking-wide">Formats</div>
              <div className="text-sm font-semibold text-white mt-0.5 truncate">
                {Object.entries(stats.formats || {}).map(([ext, count]) => `${ext} (${count})`).join(', ') || 'unknown'}
              </div>
            </div>
          </div>
        )}

        {stats && stats.samples_with_dimensions < stats.total_samples && (
          <p className="text-[10px] text-gray-500 pt-1">
            {stats.samples_with_dimensions}/{stats.total_samples} samples have known dimensions
            (older imports may predate size capture).
          </p>
        )}
      </div>
    </div>
  );
}

export default DatasetInfoPanel;
