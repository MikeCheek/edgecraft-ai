import { ImageIcon, RefreshCw, Trash2, Check, X, Tag, Edit2 } from 'lucide-react';
import React, { useState } from 'react'
import { DatasetSample } from '../../types';


interface SampleCardProps {
  sample: DatasetSample;
  allLabels: string[];
  apiBase: string;
  onRelabel: (id: string, newLabel: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  onSplitChange: (id: string, newSplit: string) => Promise<void>;
}

function SampleCard({ sample, allLabels, apiBase, onRelabel, onDelete, onSplitChange }: SampleCardProps) {
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(sample.label);
  const [customLabel, setCustomLabel] = useState('');
  const [useCustom, setUseCustom] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [imgError, setImgError] = useState(false);

  const isAudio = sample.filename?.match(/\.(wav|mp3)$/i);

  const handleSave = async () => {
    const finalLabel = useCustom ? customLabel.trim() : label;
    if (!finalLabel || finalLabel === sample.label) { setEditing(false); return; }
    setSaving(true);
    await onRelabel(sample.id, finalLabel);
    setSaving(false);
    setEditing(false);
  };

  const handleDelete = async () => {
    if (!window.confirm("Are you sure you want to delete this sample?")) return;
    setDeleting(true);
    await onDelete(sample.id);
  };

  return (
    <div className={`bg-slate-800 rounded-xl border border-slate-700 overflow-hidden group hover:border-slate-500 transition-all ${deleting ? 'opacity-50' : ''}`}>
      <div className="relative h-28 bg-slate-900 flex items-center justify-center">
        {isAudio ? (
          <div className="flex flex-col items-center gap-1 opacity-50">
            <span className="text-2xl">🎵</span>
            <span className="text-xs text-gray-400 truncate px-2 max-w-full">{sample.filename}</span>
          </div>
        ) : imgError ? (
          <div className="flex flex-col items-center gap-1 opacity-40">
            <ImageIcon className="w-8 h-8 text-gray-500" />
            <span className="text-xs text-gray-500">No preview</span>
          </div>
        ) : (
          <img
            src={`${apiBase}/datasets/image/${sample.id}`}
            alt={sample.label}
            loading="lazy"
            decoding="async"
            className="w-full h-full object-cover"
            onError={() => setImgError(true)}
          />
        )}
        <button
          onClick={handleDelete}
          disabled={deleting}
          className="absolute top-1 right-1 p-1 bg-red-600/80 hover:bg-red-500 rounded-md opacity-0 group-hover:opacity-100 transition-opacity disabled:cursor-not-allowed"
        >
          {deleting
            ? <RefreshCw className="w-3 h-3 text-white animate-spin" />
            : <Trash2 className="w-3 h-3 text-white" />}
        </button>
      </div>

      <div className="p-2">
        {editing ? (
          <div className="space-y-1.5">
            {!useCustom && (
              <select value={label} onChange={e => setLabel(e.target.value)}
                className="w-full px-2 py-1 bg-slate-700 border border-slate-500 rounded text-white text-xs focus:outline-none focus:border-purple-500">
                {allLabels.map(l => <option key={l} value={l}>{l}</option>)}
              </select>
            )}
            <div className="flex items-center gap-1">
              <input type="checkbox" checked={useCustom} onChange={e => setUseCustom(e.target.checked)}
                className="accent-purple-500" id={`custom-${sample.id}`} />
              <label htmlFor={`custom-${sample.id}`} className="text-xs text-gray-400 cursor-pointer">New label</label>
            </div>
            {useCustom && (
              <input autoFocus value={customLabel} onChange={e => setCustomLabel(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSave()}
                placeholder="Type new label..."
                className="w-full px-2 py-1 bg-slate-700 border border-slate-500 rounded text-white text-xs focus:outline-none focus:border-purple-500" />
            )}
            <div className="flex gap-1 pt-0.5">
              <button onClick={handleSave} disabled={saving}
                className="flex-1 flex items-center justify-center gap-1 py-1 bg-green-700 hover:bg-green-600 disabled:opacity-50 text-white text-xs rounded transition">
                {saving ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />} Save
              </button>
              <button onClick={() => { setEditing(false); setLabel(sample.label); setUseCustom(false); }}
                className="px-2 py-1 bg-slate-700 hover:bg-slate-600 text-gray-400 text-xs rounded transition">
                <X className="w-3 h-3" />
              </button>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-between gap-1">
            <span className="text-xs font-medium text-purple-300 truncate flex items-center gap-1">
              <Tag className="w-3 h-3 flex-shrink-0" />{sample.label}
            </span>
            <button onClick={() => { setEditing(true); setLabel(sample.label); }}
              className="p-1 text-gray-500 hover:text-white hover:bg-slate-700 rounded transition flex-shrink-0">
              <Edit2 className="w-3 h-3" />
            </button>
          </div>
        )}

        <div className="flex items-center justify-between mt-1 pt-1 border-t border-slate-700/50">
          <span className="text-[10px] text-gray-500 uppercase tracking-wider">Split</span>
          <select value={sample.split || 'unassigned'} onChange={(e) => onSplitChange(sample.id, e.target.value)}
            className={`text-[10px] px-1.5 py-0.5 rounded font-medium focus:outline-none cursor-pointer ${sample.split === 'train' ? 'bg-blue-900/40 text-blue-400' :
              sample.split === 'val' ? 'bg-amber-900/40 text-amber-400' :
                sample.split === 'test' ? 'bg-green-900/40 text-green-400' :
                  'bg-slate-700/50 text-gray-400'
              }`}>
            <option value="unassigned">Unassigned</option>
            <option value="train">Train</option>
            <option value="val">Val</option>
            <option value="test">Test</option>
          </select>
        </div>
      </div>
    </div>
  );
}

export default SampleCard