import React, { useState } from 'react';
import { FolderTree, X, FileImage, ShieldCheck } from 'lucide-react';
import { TreeItem } from '../../types';

interface ZipTreeMapperProps {
  tree: TreeItem[];
  onConfirm: (mapping: TreeItem[]) => void;
  onCancel: () => void;
}

export function ZipTreeMapper({ tree, onConfirm, onCancel }: ZipTreeMapperProps) {
  const [mapping, setMapping] = useState<TreeItem[]>(tree);

  const updateItem = (index: number, updates: Partial<TreeItem>) => {
    const newMapping = [...mapping];
    newMapping[index] = { ...newMapping[index], ...updates };
    setMapping(newMapping);
  };

  const toggleAll = (ignore: boolean) => {
    setMapping(mapping.map(m => ({ ...m, ignore })));
  };

  const totalFiles = mapping.filter(m => !m.ignore).reduce((acc, m) => acc + m.file_count, 0);

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-4xl shadow-2xl flex flex-col max-h-[90vh]">
        <div className="px-6 py-4 border-b border-slate-700 flex justify-between items-center bg-slate-800/50 rounded-t-2xl">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <FolderTree className="text-indigo-400" /> Map Dataset Folders
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Select which folders to import, assign them to a split, and explicitly name their class label.
            </p>
          </div>
          <button onClick={onCancel} className="text-gray-400 hover:text-white p-2 bg-slate-800 rounded-lg"><X /></button>
        </div>

        <div className="px-6 py-3 border-b border-slate-800 flex gap-2">
          <button onClick={() => toggleAll(false)} className="text-xs px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded">Select All</button>
          <button onClick={() => toggleAll(true)} className="text-xs px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded">Deselect All</button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-700 text-xs text-gray-400 uppercase">
                <th className="pb-3 pl-2 w-12">Import</th>
                <th className="pb-3">Origin Folder Path</th>
                <th className="pb-3 w-24">Files</th>
                <th className="pb-3 w-36">Split Tag</th>
                <th className="pb-3 w-48">Class Label Tag</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {mapping.map((item, idx) => (
                <tr key={idx} className={item.ignore ? 'opacity-40' : 'hover:bg-slate-800/30'}>
                  <td className="py-3 pl-4">
                    <input type="checkbox" checked={!item.ignore} onChange={e => updateItem(idx, { ignore: !e.target.checked })} className="w-4 h-4 accent-indigo-500 cursor-pointer" />
                  </td>
                  <td className="py-3 pr-4">
                    <span className="text-sm font-mono text-gray-300 break-all">{item.path}</span>
                  </td>
                  <td className="py-3">
                    <span className="text-xs font-semibold bg-slate-800 px-2 py-1 rounded text-cyan-400 flex items-center gap-1 w-max">
                      <FileImage size={12} /> {item.file_count}
                    </span>
                  </td>
                  <td className="py-3 pr-4">
                    <select value={item.split} onChange={e => updateItem(idx, { split: e.target.value })} disabled={item.ignore} className="w-full bg-slate-950 border border-slate-700 text-white text-xs px-2 py-1.5 rounded focus:border-indigo-500">
                      <option value="unassigned">Unassigned</option>
                      <option value="train">Train</option>
                      <option value="val">Val</option>
                      <option value="test">Test</option>
                    </select>
                  </td>
                  <td className="py-3">
                    <input type="text" value={item.label} onChange={e => updateItem(idx, { label: e.target.value })} disabled={item.ignore} className="w-full bg-slate-950 border border-slate-700 text-white text-xs px-3 py-1.5 rounded focus:border-indigo-500 font-semibold text-purple-300" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="px-6 py-4 border-t border-slate-700 bg-slate-800/50 rounded-b-2xl flex justify-between items-center">
          <span className="text-sm text-gray-400">Total ready to import: <strong className="text-white">{totalFiles}</strong> samples</span>
          <div className="flex gap-3">
            <button onClick={onCancel} className="px-4 py-2 text-sm text-gray-300 hover:text-white transition">Cancel</button>
            <button onClick={() => onConfirm(mapping)} disabled={totalFiles === 0} className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-bold rounded-lg transition shadow-lg flex items-center gap-2">
              <ShieldCheck size={18} /> Apply mapping & Extract
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}