import React, { useState, useMemo } from 'react';
import {
  FolderTree,
  X,
  FileImage,
  ShieldCheck,
  ChevronRight,
  ChevronDown,
  Folder,
  FolderOpen
} from 'lucide-react';
import { TreeItem } from '../../types';

interface ZipTreeMapperProps {
  tree: TreeItem[];
  onConfirm: (mapping: TreeItem[]) => void;
  onCancel: () => void;
}

// Data structure to represent the folder hierarchy
interface TreeNode {
  name: string;
  fullPath: string;
  itemIndex: number | null; // Maps back to the flat `mapping` state array
  children: Record<string, TreeNode>;
}

export function ZipTreeMapper({ tree, onConfirm, onCancel }: ZipTreeMapperProps) {
  const [mapping, setMapping] = useState<TreeItem[]>(tree);

  // 1. Build the hierarchical tree structure once on mount
  const { treeRoot, initialExpanded } = useMemo(() => {
    const root: TreeNode = { name: '', fullPath: 'root', itemIndex: null, children: {} };
    const expanded = new Set<string>();

    tree.forEach((item, index) => {
      const parts = item.path.split('/').filter(Boolean);
      let current = root;

      parts.forEach((part, i) => {
        const fullPath = parts.slice(0, i + 1).join('/');
        if (!current.children[part]) {
          current.children[part] = {
            name: part,
            fullPath,
            itemIndex: null,
            children: {}
          };
        }
        current = current.children[part];
        expanded.add(fullPath); // Collect paths so we can expand all by default
      });
      // Assign the mapping index to the final destination node
      current.itemIndex = index;
    });

    return { treeRoot: root, initialExpanded: expanded };
  }, [tree]);

  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(initialExpanded);

  const toggleFolder = (path: string) => {
    const next = new Set(expandedFolders);
    if (next.has(path)) next.delete(path);
    else next.add(path);
    setExpandedFolders(next);
  };

  const updateItem = (index: number, updates: Partial<TreeItem>) => {
    const newMapping = [...mapping];
    newMapping[index] = { ...newMapping[index], ...updates };
    setMapping(newMapping);
  };

  const toggleAll = (ignore: boolean) => {
    setMapping(mapping.map(m => ({ ...m, ignore })));
  };

  const totalFiles = mapping.filter(m => !m.ignore).reduce((acc, m) => acc + m.file_count, 0);

  // 2. Recursive function to render branches and leaves
  const renderNode = (node: TreeNode, depth: number) => {
    const item = node.itemIndex !== null ? mapping[node.itemIndex] : null;
    const isExpanded = expandedFolders.has(node.fullPath);
    const children = Object.values(node.children);
    const hasChildren = children.length > 0;

    return (
      <React.Fragment key={node.fullPath}>
        <div className="flex items-center hover:bg-slate-800/40 py-2 px-4 border-b border-slate-800/50 group transition-colors">

          {/* Left Side: Tree Structure */}
          <div
            className="flex-1 flex items-center gap-2 overflow-hidden pr-4"
            style={{ paddingLeft: `${depth * 1.5}rem` }}
          >
            {hasChildren ? (
              <button
                onClick={() => toggleFolder(node.fullPath)}
                className="p-0.5 hover:bg-slate-700 rounded text-gray-400 transition-colors focus:outline-none"
              >
                {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              </button>
            ) : (
              <div className="w-5" /> // Spacer for alignment
            )}

            {hasChildren ? (
              isExpanded ? <FolderOpen size={16} className="text-indigo-400 shrink-0" /> : <Folder size={16} className="text-indigo-400 shrink-0" />
            ) : (
              <Folder size={16} className="text-gray-500 shrink-0" />
            )}

            <span className={`text-sm font-mono truncate ${item && !item.ignore ? 'text-gray-200' : 'text-gray-500'}`}>
              {node.name}
            </span>
          </div>

          {/* Right Side: Map Tags & Actions (Fixed width columns) */}
          <div className={`flex items-center gap-2 ${item?.ignore ? 'opacity-40 grayscale' : ''} transition-all`}>

            {/* Import Toggle */}
            <div className="w-12 flex justify-center">
              {item && (
                <input
                  type="checkbox"
                  checked={!item.ignore}
                  onChange={e => updateItem(node.itemIndex!, { ignore: !e.target.checked })}
                  className="w-4 h-4 accent-indigo-500 cursor-pointer rounded bg-slate-900 border-slate-700"
                />
              )}
            </div>

            {/* Files Tag */}
            <div className="w-20 flex justify-end pr-2">
              {item && (
                <span className="text-[11px] font-semibold bg-slate-800 px-2 py-1 rounded text-cyan-400 flex items-center gap-1 border border-slate-700">
                  <FileImage size={12} /> {item.file_count}
                </span>
              )}
            </div>

            {/* Split Tag */}
            <div className="w-32">
              {item && (
                <select
                  value={item.split}
                  onChange={e => updateItem(node.itemIndex!, { split: e.target.value })}
                  disabled={item.ignore}
                  className="w-full bg-slate-900 border border-slate-700 text-white text-xs px-2 py-1.5 rounded focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition-all"
                >
                  <option value="unassigned">Unassigned</option>
                  <option value="train">Train</option>
                  <option value="val">Val</option>
                  <option value="test">Test</option>
                </select>
              )}
            </div>

            {/* Class Label Tag */}
            <div className="w-40">
              {item && (
                <input
                  type="text"
                  value={item.label}
                  onChange={e => updateItem(node.itemIndex!, { label: e.target.value })}
                  disabled={item.ignore}
                  className="w-full bg-slate-900 border border-slate-700 text-white text-xs px-3 py-1.5 rounded focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 font-semibold text-purple-300 outline-none transition-all"
                />
              )}
            </div>
          </div>
        </div>

        {/* Recursively render nested children */}
        {hasChildren && isExpanded && (
          <div className="flex flex-col">
            {children.map(child => renderNode(child, depth + 1))}
          </div>
        )}
      </React.Fragment>
    );
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-5xl shadow-2xl flex flex-col max-h-[90vh]">

        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-700 flex justify-between items-center bg-slate-800/50 rounded-t-2xl shrink-0">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <FolderTree className="text-indigo-400" /> Map Dataset Folders
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Select which folders to import, assign them to a split, and explicitly name their class label.
            </p>
          </div>
          <button onClick={onCancel} className="text-gray-400 hover:text-white p-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"><X /></button>
        </div>

        {/* Toolbar */}
        <div className="px-6 py-3 border-b border-slate-800 flex gap-2 shrink-0">
          <button onClick={() => toggleAll(false)} className="text-xs px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded transition-colors border border-slate-700">Select All</button>
          <button onClick={() => toggleAll(true)} className="text-xs px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded transition-colors border border-slate-700">Deselect All</button>
        </div>

        {/* TreeGrid Area */}
        <div className="flex-1 overflow-y-auto custom-scrollbar bg-slate-950/50">
          <div className="min-w-[800px]">
            {/* Tree Grid Header */}
            <div className="flex px-4 py-3 border-b border-slate-700 bg-slate-800/80 text-xs text-gray-400 uppercase font-semibold tracking-wider sticky top-0 z-10 backdrop-blur-md">
              <div className="flex-1 pl-6">Folder Structure</div>
              <div className="flex gap-2">
                <div className="w-12 text-center">Import</div>
                <div className="w-20 text-right pr-2">Files</div>
                <div className="w-32">Split Tag</div>
                <div className="w-40">Class Label Tag</div>
              </div>
            </div>

            {/* Tree Body */}
            <div className="pb-4">
              {Object.values(treeRoot.children).map(child => renderNode(child, 0))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-700 bg-slate-800/50 rounded-b-2xl flex justify-between items-center shrink-0">
          <span className="text-sm text-gray-400">Total ready to import: <strong className="text-white">{totalFiles}</strong> samples</span>
          <div className="flex gap-3">
            <button onClick={onCancel} className="px-4 py-2 text-sm text-gray-300 hover:text-white transition-colors">Cancel</button>
            <button
              onClick={() => onConfirm(mapping)}
              disabled={totalFiles === 0}
              className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-all shadow-lg flex items-center gap-2"
            >
              <ShieldCheck size={18} /> Apply mapping & Extract
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}