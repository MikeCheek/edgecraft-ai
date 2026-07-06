import React, { useState, useMemo, useEffect } from 'react';
import {
  FolderTree,
  X,
  FileImage,
  ShieldCheck,
  ChevronRight,
  ChevronDown,
  Folder,
  FolderOpen,
  File,
  Regex,
  Tag,
  Settings2,
  Loader
} from 'lucide-react';
import { useAPI } from '../../hooks/useAPI';

export interface TreeItem {
  path: string;
  file_count: number;
  split: string;
  label: string;
  ignore: boolean;
  files?: string[];
}

interface ZipTreeMapperProps {
  uploadId: string;
  tree: TreeItem[];
  onConfirm: (mapping: TreeItem[], labelStrategy: string, regexPattern: string) => void; // UPDATED
  onCancel: () => void;
}

interface TreeNode {
  name: string;
  fullPath: string;
  itemIndex: number | null;
  children: Record<string, TreeNode>;
}

const REGEX_PRESETS = [
  { label: "Custom...", value: "" },
  { label: "Before first underscore (_)", value: "^([^_]+)" },
  { label: "Before first hyphen (-)", value: "^([^-]+)" },
  { label: "Before first dot (.)", value: "^([^.]+)" },
  { label: "Before first space", value: "^([^\\s]+)" },
  { label: "Before first slash (/)", value: "^([^/]+)" },
  { label: "Extract letters only", value: "([a-zA-Z]+)" },
  { label: "Extract numbers only", value: "(\\d+)" }
];

const SPLIT_KEYWORDS = ['train', 'training', 'test', 'testing', 'val', 'validation'];

// Helper to auto-detect split from a path string
const autoDetectSplit = (path: string): string => {
  const p = path.toLowerCase();
  if (p.includes('train')) return 'train';
  if (p.includes('test')) return 'test';
  if (p.includes('val')) return 'val';
  return 'unassigned';
};

// Helper component to highlight the Regex match inside filenames
const HighlightText = ({ text, pattern, active }: { text: string; pattern: string; active: boolean }) => {
  if (!pattern || !active) return <span>{text}</span>;
  try {
    const regex = new RegExp(pattern);
    const match = text.match(regex);
    if (!match) return <span>{text}</span>;

    const matchedSubstring = match[1] || match[0];
    const index = text.indexOf(matchedSubstring);

    if (index === -1) return <span>{text}</span>;

    const before = text.substring(0, index);
    const after = text.substring(index + matchedSubstring.length);

    return (
      <span className="truncate">
        {before}
        <span className="bg-indigo-500/30 text-indigo-300 font-semibold px-1 py-0.5 rounded border border-indigo-500/40 font-mono text-[11px] inline-block mx-0.5 shadow-sm">
          {matchedSubstring}
        </span>
        {after}
      </span>
    );
  } catch (e) {
    return <span>{text}</span>;
  }
};

export function ZipTreeMapper({ uploadId, tree, onConfirm, onCancel }: ZipTreeMapperProps) {
  const { apiClient } = useAPI();
  // Initialize mapping with auto-detected splits and cleared invalid labels
  const [mapping, setMapping] = useState<TreeItem[]>(() => {
    return tree.map(item => {
      let autoSplit = item.split;
      if (!autoSplit || autoSplit === 'unassigned') {
        autoSplit = autoDetectSplit(item.path);
      }

      let autoLabel = item.label;
      const folderName = item.path.split('/').pop()?.toLowerCase() || '';
      // If the default label is just "training" or "testing", clear it.
      if (SPLIT_KEYWORDS.includes(folderName) && autoLabel.toLowerCase() === folderName) {
        autoLabel = '';
      }
      return { ...item, split: autoSplit, label: autoLabel };
    });
  });

  const [labelSource, setLabelSource] = useState<'folder' | 'filename'>('folder');
  const [regexPattern, setRegexPattern] = useState<string>('^([^_]+)');

  // NEW: State for backend preview
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [discoveredClasses, setDiscoveredClasses] = useState<{ name: string, count: number }[]>([]);
  const [regexError, setRegexError] = useState<string | null>(null);
  const [visibleFilesCount, setVisibleFilesCount] = useState<Record<string, number>>({});

  const { treeRoot, initialExpanded } = useMemo(() => {
    const root: TreeNode = { name: '', fullPath: 'root', itemIndex: null, children: {} };
    const expanded = new Set<string>();

    mapping.forEach((item, index) => {
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
        expanded.add(fullPath);
      });
      current.itemIndex = index;
    });

    return { treeRoot: root, initialExpanded: expanded };
  }, [mapping]);

  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(initialExpanded);

  const uniqueClassesSummary = useMemo(() => {
    const classesMap: Record<string, number> = {};
    mapping.forEach((item) => {
      if (!item.ignore && item.label.trim()) {
        const normalized = item.label.trim();
        classesMap[normalized] = (classesMap[normalized] || 0) + item.file_count;
      }
    });
    return Object.entries(classesMap).sort((a, b) => b[1] - a[1]);
  }, [mapping]);

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
    setMapping(mapping.map((m) => ({ ...m, ignore })));
  };

  const handleLoadMoreFiles = (path: string) => {
    setVisibleFilesCount((prev) => ({
      ...prev,
      [path]: (prev[path] || 5) + 5
    }));
  };

  const applyLabelStrategy = async () => {
    setPreviewError(null);
    setDiscoveredClasses([]);

    if (labelSource === 'folder') {
      const newMapping = mapping.map((item) => {
        if (item.ignore) return item;
        const parts = item.path.split('/').filter(Boolean);
        const validParts = parts.filter(p => !['train', 'training', 'test', 'testing', 'val', 'validation'].includes(p.toLowerCase()));
        return { ...item, label: validParts.length > 0 ? validParts[validParts.length - 1] : item.label };
      });
      setMapping(newMapping);

      // Calculate local preview for folder strategy
      const classesMap: Record<string, number> = {};
      newMapping.forEach((m) => {
        if (!m.ignore && m.label) {
          classesMap[m.label] = (classesMap[m.label] || 0) + m.file_count;
        }
      });
      setDiscoveredClasses(Object.entries(classesMap).map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count));

    } else if (labelSource === 'filename') {
      if (!regexPattern) return;

      setPreviewLoading(true);
      try {
        // Fetch real preview from Python Backend
        // Make sure to add `previewZipRegex` to your apiClient methods
        const res = await apiClient.previewZipRegex(uploadId, regexPattern);
        if (res.status === 'success') {
          setDiscoveredClasses(res.classes);
        } else {
          setPreviewError(res.message || "Failed to preview regex.");
        }
      } catch (err: any) {
        setPreviewError(err.message || "Invalid Regex.");
      } finally {
        setPreviewLoading(false);
      }
    }
  };

  const totalFiles = mapping.filter((m) => !m.ignore).reduce((acc, m) => acc + m.file_count, 0);
  const isPresetMatch = REGEX_PRESETS.some((p) => p.value === regexPattern);

  const renderNode = (node: TreeNode, depth: number) => {
    const item = node.itemIndex !== null ? mapping[node.itemIndex] : null;
    const isExpanded = expandedFolders.has(node.fullPath);
    const children = Object.values(node.children);
    const hasChildren = children.length > 0;

    const allowedFileLimit = visibleFilesCount[node.fullPath] || 5;
    const hasFilesToPreview = item && item.files && item.files.length > 0 && isExpanded && !item.ignore;

    return (
      <React.Fragment key={node.fullPath}>
        <div className="flex items-center hover:bg-slate-800/40 py-2 px-4 border-b border-slate-800/50 group transition-colors">
          <div className="flex-1 flex items-center gap-2 overflow-hidden pr-4" style={{ paddingLeft: `${depth * 1.5}rem` }}>
            {hasChildren ? (
              <button
                onClick={() => toggleFolder(node.fullPath)}
                className="p-0.5 hover:bg-slate-700 rounded text-gray-400 transition-colors focus:outline-none"
              >
                {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              </button>
            ) : (
              <div className="w-5" />
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

          <div className={`flex items-center gap-2 ${item?.ignore ? 'opacity-40 grayscale' : ''} transition-all`}>
            <div className="w-12 flex justify-center">
              {item && (
                <input
                  type="checkbox"
                  checked={!item.ignore}
                  onChange={(e) => updateItem(node.itemIndex!, { ignore: !e.target.checked })}
                  className="w-4 h-4 accent-indigo-500 cursor-pointer rounded bg-slate-900 border-slate-700"
                />
              )}
            </div>

            <div className="w-20 flex justify-end pr-2">
              {item && (
                <span className="text-[11px] font-semibold bg-slate-800 px-2 py-1 rounded text-cyan-400 flex items-center gap-1 border border-slate-700">
                  <FileImage size={12} /> {item.file_count}
                </span>
              )}
            </div>

            <div className="w-32">
              {item && (
                <select
                  value={item.split}
                  onChange={(e) => updateItem(node.itemIndex!, { split: e.target.value })}
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

            <div className="w-40">
              {item && (
                <input
                  type="text"
                  value={item.label}
                  placeholder="Auto-assigned"
                  onChange={(e) => updateItem(node.itemIndex!, { label: e.target.value })}
                  disabled={item.ignore}
                  className="w-full bg-slate-900 border border-slate-700 text-white text-xs px-3 py-1.5 rounded focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 font-semibold text-purple-300 outline-none transition-all placeholder:text-slate-600 placeholder:font-normal"
                />
              )}
            </div>
          </div>
        </div>

        {hasFilesToPreview && (
          <>
            {item!.files!.slice(0, allowedFileLimit).map((fileName, idx) => (
              <div
                key={`file-${node.fullPath}-${idx}`}
                className="flex items-center bg-slate-950/20 hover:bg-slate-900/30 py-1.5 px-4 border-b border-slate-800/30 transition-colors"
              >
                <div
                  className="flex-1 flex items-center gap-2 overflow-hidden pr-4"
                  style={{ paddingLeft: `${(depth + 1) * 1.5 + 1}rem` }}
                >
                  <File size={13} className="text-slate-500 shrink-0" />
                  <span className="text-xs font-mono text-slate-400 truncate">
                    <HighlightText
                      text={fileName}
                      pattern={regexPattern}
                      active={labelSource === 'filename'}
                    />
                  </span>
                </div>
                <div className="flex items-center gap-2 opacity-0 pointer-events-none select-none">
                  <div className="w-12" />
                  <div className="w-20" />
                  <div className="w-32" />
                  <div className="w-40" />
                </div>
              </div>
            ))}

            {item!.files!.length > allowedFileLimit && (
              <div className="flex items-center bg-slate-950/10 py-1 px-4 border-b border-slate-800/20">
                <div
                  className="flex-1"
                  style={{ paddingLeft: `${(depth + 1) * 1.5 + 1}rem` }}
                >
                  <button
                    onClick={() => handleLoadMoreFiles(node.fullPath)}
                    className="text-xs text-indigo-400 hover:text-indigo-300 font-semibold transition-colors pl-5 py-1 focus:outline-none"
                  >
                    + Load more ({item!.files!.length - allowedFileLimit} hidden)
                  </button>
                </div>
              </div>
            )}
          </>
        )}

        {hasChildren && isExpanded && (
          <div className="flex flex-col">
            {children.map((child) => renderNode(child, depth + 1))}
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

        {/* Dynamic Toolbar Strategy Section */}
        <div className="px-6 py-3 border-b border-slate-800 flex flex-col md:flex-row gap-4 items-start md:items-center bg-slate-900 shrink-0">
          <div className="flex gap-2">
            <button onClick={() => toggleAll(false)} className="text-[11px] px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded transition-colors border border-slate-700">Select All</button>
            <button onClick={() => toggleAll(true)} className="text-[11px] px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded transition-colors border border-slate-700">Deselect All</button>
          </div>

          <div className="w-px h-6 bg-slate-700 hidden md:block" />

          {/* Strategy Toggle */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400 font-medium flex items-center gap-1.5">
              <Settings2 size={14} /> Class Source:
            </span>
            <div className="flex bg-slate-950 rounded border border-slate-700 p-0.5">
              <button
                onClick={() => setLabelSource('folder')}
                className={`text-[11px] px-3 py-1 rounded transition-colors font-medium ${labelSource === 'folder' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
              >
                Folder Name
              </button>
              <button
                onClick={() => setLabelSource('filename')}
                className={`text-[11px] px-3 py-1 rounded transition-colors font-medium ${labelSource === 'filename' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
              >
                File Name Regex
              </button>
            </div>
          </div>

          {/* Regex Input Panel (Only shows if Strategy is Filename) */}
          {labelSource === 'filename' && (
            <div className="flex items-center gap-2 flex-1 animate-in fade-in slide-in-from-left-4 duration-300 min-w-[300px]">
              <div className="w-px h-6 bg-slate-700 hidden md:block mx-1" />
              <Regex size={15} className="text-indigo-400" />

              <select
                value={isPresetMatch ? regexPattern : ""}
                onChange={(e) => setRegexPattern(e.target.value)}
                className="text-[11px] px-2 py-1.5 bg-slate-950 border border-slate-700 text-slate-300 rounded focus:border-indigo-500 outline-none max-w-[140px] truncate"
              >
                {REGEX_PRESETS.map((preset, i) => (
                  <option key={i} value={preset.value}>{preset.label}</option>
                ))}
              </select>

              <input
                type="text"
                placeholder="Regex (e.g. ^([a-zA-Z]+)_)"
                value={regexPattern}
                onChange={(e) => setRegexPattern(e.target.value)}
                className={`text-xs px-3 py-1.5 bg-slate-950 border ${regexError ? 'border-red-500' : 'border-slate-700'} text-white rounded focus:border-indigo-500 outline-none flex-1 font-mono`}
              />
            </div>
          )}
          <div className="flex-1 md:flex-none flex justify-end">
            <button
              onClick={applyLabelStrategy}
              disabled={previewLoading}
              className="flex items-center gap-1.5 text-[11px] px-4 py-1.5 bg-slate-200 hover:bg-white text-slate-900 font-bold rounded transition-all whitespace-nowrap shadow-sm disabled:opacity-50"
            >
              {previewLoading && <Loader size={12} className="animate-spin" />}
              {labelSource === 'filename' ? 'Preview Backend Strategy' : 'Apply Strategy'}
            </button>
          </div>

          {previewError && (
            <div className="px-6 py-1 bg-red-500/10 border-b border-red-500/20 text-xs text-red-400 font-medium flex items-center justify-center">
              {previewError}
            </div>
          )}

        </div>
        {/* Updated Active Classes Preview Panel using Backend Data */}
        {discoveredClasses.length > 0 && (
          <div className="px-6 py-2.5 bg-slate-950/60 border-b border-slate-800 flex flex-wrap gap-2 items-center shrink-0 max-h-32 overflow-y-auto custom-scrollbar">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wider mr-2">
              <Tag size={13} className="text-purple-400" /> Discovered Classes ({discoveredClasses.length}):
            </div>
            {discoveredClasses.map((cls) => (
              <span
                key={cls.name}
                className="text-[11px] bg-purple-500/10 border border-purple-500/30 text-purple-300 px-2.5 py-0.5 rounded-full flex items-center gap-1.5 font-medium"
              >
                {cls.name}
                <span className="text-[10px] bg-purple-500/20 text-purple-400 px-1.5 py-0.5 rounded-full font-bold">
                  {cls.count}
                </span>
              </span>
            ))}
          </div>
        )}

        {/* TreeGrid Area */}
        <div className="flex-1 overflow-y-auto custom-scrollbar bg-slate-950/50">
          <div className="min-w-[800px]">
            <div className="flex px-4 py-3 border-b border-slate-700 bg-slate-800/90 text-[11px] text-gray-400 uppercase font-semibold tracking-wider sticky top-0 z-10 backdrop-blur-md">
              <div className="flex-1 pl-6">Folder Structure</div>
              <div className="flex gap-2">
                <div className="w-12 text-center">Import</div>
                <div className="w-20 text-right pr-2">Files</div>
                <div className="w-32">Split Tag</div>
                <div className="w-40">Class Label Tag</div>
              </div>
            </div>

            <div className="pb-4">
              {Object.values(treeRoot.children).map((child) => renderNode(child, 0))}
            </div>
          </div>
        </div>

        <div className="px-6 py-4 border-t border-slate-700 bg-slate-800/50 rounded-b-2xl flex justify-between items-center shrink-0">
          <span className="text-sm text-gray-400">Total ready to import: <strong className="text-white">{totalFiles}</strong> samples</span>
          <div className="flex gap-3">
            <button onClick={onCancel} className="px-4 py-2 text-sm text-gray-300 hover:text-white transition-colors">Cancel</button>
            <button
              onClick={() => onConfirm(mapping, labelSource, regexPattern)} // UPDATED
              disabled={totalFiles === 0 || previewLoading}
              className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-all shadow-lg flex items-center gap-2"
            >
              <ShieldCheck size={18} /> Apply & Extract
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}