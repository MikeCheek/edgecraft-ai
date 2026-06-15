import { Upload, X, Folder, Link as LinkIcon, File as FileIcon } from 'lucide-react';
import { useState, useRef } from 'react';
import { useAPI } from '../hooks/useAPI';
import { TinyMLTask } from '../types';

interface DataCollectorProps { datasetId: string; task: TinyMLTask; onSampleAdded?: () => void; }

export function DataCollector({ datasetId, task, onSampleAdded }: DataCollectorProps) {
  const [label, setLabel] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isFolderMode, setIsFolderMode] = useState(false);
  const [progress, setProgress] = useState(0);

  const { request, apiClient } = useAPI();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async () => {
    setIsLoading(true);
    const validExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.wav'];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];

      // Determine label: Use custom input if in file mode, use folder name if in folder mode
      let targetLabel = label;
      if (isFolderMode) {
        const pathParts = file.webkitRelativePath.split('/');
        targetLabel = pathParts.length > 2 ? pathParts[pathParts.length - 2] : 'unknown';
      }

      // Filter by extension
      const ext = `.${file.name.split('.').pop()?.toLowerCase()}`;
      if (!validExtensions.includes(ext)) continue;

      if (targetLabel.trim()) {
        await request(() => apiClient.uploadSample(datasetId, targetLabel, task, file));
      }
      setProgress(Math.round(((i + 1) / files.length) * 100));
    }
    setLabel(''); setFiles([]); onSampleAdded?.();
    setIsLoading(false);
  };

  return (
    <div className="space-y-4">
      {/* Mode Switcher */}
      <div className="flex gap-2">
        <button onClick={() => { setIsFolderMode(false); setFiles([]); }} className={`flex-1 py-1 text-sm rounded ${!isFolderMode ? 'bg-purple-600' : 'bg-slate-700'}`}>Single Files</button>
        <button onClick={() => { setIsFolderMode(true); setFiles([]); }} className={`flex-1 py-1 text-sm rounded ${isFolderMode ? 'bg-purple-600' : 'bg-slate-700'}`}>Load Folder</button>
      </div>

      {!isFolderMode && (
        <div>
          <label className="block text-sm text-gray-300 mb-1">Label</label>
          <input type="text" value={label} onChange={e => setLabel(e.target.value)} className="w-full px-3 py-2 bg-slate-700 rounded-lg text-white" placeholder="e.g. cat, noise..." />
        </div>
      )}

      <div className="border-2 border-dashed border-purple-500/50 rounded-lg p-6 text-center bg-slate-800/50">
        <input
          type="file"
          ref={fileInputRef}
          onChange={e => setFiles(Array.from(e.target.files || []))}
          className="hidden"
          // Magic attributes for folder loading
          {...(isFolderMode ? { webkitdirectory: "true", directory: "true" } : {})}
        />
        <button onClick={() => fileInputRef.current?.click()} className="px-4 py-2 bg-slate-700 text-white rounded">
          {isFolderMode ? 'Select Root Folder' : 'Select Files'}
        </button>
        {files.length > 0 && <p className="text-xs text-gray-400 mt-2">{files.length} items found</p>}
      </div>

      <button
        onClick={handleUpload}
        disabled={isLoading || files.length === 0 || (!isFolderMode && !label)}
        className="w-full py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg disabled:opacity-50"
      >
        {isLoading ? 'Uploading...' : 'Upload Data'}
      </button>
      {isLoading && (
        <div className="w-full bg-slate-700 rounded-full h-2 mt-2">
          <div className="bg-purple-500 h-2 rounded-full transition-all" style={{ width: `${progress}%` }} />
        </div>
      )}
    </div>
  );
}