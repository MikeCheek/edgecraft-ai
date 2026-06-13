import { Upload, X, Folder, Link as LinkIcon, File as FileIcon } from 'lucide-react';
import { useState, useRef } from 'react';
import { useAPI } from '../hooks/useAPI';
import { TinyMLTask } from '../types';

interface DataCollectorProps {
  task: TinyMLTask;
  onSampleAdded?: () => void;
}

type InputMode = 'file' | 'folder' | 'link';

export function DataCollector({ task, onSampleAdded }: DataCollectorProps) {
  const [mode, setMode] = useState<InputMode>('file');
  const [label, setLabel] = useState('');

  // Data states
  const [files, setFiles] = useState<File[]>([]);
  const [urls, setUrls] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const { request, error } = useAPI();

  const isImageTask = task.includes('IMAGE') || task === 'VISUAL_WAKE_WORDS' || task === 'OBJECT_DETECTION';
  const isAudioTask = task.includes('KEYWORD') || task === 'AUDIO_CLASSIFICATION';
  const acceptTypes = isImageTask ? 'image/*' : isAudioTask ? 'audio/*' : '*/*';

  // Refs for hidden inputs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length > 0) {
      setFiles(selectedFiles);
    }
  };

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length > 0) {
      setFiles(selectedFiles);

      // Auto-extract label from folder structure (e.g., "dataset/cats/img1.jpg" -> "cats")
      const pathParts = selectedFiles[0].webkitRelativePath.split('/');
      if (pathParts.length >= 2) {
        const folderName = pathParts[pathParts.length - 2];
        setLabel(folderName);
      }
    }
  };

  const clearSelection = () => {
    setFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (folderInputRef.current) folderInputRef.current.value = '';
  };

  const handleUpload = async () => {
    const hasValidFiles = (mode === 'file' || mode === 'folder') && files.length > 0;
    const hasValidLinks = mode === 'link' && urls.trim().length > 0;

    if (!label.trim()) {
      alert('Please enter a label for this data.');
      return;
    }

    if (!hasValidFiles && !hasValidLinks) {
      alert('Please provide files, a folder, or valid URLs.');
      return;
    }

    setIsLoading(true);

    // Simulate API request - You will update this payload to match your FastAPI backend
    const payload = {
      label,
      mode,
      fileCount: files.length,
      links: mode === 'link' ? urls.split('\n').filter(url => url.trim()) : [],
    };

    const result = await request(() =>
      Promise.resolve({ data: { status: 'success', data: { sample_id: 'temp', payload } } })
    );

    if (result) {
      setLabel('');
      clearSelection();
      setUrls('');
      onSampleAdded?.();
    }
    setIsLoading(false);
  };

  const isSubmitDisabled = isLoading || !label.trim() ||
    ((mode === 'file' || mode === 'folder') && files.length === 0) ||
    (mode === 'link' && urls.trim() === '');

  return (
    <div className="space-y-6">
      {/* Input Mode Selector */}
      <div className="flex bg-slate-800 p-1 rounded-lg">
        <button
          onClick={() => { setMode('file'); clearSelection(); }}
          className={`flex-1 flex items-center justify-center gap-2 py-2 text-sm font-medium rounded-md transition ${mode === 'file' ? 'bg-slate-600 text-white' : 'text-gray-400 hover:text-white'}`}
        >
          <FileIcon className="w-4 h-4" /> File
        </button>
        <button
          onClick={() => { setMode('folder'); clearSelection(); }}
          className={`flex-1 flex items-center justify-center gap-2 py-2 text-sm font-medium rounded-md transition ${mode === 'folder' ? 'bg-slate-600 text-white' : 'text-gray-400 hover:text-white'}`}
        >
          <Folder className="w-4 h-4" /> Folder
        </button>
        <button
          onClick={() => { setMode('link'); clearSelection(); }}
          className={`flex-1 flex items-center justify-center gap-2 py-2 text-sm font-medium rounded-md transition ${mode === 'link' ? 'bg-slate-600 text-white' : 'text-gray-400 hover:text-white'}`}
        >
          <LinkIcon className="w-4 h-4" /> URLs
        </button>
      </div>

      {/* Label Input */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Dataset Label</label>
        <input
          type="text"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder={mode === 'folder' ? "Auto-fills from folder name..." : "e.g., 'person', 'background', 'yes'"}
          className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500 transition"
        />
      </div>

      {/* Dynamic Input Area */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          {mode === 'file' ? `Upload Single ${isImageTask ? 'Image' : 'Audio'}` :
            mode === 'folder' ? 'Upload Directory of Samples' :
              'Paste URLs (one per line)'}
        </label>

        {mode === 'link' ? (
          <textarea
            value={urls}
            onChange={(e) => setUrls(e.target.value)}
            placeholder={`https://example.com/image1.jpg\nhttps://example.com/image2.png`}
            rows={4}
            className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 transition resize-y"
          />
        ) : (
          <div className="border-2 border-dashed border-purple-500/50 rounded-lg p-6 text-center hover:border-purple-500 transition bg-slate-800/50">
            {/* Hidden Inputs */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept={acceptTypes}
              className="hidden"
            />
            {/* webkitdirectory is required to select folders in HTML5 */}
            <input
              type="file"
              ref={folderInputRef}
              onChange={handleFolderSelect}
              accept={acceptTypes}
              webkitdirectory=""
              directory=""
              className="hidden"
            />

            {files.length > 0 ? (
              <div className="flex flex-col items-center justify-center gap-3">
                <div className="flex items-center gap-2 px-4 py-2 bg-slate-700 rounded-lg border border-slate-600">
                  {mode === 'folder' ? <Folder className="w-5 h-5 text-purple-400" /> : <FileIcon className="w-5 h-5 text-purple-400" />}
                  <span className="text-sm font-medium text-gray-200">
                    {mode === 'folder' ? `${files.length} files selected` : files[0].name}
                  </span>
                  <button onClick={clearSelection} className="ml-2 p-1 hover:bg-slate-600 rounded-full transition">
                    <X className="w-4 h-4 text-red-400" />
                  </button>
                </div>
                {mode === 'folder' && (
                  <span className="text-xs text-gray-400">
                    From: {files[0].webkitRelativePath.split('/')[0]}
                  </span>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <Upload className="w-8 h-8 text-purple-400" />
                <span className="text-sm text-gray-300">
                  {mode === 'folder' ? 'Select a folder containing your samples' : 'Click to select a file'}
                </span>
                <button
                  onClick={() => mode === 'folder' ? folderInputRef.current?.click() : fileInputRef.current?.click()}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm font-medium rounded-md border border-slate-600 transition"
                >
                  Browse {mode === 'folder' ? 'Folder' : 'File'}
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {error && <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200 text-sm">{error}</div>}

      <button
        onClick={handleUpload}
        disabled={isSubmitDisabled}
        className="w-full px-4 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:text-gray-500 disabled:cursor-not-allowed text-white font-medium rounded-lg transition"
      >
        {isLoading ? 'Processing...' : `Ingest Data to '${label || '...'}'`}
      </button>
    </div>
  );
}

// Ensure TypeScript recognizes the directory attributes for inputs
declare module 'react' {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    webkitdirectory?: string;
    directory?: string;
  }
}