import { Upload, X, File as FileIcon, FileArchive, AlertCircle } from 'lucide-react';
import { useState, useRef, useCallback } from 'react';
import { useAPI } from '../hooks/useAPI';
import { TinyMLTask } from '../types';

interface DataCollectorProps {
  datasetId: string;
  task: TinyMLTask;
  onSampleAdded?: () => void;
}

type UploadMode = 'single' | 'zip';

const TOO_MANY_FILES_THRESHOLD = 10;

// Chunk size: 8 MB — sweet spot between round-trip overhead and per-chunk latency.
// Raw PUT body means no multipart parsing cost, so larger chunks are fine.
const CHUNK_SIZE = 8 * 1024 * 1024;

// Parallel in-flight chunks. Browser limits same-origin connections to ~6,
// so 4 leaves headroom for other requests.
const PARALLEL_CHUNKS = 4;

export function DataCollector({ datasetId, task, onSampleAdded }: DataCollectorProps) {
  const [label, setLabel] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadMode, setUploadMode] = useState<UploadMode>('single');
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [tooManyWarning, setTooManyWarning] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const uploadIdRef = useRef<string | null>(null);

  const { apiClient } = useAPI();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const switchMode = useCallback((mode: UploadMode) => {
    setUploadMode(mode);
    setFiles([]);
    setLabel('');
    setTooManyWarning(false);
    setUploadError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

  // ── Chunked ZIP upload ────────────────────────────────────────────────────

  const uploadZipChunked = async (zipFile: File) => {
    const totalChunks = Math.ceil(zipFile.size / CHUNK_SIZE);
    abortRef.current = new AbortController();
    const signal = abortRef.current.signal;

    // 1. Init — backend pre-allocates the sparse file
    setProgressLabel('Initialising…');
    const initRes = await apiClient.initZipUpload({
      dataset_id: datasetId,
      task,
      filename: zipFile.name,
      total_chunks: totalChunks,
      file_size: zipFile.size,
      chunk_size: CHUNK_SIZE,
    });
    if (initRes.status === 'error') throw new Error(initRes.message || 'Init failed');
    const uploadId: string = initRes.upload_id;
    uploadIdRef.current = uploadId;

    // 2. Resume: find which chunks the server already has
    let alreadyReceived: Set<number> = new Set();
    try {
      const st = await apiClient.getZipUploadStatus(uploadId);
      alreadyReceived = new Set(st.received_chunks ?? []);
    } catch { /* start fresh */ }

    // 3. Parallel upload with a work-stealing queue
    const pending = Array.from({ length: totalChunks }, (_, i) => i)
      .filter(i => !alreadyReceived.has(i));

    let completed = alreadyReceived.size;
    let firstError: Error | null = null;

    const uploadOne = async (i: number): Promise<void> => {
      if (signal.aborted || firstError) return;
      const start = i * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, zipFile.size);
      const blob = zipFile.slice(start, end);

      for (let attempt = 0; attempt < 3; attempt++) {
        if (signal.aborted) throw new Error('Upload cancelled');
        if (firstError) return;
        try {
          // Raw PUT — no FormData, no multipart, maximum throughput
          await apiClient.putZipChunk(uploadId, i, blob, signal);
          completed++;
          setProgress(Math.round((completed / totalChunks) * 90));
          setProgressLabel(`${completed} / ${totalChunks} chunks  (${((completed / totalChunks) * 90).toFixed(0)}%)`);
          return;
        } catch (err: any) {
          if (signal.aborted) throw new Error('Upload cancelled');
          if (attempt === 2) {
            firstError = new Error(`Chunk ${i} failed: ${err.message}`);
            return;
          }
          await new Promise(r => setTimeout(r, 400 * (attempt + 1)));
        }
      }
    };

    // Work-stealing: each worker pulls next item from the queue
    const queue = [...pending];
    await Promise.all(
      Array.from({ length: Math.min(PARALLEL_CHUNKS, queue.length || 1) }, async () => {
        while (queue.length > 0 && !firstError && !signal.aborted) {
          const idx = queue.shift();
          if (idx === undefined) break;
          await uploadOne(idx);
        }
      })
    );

    if (signal.aborted) throw new Error('Upload cancelled');
    if (firstError) throw firstError;

    // 4. Finalize
    setProgressLabel('Processing on server…');
    setProgress(95);
    const finalRes = await apiClient.finalizeZipUpload({
      upload_id: uploadId,
      dataset_id: datasetId,
      task,
      total_chunks: totalChunks,
    });
    if (finalRes.status === 'error') throw new Error(finalRes.message || 'Finalize failed');

    uploadIdRef.current = null;
    setProgress(100);
    return finalRes;
  };

  // ── Main upload handler ───────────────────────────────────────────────────

  const handleUpload = async () => {
    setIsLoading(true);
    setUploadError(null);
    setProgress(0);
    setProgressLabel('');

    try {
      if (uploadMode === 'zip') {
        const zipFile = files.find(f => f.name.toLowerCase().endsWith('.zip'));
        if (!zipFile) { setUploadError('No ZIP file selected.'); return; }
        await uploadZipChunked(zipFile);
      } else {
        const validExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.wav'];
        const validFiles = files.filter(f => {
          const ext = `.${f.name.split('.').pop()?.toLowerCase()}`;
          return validExtensions.includes(ext);
        });
        if (validFiles.length === 0) {
          setUploadError('No valid files found (jpg, png, bmp, wav).');
          return;
        }
        for (let i = 0; i < validFiles.length; i++) {
          setProgressLabel(`File ${i + 1} / ${validFiles.length}`);
          const response = await apiClient.uploadSample(datasetId, label.trim(), task, validFiles[i]);

          // Safety check for backend implementations returning raw response status or bundled data wrapper
          const responseStatus = response?.data?.status || response?.status;
          const responseMessage = response?.data?.message || response?.statusText;

          if (responseStatus === 'error') {
            setUploadError(responseMessage || 'Upload failed');
            return;
          }
          setProgress(Math.round(((i + 1) / validFiles.length) * 100));
        }
      }

      setLabel('');
      setFiles([]);
      setTooManyWarning(false);
      onSampleAdded?.();
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      if (err.message === 'Upload cancelled') {
        setUploadError('Upload was cancelled.');
      } else {
        setUploadError(err.message || 'Upload failed.');
      }
      if (uploadIdRef.current) {
        apiClient.abortZipUpload(uploadIdRef.current).catch(() => { });
        uploadIdRef.current = null;
      }
    } finally {
      setIsLoading(false);
      setProgress(0);
      setProgressLabel('');
      abortRef.current = null;
    }
  };

  const handleCancel = () => abortRef.current?.abort();

  // ── File selection ────────────────────────────────────────────────────────

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const incoming = Array.from(e.target.files || []);
    setTooManyWarning(false);
    setUploadError(null);
    if (uploadMode === 'single' && incoming.length > TOO_MANY_FILES_THRESHOLD) {
      setTooManyWarning(true);
      return;
    }
    if (uploadMode === 'single') {
      setFiles(prev => {
        const names = new Set(prev.map(f => f.name));
        return [...prev, ...incoming.filter(f => !names.has(f.name))];
      });
    } else {
      setFiles(incoming);
    }
  };

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); };
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    setTooManyWarning(false);
    setUploadError(null);
    const dropped = Array.from(e.dataTransfer.files);
    const isZip = dropped.some(f => f.name.toLowerCase().endsWith('.zip'));
    if (isZip) {
      setUploadMode('zip');
      setFiles(dropped.filter(f => f.name.toLowerCase().endsWith('.zip')));
      return;
    }
    if (uploadMode === 'single' && dropped.length > TOO_MANY_FILES_THRESHOLD) {
      setTooManyWarning(true);
      return;
    }
    setFiles(dropped);
  };

  const removeFile = (index: number) => setFiles(prev => prev.filter((_, i) => i !== index));

  const canUpload = !isLoading && files.length > 0 && (uploadMode !== 'single' || label.trim().length > 0);
  const dropZoneBorder = isDragging ? 'border-purple-400 bg-purple-900/20' : 'border-purple-500/50 bg-slate-800/50';

  return (
    <div className="space-y-4">
      {/* Mode tabs */}
      <div className="flex gap-2">
        <button onClick={() => switchMode('single')} type="button"
          className={`flex-1 py-1.5 text-sm rounded font-medium transition-colors flex items-center justify-center gap-1 ${uploadMode === 'single' ? 'bg-purple-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}>
          <FileIcon size={14} /> Single File
        </button>
        <button onClick={() => switchMode('zip')} type="button"
          className={`flex-1 py-1.5 text-sm rounded font-medium transition-colors flex items-center justify-center gap-1 ${uploadMode === 'zip' ? 'bg-purple-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}>
          <FileArchive size={14} /> ZIP Archive
        </button>
      </div>

      {/* Label (single mode) */}
      {uploadMode === 'single' && (
        <div>
          <label className="block text-sm text-gray-300 mb-1">Label</label>
          <input type="text" value={label} onChange={e => setLabel(e.target.value)}
            className="w-full px-3 py-2 bg-slate-700 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500" placeholder="e.g. cat, noise…" />
        </div>
      )}

      {/* ZIP hint */}
      {uploadMode === 'zip' && (
        <div className="text-sm text-gray-400 bg-slate-800/80 p-3 rounded border border-slate-700 space-y-1">
          <p>Upload a <strong>.zip</strong> — subfolder names become labels.</p>
          <p className="text-xs text-gray-500">Large archives (5–10 GB) use resumable chunked upload. If interrupted, just retry.</p>
        </div>
      )}

      {/* Too-many-files warning */}
      {tooManyWarning && (
        <div className="flex items-start gap-2 p-3 bg-amber-900/20 border border-amber-500/40 rounded-lg text-sm text-amber-300 animate-fadeIn">
          <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Too many files!</p>
            <p className="text-xs mt-0.5">For more than {TOO_MANY_FILES_THRESHOLD} files, zip them into a single archive organised by class folder.</p>
            <button className="mt-2 text-xs underline text-amber-400 hover:text-amber-200 block text-left"
              onClick={() => { setTooManyWarning(false); switchMode('zip'); }}>
              Switch to ZIP mode →
            </button>
          </div>
        </div>
      )}

      {/* Drop zone */}
      <div className={`border-2 border-dashed rounded-lg p-6 text-center transition-all ${dropZoneBorder}`}
        onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
        <input type="file" ref={fileInputRef} onChange={handleFileInputChange} className="hidden"
          accept={uploadMode === 'zip' ? '.zip' : undefined} multiple={uploadMode === 'single'} />
        <Upload className="w-6 h-6 text-gray-400 mx-auto mb-2" />
        <button onClick={() => fileInputRef.current?.click()} type="button"
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 transition-colors text-white text-sm rounded font-medium">
          {uploadMode === 'zip' ? 'Select ZIP File' : 'Select File(s)'}
        </button>
        <p className="text-xs text-gray-500 mt-2">
          {isDragging ? 'Release to load…' : uploadMode === 'zip' ? 'or drag & drop a .zip file here' : 'or drag & drop here (zip auto-detected)'}
        </p>
        {files.length > 0 && (
          <ul className="mt-3 max-h-32 overflow-y-auto text-left space-y-1 pr-1 custom-scrollbar">
            {files.map((f, i) => (
              <li key={i} className="flex items-center justify-between text-xs text-gray-300 bg-slate-700/60 px-2 py-1 rounded">
                <span className="truncate max-w-[85%]">{f.name}</span>
                <button onClick={() => removeFile(i)} type="button" className="text-gray-400 hover:text-red-400 ml-2 flex-shrink-0 transition-colors"><X size={12} /></button>
              </li>
            ))}
          </ul>
        )}
        {files.length > 0 && (
          <p className="text-xs text-gray-400 mt-2">
            {files.length} {files.length === 1 ? 'file' : 'files'} ready
            {uploadMode === 'zip' && files[0] ? ` · ${(files[0].size / 1024 / 1024).toFixed(1)} MB` : ''}
          </p>
        )}
      </div>

      {uploadError && (
        <div className="text-sm text-red-400 bg-red-900/20 border border-red-800 px-3 py-2 rounded">{uploadError}</div>
      )}

      <div className="flex gap-2">
        <button onClick={handleUpload} disabled={!canUpload} type="button"
          className="flex-1 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium disabled:opacity-50 transition-colors">
          {isLoading ? (uploadMode === 'zip' ? 'Uploading ZIP…' : 'Uploading…') : 'Upload Data'}
        </button>
        {isLoading && (
          <button onClick={handleCancel} type="button"
            className="px-4 py-2 bg-red-600/20 hover:bg-red-600/40 border border-red-500/50 text-red-400 text-sm font-semibold rounded-lg transition-all">
            Cancel
          </button>
        )}
      </div>

      {isLoading && (
        <div className="space-y-1 pt-1">
          {progressLabel && <p className="text-xs text-gray-400">{progressLabel}</p>}
          <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
            <div className="bg-purple-500 h-2 rounded-full transition-all duration-150 ease-out" style={{ width: `${progress}%` }} />
          </div>
          <p className="text-xs text-gray-500 text-right">{progress}%</p>
        </div>
      )}
    </div>
  );
}