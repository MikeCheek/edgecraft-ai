// ---------------------------------------------------------------------------

import { Upload, Camera, Square, Play, RefreshCw } from "lucide-react";
import { useState, useRef, useCallback, useEffect } from "react";

// Image Input Modes// ---------------------------------------------------------------------------
const ImageUploadTab: React.FC<{ onData: (blob: Blob) => void }> = ({ onData,
}) => {
  const [preview, setPreview] = useState<string | null>(null); const [dragging, setDragging] = useState(false); const inputRef = useRef<HTMLInputElement>(null);
  const handleFile = (file: File) => {
    if (!file.type.startsWith("image/")) return; setPreview(URL.createObjectURL(file));
    onData(file);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false); const file = e.dataTransfer.files[0]; if (file) handleFile(file);
  };
  return (<div className="space-y-3">
    <div onClick={() => inputRef.current?.click()} onDragOver={(e) => {
      e.preventDefault(); setDragging(true);
    }}
      onDragLeave={() => setDragging(false)} onDrop={handleDrop} className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-colors ${dragging ? "border-indigo-400 bg-indigo-500/10" : "border-white/20 hover:border-indigo-400/60 hover:bg-white/5"
        }`}
    >        <Upload size={28} className="text-slate-400 mb-2" />        <p className="text-sm text-slate-300 font-medium">          Drop image or click to browse
      </p>        <p className="text-xs text-slate-500 mt-1">Supports PNG, JPG, WEBP, GIF</p>
    </div>
    <input
      ref={inputRef}
      type="file" accept="image/*"
      className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
    />

    {preview && (
      <img
        src={preview} alt="Preview" className="w-full max-h-48 object-contain rounded-xl border border-white/10"
      />
    )}
  </div>
  );
};
const ImageUrlTab: React.FC<{ onData: (url: string) => void }> = ({
  onData,
}) => {
  const [url, setUrl] = useState(""); const [preview, setPreview] = useState<string | null>(null); const [imgError, setImgError] = useState(false);
  const handleLoad = () => {
    if (!url.trim()) return; setImgError(false); setPreview(url.trim());
    onData(url.trim());
  };
  return (
    <div className="space-y-3">      <div className="flex gap-2">
      <input type="url"
        value={url} onChange={(e) => setUrl(e.target.value)}
        placeholder="https://example.com/image.jpg" className="flex-1 bg-white/5 border border-white/15 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500" onKeyDown={(e) => e.key === "Enter" && handleLoad()}
      />
      <button
        onClick={handleLoad} className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm rounded-lg font-medium transition-colors"
      >
        Load
      </button>      </div>

      {preview && !imgError && (
        <img
          src={preview} alt="URL Preview" onError={() => setImgError(true)} className="w-full max-h-48 object-contain rounded-xl border border-white/10"
        />
      )}
      {imgError && (<p className="text-sm text-red-400">          Failed to load image from URL.
      </p>
      )}
    </div>
  );
};
const ImageCameraTab: React.FC<{ onData: (blob: Blob) => void }> = ({
  onData,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [streaming, setStreaming] = useState(false); const [liveActive, setLiveActive] = useState(false); const liveIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream; if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setStreaming(true);
    } catch {
      alert("Could not access camera. Please check permissions.");
    }
  };
  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStreaming(false); setLiveActive(false); if (liveIntervalRef.current) clearInterval(liveIntervalRef.current);
  }, []);
  useEffect(() => () => stopCamera(), [stopCamera]);
  const captureBlob = (): Blob | null => {
    const video = videoRef.current; const canvas = canvasRef.current;
    if (!video || !canvas) return null; canvas.width = video.videoWidth;
    canvas.height = video.videoHeight; canvas.getContext("2d")?.drawImage(video, 0, 0); let captured: Blob | null = null; canvas.toBlob((b) => {
      if (b) captured = b;
    }, "image/jpeg");
    return captured;
  };
  const handleCapture = () => {
    const blob = captureBlob();
    if (blob) onData(blob);
  };
  const toggleLive = () => {
    if (liveActive) {
      if (liveIntervalRef.current) clearInterval(liveIntervalRef.current);
      setLiveActive(false);
    } else {
      setLiveActive(true); liveIntervalRef.current = setInterval(() => {
        const blob = captureBlob();
        if (blob) onData(blob);
      }, 2000);
    }
  };
  return (
    <div className="space-y-3">      <div className="relative rounded-xl overflow-hidden bg-black border border-white/10 aspect-video flex items-center justify-center">        {!streaming && (
      <button onClick={startCamera} className="flex flex-col items-center gap-2 text-slate-400 hover:text-white transition-colors"
      >            <Camera size={40} />            <span className="text-sm font-medium">Start Camera</span>
      </button>
    )}
      <video ref={videoRef} className={`w-full h-full object-cover ${!streaming ? "hidden" : ""}`} muted
        playsInline />
      {liveActive && (<div className="absolute top-2 right-2 flex items-center gap-1 bg-red-600/80 rounded-full px-2 py-0.5">            <span className="w-2 h-2 bg-white rounded-full animate-pulse" />            <span className="text-xs text-white font-semibold">LIVE</span>
      </div>
      )}      </div>

      <canvas ref={canvasRef} className="hidden" />
      {streaming && (
        <div className="flex gap-2">
          <button onClick={handleCapture} className="flex-1 flex items-center justify-center gap-2 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-colors"
          >            <Camera size={15} /> Capture & Predict
          </button>
          <button
            onClick={toggleLive} className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-colors ${liveActive ? "bg-red-600 hover:bg-red-500 text-white" : "bg-white/10 hover:bg-white/15 text-slate-300"
              }`}
          >            {liveActive ? <Square size={15} /> : <Play size={15} />}            {liveActive ? "Stop Live" : "Live Predict"}
          </button>
          <button
            onClick={stopCamera} className="px-3 py-2 bg-white/10 hover:bg-white/15 rounded-lg transition-colors"
            title="Stop camera"
          >            <RefreshCw size={15} className="text-slate-400" />
          </button>
        </div>)}
    </div>
  );
};

export { ImageUploadTab, ImageUrlTab, ImageCameraTab };
