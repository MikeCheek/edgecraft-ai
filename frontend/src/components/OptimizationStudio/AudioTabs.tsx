// ---------------------------------------------------------------------------
// AudioTabs.tsx
// Three audio input modes: upload a file, load from URL, record via microphone.
// Each fires onData(blob) when audio is ready, mirroring ImageTabs.tsx conventions.
// ---------------------------------------------------------------------------

import React, { useState, useRef, useCallback, useEffect } from "react";
import { Upload, Mic, Square, Play } from "lucide-react";

// ---------------------------------------------------------------------------
// Upload tab
// ---------------------------------------------------------------------------
export const AudioUploadTab: React.FC<{ onData: (blob: Blob) => void }> = ({ onData }) => {
  const [filename, setFilename] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    if (!file.type.startsWith("audio/") && !file.name.endsWith(".wav")) return;
    setFilename(file.name);
    onData(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="space-y-3">
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-colors ${dragging
            ? "border-purple-400 bg-purple-500/10"
            : "border-white/20 hover:border-purple-400/60 hover:bg-white/5"
          }`}
      >
        <Upload size={28} className="text-slate-400 mb-2" />
        <p className="text-sm text-slate-300 font-medium">Drop audio or click to browse</p>
        <p className="text-xs text-slate-500 mt-1">Supports WAV, MP3, OGG, FLAC, M4A</p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />

      {filename && (
        <div className="flex items-center gap-2 rounded-lg bg-purple-500/10 border border-purple-500/30 px-3 py-2">
          <Mic size={14} className="text-purple-400 shrink-0" />
          <span className="text-sm text-slate-300 truncate">{filename}</span>
        </div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// URL tab
// ---------------------------------------------------------------------------
export const AudioUrlTab: React.FC<{ onData: (url: string) => void }> = ({ onData }) => {
  const [url, setUrl] = useState("");

  const handleLoad = () => {
    const trimmed = url.trim();
    if (!trimmed) return;
    onData(trimmed);
  };

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com/sound.wav"
          className="flex-1 bg-white/5 border border-white/15 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500"
          onKeyDown={(e) => e.key === "Enter" && handleLoad()}
        />
        <button
          onClick={handleLoad}
          className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white text-sm rounded-lg font-medium transition-colors"
        >
          Load
        </button>
      </div>
      <p className="text-xs text-slate-500">
        The URL is fetched server-side. Must be publicly accessible.
      </p>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Microphone tab
// ---------------------------------------------------------------------------
export const AudioMicTab: React.FC<{ onData: (blob: Blob) => void }> = ({ onData }) => {
  const [recording, setRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [blobReady, setBlobReady] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (timerRef.current) clearInterval(timerRef.current);
    setRecording(false);
  }, []);

  // Auto-stop after 10 s to avoid huge uploads
  useEffect(() => {
    if (recording && duration >= 10) stopRecording();
  }, [duration, recording, stopRecording]);

  // Cleanup on unmount
  useEffect(() => () => stopRecording(), [stopRecording]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      setBlobReady(false);
      setDuration(0);

      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setBlobReady(true);
        onData(blob);
      };

      mr.start(250); // collect in 250 ms chunks
      setRecording(true);

      timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000);
    } catch {
      alert("Could not access microphone. Please check permissions.");
    }
  };

  return (
    <div className="space-y-3">
      {/* Waveform / status area */}
      <div className="relative rounded-xl bg-black/40 border border-white/10 p-6 flex flex-col items-center justify-center min-h-[80px]">
        {recording ? (
          <>
            <div className="flex items-center gap-1 mb-2">
              {Array.from({ length: 12 }).map((_, i) => (
                <div
                  key={i}
                  className="w-1 rounded-full bg-purple-400 animate-pulse"
                  style={{
                    height: `${8 + Math.random() * 20}px`,
                    animationDelay: `${i * 60}ms`,
                  }}
                />
              ))}
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              <span className="text-sm font-semibold text-red-400">
                Recording — {duration}s / 10s
              </span>
            </div>
          </>
        ) : blobReady ? (
          <p className="text-sm text-green-400 font-medium">
            ✓ Audio captured — running inference…
          </p>
        ) : (
          <p className="text-sm text-slate-500">Press Record to begin</p>
        )}
      </div>

      <div className="flex gap-2">
        {!recording ? (
          <button
            onClick={startRecording}
            className="flex-1 flex items-center justify-center gap-2 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            <Mic size={15} /> Record
          </button>
        ) : (
          <button
            onClick={stopRecording}
            className="flex-1 flex items-center justify-center gap-2 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            <Square size={15} /> Stop & Predict
          </button>
        )}
      </div>

      <p className="text-xs text-slate-500">Max 10 seconds. Audio is sent to your local backend.</p>
    </div>
  );
};