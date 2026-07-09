import React, { useRef, useState, useEffect, useCallback } from 'react';
import { X, Crop as CropIcon, Check, RefreshCw, Maximize2 } from 'lucide-react';

interface CropBox { x: number; y: number; width: number; height: number; }

interface ImageEditorModalProps {
  imageUrl: string; // the API URL (with cache-bust query), fetched as a blob internally
  sampleLabel?: string;
  onClose: () => void;
  onSaveCrop: (blob: Blob) => Promise<void>;
}

const HANDLE_SIZE = 12;

function ImageEditorModal({ imageUrl, sampleLabel, onClose, onSaveCrop }: ImageEditorModalProps) {
  const [mode, setMode] = useState<'view' | 'crop'>('view');
  const [blobUrl, setBlobUrl] = useState<string | null>(null); // NEW
  const [loadError, setLoadError] = useState(false); // NEW
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 });
  const [box, setBox] = useState<CropBox>({ x: 0, y: 0, width: 0, height: 0 });
  const [drag, setDrag] = useState<{ mode: 'move' | 'resize'; handle?: string; startX: number; startY: number; origBox: CropBox } | null>(null);
  const [saving, setSaving] = useState(false);

  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // NEW: fetch the image as a blob and load from an object URL instead of
  // the raw API URL. A blob: URL is always same-origin for canvas purposes,
  // so canvas.toBlob() in handleApplyCrop won't throw "Tainted canvases may
  // not be exported" - which <img src={crossOriginUrl}> would trigger even
  // though the browser happily *displays* a cross-origin image fine.
  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;

    setBlobUrl(null);
    setLoadError(false);

    fetch(imageUrl)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.blob();
      })
      .then(blob => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl(objectUrl);
      })
      .catch(() => { if (!cancelled) setLoadError(true); });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [imageUrl]);

  const handleImgLoad = () => {
    const img = imgRef.current;
    if (!img) return;
    setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
    const rect = img.getBoundingClientRect();
    setDisplaySize({ w: rect.width, h: rect.height });
    setBox({ x: rect.width * 0.1, y: rect.height * 0.1, width: rect.width * 0.8, height: rect.height * 0.8 });
  };

  const clamp = useCallback((b: CropBox): CropBox => {
    let { x, y, width, height } = b;
    width = Math.max(20, Math.min(width, displaySize.w));
    height = Math.max(20, Math.min(height, displaySize.h));
    x = Math.max(0, Math.min(x, displaySize.w - width));
    y = Math.max(0, Math.min(y, displaySize.h - height));
    return { x, y, width, height };
  }, [displaySize]);

  const onMouseDownMove = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDrag({ mode: 'move', startX: e.clientX, startY: e.clientY, origBox: box });
  };

  const onMouseDownHandle = (handle: string) => (e: React.MouseEvent) => {
    e.stopPropagation();
    setDrag({ mode: 'resize', handle, startX: e.clientX, startY: e.clientY, origBox: box });
  };

  useEffect(() => {
    if (!drag) return;
    const onMove = (e: MouseEvent) => {
      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      const o = drag.origBox;
      if (drag.mode === 'move') {
        setBox(clamp({ ...o, x: o.x + dx, y: o.y + dy }));
        return;
      }
      let { x, y, width, height } = o;
      const h = drag.handle!;
      if (h.includes('e')) width = o.width + dx;
      if (h.includes('s')) height = o.height + dy;
      if (h.includes('w')) { width = o.width - dx; x = o.x + dx; }
      if (h.includes('n')) { height = o.height - dy; y = o.y + dy; }
      setBox(clamp({ x, y, width, height }));
    };
    const onUp = () => setDrag(null);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [drag, clamp]);

  const handleApplyCrop = async () => {
    const img = imgRef.current;
    if (!img || !naturalSize.w) return;
    const scaleX = naturalSize.w / displaySize.w;
    const scaleY = naturalSize.h / displaySize.h;

    const canvas = document.createElement('canvas');
    canvas.width = Math.round(box.width * scaleX);
    canvas.height = Math.round(box.height * scaleY);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(
      img,
      box.x * scaleX, box.y * scaleY, box.width * scaleX, box.height * scaleY,
      0, 0, canvas.width, canvas.height
    );

    setSaving(true);
    canvas.toBlob(async (blob) => {
      if (blob) {
        try {
          await onSaveCrop(blob);
          onClose();
        } finally {
          setSaving(false);
        }
      } else {
        setSaving(false);
      }
    }, 'image/png');
  };

  const handles = ['nw', 'ne', 'sw', 'se'];
  const handlePos: Record<string, React.CSSProperties> = {
    nw: { left: -HANDLE_SIZE / 2, top: -HANDLE_SIZE / 2, cursor: 'nwse-resize' },
    ne: { right: -HANDLE_SIZE / 2, top: -HANDLE_SIZE / 2, cursor: 'nesw-resize' },
    sw: { left: -HANDLE_SIZE / 2, bottom: -HANDLE_SIZE / 2, cursor: 'nesw-resize' },
    se: { right: -HANDLE_SIZE / 2, bottom: -HANDLE_SIZE / 2, cursor: 'nwse-resize' },
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/90 backdrop-blur-sm p-4">
      <div className="w-full max-w-4xl max-h-[90vh] flex flex-col bg-slate-900 rounded-2xl border border-slate-700 shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700 bg-slate-800/50">
          <span className="text-sm font-semibold text-white truncate">{sampleLabel}</span>
          <div className="flex items-center gap-2">
            {mode === 'view' ? (
              <button onClick={() => setMode('crop')} disabled={!blobUrl}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white transition">
                <CropIcon className="w-3.5 h-3.5" /> Crop
              </button>
            ) : (
              <>
                <button onClick={() => setMode('view')}
                  className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-700 hover:bg-slate-600 text-gray-300 transition">
                  Cancel
                </button>
                <button onClick={handleApplyCrop} disabled={saving}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-green-700 hover:bg-green-600 disabled:opacity-50 text-white transition">
                  {saving ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
                  Save Crop
                </button>
              </>
            )}
            <button onClick={onClose} className="p-1.5 text-gray-400 hover:text-white hover:bg-slate-700 rounded-lg transition">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center p-6 overflow-auto bg-black/40">
          {loadError ? (
            <div className="flex flex-col items-center gap-2 text-gray-400">
              <RefreshCw className="w-6 h-6" />
              <p className="text-sm">Failed to load image</p>
            </div>
          ) : !blobUrl ? (
            <RefreshCw className="w-6 h-6 text-gray-400 animate-spin" />
          ) : (
            <div ref={containerRef} className="relative inline-block select-none">
              <img
                ref={imgRef}
                src={blobUrl}
                onLoad={handleImgLoad}
                alt={sampleLabel}
                className="max-h-[70vh] max-w-full block"
                draggable={false}
              />
              {mode === 'crop' && naturalSize.w > 0 && (
                <>
                  <div className="absolute inset-0 pointer-events-none">
                    <div className="absolute bg-black/60" style={{ left: 0, top: 0, width: '100%', height: box.y }} />
                    <div className="absolute bg-black/60" style={{ left: 0, top: box.y + box.height, width: '100%', bottom: 0, height: `calc(100% - ${box.y + box.height}px)` }} />
                    <div className="absolute bg-black/60" style={{ left: 0, top: box.y, width: box.x, height: box.height }} />
                    <div className="absolute bg-black/60" style={{ left: box.x + box.width, top: box.y, width: `calc(100% - ${box.x + box.width}px)`, height: box.height }} />
                  </div>
                  <div
                    onMouseDown={onMouseDownMove}
                    className="absolute border-2 border-indigo-400 cursor-move"
                    style={{ left: box.x, top: box.y, width: box.width, height: box.height }}
                  >
                    {handles.map(h => (
                      <div key={h}
                        onMouseDown={onMouseDownHandle(h)}
                        className="absolute bg-indigo-400 border border-white rounded-sm"
                        style={{ width: HANDLE_SIZE, height: HANDLE_SIZE, ...handlePos[h] }}
                      />
                    ))}
                    <div className="absolute -top-6 left-0 text-[10px] text-indigo-300 font-mono">
                      {Math.round(box.width * (naturalSize.w / displaySize.w))} × {Math.round(box.height * (naturalSize.h / displaySize.h))}px
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ImageEditorModal;