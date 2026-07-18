import React, { useRef, useState, useEffect, useCallback } from 'react';
import { X, Crop as CropIcon, Check, RefreshCw, Maximize2, BoxSelect, Pencil, Trash2 } from 'lucide-react';
import { BoundingBox } from '../../types';
import AnnotationOverlay from './AnnotationOverlay';
import { useAPI } from '../../hooks/useAPI';

interface CropBox { x: number; y: number; width: number; height: number; }

interface ImageEditorModalProps {
  imageUrl: string;
  sampleId?: string;
  sampleLabel?: string;
  annotations?: BoundingBox[];
  onClose: () => void;
  onSaveCrop: (blob: Blob) => Promise<void>;
  onAnnotationsSaved?: () => void;
}

const HANDLE_SIZE = 12;
const CLASS_COLORS = ['#22c55e','#3b82f6','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#84cc16','#f97316','#6366f1'];

function ImageEditorModal({ imageUrl, sampleId, sampleLabel, annotations, onClose, onSaveCrop, onAnnotationsSaved }: ImageEditorModalProps) {
  const { apiClient } = useAPI();
  const [mode, setMode] = useState<'view' | 'crop' | 'annotate'>('view');
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 });
  const [box, setBox] = useState<CropBox>({ x: 0, y: 0, width: 0, height: 0 });
  const [drag, setDrag] = useState<{ mode: 'move' | 'resize'; handle?: string; startX: number; startY: number; origBox: CropBox } | null>(null);
  const [saving, setSaving] = useState(false);

  // Annotate mode state
  const [editAnnotations, setEditAnnotations] = useState<BoundingBox[]>(annotations || []);
  const [activeClass, setActiveClass] = useState('');
  const [drawingBox, setDrawingBox] = useState<CropBox | null>(null);
  const [drawDrag, setDrawDrag] = useState<{ startX: number; startY: number } | null>(null);
  const [savingAnnotations, setSavingAnnotations] = useState(false);

  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    setEditAnnotations(annotations || []);
  }, [annotations]);

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

  // --- Crop mode drag handlers ---
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

  // --- Annotate mode draw handlers ---
  const onAnnotateMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const startX = e.clientX - rect.left;
    const startY = e.clientY - rect.top;
    setDrawingBox({ x: startX, y: startY, width: 0, height: 0 });
    setDrawDrag({ startX, startY });
  };

  useEffect(() => {
    if (!drawDrag || !drawingBox) return;
    const onMove = (e: MouseEvent) => {
      const container = containerRef.current?.querySelector('.annotate-target') as HTMLElement;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const curX = Math.max(0, Math.min(e.clientX - rect.left, displaySize.w));
      const curY = Math.max(0, Math.min(e.clientY - rect.top, displaySize.h));
      const x = Math.min(drawDrag.startX, curX);
      const y = Math.min(drawDrag.startY, curY);
      const width = Math.abs(curX - drawDrag.startX);
      const height = Math.abs(curY - drawDrag.startY);
      setDrawingBox({ x, y, width, height });
    };
    const onUp = () => {
      if (drawingBox && drawingBox.width > 5 && drawingBox.height > 5) {
        const scaleX = naturalSize.w / displaySize.w;
        const scaleY = naturalSize.h / displaySize.h;
        const newBbox: BoundingBox = {
          class_name: activeClass || `class_${editAnnotations.length + 1}`,
          cx: ((drawingBox.x + drawingBox.width / 2) * scaleX) / naturalSize.w,
          cy: ((drawingBox.y + drawingBox.height / 2) * scaleY) / naturalSize.h,
          w: (drawingBox.width * scaleX) / naturalSize.w,
          h: (drawingBox.height * scaleY) / naturalSize.h,
        };
        setEditAnnotations(prev => [...prev, newBbox]);
      }
      setDrawingBox(null);
      setDrawDrag(null);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [drawDrag, drawingBox, displaySize, naturalSize, activeClass, editAnnotations.length]);

  const handleSaveAnnotations = async () => {
    if (!sampleId) return;
    setSavingAnnotations(true);
    try {
      await apiClient.updateSampleAnnotations(sampleId, editAnnotations);
      onAnnotationsSaved?.();
      setMode('view');
    } catch {
      // error handled by API client
    } finally {
      setSavingAnnotations(false);
    }
  };

  const handleDeleteAnnotation = (index: number) => {
    setEditAnnotations(prev => prev.filter((_, i) => i !== index));
  };

  // Discover unique class names from existing annotations
  const knownClasses = [...new Set(editAnnotations.map(a => a.class_name))];
  if (activeClass && !knownClasses.includes(activeClass)) knownClasses.push(activeClass);

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
          <div className="flex items-center gap-3 min-w-0">
            <span className="text-sm font-semibold text-white truncate">{sampleLabel}</span>
            {mode === 'annotate' && (
              <span className="flex items-center gap-1 px-2 py-0.5 bg-indigo-600/80 text-white text-[10px] font-bold rounded-full shrink-0">
                <Pencil className="w-3 h-3" /> Annotate ({editAnnotations.length} boxes)
              </span>
            )}
            {mode !== 'annotate' && annotations && annotations.length > 0 && (
              <span className="flex items-center gap-1 px-2 py-0.5 bg-emerald-600/80 text-white text-[10px] font-bold rounded-full shrink-0">
                <BoxSelect className="w-3 h-3" /> {annotations.length} bbox
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {mode === 'view' && (
              <>
                {sampleId && (
                  <button onClick={() => { setMode('annotate'); setEditAnnotations(annotations || []); }}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition">
                    <Pencil className="w-3.5 h-3.5" /> Annotate
                  </button>
                )}
                <button onClick={() => setMode('crop')} disabled={!blobUrl}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white transition">
                  <CropIcon className="w-3.5 h-3.5" /> Crop
                </button>
              </>
            )}
            {mode === 'annotate' && (
              <>
                <button onClick={() => setMode('view')}
                  className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-700 hover:bg-slate-600 text-gray-300 transition">
                  Cancel
                </button>
                <button onClick={handleSaveAnnotations} disabled={savingAnnotations}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-green-700 hover:bg-green-600 disabled:opacity-50 text-white transition">
                  {savingAnnotations ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
                  Save Annotations
                </button>
              </>
            )}
            {mode === 'crop' && (
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

        {/* Annotate mode toolbar */}
        {mode === 'annotate' && (
          <div className="px-5 py-2 border-b border-slate-700 bg-slate-800/30 flex items-center gap-3 flex-wrap">
            <span className="text-[11px] text-gray-400 font-medium">Class:</span>
            {knownClasses.map(cls => {
              const colorIdx = knownClasses.indexOf(cls) % CLASS_COLORS.length;
              return (
                <button key={cls} onClick={() => setActiveClass(cls)}
                  className={`text-[11px] px-2 py-1 rounded border transition font-medium ${activeClass === cls
                    ? 'bg-white/10 border-white/30 text-white'
                    : 'bg-slate-900 border-slate-700 text-gray-400 hover:text-white'}`}
                  style={activeClass === cls ? { borderColor: CLASS_COLORS[colorIdx] } : {}}>
                  <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: CLASS_COLORS[colorIdx] }} />
                  {cls}
                </button>
              );
            })}
            <input type="text" value={activeClass} onChange={e => setActiveClass(e.target.value)}
              placeholder="New class..."
              className="text-[11px] px-2 py-1 bg-slate-900 border border-slate-700 rounded text-white placeholder-gray-500 focus:border-indigo-500 outline-none w-28" />
            <span className="text-[10px] text-gray-500 ml-auto">Click & drag on the image to draw a bounding box</span>
          </div>
        )}

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
              {mode === 'view' && annotations && annotations.length > 0 && naturalSize.w > 0 && (
                <AnnotationOverlay
                  annotations={annotations}
                  imageWidth={naturalSize.w}
                  imageHeight={naturalSize.h}
                />
              )}
              {mode === 'annotate' && naturalSize.w > 0 && (
                <>
                  {/* Existing annotations */}
                  {editAnnotations.length > 0 && (
                    <AnnotationOverlay
                      annotations={editAnnotations}
                      imageWidth={naturalSize.w}
                      imageHeight={naturalSize.h}
                    />
                  )}
                  {/* Delete buttons for each annotation */}
                  {editAnnotations.map((ann, idx) => {
                    const imgEl = imgRef.current;
                    if (!imgEl) return null;
                    const rect = imgEl.getBoundingClientRect();
                    const imgLeft = imgEl.offsetLeft;
                    const imgTop = imgEl.offsetTop;
                    const scaleX = rect.width / naturalSize.w;
                    const scaleY = rect.height / naturalSize.h;
                    const cx = ann.cx * naturalSize.w * scaleX + imgLeft;
                    const cy = ann.cy * naturalSize.h * scaleY + imgTop;
                    return (
                      <button key={idx}
                        onClick={(e) => { e.stopPropagation(); handleDeleteAnnotation(idx); }}
                        className="absolute z-20 p-0.5 bg-red-600 hover:bg-red-500 rounded text-white shadow-md"
                        style={{ left: cx - 8, top: cy - 8 }}>
                        <Trash2 className="w-3 h-3" />
                      </button>
                    );
                  })}
                  {/* Draw target overlay */}
                  <div
                    className="annotate-target absolute inset-0 z-10"
                    style={{ cursor: 'crosshair' }}
                    onMouseDown={onAnnotateMouseDown}
                  />
                  {/* In-progress drawing box */}
                  {drawingBox && drawingBox.width > 0 && drawingBox.height > 0 && (
                    <div className="absolute border-2 border-dashed border-cyan-400 bg-cyan-400/10 pointer-events-none z-20"
                      style={{ left: drawingBox.x, top: drawingBox.y, width: drawingBox.width, height: drawingBox.height }} />
                  )}
                </>
              )}
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