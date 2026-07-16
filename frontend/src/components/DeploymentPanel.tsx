import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Cpu, Download, RefreshCw, AlertTriangle, CheckCircle2, Camera, Monitor, Code2, Copy, Check, GitBranch, ChevronDown } from 'lucide-react';
import { useAPI, API_BASE } from '../hooks/useAPI';
import { useToast } from '../context/ToastContext';
import { TargetBoard } from '../types';
import { ModelTree } from './ModelTree';

interface DeploymentPanelProps {
  board: TargetBoard;
}

const DEFAULT_CAMERA_PINS = {
  pwdn: 32, reset: -1, xclk: 0, siod: 26, sioc: 27,
  y9: 35, y8: 34, y7: 39, y6: 36, y5: 21, y4: 19, y3: 18, y2: 5,
  vsync: 25, href: 23, pclk: 22,
};

const DEFAULT_DISPLAY_PINS = {
  cs: 15, dc: 2, rst: 4, sck: 18, mosi: 23, backlight: '',
};

interface BoardEvaluation {
  board: string;
  board_name: string;
  ram_usage_kb: number;
  flash_usage_kb: number;
  ram_percentage: number;
  flash_percentage: number;
  measured_inference_ms_on_host?: number;
  estimated_inference_ms_on_device?: number;
  estimation_note?: string;
  warnings: string[];
  suggestions: string[];
  deployment_feasible: boolean;
}

export function DeploymentPanel({ board }: DeploymentPanelProps) {
  const { apiClient } = useAPI();
  const { toast } = useToast();
  const [searchParams, setSearchParams] = useSearchParams();

  // NOTE: this used to read `state.currentOptimization?.id` from
  // AppContext - but nothing in the app ever dispatched SET_OPTIMIZATION,
  // so that value was permanently undefined and this panel always showed
  // "complete optimization first" even when completed optimizations
  // existed. It now has its own real selector (backed by the same
  // dataset -> model -> optimization tree used in the Models Explorer),
  // and supports deep-linking via ?optimization=<id> (e.g. from clicking
  // a node in that tree).
  const [optimizationId, setOptimizationIdState] = useState<string | null>(searchParams.get('optimization'));
  const [showPicker, setShowPicker] = useState(!optimizationId);
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);

  const setOptimizationId = (id: string | null, label?: string) => {
    setOptimizationIdState(id);
    setSelectedLabel(label ?? null);
    setShowPicker(false);
    setSearchParams(id ? { optimization: id } : {});
  };

  const [cameraPins, setCameraPins] = useState({ ...DEFAULT_CAMERA_PINS });
  // ESP32-CAM has an integrated camera, so it's on by default there.
  // Any other board can still opt in to an externally-wired camera module
  // for live inference - this is what drives the "attach by pins or
  // integrated module" choice.
  const [cameraEnabled, setCameraEnabled] = useState(board === 'ESP32_CAM');
  const isIntegratedCamera = board === 'ESP32_CAM';
  const [displayEnabled, setDisplayEnabled] = useState(false);
  const [displayPins, setDisplayPins] = useState({ ...DEFAULT_DISPLAY_PINS });

  // Keep the camera toggle in sync when the globally-selected board changes
  // (e.g. switching to ESP32-CAM should turn its integrated camera on;
  // switching away shouldn't leave a stale "external camera" config active
  // for a board that may not have one wired up).
  useEffect(() => {
    setCameraEnabled(board === 'ESP32_CAM');
  }, [board]);

  const [evaluation, setEvaluation] = useState<BoardEvaluation | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);

  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const [preview, setPreview] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const buildDisplayConfig = useCallback(() => {
    if (!displayEnabled) return { enabled: false };
    return {
      enabled: true,
      cs: Number(displayPins.cs),
      dc: Number(displayPins.dc),
      rst: Number(displayPins.rst),
      sck: Number(displayPins.sck),
      mosi: Number(displayPins.mosi),
      backlight: displayPins.backlight === '' ? null : Number(displayPins.backlight),
    };
  }, [displayEnabled, displayPins]);

  const buildCameraPins = useCallback(() => {
    const out: Record<string, number> = {};
    Object.entries(cameraPins).forEach(([k, v]) => { out[k] = Number(v); });
    return out;
  }, [cameraPins]);

  const buildCameraConfig = useCallback(() => ({
    enabled: cameraEnabled,
    module_type: isIntegratedCamera ? 'integrated' : 'external',
  }), [cameraEnabled, isIntegratedCamera]);

  // Live "ready to flash" preview - refetched whenever board/pins/optimization change
  useEffect(() => {
    if (!optimizationId) { setPreview(null); return; }
    setPreviewLoading(true);
    const body: any = { board, display_config: buildDisplayConfig(), camera_config: buildCameraConfig() };
    if (cameraEnabled) body.camera_pins = buildCameraPins();

    const timeout = setTimeout(() => {
      fetch(`${API_BASE}/optimization/export-preview/${optimizationId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
        .then((r) => r.json())
        .then((j) => setPreview(j.status === 'success' ? j.sketch : null))
        .catch(() => {
          setPreview(null);
          toast('error', 'Failed to load sketch preview');
        })
        .finally(() => setPreviewLoading(false));
    }, 400); // debounce pin edits

    return () => clearTimeout(timeout);
  }, [optimizationId, board, cameraPins, cameraEnabled, displayEnabled, displayPins, buildDisplayConfig, buildCameraPins, buildCameraConfig]);

  const handleEvaluate = async () => {
    if (!optimizationId) return;
    setIsEvaluating(true);
    setEvalError(null);
    try {
      const resp = await fetch(`${API_BASE}/optimization/evaluate-board`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ optimization_id: optimizationId, board }),
      });
      const json = await resp.json();
      if (json.status === 'success') setEvaluation(json.recommendation);
      else setEvalError(json.message ?? 'Board evaluation failed.');
    } catch (e: any) {
      setEvalError(e.message ?? 'Board evaluation failed.');
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleExport = async () => {
    if (!optimizationId) return;
    setIsExporting(true);
    setExportError(null);
    try {
      const body: any = { board, display_config: buildDisplayConfig(), camera_config: buildCameraConfig() };
      if (cameraEnabled) body.camera_pins = buildCameraPins();

      const resp = await fetch(`${API_BASE}/optimization/export/${optimizationId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err?.detail ?? `HTTP ${resp.status}`);
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `edgecraft_export_${optimizationId.slice(0, 8)}_${board}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setExportError(e.message ?? 'Export failed.');
    } finally {
      setIsExporting(false);
    }
  };

  const handleCopyPreview = () => {
    if (!preview) return;
    navigator.clipboard.writeText(preview);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  if (!optimizationId || showPicker) {
    return (
      <div className="space-y-4">
        {optimizationId && (
          <button
            onClick={() => setShowPicker(false)}
            className="text-xs text-slate-400 hover:text-white flex items-center gap-1"
          >
            ← Back to configuration
          </button>
        )}
        <div className="flex items-center gap-2 mb-1">
          <GitBranch size={16} className="text-emerald-400" />
          <h3 className="text-sm font-semibold text-white">Select an Optimized Model to Deploy</h3>
        </div>
        <p className="text-xs text-gray-500 mb-3">
          Pick any completed optimization below - only optimizations that finished successfully can be exported.
        </p>
        <ModelTree
          selectedOptimizationId={optimizationId}
          onSelectOptimization={(opt, model) => setOptimizationId(opt.id, `${model.name} · ${opt.method}`)}
        />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <GitBranch size={13} className="text-emerald-400" />
          {selectedLabel && <span className="text-slate-300 font-medium">{selectedLabel}</span>}
        </div>
        <button
          onClick={() => setShowPicker(true)}
          className="flex items-center gap-1 text-xs text-violet-400 hover:text-violet-300"
        >
          Change model <ChevronDown size={12} />
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left: configuration */}
      <div className="space-y-5">
        <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700">
          <p className="text-sm text-gray-400">
            Target board: <span className="text-white font-semibold">{board.replace(/_/g, ' ')}</span>
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Change the board from the Global Config menu in the header if needed.
          </p>
        </div>

        <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700 space-y-3">
          {isIntegratedCamera ? (
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <Camera className="w-4 h-4 text-cyan-400" /> Integrated Camera (OV2640)
            </h4>
          ) : (
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={cameraEnabled}
                onChange={(e) => setCameraEnabled(e.target.checked)}
                className="rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500 focus:ring-offset-0"
              />
              <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                <Camera className="w-4 h-4 text-cyan-400" /> Attach an external camera module (via GPIO) for live inference
              </h4>
            </label>
          )}

          {cameraEnabled && (
            <>
              <p className="text-xs text-gray-500">
                {isIntegratedCamera
                  ? "Defaults match the common AI-Thinker ESP32-CAM module. Adjust if your board wires the camera differently."
                  : "Wire an OV2640 (or compatible) camera module to these GPIOs. Defaults follow the common AI-Thinker pinout as a starting point - adjust to match your actual wiring."}
              </p>
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(cameraPins).map(([key, value]) => (
                  <label key={key} className="text-xs text-gray-400">
                    {key.toUpperCase()}
                    <input
                      type="number"
                      value={value}
                      onChange={(e) => setCameraPins((prev) => ({ ...prev, [key]: e.target.value === '' ? '' : Number(e.target.value) }))}
                      className="mt-1 w-full px-2 py-1.5 bg-slate-800 border border-slate-600 rounded text-white text-sm focus:border-cyan-500 focus:outline-none"
                    />
                  </label>
                ))}
              </div>
              <p className="text-[11px] text-slate-500">
                The exported sketch streams live inference from the camera and prints every class's confidence
                to Serial each cycle, plus the best pick. If a display is also enabled below, it shows a live
                preview of the camera feed with the current prediction overlaid.
              </p>
            </>
          )}
        </div>

        <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700 space-y-3">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={displayEnabled}
              onChange={(e) => setDisplayEnabled(e.target.checked)}
              className="rounded border-slate-600 bg-slate-800 text-pink-500 focus:ring-pink-500 focus:ring-offset-0"
            />
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <Monitor className="w-4 h-4 text-pink-400" /> Attach a status display (SPI TFT, ST7735)
            </h4>
          </label>
          {displayEnabled && (
            <>
              <p className="text-xs text-gray-500">
                {cameraEnabled
                  ? "Shows a live preview of the camera feed with the prediction + confidence overlaid. Requires the Adafruit GFX + ST7735 libraries."
                  : "Shows the current prediction + confidence as text on a small SPI TFT. Requires the Adafruit GFX + ST7735 libraries."}
              </p>
              <div className="grid grid-cols-3 gap-2">
                {(['cs', 'dc', 'rst', 'sck', 'mosi'] as const).map((key) => (
                  <label key={key} className="text-xs text-gray-400">
                    {key.toUpperCase()}
                    <input
                      type="number"
                      value={displayPins[key]}
                      onChange={(e) => setDisplayPins((prev) => ({ ...prev, [key]: e.target.value === '' ? '' : Number(e.target.value) }))}
                      className="mt-1 w-full px-2 py-1.5 bg-slate-800 border border-slate-600 rounded text-white text-sm focus:border-pink-500 focus:outline-none"
                    />
                  </label>
                ))}
                <label className="text-xs text-gray-400">
                  BACKLIGHT (optional)
                  <input
                    type="number"
                    value={displayPins.backlight}
                    placeholder="tied to 3.3V"
                    onChange={(e) => setDisplayPins((prev) => ({ ...prev, backlight: e.target.value }))}
                    className="mt-1 w-full px-2 py-1.5 bg-slate-800 border border-slate-600 rounded text-white text-sm focus:border-pink-500 focus:outline-none"
                  />
                </label>
              </div>
            </>
          )}
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleEvaluate}
            disabled={isEvaluating}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-semibold bg-cyan-600 hover:bg-cyan-700 disabled:opacity-50 text-white transition-colors"
          >
            {isEvaluating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Cpu className="w-4 h-4" />}
            Evaluate for Board
          </button>
          <button
            onClick={handleExport}
            disabled={isExporting}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-semibold bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 disabled:opacity-50 text-white transition-colors"
          >
            {isExporting ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            Export Arduino Project
          </button>
        </div>

        {evalError && <p className="text-xs text-red-400">{evalError}</p>}
        {exportError && <p className="text-xs text-red-400">{exportError}</p>}

        {evaluation && (
          <div className="space-y-3 p-4 bg-slate-900/50 rounded-lg border border-slate-700">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-gray-400 block mb-1">RAM (est.)</span>
                <div className="flex justify-between items-end">
                  <span className="text-white font-mono">{evaluation.ram_usage_kb} KB</span>
                  <span className={evaluation.ram_percentage > 80 ? 'text-red-400' : 'text-green-400'}>{evaluation.ram_percentage.toFixed(1)}%</span>
                </div>
              </div>
              <div>
                <span className="text-gray-400 block mb-1">Flash</span>
                <div className="flex justify-between items-end">
                  <span className="text-white font-mono">{evaluation.flash_usage_kb} KB</span>
                  <span className={evaluation.flash_percentage > 80 ? 'text-red-400' : 'text-green-400'}>{evaluation.flash_percentage.toFixed(1)}%</span>
                </div>
              </div>
            </div>
            {evaluation.warnings.length > 0 && (
              <div className="space-y-1">
                {evaluation.warnings.map((w, i) => (
                  <p key={i} className="text-xs text-amber-300 flex items-start gap-1.5"><AlertTriangle className="w-3 h-3 mt-0.5 shrink-0" /> {w}</p>
                ))}
              </div>
            )}
            <p className={`text-xs font-semibold flex items-center gap-1.5 ${evaluation.deployment_feasible ? 'text-emerald-400' : 'text-red-400'}`}>
              <CheckCircle2 className="w-3.5 h-3.5" />
              {evaluation.deployment_feasible ? 'Fits within board constraints' : 'May not fit - consider a lighter optimization'}
            </p>
          </div>
        )}
      </div>

      {/* Right: live sketch preview */}
      <div className="bg-slate-900/70 rounded-xl border border-slate-700 overflow-hidden flex flex-col max-h-[720px]">
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700 bg-slate-900">
          <h4 className="text-sm font-semibold text-white flex items-center gap-2">
            <Code2 className="w-4 h-4 text-emerald-400" /> sketch.ino (live preview)
          </h4>
          <button
            onClick={handleCopyPreview}
            disabled={!preview}
            className="flex items-center gap-1.5 text-xs px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-gray-300 transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
        <div className="flex-1 overflow-auto p-4">
          {previewLoading && !preview ? (
            <p className="text-sm text-gray-500">Generating preview...</p>
          ) : preview ? (
            <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap leading-relaxed">{preview}</pre>
          ) : (
            <p className="text-sm text-gray-500">Preview will appear here once configuration is valid.</p>
          )}
        </div>
      </div>
      </div>
    </div>
  );
}
