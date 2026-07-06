import { useState, useEffect, useCallback } from 'react';
import { Cpu, Download, RefreshCw, AlertTriangle, CheckCircle2, Camera, Monitor, Code2, Copy, Check } from 'lucide-react';
import { useAppContext } from '../context/AppContext';
import { useAPI } from '../hooks/useAPI';
import { TargetBoard } from '../types';

const API_BASE = 'http://localhost:8000/api';

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
  const { state } = useAppContext();
  const { apiClient } = useAPI();
  const optimizationId = state.currentOptimization?.id;

  const [cameraPins, setCameraPins] = useState({ ...DEFAULT_CAMERA_PINS });
  const [displayEnabled, setDisplayEnabled] = useState(false);
  const [displayPins, setDisplayPins] = useState({ ...DEFAULT_DISPLAY_PINS });

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

  // Live "ready to flash" preview - refetched whenever board/pins/optimization change
  useEffect(() => {
    if (!optimizationId) { setPreview(null); return; }
    setPreviewLoading(true);
    const body: any = { board, display_config: buildDisplayConfig() };
    if (board === 'ESP32_CAM') body.camera_pins = buildCameraPins();

    const timeout = setTimeout(() => {
      fetch(`${API_BASE}/optimization/export-preview/${optimizationId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
        .then((r) => r.json())
        .then((j) => setPreview(j.status === 'success' ? j.sketch : null))
        .catch(() => setPreview(null))
        .finally(() => setPreviewLoading(false));
    }, 400); // debounce pin edits

    return () => clearTimeout(timeout);
  }, [optimizationId, board, cameraPins, displayEnabled, displayPins, buildDisplayConfig, buildCameraPins]);

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
      const body: any = { board, display_config: buildDisplayConfig() };
      if (board === 'ESP32_CAM') body.camera_pins = buildCameraPins();

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

  if (!optimizationId) {
    return (
      <div className="p-8 text-center text-gray-400">
        <Cpu className="w-10 h-10 mx-auto mb-3 opacity-40" />
        <p>Complete an optimization in the Optimization tab first, then come back here to configure your board and export a ready-to-flash sketch.</p>
      </div>
    );
  }

  return (
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

        {board === 'ESP32_CAM' && (
          <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700 space-y-3">
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <Camera className="w-4 h-4 text-cyan-400" /> Camera Pins
            </h4>
            <p className="text-xs text-gray-500">
              Defaults match the common AI-Thinker ESP32-CAM module. Adjust if your board wires the camera differently.
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
          </div>
        )}

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
                Shows the live prediction + confidence on a small SPI TFT. Requires the Adafruit GFX + ST7735 libraries.
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
  );
}
