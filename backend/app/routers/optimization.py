from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io

from app.services.shared_state import trainer
from app.services.mcu_advisor import MCUAdvisor
from app.services.llm_advisor import LLMAdvisor
from app.services.local_llm_advisor import LocalLLMAdvisor
from app.services.optimizer import (
    create_optimization_session,
    optimize,
    get_optimization_status,
    get_optimization_result,
    get_output_path,
    list_optimization_sessions,
    OPTIMIZATION_DIR,
)
from app.services import exporter

router = APIRouter()
mcu_advisor = MCUAdvisor()
llm_advisor = LLMAdvisor()
local_llm = LocalLLMAdvisor()

# ---------------------------------------------------------------------------
# Method-name mapping layer
# ---------------------------------------------------------------------------
# The frontend (and app/models.py's QuantizationMethod) uses uppercase enum
# names like "INT8_QUANTIZATION". The optimizer's internal dispatch uses
# short lowercase keys. This is the single place that translates between
# the two, so neither side has to change its existing naming convention.
FRONTEND_TO_INTERNAL_METHOD = {
    "INT8_QUANTIZATION": "int8",
    "FLOAT16_QUANTIZATION": "float16",
    "DYNAMIC_QUANTIZATION": "dynamic_range",
    "PRUNING": "pruning",
    "WEIGHT_CLUSTERING": "weight_clustering",
    "TRANSFER_LEARNING": "transfer_learning",
}
INTERNAL_TO_FRONTEND_METHOD = {v: k for k, v in FRONTEND_TO_INTERNAL_METHOD.items()}

def _to_internal_method(method: str) -> str:
    # Accept either naming style so older/other callers don't break.
    if method in FRONTEND_TO_INTERNAL_METHOD:
        return FRONTEND_TO_INTERNAL_METHOD[method]
    if method in INTERNAL_TO_FRONTEND_METHOD:
        return method
    raise HTTPException(
        status_code=400,
        detail=f"Unknown optimization method '{method}'. "
               f"Expected one of {list(FRONTEND_TO_INTERNAL_METHOD.keys())}",
    )

# --- Request Schemas ---
class OptimizationRequest(BaseModel):
    training_id: str
    method: str
    sparsity_level: float = 0.5
    representative_dataset_size: int = 100

class BoardEvaluationRequest(BaseModel):
    optimization_id: str
    board: str

class LLMSuggestRequest(BaseModel):
    training_id: str
    provider: str = "openrouter"
    model_name: str = "openrouter/free"

class LLMOptimizeRequest(BaseModel):
    optimization_id: str
    board: str
    use_local_llm: bool = False
    local_model_name: str = "phi3"
# -----------------------
# NOTE: pre-training recommendations (base_model/hyperparameter suggestions
# BEFORE a run starts) live on the training router as POST /api/training/recommend,
# since they're conceptually part of setting up a training run, not tuning
# an already-optimized model.

@router.post("/quantize")
async def quantize_model(
    request: OptimizationRequest, background_tasks: BackgroundTasks
):
    internal_method = _to_internal_method(request.method)
    optimization_id = create_optimization_session(
        training_id=request.training_id,
        method=internal_method,
        sparsity_level=request.sparsity_level,
        frontend_method=request.method,
    )
    background_tasks.add_task(
        optimize, optimization_id, trainer.storage_dir
    )
    return {"status": "success", "optimization_id": optimization_id}

@router.get("/history")
async def get_optimization_history():
    """Return all past optimization sessions sorted newest-first."""
    try:
        sessions = list_optimization_sessions()
        return {"status": "success", "sessions": sessions}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/active")
async def get_active_optimization(training_id: str = None):
    """
    Returns the currently running optimization session, if any -
    optionally filtered to a specific training_id. Used by the frontend to
    reattach to an in-progress job after navigating away and back (or
    reloading the page).
    """
    try:
        sessions = list_optimization_sessions()
        active = [s for s in sessions if s.get("status") == "running"]
        if training_id:
            active = [s for s in active if s.get("training_id") == training_id]
        active.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        return {"status": "success", "session": active[0] if active else None}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# NOTE: path order matches the frontend's apiClient
# (`/optimization/status/${id}`, not `/optimization/${id}/status`).
@router.get("/status/{optimization_id}")
async def optimization_status(optimization_id: str):
    try:
        return {"status": "success", **get_optimization_status(optimization_id)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/result/{optimization_id}")
async def optimization_result(optimization_id: str):
    try:
        return {"status": "success", "result": get_optimization_result(optimization_id)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/downloads/{optimization_id}/model.tflite")
async def download_optimized_model(optimization_id: str):
    """Serve the real .tflite binary produced by the optimizer."""
    output_path = get_output_path(optimization_id)

    if output_path is None:
        raise HTTPException(
            status_code=404,
            detail="Optimized model not found. Optimization may still be running or failed.",
        )
    if not output_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Optimization completed but file missing at {output_path}",
        )

    return FileResponse(
        path=str(output_path),
        media_type="application/octet-stream",
        filename=f"model_{optimization_id[:8]}.tflite",
        headers={"Content-Disposition": f"attachment; filename=model_{optimization_id[:8]}.tflite"},
    )

@router.post("/to-c-array/{optimization_id}")
async def export_as_c_array(optimization_id: str):
    """Real C array generated from the actual optimized model bytes
    (previously this returned a placeholder 0x00,0x01,0x02... array)."""
    try:
        return {"status": "success", "c_array": exporter.generate_c_array_only(optimization_id)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/hardware_presets")
async def get_hardware_presets():
    """Catalog of known camera modules and display modules (id, human
    label, default pins, style) for populating the Deployment tab's
    dropdowns. The frontend picks one, sends its id back as
    `camera_config.module_preset` / `display_config.module_preset` in the
    export/preview request, and can still override individual pins on top."""
    try:
        return {"status": "success", **exporter.list_hardware_presets()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class ExportConfigRequest(BaseModel):
    board: str
    # Legacy manual pin override, applied on top of whatever the chosen
    # camera_config.module_preset resolves to. Kept for backward
    # compatibility with any caller not yet using presets.
    camera_pins: Optional[dict] = None
    # {
    #   "enabled": bool,                     # optional; defaults to True for integrated presets
    #   "module_preset": "AI_THINKER_ESP32_CAM" | "ESP32_S3_CAM_OV3660" | "CUSTOM",
    #   "module_type": "integrated" | "external",   # optional override of the preset's default
    #   "pins_override": {"xclk": 15, ...},         # optional per-pin overrides
    # }
    # See GET /api/optimization/hardware_presets for the full list of preset ids.
    camera_config: Optional[dict] = None
    # {
    #   "enabled": bool,
    #   "module_preset": "NONE" | "ST7735_TEXT_HUD" | "ST7735_PIXEL_HUD",
    #   "cs": .., "dc": .., "rst": .., "sck": .., "mosi": .., "backlight": ..  # optional pin overrides
    # }
    display_config: Optional[dict] = None

@router.post("/export/{optimization_id}")
async def export_project(optimization_id: str, request: ExportConfigRequest):
    """Download a full Arduino/C++ project (model_data.h + sketch.ino +
    README.md) for the given board, wired for the user's actual hardware
    configuration (a named camera module preset or hand-wired pins, and an
    optional attached status display, also from a named preset)."""
    try:
        zip_bytes = exporter.generate_export_package(
            optimization_id, request.board,
            camera_pins=request.camera_pins,
            display_config=request.display_config,
            camera_config=request.camera_config,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = f"edgecraft_export_{optimization_id[:8]}_{request.board}.zip"
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

@router.post("/export-preview/{optimization_id}")
async def preview_export_sketch(optimization_id: str, request: ExportConfigRequest):
    """Return just the generated .ino source (no zip) so the Deployment tab
    can show a live 'ready to flash' preview as pin values are edited."""
    try:
        sketch = exporter.preview_sketch(
            optimization_id, request.board,
            camera_pins=request.camera_pins,
            display_config=request.display_config,
            camera_config=request.camera_config,
        )
        return {"status": "success", "sketch": sketch}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/evaluate-board")
async def evaluate_for_board(request: BoardEvaluationRequest):
    try:
        recommendation = mcu_advisor.evaluate_model(
            optimization_id=request.optimization_id,
            board=request.board,
        )
        return {"status": "success", "recommendation": recommendation}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/boards")
async def get_supported_boards():
    try:
        return {"status": "success", "boards": mcu_advisor.get_supported_boards()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/llm-status")
async def get_llm_status():
    """Check whether a local Ollama LLM is available."""
    try:
        return {"status": "success", "llm": local_llm.get_status()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/llm-suggest")
async def get_llm_suggestions(request: LLMSuggestRequest):
    """Post-training suggestions based on the actual training run's metrics history."""
    try:
        session_context = trainer.get_training_status(request.training_id)

        suggestions = await llm_advisor.generate_suggestions(
            context=session_context,
            provider=request.provider,
            model_name=request.model_name,
        )
        return {"status": "success", "suggestions": suggestions}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/llm-optimize")
async def get_llm_optimization_advice(request: LLMOptimizeRequest):
    """Board-specific deployment advice grounded in the real optimization
    session (size, compression ratio, measured test-set inference time)."""
    try:
        advice = await llm_advisor.get_optimization_advice(
            optimization_id=request.optimization_id,
            board=request.board,
            use_local_llm=request.use_local_llm,
            model_name=request.local_model_name,
        )
        return {"status": "success", "advice": advice}
    except Exception as e:
        return {"status": "error", "message": str(e)}
