"""
app/routers/inference.py

REST endpoints for running real model inference.

POST /api/inference/run
    Accepts multipart/form-data:
        file        : UploadFile  — image or audio bytes
        training_id : str         — which trained model to use
        optimization_id : str?    — if set, use the .tflite instead of .keras
        top_k       : int = 5

GET /api/inference/history?limit=100
    Returns the last N inference log entries.

DELETE /api/inference/history
    Clears the inference log (useful for testing).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.shared_state import trainer
from app.services.optimizer import get_output_path, OPTIMIZATION_DIR
from app.services.inference_engine import (
    run_inference,
    get_inference_history,
    INFERENCE_LOG_PATH,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_path(training_id: str, optimization_id: Optional[str]) -> tuple[str, str]:
    """
    Return (absolute_model_path, model_kind) where model_kind is 'keras' or 'tflite'.
    Raises HTTPException if nothing is found.
    """
    if optimization_id:
        tflite_path = get_output_path(optimization_id)
        if tflite_path and tflite_path.exists():
            return str(tflite_path), "tflite"
        raise HTTPException(
            status_code=404,
            detail=f"Optimized model for optimization_id={optimization_id!r} not found. "
                   "Make sure the optimization has completed successfully.",
        )

    # Keras model. Trainer.train() actually saves a FLAT file at
    # `<storage_dir>/<training_id>.keras` (not `<storage_dir>/<training_id>/model.keras`).
    # Look it up via the trainer's own registry first (authoritative), then
    # fall back to the flat-file convention directly.
    for model in trainer.trained_models.values():
        if model.get("training_id") == training_id and model.get("path"):
            p = Path(model["path"])
            if p.exists():
                return str(p), "keras"

    for ext in (f"{training_id}.keras", f"{training_id}.h5"):
        p = Path(trainer.storage_dir) / ext
        if p.exists():
            return str(p), "keras"

    raise HTTPException(
        status_code=404,
        detail=f"No trained model found for training_id={training_id!r}. "
               "Check that training has completed.",
    )

def _resolve_labels_and_task(training_id: str) -> tuple[list[str], str]:
    """
    Look up the sorted label list and task string from the trainer's session metadata.
    Falls back to empty labels if the session cannot be found (inference still works,
    classes will be labelled class_0, class_1, …).
    """
    try:
        models = trainer.get_trained_models()
        for m in models:
            if m.get("training_id") == training_id or m.get("id") == training_id:
                # IMPORTANT: do NOT alphabetically sort these. Class index i
                # was assigned during training using this exact list order
                # (see Trainer._build_image_pipeline / _build_audio_pipeline,
                # which enumerate `class_names` as returned by
                # data_manager.get_dataset_labels()). Sorting here would
                # silently scramble which label corresponds to which output
                # neuron whenever the original order wasn't alphabetical.
                labels = m.get("labels") or []
                task   = m.get("task", "IMAGE_CLASSIFICATION")
                return labels, task
    except Exception as exc:
        logger.warning(f"Could not fetch model metadata for {training_id}: {exc}")

    return [], "IMAGE_CLASSIFICATION"

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/run")
async def run_model_inference(
    training_id: str = Form(...),
    optimization_id: Optional[str] = Form(None),
    top_k: int = Form(5),
    file: Optional[UploadFile] = File(None),
    input_url: Optional[str] = Form(None),
):
    """
    Run real inference.

    Supply the input as either:
      • `file`      — uploaded image / audio file
      • `input_url` — publicly accessible URL (fetched server-side)

    Returns the top-K predictions plus timing info.
    """
    # 1. Resolve the model file path
    model_path, _ = _resolve_model_path(training_id, optimization_id)

    # 2. Fetch raw bytes
    if file is not None:
        raw_bytes = await file.read()
    elif input_url:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(input_url, follow_redirects=True)
                resp.raise_for_status()
                raw_bytes = resp.content
                # Guard against absurdly large downloads
                if len(raw_bytes) > 50 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Input file exceeds 50 MB limit.")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {exc}") from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Network error fetching URL: {exc}") from exc
    else:
        raise HTTPException(status_code=422, detail="Provide either 'file' or 'input_url'.")

    # 3. Resolve labels and task
    labels, task = _resolve_labels_and_task(training_id)

    # 4. Run inference (CPU-bound but fast; run synchronously inside async route)
    try:
        result = run_inference(
            model_path=model_path,
            raw_input=raw_bytes,
            task=task,
            labels=labels,
            top_k=min(top_k, 10),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Inference error for training_id={training_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {"status": "success", "result": result}

@router.get("/history")
async def inference_history(limit: int = 100):
    """Return the last `limit` inference log entries, newest first."""
    try:
        entries = get_inference_history(limit=limit)
        return {"status": "success", "entries": entries, "total": len(entries)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.delete("/history")
async def clear_inference_history():
    """Wipe the inference log (useful during development / testing)."""
    try:
        if INFERENCE_LOG_PATH.exists():
            INFERENCE_LOG_PATH.unlink()
        return {"status": "success", "message": "Inference history cleared."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
