"""
inference_engine.py

Real inference engine for EdgeCraft AI.
Supports Keras (.keras / .h5) and TFLite (.tflite) models.
Preprocessing reuses the same logic as the training pipeline (DataProcessor).
Results include per-run timing and are logged to a persistent JSON file.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import tempfile

INFERENCE_LOG_PATH = Path(tempfile.gettempdir()) / "edgecraft_inference_log.json"
OPTIMIZATION_DIR   = Path(tempfile.gettempdir()) / "edgecraft_optimizations"

# ---------------------------------------------------------------------------
# Lazy TF / TFLite imports (avoid import-time crash if TF not installed)
# ---------------------------------------------------------------------------
def _load_tf():
    import tensorflow as tf  # noqa: PLC0415
    return tf

def _load_tflite_interpreter(model_path: str):
    tf = _load_tf()
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# ---------------------------------------------------------------------------
# Preprocessing helpers
# (mirrors DataProcessor — kept inline so this module has no circular import)
# ---------------------------------------------------------------------------

# NOTE: previously this hardcoded (64, 64) / (96, 96) per task regardless of
# what input_shape a given model was actually trained with, causing
# "expected shape=(None, 224, 224, 3), found shape=(1, 64, 64, 3)" (Keras)
# and "Dimension mismatch. Got 64 but expected 224" (TFLite) errors on any
# model trained with a non-default input_shape. _preprocess_image now takes
# the model's real resolved input_shape instead (see run_inference below).

AUDIO_PARAMS: Dict[str, Dict[str, int]] = {
    "KEYWORD_SPOTTING":     {"n_mfcc": 40, "n_fft": 512, "hop_length": 512, "time_frames": 101},
    "AUDIO_CLASSIFICATION": {"n_mfcc": 64, "n_fft": 512, "hop_length": 256, "time_frames": 101},
}

def _preprocess_image(image_bytes: bytes, input_shape: Tuple[int, ...]) -> np.ndarray:
    """Decode + resize + normalise image to [0, 1] float32, matching the
    model's ACTUAL (h, w, channels) - not a hardcoded per-task guess."""
    tf = _load_tf()
    h, w = input_shape[0], input_shape[1]
    channels = input_shape[2] if len(input_shape) >= 3 else 3
    decode_channels = 1 if channels == 1 else 3
    tensor = tf.io.decode_image(image_bytes, channels=decode_channels, expand_animations=False)
    tensor = tf.image.resize(tensor, [h, w])
    tensor = tf.cast(tensor, tf.float32) / 255.0
    return tensor.numpy()  # (H, W, channels)

def _preprocess_audio(audio_bytes: bytes, task: str, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Decode WAV ? MFCC ? pad/truncate to match model's expected input shape."""
    try:
        import librosa  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("librosa is required for audio inference. Install with: pip install librosa") from exc

    import io
    import soundfile as sf  # noqa: PLC0415

    params = AUDIO_PARAMS.get(task, {"n_mfcc": 40, "n_fft": 512, "hop_length": 512, "time_frames": 101})
    time_frames = params["time_frames"]

    # Decode audio
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo ? mono
    audio = audio.astype(np.float32)

    # Resample to 16 kHz (standard for keyword spotting / audio classification)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Compute MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=params["n_mfcc"],
                                 n_fft=params["n_fft"], hop_length=params["hop_length"])

    # Pad or truncate time axis to match model's expected time_frames
    # Use the actual model input shape when available
    expected_time = target_shape[1] if len(target_shape) >= 2 else time_frames
    if mfcc.shape[1] < expected_time:
        pad_width = expected_time - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :expected_time]

    # Add channel dimension: (n_mfcc, time_frames) ? (n_mfcc, time_frames, 1)
    mfcc = mfcc[:, :, np.newaxis].astype(np.float32)
    return mfcc

# ---------------------------------------------------------------------------
# Inference log helpers
# ---------------------------------------------------------------------------

def _load_log() -> List[Dict[str, Any]]:
    if INFERENCE_LOG_PATH.exists():
        try:
            with open(INFERENCE_LOG_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _append_log(entry: Dict[str, Any]) -> None:
    log = _load_log()
    log.append(entry)
    # Keep last 1 000 entries to avoid unbounded growth
    if len(log) > 1000:
        log = log[-1000:]
    try:
        with open(INFERENCE_LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write inference log: {e}")

def get_inference_history(limit: int = 100) -> List[Dict[str, Any]]:
    log = _load_log()
    return list(reversed(log[-limit:]))

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_model_cache: Dict[str, Any] = {}   # cache_key ? interpreter OR keras model

def _get_or_load_tflite(path: str):
    if path not in _model_cache:
        logger.info(f"Loading TFLite model from {path}")
        _model_cache[path] = _load_tflite_interpreter(path)
    return _model_cache[path]

def _get_or_load_keras(path: str):
    if path not in _model_cache:
        tf = _load_tf()
        logger.info(f"Loading Keras model from {path}")
        _model_cache[path] = tf.keras.models.load_model(path)
    return _model_cache[path]

# ---------------------------------------------------------------------------
# Core inference runners
# ---------------------------------------------------------------------------

def _run_tflite(interpreter, input_array: np.ndarray) -> Tuple[List[str], List[float], float]:
    """Run a single inference pass on a TFLite interpreter. Returns (class_names, confidences, ms)."""
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Quantised INT8 input: scale / zero_point
    inp_detail = input_details[0]
    if inp_detail["dtype"] == np.int8:
        scale, zero_point = inp_detail["quantization"]
        if scale == 0:
            scale = 1.0
        inp = (input_array / scale + zero_point).astype(np.int8)
    else:
        inp = input_array.astype(inp_detail["dtype"])

    # Add batch dimension
    inp = np.expand_dims(inp, axis=0)
    interpreter.set_tensor(inp_detail["index"], inp)

    t0 = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out = interpreter.get_tensor(output_details[0]["index"])[0]

    # Dequantise INT8 output
    out_detail = output_details[0]
    if out_detail["dtype"] == np.int8:
        scale, zero_point = out_detail["quantization"]
        if scale == 0:
            scale = 1.0
        out = (out.astype(np.float32) - zero_point) * scale

    out = out.astype(np.float32)
    # Softmax if not already a probability distribution
    if out.max() > 1.0 or out.min() < 0.0:
        exp = np.exp(out - out.max())
        out = exp / exp.sum()

    return out, elapsed_ms

def _run_keras(model, input_array: np.ndarray) -> Tuple[np.ndarray, float]:
    inp = np.expand_dims(input_array, axis=0).astype(np.float32)
    t0 = time.perf_counter()
    preds = model.predict(inp, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return preds.astype(np.float32), elapsed_ms

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    *,
    model_path: str,
    raw_input: bytes,
    task: str,
    labels: List[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run inference on a model file (Keras or TFLite) with raw bytes input.

    Args:
        model_path: Absolute path to .keras, .h5, or .tflite file.
        raw_input:  Raw file bytes (image bytes or audio bytes).
        task:       One of IMAGE_CLASSIFICATION, OBJECT_DETECTION,
                    VISUAL_WAKE_WORDS, KEYWORD_SPOTTING, AUDIO_CLASSIFICATION.
        labels:     Ordered list of class names (same order as training).
        top_k:      How many top predictions to return.

    Returns:
        Dict with keys: top_class, confidence, inference_time_ms, top_k_results,
        model_kind, model_path, task, timestamp.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suffix = path.suffix.lower()
    is_audio = task in ("KEYWORD_SPOTTING", "AUDIO_CLASSIFICATION")

    # ---- Load model and determine input shape --------------------------------
    if suffix == ".tflite":
        interpreter = _get_or_load_tflite(model_path)
        input_shape = tuple(interpreter.get_input_details()[0]["shape"][1:])  # drop batch dim
        model_kind = "tflite"
    else:
        keras_model = _get_or_load_keras(model_path)
        input_shape = tuple(keras_model.input_shape[1:])
        model_kind = "keras"

    # ---- Preprocess ----------------------------------------------------------
    if is_audio:
        processed = _preprocess_audio(raw_input, task, input_shape)
    else:
        processed = _preprocess_image(raw_input, input_shape)

    # ---- Inference -----------------------------------------------------------
    t_total_start = time.perf_counter()
    if model_kind == "tflite":
        probs, inference_ms = _run_tflite(interpreter, processed)
    else:
        probs, inference_ms = _run_keras(keras_model, processed)
    total_ms = (time.perf_counter() - t_total_start) * 1000.0

    # ---- Map to labels -------------------------------------------------------
    num_classes = len(probs)
    if len(labels) != num_classes:
        # Fallback: auto-generate class labels
        labels = [f"class_{i}" for i in range(num_classes)]

    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    top_k_results = [
        {"class_name": labels[i], "confidence": float(p)}
        for i, p in indexed[:top_k]
    ]
    top_class = top_k_results[0]["class_name"] if top_k_results else "unknown"
    top_conf  = top_k_results[0]["confidence"]  if top_k_results else 0.0

    result: Dict[str, Any] = {
        "inference_id":       str(uuid.uuid4()),
        "top_class":          top_class,
        "confidence":         round(top_conf, 6),
        "inference_time_ms":  round(inference_ms, 3),
        "total_time_ms":      round(total_ms, 3),
        "top_k_results":      top_k_results,
        "model_kind":         model_kind,
        "model_path":         model_path,
        "task":               task,
        "num_classes":        num_classes,
        "timestamp":          time.time(),
    }

    _append_log(result)
    return result
