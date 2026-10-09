"""
Live inference on a trained (.keras) or optimized (.tflite) model.

Preprocessing is app.services.preprocessing - byte-for-byte the same path
training and evaluation use. Results are logged to
<STORAGE_DIR>/inference_log.json.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app import config
from app.services import preprocessing
from app.services.detection import decode_predictions
from app.services.json_store import atomic_write_json, read_json

logger = logging.getLogger(__name__)

INFERENCE_LOG_PATH = config.STORAGE_DIR / "inference_log.json"
_log_lock = threading.Lock()
_model_lock = threading.Lock()


def _load_log() -> List[Dict[str, Any]]:
    return read_json(str(INFERENCE_LOG_PATH), [])


def _append_log(entry: Dict[str, Any]) -> None:
    with _log_lock:
        log = _load_log()
        log.append(entry)
        try:
            atomic_write_json(str(INFERENCE_LOG_PATH), log[-1000:])
        except Exception as e:
            logger.warning(f"Could not write inference log: {e}")


def get_inference_history(limit: int = 100) -> List[Dict[str, Any]]:
    return list(reversed(_load_log()[-limit:]))


def clear_inference_history() -> None:
    with _log_lock:
        atomic_write_json(str(INFERENCE_LOG_PATH), [])


class _LRUModelCache(OrderedDict):
    maxsize = 5

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)


_model_cache = _LRUModelCache()


def _get_model(path: str):
    """(kind, model_or_interpreter); cache key includes mtime so a re-trained
    file at the same path is reloaded."""
    import tensorflow as tf

    key = f"{path}:{Path(path).stat().st_mtime}"
    with _model_lock:
        if key not in _model_cache:
            if path.endswith(".tflite"):
                from app.services.evaluator import make_interpreter

                _model_cache[key] = ("tflite", make_interpreter(model_path=path))
            else:
                _model_cache[key] = ("keras", tf.keras.models.load_model(path, safe_mode=False, compile=False))
        else:
            _model_cache.move_to_end(key)
        return _model_cache[key]


def run_inference(
    *, model_path: str, raw_input: bytes, task: str, labels: List[str], top_k: int = 5,
) -> Dict[str, Any]:
    from app.services.evaluator import run_tflite_single

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    kind, model = _get_model(model_path)
    if kind == "tflite":
        input_shape = tuple(int(d) for d in model.get_input_details()[0]["shape"][1:])
    else:
        input_shape = tuple(int(d) for d in model.input_shape[1:])

    processed = preprocessing.preprocess(raw_input, task, input_shape)

    with _model_lock:  # TFLite interpreters are not thread-safe
        t0 = time.perf_counter()
        if kind == "tflite":
            out, inference_ms = run_tflite_single(model, processed)
        else:
            t_inf = time.perf_counter()
            out = model(np.expand_dims(processed, 0).astype(np.float32), training=False).numpy()[0]
            inference_ms = (time.perf_counter() - t_inf) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

    result: Dict[str, Any] = {
        "inference_id": str(uuid.uuid4()),
        "inference_time_ms": round(inference_ms, 3),
        "total_time_ms": round(total_ms, 3),
        "model_kind": kind,
        "model_path": model_path,
        "task": task,
        "timestamp": time.time(),
    }

    if task == "OBJECT_DETECTION" and out.ndim == 3:
        dets = decode_predictions(out)
        for d in dets:
            k = d["class_index"]
            d["label"] = labels[k] if k < len(labels) else f"class_{k}"
        dets.sort(key=lambda d: -d["confidence"])
        result.update({
            "detections": dets,
            "grid": [int(out.shape[0]), int(out.shape[1])],
            "top_class": dets[0]["label"] if dets else "none",
            "confidence": round(dets[0]["confidence"], 6) if dets else 0.0,
            "top_k_results": [{"class_name": d["label"], "confidence": d["confidence"]} for d in dets[:top_k]],
            "num_classes": int(out.shape[-1]) - 1,
        })
    else:
        probs = np.asarray(out, dtype=np.float32).reshape(-1)
        if probs.size == 1:  # legacy single-sigmoid binary model
            probs = np.array([1 - probs[0], probs[0]], dtype=np.float32)
        if len(labels) != probs.size:
            labels = [f"class_{i}" for i in range(probs.size)]
        order = np.argsort(-probs)
        top = [{"class_name": labels[i], "confidence": float(probs[i])} for i in order[:top_k]]
        result.update({
            "top_class": top[0]["class_name"],
            "confidence": round(top[0]["confidence"], 6),
            "top_k_results": top,
            "num_classes": int(probs.size),
        })

    _append_log(result)
    return result
