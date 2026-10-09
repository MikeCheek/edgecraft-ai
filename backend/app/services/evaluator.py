"""
Honest original-vs-optimized comparison on the held-out split.

* Accuracy/F1 of the original model comes from the Keras model (batched).
* Size and latency of the original come from a plain float32 TFLite
  conversion of the same model, so both sides are measured with the same
  interpreter, single-sample, the way a microcontroller runs them. (Timing
  Keras' predict() per sample measured ~100x framework overhead, and the
  .keras file also stores optimizer state, which made earlier "speedup" and
  "size reduction" numbers meaningless.)
* Quantized inputs are rounded and saturated (not truncated / wrapped).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from app.services import dataset_loader
from app.services.detection import decode_predictions, score_detections
from app.services.preprocessing import dequantize_output, quantize_input

logger = logging.getLogger(__name__)

MAX_EVAL_SAMPLES_DEFAULT = 300


def make_interpreter(tflite_bytes: bytes = None, model_path: str = None):
    """TFLite interpreter, retrying without the default XNNPACK delegate when
    it rejects a graph (it can't prepare some int8 ops, e.g. hard-swish)."""
    kwargs = {"model_content": tflite_bytes} if tflite_bytes is not None else {"model_path": model_path}
    try:
        interpreter = tf.lite.Interpreter(**kwargs)
        interpreter.allocate_tensors()
        return interpreter
    except RuntimeError:
        interpreter = tf.lite.Interpreter(
            **kwargs,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interpreter.allocate_tensors()
        return interpreter


def run_tflite_single(interpreter, sample: np.ndarray) -> Tuple[np.ndarray, float]:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    x = sample
    if inp["dtype"] in (np.int8, np.uint8):
        scale, zp = inp["quantization"]
        x = quantize_input(sample, scale, zp, inp["dtype"])
    else:
        x = sample.astype(inp["dtype"])
    interpreter.set_tensor(inp["index"], np.expand_dims(x, 0))
    t0 = time.perf_counter()
    interpreter.invoke()
    elapsed = (time.perf_counter() - t0) * 1000.0
    y = interpreter.get_tensor(out["index"])[0]
    if out["dtype"] in (np.int8, np.uint8):
        scale, zp = out["quantization"]
        y = dequantize_output(y, scale, zp)
    return y.astype(np.float32), elapsed


def run_tflite(tflite_bytes: bytes, X: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    interpreter = make_interpreter(tflite_bytes)
    outs, times = [], []
    if len(X):  # warm-up so the first sample's allocation isn't timed
        run_tflite_single(interpreter, X[0])
    for x in X:
        y, ms = run_tflite_single(interpreter, x)
        outs.append(y)
        times.append(ms)
    return np.asarray(outs), times


def _class_predictions(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """argmax + confidence, also handling legacy single-sigmoid models."""
    if probs.ndim == 2 and probs.shape[1] == 1:
        p = probs[:, 0]
        pred = (p >= 0.5).astype(int)
        return pred, np.where(pred == 1, p, 1 - p)
    pred = probs.argmax(axis=-1)
    return pred, probs[np.arange(len(probs)), pred]


def _metrics(task: str, y: np.ndarray, probs: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    from app.services.trainer import classification_metrics

    if task == "OBJECT_DETECTION":
        m = score_detections(list(y), list(probs), len(labels))
        m["labels"] = labels
        m["accuracy"] = m["f1"]  # headline metric used across the UI
        return m
    pred, _ = _class_predictions(probs)
    return classification_metrics(y, pred, labels)


def evaluate_original_vs_optimized(
    *,
    keras_model: tf.keras.Model,
    tflite_bytes: bytes,
    baseline_tflite_bytes: bytes,
    dataset_id: str,
    task: str,
    input_shape: Sequence[int],
    labels: List[str],
    max_samples: int = MAX_EVAL_SAMPLES_DEFAULT,
    keras_model_path: str = None,
) -> Dict[str, Any]:
    split = dataset_loader.evaluation_split(dataset_id)
    X, y, meta = dataset_loader.load_split(dataset_id, task, input_shape, labels, split, max_samples=max_samples)
    if len(X) == 0:
        raise ValueError(
            "No test or validation samples available to evaluate against. "
            "Split your dataset (Auto Split) before running optimization."
        )

    keras_probs = keras_model.predict(X, verbose=0, batch_size=64)
    _, base_times = run_tflite(baseline_tflite_bytes, X)
    opt_probs, opt_times = run_tflite(tflite_bytes, X)

    orig_metrics = _metrics(task, y, keras_probs, labels)
    opt_metrics = _metrics(task, y, opt_probs, labels)
    base_ms = float(np.median(base_times))
    opt_ms = float(np.median(opt_times))
    base_size, opt_size = len(baseline_tflite_bytes), len(tflite_bytes)

    sample_results = []
    for i, m in enumerate(meta):
        if task == "OBJECT_DETECTION":
            entry = {
                "original": {"detections": decode_predictions(keras_probs[i])},
                "optimized": {"detections": decode_predictions(opt_probs[i])},
            }
            for side in ("original", "optimized"):
                for d in entry[side]["detections"]:
                    d["label"] = labels[d["class_index"]] if d["class_index"] < len(labels) else str(d["class_index"])
        else:
            entry = {}
            for side, probs in (("original", keras_probs), ("optimized", opt_probs)):
                pred, conf = _class_predictions(probs[i:i + 1])
                k = int(pred[0])
                entry[side] = {
                    "predicted_label": labels[k] if k < len(labels) else f"class_{k}",
                    "confidence": round(float(conf[0]), 4),
                    "correct": bool(k == int(y[i])),
                }
        sample_results.append({"sample_id": m["id"], "filename": m["filename"], "true_label": m["label"], **entry})

    def _side(metrics, ms, size, extra=None):
        d = {"accuracy": metrics.get("accuracy", 0.0), "avg_inference_ms": round(ms, 4),
             "size_bytes": size, "metrics": metrics}
        d.update(extra or {})
        return d

    return {
        "test_split_used": split,
        "num_samples_evaluated": int(len(X)),
        "metric_name": "f1" if task == "OBJECT_DETECTION" else "accuracy",
        "original": _side(orig_metrics, base_ms, base_size,
                          {"baseline": "float32 TFLite (same weights as the trained Keras model)"}),
        "optimized": _side(opt_metrics, opt_ms, opt_size),
        "deltas": {
            "accuracy_delta": round(opt_metrics.get("accuracy", 0.0) - orig_metrics.get("accuracy", 0.0), 4),
            "speedup_factor": round(base_ms / opt_ms, 2) if opt_ms > 0 else 0.0,
            "size_reduction_pct": round((1 - opt_size / base_size) * 100.0, 2) if base_size else 0.0,
        },
        "timing_note": "Median single-sample latency on this host's TFLite interpreter; MCU timings differ.",
        "sample_results": sample_results,
    }
