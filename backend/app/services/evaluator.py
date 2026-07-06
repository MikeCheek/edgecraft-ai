"""
evaluator.py
------------
Runs the ORIGINAL (.keras) model and an OPTIMIZED (.tflite) model over the
same held-out test set and reports accuracy, per-sample inference latency,
and size — so the optimization pipeline can show a real, honest comparison
instead of guessed numbers.

Design notes:
- Reuses the exact preprocessing conventions already used by Trainer
  (PIL resize + /255 normalise for images, DataProcessor.preprocess_audio
  for audio) so evaluation sees the same kind of tensors the model was
  trained on.
- Falls back to the "val" split if no "test" split exists yet, and reports
  which split was actually used so the numbers are never silently misleading.
- Caps the number of evaluated samples (default 200) to keep this fast
  enough to run synchronously inside the optimization background task.
- Per-sample (batch-of-1) timing is used for inference latency because that
  mirrors how the model will actually be invoked on an MCU (single sample
  at a time), rather than reporting large-batch throughput.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

MAX_EVAL_SAMPLES_DEFAULT = 200

IMAGE_TASKS = {"IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "VISUAL_WAKE_WORDS"}
AUDIO_TASKS = {"AUDIO_CLASSIFICATION", "KEYWORD_SPOTTING"}


# ---------------------------------------------------------------------------
# Test-set loading
# ---------------------------------------------------------------------------

def _load_test_samples(
    dataset_id: str, task: str, input_shape: Tuple[int, ...], labels: List[str],
    max_samples: int = MAX_EVAL_SAMPLES_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, str, List[Dict[str, Any]]]:
    """
    Returns (X, y, split_used, sample_meta). X is float32, already
    preprocessed to `input_shape`. y is int32 class indices matching
    `labels` order. sample_meta[i] = {"id", "filename", "label"} for the
    sample at X[i]/y[i], used to build per-sample prediction results (with
    a thumbnail fetchable via GET /api/datasets/image/{sample_id}).
    Falls back to the 'val' split (with a warning) if 'test' is empty.
    """
    from app.services.shared_state import data_manager
    from app.utils.data_processor import DataProcessor

    label_to_idx = {name: i for i, name in enumerate(labels)}
    samples = data_manager.get_samples(dataset_id)

    def _collect(split_name: str):
        subset = [s for s in samples if s.get("split") == split_name]
        return subset

    split_used = "test"
    subset = _collect("test")
    if len(subset) == 0:
        logger.warning(
            f"Dataset {dataset_id} has no 'test' split samples; "
            f"falling back to 'val' split for evaluation."
        )
        split_used = "val"
        subset = _collect("val")

    if len(subset) == 0:
        raise ValueError(
            "No test or validation samples available to evaluate against. "
            "Split your dataset (Auto Split) before running optimization."
        )

    if len(subset) > max_samples:
        # Deterministic, evenly-spaced sampling rather than the first N,
        # so we don't just evaluate on one class if samples are grouped.
        idx = np.linspace(0, len(subset) - 1, max_samples).astype(int)
        subset = [subset[i] for i in idx]

    X: List[np.ndarray] = []
    y: List[int] = []
    sample_meta: List[Dict[str, Any]] = []

    is_audio = task in AUDIO_TASKS

    for sample in subset:
        label = sample.get("label")
        if label not in label_to_idx:
            continue
        raw = data_manager.get_sample_data(sample["id"])
        if not raw:
            continue

        try:
            if is_audio:
                mfcc = DataProcessor.preprocess_audio(raw, task)
                mfcc = np.squeeze(mfcc)
                if mfcc.ndim != 2:
                    continue
                n_mfcc_expected, time_expected = input_shape[0], input_shape[1]
                actual_n_mfcc, actual_time = mfcc.shape
                if actual_n_mfcc != n_mfcc_expected:
                    continue
                if actual_time > time_expected:
                    mfcc = mfcc[:, :time_expected]
                elif actual_time < time_expected:
                    mfcc = np.pad(mfcc, ((0, 0), (0, time_expected - actual_time)), mode="constant")
                arr = np.expand_dims(mfcc, axis=-1).astype(np.float32)
            else:
                h, w, c = input_shape[0], input_shape[1], input_shape[2]
                from PIL import Image as PILImage
                with PILImage.open(io.BytesIO(raw)) as img:
                    img = img.convert("L" if c == 1 else "RGB")
                    img = img.resize((w, h), PILImage.BILINEAR)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    if c == 1:
                        arr = arr[:, :, np.newaxis]
        except Exception as e:
            logger.warning(f"Skipping unreadable sample {sample.get('id')}: {e}")
            continue

        X.append(arr)
        y.append(label_to_idx[label])
        sample_meta.append({
            "id": sample["id"],
            "filename": sample.get("filename"),
            "label": label,
        })

    if not X:
        raise ValueError(
            f"Could not decode any samples from the '{split_used}' split for evaluation."
        )

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), split_used, sample_meta


# ---------------------------------------------------------------------------
# TFLite helpers (quantisation-aware single-sample inference)
# ---------------------------------------------------------------------------

def _run_tflite_single(interpreter, sample: np.ndarray) -> Tuple[np.ndarray, float]:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    inp_detail = input_details[0]

    if inp_detail["dtype"] == np.int8:
        scale, zero_point = inp_detail["quantization"]
        scale = scale or 1.0
        inp = (sample / scale + zero_point).astype(np.int8)
    elif inp_detail["dtype"] == np.uint8:
        scale, zero_point = inp_detail["quantization"]
        scale = scale or 1.0
        inp = (sample / scale + zero_point).astype(np.uint8)
    else:
        inp = sample.astype(inp_detail["dtype"])

    inp = np.expand_dims(inp, axis=0)
    interpreter.set_tensor(inp_detail["index"], inp)

    t0 = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out_detail = output_details[0]
    out = interpreter.get_tensor(out_detail["index"])[0]
    if out_detail["dtype"] in (np.int8, np.uint8):
        scale, zero_point = out_detail["quantization"]
        scale = scale or 1.0
        out = (out.astype(np.float32) - zero_point) * scale

    return out.astype(np.float32), elapsed_ms


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_original_vs_optimized(
    *,
    keras_model: tf.keras.Model,
    keras_model_path: str,
    tflite_bytes: bytes,
    dataset_id: str,
    task: str,
    input_shape: Tuple[int, ...],
    labels: List[str],
    max_samples: int = MAX_EVAL_SAMPLES_DEFAULT,
) -> Dict[str, Any]:
    """
    Evaluate both models on the same test-set slice.
    Returns a dict with 'original', 'optimized', and 'deltas' sub-dicts,
    plus metadata about which split/how many samples were used.
    """
    import os
    import tempfile

    X, y, split_used, sample_meta = _load_test_samples(dataset_id, task, input_shape, labels, max_samples)
    n = len(X)
    is_binary = task == "VISUAL_WAKE_WORDS"

    # Per-sample records power the "testing" gallery (thumbnail + true label +
    # each model's prediction). Populated incrementally below as each model
    # runs, then merged into a single list per sample at the end.
    orig_preds: List[Dict[str, Any]] = [None] * n
    opt_preds: List[Dict[str, Any]] = [None] * n

    # ---- Original Keras model ------------------------------------------------
    orig_correct = 0
    orig_times_ms: List[float] = []
    orig_losses: List[float] = []
    for i in range(n):
        x = np.expand_dims(X[i], axis=0)
        t0 = time.perf_counter()
        pred = keras_model.predict(x, verbose=0)[0]
        orig_times_ms.append((time.perf_counter() - t0) * 1000.0)

        if is_binary:
            p = float(pred[0]) if np.ndim(pred) > 0 else float(pred)
            pred_class = 1 if p >= 0.5 else 0
            confidence = p if pred_class == 1 else (1 - p)
            eps = 1e-7
            p_clipped = min(max(p, eps), 1 - eps)
            orig_losses.append(-(y[i] * np.log(p_clipped) + (1 - y[i]) * np.log(1 - p_clipped)))
        else:
            pred_class = int(np.argmax(pred))
            confidence = float(pred[pred_class])
            eps = 1e-7
            p_true = float(pred[y[i]]) if y[i] < len(pred) else eps
            orig_losses.append(-np.log(max(p_true, eps)))

        is_correct = bool(pred_class == y[i])
        orig_correct += int(is_correct)
        predicted_label = labels[pred_class] if pred_class < len(labels) else f"class_{pred_class}"
        orig_preds[i] = {
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "correct": is_correct,
        }

    orig_accuracy = orig_correct / n
    orig_avg_loss = float(np.mean(orig_losses))
    orig_avg_ms = float(np.mean(orig_times_ms))
    orig_size_bytes = os.path.getsize(keras_model_path) if os.path.exists(keras_model_path) else 0

    # ---- Optimized TFLite model -----------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp:
        tmp.write(tflite_bytes)
        tmp_path = tmp.name

    try:
        interpreter = tf.lite.Interpreter(model_path=tmp_path)
        interpreter.allocate_tensors()

        opt_correct = 0
        opt_times_ms: List[float] = []
        for i in range(n):
            out, ms = _run_tflite_single(interpreter, X[i])
            opt_times_ms.append(ms)
            if is_binary:
                p = float(out[0]) if np.ndim(out) > 0 else float(out)
                pred_class = 1 if p >= 0.5 else 0
                confidence = p if pred_class == 1 else (1 - p)
            else:
                pred_class = int(np.argmax(out))
                confidence = float(out[pred_class]) if pred_class < len(out) else 0.0
            is_correct = bool(pred_class == y[i])
            opt_correct += int(is_correct)
            predicted_label = labels[pred_class] if pred_class < len(labels) else f"class_{pred_class}"
            opt_preds[i] = {
                "predicted_label": predicted_label,
                "confidence": round(confidence, 4),
                "correct": is_correct,
            }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    opt_accuracy = opt_correct / n
    opt_avg_ms = float(np.mean(opt_times_ms))
    opt_size_bytes = len(tflite_bytes)

    speedup = (orig_avg_ms / opt_avg_ms) if opt_avg_ms > 0 else 0.0
    size_reduction_pct = (
        (1 - (opt_size_bytes / orig_size_bytes)) * 100.0 if orig_size_bytes > 0 else 0.0
    )

    sample_results = [
        {
            "sample_id": sample_meta[i]["id"],
            "filename": sample_meta[i]["filename"],
            "true_label": sample_meta[i]["label"],
            "original": orig_preds[i],
            "optimized": opt_preds[i],
        }
        for i in range(n)
    ]

    return {
        "test_split_used": split_used,
        "num_samples_evaluated": n,
        "original": {
            "accuracy": round(orig_accuracy, 4),
            "loss": round(orig_avg_loss, 4),
            "avg_inference_ms": round(orig_avg_ms, 4),
            "size_bytes": orig_size_bytes,
        },
        "optimized": {
            "accuracy": round(opt_accuracy, 4),
            "avg_inference_ms": round(opt_avg_ms, 4),
            "size_bytes": opt_size_bytes,
        },
        "deltas": {
            "accuracy_delta": round(opt_accuracy - orig_accuracy, 4),
            "speedup_factor": round(speedup, 2),
            "size_reduction_pct": round(size_reduction_pct, 2),
        },
        # Per-sample breakdown for the "testing" gallery view (thumbnail via
        # GET /api/datasets/image/{sample_id} + true label + both models'
        # predictions). Kept out of the lightweight /status polling response
        # (see optimizer.get_optimization_status) and only returned from
        # /result, since this list can be a few hundred entries.
        "sample_results": sample_results,
    }
