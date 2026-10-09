"""
Optimization engine: TFLite conversion (+ optional pruning / clustering with
real fine-tuning), then an honest evaluation against a float32 baseline.

session["method"] stores the internal key: int8, float16, dynamic_range,
pruning, weight_clustering. The router maps the UI enum names.
"""

import gzip
import logging
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from app import config
from app.services import dataset_loader
from app.services.job_logs import job_log_broker
from app.services.json_store import atomic_write_json, read_json
from app.services.model_factory import AudioToRGB  # noqa: F401  (registers legacy layer for load_model)

logger = logging.getLogger(__name__)

OPTIMIZATION_DIR = config.OPTIMIZATION_DIR
_SESSION_DB = OPTIMIZATION_DIR / "sessions.json"
_LEGACY_DIR = Path(tempfile.gettempdir()) / "edgecraft_optimizations"

ACTIVE_STATUSES = ("queued", "pending", "running")
CALIBRATION_SAMPLES = 200

_lock = threading.RLock()
_optimization_sessions: Dict[str, Dict[str, Any]] = {}


def _migrate_legacy_dir() -> None:
    """Earlier versions kept optimized models under the OS temp dir, which is
    wiped on reboot. Copy anything still there into STORAGE_DIR once."""
    legacy_db = _LEGACY_DIR / "sessions.json"
    if _SESSION_DB.exists() or not legacy_db.exists():
        return
    try:
        sessions = read_json(str(legacy_db), {})
        for oid, s in sessions.items():
            src = _LEGACY_DIR / oid
            if src.is_dir():
                shutil.copytree(src, OPTIMIZATION_DIR / oid, dirs_exist_ok=True)
            if s.get("output_path"):
                s["output_path"] = str(OPTIMIZATION_DIR / oid / "model.tflite")
        atomic_write_json(str(_SESSION_DB), sessions)
        logger.info("Migrated %d optimization sessions from %s", len(sessions), _LEGACY_DIR)
    except Exception as exc:
        logger.warning("Could not migrate legacy optimization dir: %s", exc)


def _load_sessions() -> None:
    global _optimization_sessions
    _migrate_legacy_dir()
    _optimization_sessions = read_json(str(_SESSION_DB), {})
    changed = False
    for s in _optimization_sessions.values():
        if s.get("status") in ACTIVE_STATUSES:
            s["status"] = "failed"
            s["error"] = "Interrupted: the backend restarted while this job was queued or running."
            changed = True
    if changed:
        _save_sessions()


def _save_sessions() -> None:
    with _lock:
        try:
            atomic_write_json(str(_SESSION_DB), _optimization_sessions)
        except Exception as e:
            logger.warning(f"Could not persist optimization session DB: {e}")


_load_sessions()


def create_optimization_session(
    training_id: str,
    method: str,
    sparsity_level: float = 0.5,
    frontend_method: Optional[str] = None,
    fine_tune_epochs: int = 2,
    quantization: str = "dynamic",
    num_clusters: int = 16,
) -> str:
    optimization_id = str(uuid.uuid4())
    with _lock:
        _optimization_sessions[optimization_id] = {
            "id": optimization_id,
            "training_id": training_id,
            "method": method,
            "frontend_method": frontend_method or method,
            "sparsity_level": float(sparsity_level),
            "fine_tune_epochs": int(fine_tune_epochs),
            "quantization": quantization if quantization in ("dynamic", "int8", "none") else "dynamic",
            "num_clusters": int(num_clusters),
            "status": "queued",
            "output_path": None,
            "error": None,
            "metrics": {},
            "comparison": None,
            "created_at": time.time(),
            "completed_at": None,
            "original_size_bytes": 0,
            "optimized_size_bytes": 0,
            "compression_ratio": 0.0,
        }
    _save_sessions()
    return optimization_id


def get_optimization_status(optimization_id: str) -> Dict[str, Any]:
    session = _optimization_sessions.get(optimization_id)
    if not session:
        raise ValueError(f"Optimization session {optimization_id} not found")
    comparison = session.get("comparison")
    summary = None
    if isinstance(comparison, dict):
        summary = {k: v for k, v in comparison.items() if k != "sample_results"}
    result = {
        "status": session["status"],
        "error": session.get("error"),
        "metrics": session.get("metrics", {}),
        "comparison": summary,
        "original_size_bytes": session.get("original_size_bytes", 0),
        "optimized_size_bytes": session.get("optimized_size_bytes", 0),
        "compression_ratio": session.get("compression_ratio", 0.0),
    }
    if session["status"] == "queued":
        from app.services.job_queue import job_queue

        result["queue_position"] = job_queue.position(optimization_id)
    return result


def get_optimization_result(optimization_id: str) -> Dict[str, Any]:
    session = _optimization_sessions.get(optimization_id)
    if not session:
        raise ValueError(f"Optimization session {optimization_id} not found")
    if session["status"] != "completed":
        raise ValueError("Optimization not completed yet")
    return {
        "optimization_id": optimization_id,
        "training_id": session.get("training_id"),
        "method": session.get("frontend_method", session["method"]),
        "metrics": session["metrics"],
        "comparison": session.get("comparison"),
        "output_path": str(session["output_path"]),
        "original_size_bytes": session.get("original_size_bytes", 0),
        "optimized_size_bytes": session.get("optimized_size_bytes", 0),
        "compression_ratio": session.get("compression_ratio", 0.0),
    }


def get_session(optimization_id: str) -> Optional[Dict[str, Any]]:
    return _optimization_sessions.get(optimization_id)


def get_output_path(optimization_id: str) -> Optional[Path]:
    session = _optimization_sessions.get(optimization_id)
    if not session or session["status"] != "completed" or not session.get("output_path"):
        return None
    return Path(session["output_path"])


def list_optimization_sessions() -> List[Dict[str, Any]]:
    result = []
    with _lock:
        items = list(_optimization_sessions.items())
    for opt_id, session in items:
        entry = {k: v for k, v in session.items() if k != "comparison"}
        comparison = session.get("comparison")
        if isinstance(comparison, dict):
            entry["comparison"] = {k: v for k, v in comparison.items() if k != "sample_results"}
        if entry.get("status") == "completed" and entry.get("output_path"):
            entry["download_url"] = f"/api/optimization/downloads/{opt_id}/model.tflite"
        result.append(entry)
    result.sort(key=lambda s: s.get("created_at", 0), reverse=True)
    return result


def cancel_optimization(optimization_id: str) -> bool:
    """Only queued jobs can be cancelled (conversion itself isn't interruptible)."""
    session = _optimization_sessions.get(optimization_id)
    if not session or session.get("status") != "queued":
        return False
    with _lock:
        session["status"] = "cancelled"
        session["completed_at"] = time.time()
    _save_sessions()
    return True


def delete_optimization(optimization_id: str) -> bool:
    session = _optimization_sessions.get(optimization_id)
    if not session:
        return False
    if session.get("status") == "running":
        raise ValueError("Cannot delete a running optimization.")
    shutil.rmtree(OPTIMIZATION_DIR / optimization_id, ignore_errors=True)
    with _lock:
        _optimization_sessions.pop(optimization_id, None)
    _save_sessions()
    return True


# ---------------------------------------------------------------------------
# Training context
# ---------------------------------------------------------------------------

def _resolve_trained_model_path(training_id: str) -> Path:
    from app.services.shared_state import trainer

    for model in trainer.trained_models.values():
        if model.get("training_id") == training_id and model.get("path"):
            p = Path(model["path"])
            if p.exists():
                return p
    for ext in (".keras", ".h5"):
        candidate = Path(trainer.storage_dir) / f"{training_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Trained model for training_id={training_id} not found.")


def _get_training_context(training_id: str) -> Dict[str, Any]:
    from app.services.shared_state import data_manager, trainer

    session = trainer.training_sessions.get(training_id, {})
    task = session.get("task")
    dataset_id = session.get("dataset_id")
    input_shape = tuple(session.get("input_shape", []))
    labels: List[str] = list(session.get("labels") or [])
    if not labels:
        for model in trainer.trained_models.values():
            if model.get("training_id") == training_id:
                labels = model.get("labels", [])
                break
    if not labels and dataset_id:
        labels = (dataset_loader.detection_classes(dataset_id) if task == "OBJECT_DETECTION"
                  else data_manager.get_dataset_labels(dataset_id))
    return {"task": task, "dataset_id": dataset_id, "input_shape": input_shape, "labels": labels}


def _load_training_data(ctx: Dict[str, Any], split: str, max_samples: Optional[int] = None):
    if not ctx["dataset_id"] or not ctx["labels"]:
        return np.zeros((0,)), np.zeros((0,))
    X, y, _ = dataset_loader.load_split(
        ctx["dataset_id"], ctx["task"], ctx["input_shape"], ctx["labels"], split, max_samples=max_samples
    )
    return X, y


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _fixed_batch(model: keras.Model) -> keras.Model:
    """Re-wrap the model with a static batch size of 1. Static shapes let the
    converter fold away SHAPE / FILL / STRIDED_SLICE ops (dynamic batch
    handling, RNN initial states) that TensorFlow Lite Micro can't run."""
    inp = keras.Input(shape=model.input_shape[1:], batch_size=1, name="input")
    return keras.Model(inp, model(inp, training=False))


def _convert(model: keras.Model, quantization: str, calibration: Optional[np.ndarray] = None) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(_fixed_batch(model))
    if quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        if calibration is None or len(calibration) == 0:
            raise ValueError("INT8 quantization needs calibration samples.")

        def representative_data_gen():
            for sample in calibration:
                yield [np.expand_dims(sample, 0).astype(np.float32)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


def _calibration_data(ctx, log) -> Optional[np.ndarray]:
    X, _ = _load_training_data(ctx, "train", CALIBRATION_SAMPLES)
    if len(X) == 0:
        X, _ = _load_training_data(ctx, "val", CALIBRATION_SAMPLES)
    if len(X) == 0:
        log("No dataset samples available for INT8 calibration.", "error")
        return None
    log(f"Calibrating INT8 ranges on {len(X)} real samples.")
    return X


def _clone_with_weights(model: keras.Model) -> keras.Model:
    cloned = keras.models.clone_model(model)
    cloned.set_weights(model.get_weights())
    return cloned


def _kernel_variables(model: keras.Model) -> List:
    """Conv / dense / depthwise kernels only - biases and BN params stay dense."""
    out = []
    for w in model.weights:
        name = w.path if hasattr(w, "path") else w.name
        if "kernel" in name.lower() and len(w.shape) >= 2:
            out.append(w)
    return out


def _loss_for(task: str):
    if task == "OBJECT_DETECTION":
        from app.services.trainer import make_fomo_loss

        return make_fomo_loss()
    return "sparse_categorical_crossentropy"


class _ApplyMasks(keras.callbacks.Callback):
    """Re-zeroes pruned weights after every batch so fine-tuning can't regrow them."""

    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs

    def on_train_batch_end(self, batch, logs=None):
        for var, mask in self.pairs:
            var.assign(var.numpy() * mask)


def _fine_tune(model: keras.Model, ctx, epochs: int, log, callbacks=None) -> None:
    if epochs <= 0:
        return
    X, y = _load_training_data(ctx, "train")
    if len(X) == 0:
        log("No training samples available - skipping fine-tuning.", "warning")
        return
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=_loss_for(ctx["task"]))
    log(f"Fine-tuning for {epochs} epoch(s) on {len(X)} samples...")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=callbacks or [])


def _magnitude_pruning(model, ctx, sparsity: float, fine_tune_epochs: int, log):
    pruned = _clone_with_weights(model)
    pairs = []
    pruned_count = total = 0
    for w in _kernel_variables(pruned):
        values = w.numpy()
        threshold = np.percentile(np.abs(values), sparsity * 100)
        mask = (np.abs(values) > threshold).astype(values.dtype)
        w.assign(values * mask)
        pairs.append((w, mask))
        pruned_count += int((mask == 0).sum())
        total += values.size
    log(f"Zeroed {pruned_count}/{total} kernel weights ({pruned_count / max(total, 1) * 100:.1f}%).")
    _fine_tune(pruned, ctx, fine_tune_epochs, log, callbacks=[_ApplyMasks(pairs)])
    return pruned, {"sparsity_target": sparsity, "sparsity_actual": round(pruned_count / max(total, 1), 4),
                    "fine_tune_epochs": fine_tune_epochs}


def _weight_clustering(model, ctx, num_clusters: int, fine_tune_epochs: int, log):
    from scipy.cluster.vq import kmeans2

    clustered = _clone_with_weights(model)
    _fine_tune(clustered, ctx, fine_tune_epochs, log)  # adapt first, then snap to centroids

    for w in _kernel_variables(clustered):
        values = w.numpy()
        flat = values.flatten().astype(np.float64)
        k = min(num_clusters, len(np.unique(flat)))
        if k < 2:
            continue
        try:
            centroids, assignment = kmeans2(flat, k, minit="++", seed=42)
            w.assign(centroids[assignment].astype(values.dtype).reshape(values.shape))
        except Exception as e:
            log(f"Skipping clustering for {w.name}: {e}", "warning")
    return clustered, {"num_clusters": num_clusters, "fine_tune_epochs": fine_tune_epochs}


# ---------------------------------------------------------------------------
# Job entry point
# ---------------------------------------------------------------------------

def optimize(optimization_id: str, model_base_dir: str = None) -> None:
    session = _optimization_sessions.get(optimization_id)
    if not session or session.get("status") == "cancelled":
        return

    def log(msg: str, level: str = "info"):
        job_log_broker.log(optimization_id, msg, level=level)

    with _lock:
        session["status"] = "running"
        session["started_at"] = time.time()
    _save_sessions()
    log(f"Optimization {optimization_id} starting - method: {session.get('frontend_method')}")

    try:
        training_id = session["training_id"]
        model_path = _resolve_trained_model_path(training_id)
        ctx = _get_training_context(training_id)
        model = keras.models.load_model(str(model_path), safe_mode=False, compile=False)
        method = session["method"]
        output_dir = OPTIMIZATION_DIR / optimization_id
        output_dir.mkdir(parents=True, exist_ok=True)

        log("Converting the float32 baseline...")
        baseline = _convert(model, "none")
        session["original_size_bytes"] = len(baseline)
        session["keras_size_bytes"] = model_path.stat().st_size

        calibration = None
        quant = session.get("quantization", "dynamic")
        needs_int8 = method == "int8" or (method in ("pruning", "weight_clustering") and quant == "int8")
        if needs_int8:
            calibration = _calibration_data(ctx, log)
            if calibration is None:
                raise ValueError("INT8 needs real calibration samples, but none could be loaded from the dataset.")

        extra: Dict[str, Any] = {}
        if method == "dynamic_range":
            tflite_data = _convert(model, "dynamic")
            note = "Weights INT8, activations float32"
        elif method == "float16":
            tflite_data = _convert(model, "float16")
            note = "Weights FLOAT16"
        elif method == "int8":
            tflite_data = _convert(model, "int8", calibration)
            note = "Full INT8 - weights and activations, calibrated on real samples"
        elif method == "pruning":
            pruned, extra = _magnitude_pruning(model, ctx, session["sparsity_level"],
                                               session.get("fine_tune_epochs", 2), log)
            tflite_data = _convert(pruned, quant, calibration)
            note = (f"{extra['sparsity_actual'] * 100:.0f}% of kernel weights zeroed, {quant} quantized. "
                    "TFLite stores zeros densely: the gain shows in the compressed size.")
        elif method == "weight_clustering":
            clustered, extra = _weight_clustering(model, ctx, session.get("num_clusters", 16),
                                                  session.get("fine_tune_epochs", 2), log)
            tflite_data = _convert(clustered, quant, calibration)
            note = (f"Kernels snapped to {extra['num_clusters']} centroids per layer, {quant} quantized. "
                    "The gain shows in the compressed size.")
        elif method == "transfer_learning":
            raise ValueError(
                "Transfer learning is a training option, not an optimization: pick a pretrained "
                "backbone (e.g. MobileNetV3Small) in Model Training instead."
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        output_path = output_dir / "model.tflite"
        output_path.write_bytes(tflite_data)
        compressed = len(gzip.compress(tflite_data, compresslevel=9))
        log(f"Conversion done - {len(tflite_data) / 1024:.1f} KB ({compressed / 1024:.1f} KB gzipped)")

        from app.services.mcu_advisor import tflm_verify

        tflm = tflm_verify(tflite_data)
        if tflm.get("available"):
            if tflm.get("supported"):
                log(f"TensorFlow Lite Micro check passed - measured arena {tflm['arena_bytes'] / 1024:.1f} KB, "
                    f"max output diff vs TFLite {tflm.get('max_abs_diff')}")
            else:
                log(f"TensorFlow Lite Micro can't run this model: {tflm.get('error')}", "warning")
        elif tflm.get("unsupported_ops"):
            log(f"Ops not in TensorFlow Lite Micro: {tflm['unsupported_ops']}", "warning")

        with _lock:
            session["output_path"] = str(output_path)
            session["metrics"] = {
                "method": method,
                "size_bytes": len(tflite_data),
                "compressed_size_bytes": compressed,
                "baseline_compressed_size_bytes": len(gzip.compress(baseline, compresslevel=9)),
                "note": note,
                "tflm": tflm,
                "ops": tflm.get("ops", []),
                **extra,
            }
            session["optimized_size_bytes"] = len(tflite_data)
            session["compression_ratio"] = round(len(tflite_data) / len(baseline), 4) if baseline else 0.0

        try:
            log("Evaluating baseline vs optimized on the held-out split...")
            if ctx["task"] and ctx["dataset_id"] and ctx["input_shape"] and ctx["labels"]:
                from app.services.evaluator import evaluate_original_vs_optimized

                comparison = evaluate_original_vs_optimized(
                    keras_model=model, tflite_bytes=tflite_data, baseline_tflite_bytes=baseline,
                    dataset_id=ctx["dataset_id"], task=ctx["task"], input_shape=ctx["input_shape"],
                    labels=ctx["labels"],
                )
                session["comparison"] = comparison
                d = comparison["deltas"]
                log(f"Evaluation done - {comparison['metric_name']} delta: {d['accuracy_delta'] * 100:+.2f}pp, "
                    f"speedup: {d['speedup_factor']}x, size reduction: {d['size_reduction_pct']}%")
            else:
                session["comparison"] = {"error": "Insufficient training context to evaluate on the test set."}
                log("Skipped evaluation - insufficient training context.", "warning")
        except Exception as eval_exc:
            session["comparison"] = {"error": str(eval_exc)}
            log(f"Evaluation failed: {eval_exc}", "warning")

        with _lock:
            session["status"] = "completed"
            session["completed_at"] = time.time()
        _save_sessions()
        log("Optimization completed successfully.")
    except Exception as exc:
        with _lock:
            session["status"] = "failed"
            session["error"] = str(exc)
            session["completed_at"] = time.time()
        _save_sessions()
        logger.exception(f"[{optimization_id}] Optimization failed: {exc}")
        log(f"Optimization FAILED: {exc}", "error")
        log(traceback.format_exc(), "error")
