# optimizer.py
# Real optimization engine: TFLite conversion + test-set evaluation.
# session["method"] always stores the INTERNAL key (int8, float16,
# dynamic_range, pruning, weight_clustering, transfer_learning).
# The router is responsible for mapping the frontend's uppercase enum
# (INT8_QUANTIZATION, etc.) to these internal keys, and for translating
# back for display.

import os, uuid, tempfile, logging, json, time
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# --- Paths -------------------------------------------------------------------
OPTIMIZATION_DIR = Path(tempfile.gettempdir()) / "edgecraft_optimizations"
OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_DB = OPTIMIZATION_DIR / "sessions.json"

_optimization_sessions: Dict[str, Dict[str, Any]] = {}

def _load_sessions() -> None:
    global _optimization_sessions
    if _SESSION_DB.exists():
        try:
            with open(_SESSION_DB) as f:
                _optimization_sessions = json.load(f)
        except Exception:
            _optimization_sessions = {}

def _save_sessions() -> None:
    try:
        serializable = {}
        for k, v in _optimization_sessions.items():
            entry = dict(v)
            if isinstance(entry.get("output_path"), Path):
                entry["output_path"] = str(entry["output_path"])
            serializable[k] = entry
        with open(_SESSION_DB, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not persist session DB: {e}")

_load_sessions()

def create_optimization_session(
    training_id: str,
    method: str,
    sparsity_level: float = 0.5,
    frontend_method: Optional[str] = None,
) -> str:
    optimization_id = str(uuid.uuid4())
    _optimization_sessions[optimization_id] = {
        "id": optimization_id,
        "training_id": training_id,
        "method": method,
        "frontend_method": frontend_method or method,
        "sparsity_level": sparsity_level,
        "status": "pending",
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

    # Strip the (potentially large) per-sample breakdown from this response -
    # it's polled every few seconds by the frontend while an optimization
    # runs, and the full per-sample list (with predictions for every test
    # sample) is only needed once, via get_optimization_result().
    comparison = session.get("comparison")
    comparison_summary = None
    if isinstance(comparison, dict):
        comparison_summary = {k: v for k, v in comparison.items() if k != "sample_results"}

    return {
        "status": session["status"],
        "error": session.get("error"),
        "metrics": session.get("metrics", {}),
        "comparison": comparison_summary,
        "original_size_bytes": session.get("original_size_bytes", 0),
        "optimized_size_bytes": session.get("optimized_size_bytes", 0),
        "compression_ratio": session.get("compression_ratio", 0.0),
    }

def get_optimization_result(optimization_id: str) -> Dict[str, Any]:
    session = _optimization_sessions.get(optimization_id)
    if not session:
        raise ValueError(f"Optimization session {optimization_id} not found")
    if session["status"] != "completed":
        raise ValueError("Optimization not completed yet")
    return {
        "optimization_id": optimization_id,
        "method": session.get("frontend_method", session["method"]),
        "metrics": session["metrics"],
        "comparison": session.get("comparison"),
        "output_path": str(session["output_path"]),
        "original_size_bytes": session.get("original_size_bytes", 0),
        "optimized_size_bytes": session.get("optimized_size_bytes", 0),
        "compression_ratio": session.get("compression_ratio", 0.0),
    }

def get_session(optimization_id: str) -> Optional[Dict[str, Any]]:
    """Raw session access, used by mcu_advisor / exporter / llm_advisor."""
    return _optimization_sessions.get(optimization_id)

def get_output_path(optimization_id: str) -> Optional[Path]:
    session = _optimization_sessions.get(optimization_id)
    if not session or session["status"] != "completed":
        return None
    op = session["output_path"]
    return Path(op) if op else None

def list_optimization_sessions() -> List[Dict[str, Any]]:
    result = []
    for opt_id, session in _optimization_sessions.items():
        entry = dict(session)
        if isinstance(entry.get("output_path"), Path):
            entry["output_path"] = str(entry["output_path"])
        if entry.get("status") == "completed" and entry.get("output_path"):
            entry["download_url"] = f"/api/optimization/downloads/{opt_id}/model.tflite"
        result.append(entry)
    result.sort(key=lambda s: s.get("created_at", 0), reverse=True)
    return result

# --- Locate the trained model file --------------------------------------------

def _resolve_trained_model_path(training_id: str) -> Path:
    """
    Trainer.train() saves models as `<storage_dir>/<training_id>.keras`
    (a flat file, NOT `<storage_dir>/<training_id>/model.keras`).
    We look it up via the Trainer's own registry first (authoritative),
    then fall back to the flat-file convention directly.
    """
    from app.services.shared_state import trainer

    for model in trainer.trained_models.values():
        if model.get("training_id") == training_id and model.get("path"):
            p = Path(model["path"])
            if p.exists():
                return p

    candidate = Path(trainer.storage_dir) / f"{training_id}.keras"
    if candidate.exists():
        return candidate
    candidate_h5 = Path(trainer.storage_dir) / f"{training_id}.h5"
    if candidate_h5.exists():
        return candidate_h5

    raise FileNotFoundError(
        f"Trained model for training_id={training_id} not found. "
        f"Looked in trainer registry and at {candidate}"
    )

def _get_training_context(training_id: str) -> Dict[str, Any]:
    """Task / dataset_id / input_shape / labels needed for test-set evaluation."""
    from app.services.shared_state import trainer

    session = trainer.training_sessions.get(training_id, {})
    task = session.get("task")
    dataset_id = session.get("dataset_id")
    input_shape = tuple(session.get("input_shape", []))

    labels: List[str] = []
    for model in trainer.trained_models.values():
        if model.get("training_id") == training_id:
            labels = model.get("labels", [])
            break

    if not labels and dataset_id:
        from app.services.shared_state import data_manager
        labels = data_manager.get_dataset_labels(dataset_id)

    return {"task": task, "dataset_id": dataset_id, "input_shape": input_shape, "labels": labels}

# --- Core optimization dispatcher ---------------------------------------------

def optimize(optimization_id: str, model_base_dir: str) -> None:
    """Background task: run the selected optimization, save the .tflite file,
    then evaluate original vs optimized on the dataset's test set."""
    session = _optimization_sessions[optimization_id]
    session["status"] = "running"
    _save_sessions()

    try:
        training_id = session["training_id"]
        model_path = _resolve_trained_model_path(training_id)
        session["original_size_bytes"] = model_path.stat().st_size

        model = tf.keras.models.load_model(str(model_path))

        method = session["method"]
        output_dir = OPTIMIZATION_DIR / optimization_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model.tflite"

        if method == "dynamic_range":
            tflite_data, metrics = _dynamic_range_quantization(model)
        elif method == "float16":
            tflite_data, metrics = _float16_quantization(model)
        elif method == "int8":
            tflite_data, metrics = _int8_quantization(model)
        elif method == "pruning":
            tflite_data, metrics = _magnitude_pruning(
                model, sparsity=session["sparsity_level"]
            )
        elif method == "weight_clustering":
            tflite_data, metrics = _weight_clustering(model)
        elif method == "transfer_learning":
            tflite_data, metrics = _transfer_learning_optimization(model)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        output_path.write_bytes(tflite_data)

        session["output_path"] = str(output_path)
        session["metrics"] = metrics
        session["optimized_size_bytes"] = len(tflite_data)
        orig = session.get("original_size_bytes", 0)
        session["compression_ratio"] = round(len(tflite_data) / orig, 4) if orig else 0.0

        # --- Real test-set evaluation: original vs optimized -------------------
        try:
            ctx = _get_training_context(training_id)
            if ctx["task"] and ctx["dataset_id"] and ctx["input_shape"] and ctx["labels"]:
                from app.services.evaluator import evaluate_original_vs_optimized
                comparison = evaluate_original_vs_optimized(
                    keras_model=model,
                    keras_model_path=str(model_path),
                    tflite_bytes=tflite_data,
                    dataset_id=ctx["dataset_id"],
                    task=ctx["task"],
                    input_shape=ctx["input_shape"],
                    labels=ctx["labels"],
                )
                session["comparison"] = comparison
            else:
                session["comparison"] = {"error": "Insufficient training context to evaluate on test set."}
        except Exception as eval_exc:
            logger.warning(f"[{optimization_id}] Test-set evaluation failed: {eval_exc}")
            session["comparison"] = {"error": str(eval_exc)}

        session["status"] = "completed"
        session["completed_at"] = time.time()
        _save_sessions()
        logger.info(f"[{optimization_id}] Optimization completed -> {output_path}")

    except Exception as exc:
        session["status"] = "failed"
        session["error"] = str(exc)
        session["completed_at"] = time.time()
        _save_sessions()
        logger.exception(f"[{optimization_id}] Optimization failed: {exc}")

# --- Method implementations ----------------------------------------------------

def _dynamic_range_quantization(model: tf.keras.Model):
    """Weights quantized to INT8; activations remain float - best for CPU."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_data = converter.convert()
    return tflite_data, {
        "method": "dynamic_range",
        "size_bytes": len(tflite_data),
        "note": "Weights INT8, activations float32 (dynamic range quantization)",
    }

def _float16_quantization(model: tf.keras.Model):
    """Weights quantized to FLOAT16 - good balance for GPU-accelerated MCUs."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_data = converter.convert()
    return tflite_data, {
        "method": "float16",
        "size_bytes": len(tflite_data),
        "note": "Weights FLOAT16",
    }

def _int8_quantization(model: tf.keras.Model):
    """
    Full INT8 post-training quantization with a representative dataset.
    Weights AND activations are INT8 - smallest model, fastest on MCUs.
    """
    input_shape = model.input_shape

    def representative_data_gen():
        for _ in range(200):
            sample = np.random.rand(
                1, *input_shape[1:]
            ).astype(np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_data = converter.convert()
    return tflite_data, {
        "method": "int8",
        "size_bytes": len(tflite_data),
        "note": "Full INT8 - weights and activations",
    }

def _clone_with_weights(model: tf.keras.Model) -> tf.keras.Model:
    """Deep-copy a Keras model (architecture + weights) so in-place weight
    mutation (pruning/clustering) never touches the original model object -
    which is also used for the "original" side of the test-set comparison."""
    cloned = tf.keras.models.clone_model(model)
    cloned.set_weights(model.get_weights())
    return cloned

def _get_prunable_weight_variables(model: tf.keras.Model) -> List:
    """Only kernel/weight matrices (Conv2D, Dense, depthwise kernels) are
    pruned/clustered - biases and batch-norm scale/shift are left alone
    since zeroing/clustering those has an outsized accuracy impact for
    negligible size savings."""
    result = []
    for w in model.weights:
        name = w.name.lower()
        if ("kernel" in name) and len(w.shape) >= 2:
            result.append(w)
    return result

def _magnitude_pruning(model: tf.keras.Model, sparsity: float = 0.5):
    """
    Magnitude-based weight pruning, applied directly (no tfmot dependency).

    NOTE: this used to go through tensorflow_model_optimization's
    prune_low_magnitude/strip_pruning, but that library's Sequential/Functional
    isinstance checks predate Keras 3 and raise
    "can only prune an object of the following types... You passed an object
    of type: Sequential" even for perfectly ordinary Sequential models on
    current TensorFlow/Keras versions. Since we only need POST-training
    pruning here (not pruning-aware fine-tuning), zeroing the smallest-
    magnitude weights per kernel directly is equivalent and avoids the
    incompatibility entirely.
    """
    cloned = _clone_with_weights(model)

    pruned_count = 0
    total_count = 0
    for w in _get_prunable_weight_variables(cloned):
        values = w.numpy()
        if values.size == 0:
            continue
        threshold = np.percentile(np.abs(values), sparsity * 100)
        mask = np.abs(values) >= threshold
        pruned_count += int(np.sum(~mask))
        total_count += values.size
        w.assign(values * mask)

    converter = tf.lite.TFLiteConverter.from_keras_model(cloned)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_data = converter.convert()

    actual_sparsity = (pruned_count / total_count) if total_count else 0.0
    return tflite_data, {
        "method": "pruning",
        "sparsity_target": sparsity,
        "sparsity_actual": round(actual_sparsity, 4),
        "size_bytes": len(tflite_data),
        "note": f"{actual_sparsity*100:.1f}% of kernel weights zeroed, then quantized",
    }

def _weight_clustering(model: tf.keras.Model, num_clusters: int = 16):
    """
    Weight clustering, applied directly via k-means on each kernel's values
    (no tfmot dependency - see the note in _magnitude_pruning for why).
    Groups each layer's weights into `num_clusters` centroids so far fewer
    unique float values need to be stored, then applies a follow-up
    quantization pass since clustered weights compress very well.
    """
    from scipy.cluster.vq import kmeans2

    cloned = _clone_with_weights(model)

    for w in _get_prunable_weight_variables(cloned):
        values = w.numpy()
        flat = values.flatten().astype(np.float64)
        unique_vals = np.unique(flat)
        k = min(num_clusters, len(unique_vals))
        if k < 2:
            continue
        try:
            centroids, labels = kmeans2(flat, k, minit="++", seed=42)
            clustered_flat = centroids[labels].astype(values.dtype)
            w.assign(clustered_flat.reshape(values.shape))
        except Exception as e:
            logger.warning(f"Skipping clustering for weight {w.name} ({values.shape}): {e}")
            continue

    converter = tf.lite.TFLiteConverter.from_keras_model(cloned)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_data = converter.convert()

    return tflite_data, {
        "method": "weight_clustering",
        "num_clusters": num_clusters,
        "size_bytes": len(tflite_data),
        "note": f"Weights grouped into up to {num_clusters} clusters per layer, then quantized",
    }

def _transfer_learning_optimization(model: tf.keras.Model):
    """
    Transfer Learning optimization:
    1. Detects the task type from model output shape.
    2. Replaces the classification head with a frozen MobileNetV2 backbone.
    3. Fine-tunes only the new head for a few synthetic steps.
    4. Exports as INT8 TFLite.

    In production, pass real fine-tuning data instead of synthetic samples.
    """
    input_shape = model.input_shape[1:]
    num_classes = model.output_shape[-1]

    is_audio = (len(input_shape) == 3 and input_shape[-1] == 1)

    if is_audio:
        inp = tf.keras.Input(shape=input_shape, name="mfcc_input")
        x = tf.keras.layers.Lambda(
            lambda t: tf.image.resize(
                tf.repeat(t, 3, axis=-1), [96, 96]
            ),
            name="audio_to_rgb",
        )(inp)
    else:
        inp = tf.keras.Input(shape=input_shape, name="image_input")
        x = tf.keras.layers.Resizing(96, 96)(inp)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="head")(x)

    tl_model = tf.keras.Model(inp, output, name="transfer_learning_model")

    try:
        orig_dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
        if orig_dense:
            tl_model.get_layer("head").set_weights(orig_dense[-1].get_weights())
    except Exception:
        pass

    tl_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    x_dummy = np.random.rand(64, *input_shape).astype(np.float32)
    y_dummy = np.random.randint(0, num_classes, size=(64,))
    tl_model.fit(x_dummy, y_dummy, epochs=3, batch_size=16, verbose=0)

    def representative_data_gen():
        for i in range(0, len(x_dummy), 1):
            yield [x_dummy[i : i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(tl_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_data = converter.convert()

    return tflite_data, {
        "method": "transfer_learning",
        "backbone": "MobileNetV2 (imagenet, frozen)",
        "fine_tune_epochs": 3,
        "size_bytes": len(tflite_data),
        "note": "MobileNetV2 backbone + fine-tuned head -> INT8 TFLite",
    }
