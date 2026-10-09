"""
Training service.

Sessions are persisted to <STORAGE_DIR>/training_db.json (atomic writes,
guarded by a lock because the training thread and API requests both touch
them). Jobs run one at a time through app.services.job_queue.
"""

import contextlib
import logging
import os
import threading
import time
import traceback
import uuid
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from app import config
from app.services import dataset_loader, preprocessing
from app.services.detection import FOMO_STRIDE, score_detections
from app.services.job_logs import job_log_broker
from app.services.json_store import atomic_write_json, read_json
from app.services.model_factory import ModelFactory, set_backbone_trainable

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("queued", "initialized", "running")
FOMO_OBJECT_WEIGHT = 25.0


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict:
    """Accuracy, confusion matrix and per-class precision/recall/F1."""
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[int(t), int(p)] += 1
    per_class = []
    for k in range(n):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append({"label": labels[k], "precision": round(float(precision), 4),
                          "recall": round(float(recall), 4), "f1": round(float(f1), 4),
                          "support": int(cm[k, :].sum())})
    total = cm.sum()
    return {
        "accuracy": round(float(np.trace(cm) / total), 4) if total else 0.0,
        "macro_f1": round(float(np.mean([c["f1"] for c in per_class])), 4) if per_class else 0.0,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "per_class": per_class,
    }


def make_fomo_loss(object_weight: float = FOMO_OBJECT_WEIGHT):
    """Cross-entropy over the FOMO grid, up-weighting the (rare) object cells."""

    def fomo_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        w = tf.where(y_true > 0, object_weight, 1.0)
        return tf.reduce_sum(ce * w) / tf.reduce_sum(w)

    fomo_loss.__name__ = "fomo_loss"
    return fomo_loss


LIVE_EVAL_MAX_SAMPLES = 2000   # val samples scored per epoch for the dashboard
LIVE_UPDATE_SECONDS = 0.5      # throttle for intra-epoch (batch) progress
BATCH_HISTORY_MAX = 600        # batch points kept for the live loss curve


def _memory_mb() -> Optional[float]:
    """Resident memory of the backend process (best effort, no psutil)."""
    try:
        with open("/proc/self/statm") as f:
            return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except Exception:
        try:
            import resource
            import sys

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return round(peak / (1e6 if sys.platform == "darwin" else 1e3), 1)
        except Exception:
            return None


def _gpu_memory_mb() -> Optional[float]:
    try:
        if tf.config.list_physical_devices("GPU"):
            return round(tf.config.experimental.get_memory_info("GPU:0")["current"] / 1e6, 1)
    except Exception:
        pass
    return None


def calibration_stats(probs: np.ndarray, y_true: np.ndarray, bins: int = 10) -> Dict:
    """Mean confidence and expected calibration error (ECE) of softmax outputs."""
    conf = probs.max(axis=-1)
    correct = probs.argmax(axis=-1) == y_true
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return {"confidence": float(conf.mean()), "ece": float(ece)}


class TrainingCallback(keras.callbacks.Callback):
    """Records per-epoch metrics on the session, streams them to the job
    console, publishes intra-epoch progress for the live dashboard, and
    honours cancellation at epoch boundaries."""

    def __init__(self, session: dict, trainer: "Trainer", od_eval: Optional[dict] = None,
                 val_eval: Optional[dict] = None, num_train: int = 0):
        super().__init__()
        self.session = session
        self.trainer = trainer
        self.od_eval = od_eval
        self.val_eval = val_eval
        self.num_train = num_train
        self.phase = "train"
        self.epoch_start = 0.0
        self.current_epoch = 0
        self._last_live = 0.0
        self._prev_weights: Optional[List[np.ndarray]] = None

    def on_train_begin(self, logs=None):
        # A new fit() (e.g. phase 2) may train a different set of weights.
        self._prev_weights = [np.array(w) for w in self.model.trainable_weights]

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        self.current_epoch = epoch + 1
        self._last_live = 0.0

    def on_train_batch_end(self, batch, logs=None):
        now = time.time()
        steps = int((self.params or {}).get("steps") or 0)
        last = steps and batch + 1 >= steps
        if now - self._last_live < LIVE_UPDATE_SECONDS and not last:
            return
        self._last_live = now
        logs = logs or {}
        loss = float(logs.get("loss", 0.0))
        acc = logs.get("accuracy", logs.get("acc"))
        point = {"step": round(self.current_epoch - 1 + (batch + 1) / max(1, steps), 4), "loss": loss}
        if acc is not None:
            point["accuracy"] = float(acc)
        s = self.session
        with self.trainer.lock:
            s["live"] = {
                "epoch": self.current_epoch, "batch": batch + 1, "batches": steps,
                "loss": loss, "accuracy": point.get("accuracy"), "phase": self.phase,
                "epoch_elapsed": now - self.epoch_start, "updated_at": now,
            }
            hist = s.setdefault("batch_history", [])
            hist.append(point)
            if len(hist) > BATCH_HISTORY_MAX:
                s["batch_history"] = hist[::2]

    def _od_scores(self, X, Y) -> Dict:
        if X is None or len(X) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        probs = self.model.predict(X, verbose=0, batch_size=64)
        return score_detections(list(Y), list(probs), self.od_eval["num_classes"])

    def _learning_rate(self) -> Optional[float]:
        try:
            return float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        except Exception:
            return None

    def _weight_stats(self) -> Dict:
        cur = [np.array(w) for w in self.model.trainable_weights]
        norm = float(np.sqrt(sum(float(np.sum(np.square(w, dtype=np.float64))) for w in cur)))
        update = None
        if self._prev_weights is not None and len(self._prev_weights) == len(cur):
            delta = sum(float(np.sum(np.square(c.astype(np.float64) - p))) for c, p in zip(cur, self._prev_weights))
            update = float(np.sqrt(delta) / norm) if norm else None
        self._prev_weights = cur
        return {"weight_norm": norm, "update_ratio": update}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        now = time.time()
        s = self.session
        total = s.get("total_epochs", 1)
        done = epoch + 1
        started = s.get("started_at") or now
        elapsed = now - started
        epoch_seconds = now - self.epoch_start
        accuracy = float(logs.get("accuracy", logs.get("acc", 0.0)))
        val_accuracy = float(logs.get("val_accuracy", logs.get("val_acc", 0.0)))
        extra: Dict = {}
        live_eval = None
        if self.od_eval:
            # For detection, "accuracy" means object-level F1 (centroid matching).
            accuracy = self._od_scores(self.od_eval["X_train"], self.od_eval["Y_train"])["f1"]
            val = self._od_scores(self.od_eval["X_val"], self.od_eval["Y_val"])
            val_accuracy = val["f1"]
            extra.update(val_precision=val["precision"], val_recall=val["recall"], val_f1=val["f1"])
            live_eval = {"epoch": done, "per_class": [
                {"label": lbl, **pc} for lbl, pc in zip(self.od_eval.get("labels", []), val["per_class"])]}
        elif self.val_eval and len(self.val_eval["X"]):
            X, y, labels = self.val_eval["X"], self.val_eval["y"], self.val_eval["labels"]
            probs = np.asarray(self.model.predict(X, verbose=0, batch_size=64), dtype=np.float32)
            cm = classification_metrics(y, probs.argmax(axis=-1), labels)
            pcs = cm["per_class"]
            extra.update(
                val_precision=float(np.mean([c["precision"] for c in pcs])),
                val_recall=float(np.mean([c["recall"] for c in pcs])),
                val_f1=cm["macro_f1"],
                **{f"val_{k}": v for k, v in calibration_stats(probs, y).items()},
            )
            live_eval = {"epoch": done, "labels": labels, "confusion_matrix": cm["confusion_matrix"],
                         "per_class": pcs, "num_samples": int(len(X))}

        entry = {
            "epoch": done,
            "accuracy": accuracy,
            "val_accuracy": val_accuracy,
            "loss": float(logs.get("loss", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            **extra,
            "learning_rate": self._learning_rate(),
            "phase": self.phase,
            "time_ms": epoch_seconds * 1000,
            "samples_per_sec": self.num_train / epoch_seconds if epoch_seconds > 0 else None,
            **self._weight_stats(),
            "memory_mb": _memory_mb(),
            "gpu_memory_mb": _gpu_memory_mb(),
            "timestamp": now,
        }
        with self.trainer.lock:
            s["metrics"].append(entry)
            s["current_epoch"] = done
            s["progress"] = int(done / total * 100)
            s["elapsed_seconds"] = elapsed
            s["remaining_seconds"] = max(0.0, elapsed / max(1, len(s["metrics"])) * (total - done))
            if live_eval:
                s["live_eval"] = live_eval

        metric_name = "f1" if self.od_eval else "accuracy"
        msg = (f"Epoch {done}/{total} - loss: {entry['loss']:.4f} - {metric_name}: {accuracy:.4f} - "
               f"val_loss: {entry['val_loss']:.4f} - val_{metric_name}: {val_accuracy:.4f}")
        if "val_f1" in extra and not self.od_eval:
            msg += f" - val_macro_f1: {extra['val_f1']:.4f}"
        if entry["learning_rate"] is not None:
            msg += f" - lr: {entry['learning_rate']:.2e}"
        job_log_broker.log(s["id"], f"{msg} ({epoch_seconds:.1f}s)")
        if s.get("stop_requested"):
            self.model.stop_training = True
            job_log_broker.log(s["id"], "Cancellation requested - stopping after this epoch.", level="warning")
        self.trainer._save_to_disk()


class Trainer:
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = str(storage_dir or config.STORAGE_DIR)
        os.makedirs(self.storage_dir, exist_ok=True)
        self.db_file = os.path.join(self.storage_dir, "training_db.json")
        self.lock = threading.RLock()
        self.training_sessions: Dict[str, dict] = {}
        self.trained_models: Dict[str, dict] = {}
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_to_disk(self):
        with self.lock:
            atomic_write_json(self.db_file, {
                "training_sessions": self.training_sessions,
                "trained_models": self.trained_models,
            })

    def _load_from_disk(self):
        data = read_json(self.db_file, {})
        self.training_sessions = data.get("training_sessions", {})
        self.trained_models = data.get("trained_models", {})
        # A job can't survive a backend restart; don't leave ghost "running"
        # sessions that the UI would try to reattach to forever.
        changed = False
        for s in self.training_sessions.values():
            if s.get("status") in ACTIVE_STATUSES:
                s["status"] = "failed"
                s["error"] = "Interrupted: the backend restarted while this job was queued or running."
                changed = True
        if changed:
            self._save_to_disk()

    @property
    def active_training(self) -> Dict[str, bool]:
        return {k: s.get("status") in ACTIVE_STATUSES for k, s in self.training_sessions.items()}

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_training_session(
        self, task, dataset_id, epochs, batch_size, learning_rate, base_model, input_shape,
        early_stopping: bool = False, early_stopping_patience: int = 5,
        early_stopping_monitor: str = "val_loss", dropout_rate: float = 0.3, l2_reg: float = 0.0,
        trainable_layers: int = 0, freeze_encoder_epochs: Optional[int] = None, augmentation: dict = None,
        device: str = "auto", seed: Optional[int] = None, class_weighting: bool = False,
        name: Optional[str] = None,
    ) -> str:
        if device not in ("auto", "cpu", "gpu"):
            device = "auto"
        base_model = ModelFactory.resolve_name(base_model)
        input_shape = [int(d) for d in (input_shape or preprocessing.default_input_shape(task))]
        if preprocessing.is_audio_task(task) and len(input_shape) == 2:
            input_shape.append(1)
        if task == "OBJECT_DETECTION":
            input_shape = [max(FOMO_STRIDE, d - d % FOMO_STRIDE) for d in input_shape[:2]] + input_shape[2:]
        if freeze_encoder_epochs is None:
            # Auto: warm up the new head on a frozen pretrained backbone before
            # fine-tuning it - fine-tuning everything from epoch 1 with a
            # randomly initialised head wrecks the ImageNet features.
            pretrained = base_model in ("MobileNetV2", "MobileNetV1_0.25", "MobileNetV3Small",
                                        "EfficientNet", "ResNet50V2", "FOMO_MobileNetV2")
            freeze_encoder_epochs = max(1, int(epochs) // 3) if pretrained and int(epochs) > 1 else 0

        session_id = str(uuid.uuid4())
        with self.lock:
            self.training_sessions[session_id] = {
                "id": session_id,
                "name": name or "",
                "task": task,
                "dataset_id": dataset_id,
                "epochs": int(epochs),
                "total_epochs": int(epochs),
                "current_epoch": 0,
                "progress": 0,
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "base_model": base_model,
                "input_shape": input_shape,
                "dropout_rate": float(dropout_rate),
                "l2_reg": float(l2_reg),
                "status": "queued",
                "created_at": time.time(),
                "elapsed_seconds": 0,
                "remaining_seconds": 0,
                "metrics": [],
                "stop_requested": False,
                "early_stopping": bool(early_stopping),
                "early_stopping_patience": int(early_stopping_patience),
                "early_stopping_monitor": early_stopping_monitor,
                "trainable_layers": int(trainable_layers),
                "freeze_encoder_epochs": int(freeze_encoder_epochs),
                "augmentation": augmentation or {},
                "device": device,
                "device_used": None,
                "seed": seed,
                "class_weighting": bool(class_weighting),
                "archived": False,
            }
        self._save_to_disk()
        return session_id

    @staticmethod
    def _resolve_device(preference: str):
        if preference == "cpu":
            return "/CPU:0", "cpu"
        gpus = tf.config.list_physical_devices("GPU")
        if preference == "gpu":
            if gpus:
                return "/GPU:0", "gpu"
            logger.warning("GPU training requested but no GPU is visible; using CPU.")
            return "/CPU:0", "cpu"
        return None, ("gpu" if gpus else "cpu")

    @staticmethod
    def get_available_devices() -> Dict:
        gpus = tf.config.list_physical_devices("GPU")
        details = []
        for gpu in gpus:
            try:
                d = tf.config.experimental.get_device_details(gpu)
                details.append({"name": d.get("device_name", gpu.name), "compute_capability": d.get("compute_capability")})
            except Exception:
                details.append({"name": gpu.name, "compute_capability": None})
        return {"cpu_available": True, "gpu_available": bool(gpus), "gpus": details}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _log(self, training_id: str, msg: str, level: str = "info"):
        job_log_broker.log(training_id, msg, level=level)

    def train(self, training_id: str):
        session = self.training_sessions.get(training_id)
        if session is None:
            return
        if session.get("stop_requested"):
            with self.lock:
                session["status"] = "cancelled"
                session["completed_at"] = time.time()
            self._save_to_disk()
            self._log(training_id, "Cancelled before it started.", "warning")
            return

        original_policy = keras.mixed_precision.global_policy()
        try:
            with self.lock:
                session["status"] = "running"
                session["started_at"] = time.time()
            self._save_to_disk()
            self._run_training(training_id, session)
        except Exception as e:
            with self.lock:
                session["status"] = "failed"
                session["error"] = str(e)
                session["completed_at"] = time.time()
            self._save_to_disk()
            self._log(training_id, f"Training FAILED: {e}", "error")
            self._log(training_id, traceback.format_exc(), "error")
        finally:
            keras.mixed_precision.set_global_policy(original_policy)

    def _run_training(self, training_id: str, session: dict):
        from app.services.shared_state import data_manager

        task = session["task"]
        dataset_id = session["dataset_id"]
        input_shape = tuple(session["input_shape"])
        is_od = task == "OBJECT_DETECTION"

        if session.get("seed") is not None:
            keras.utils.set_random_seed(int(session["seed"]))

        self._log(training_id, f"Task: {task} | Model: {session['base_model']} | Input: {list(input_shape)}")
        if not data_manager.is_split_ready(dataset_id):
            raise ValueError("Dataset has no complete train/val split. Split it via Auto Split before training.")

        if is_od:
            labels = dataset_loader.detection_classes(dataset_id)
            if not labels:
                raise ValueError("No bounding-box annotations found. Import or draw annotations before training a detector.")
        else:
            labels, dropped = dataset_loader.classification_labels(dataset_id)
            if dropped:
                self._log(training_id, f"Ignoring labels with no training samples: {dropped}", "warning")
            if len(labels) < 2:
                raise ValueError(f"Need at least 2 classes with training samples, found {labels}.")
        self._log(training_id, f"{len(labels)} classes: {labels}")

        skip = lambda m: self._log(training_id, m, "warning")  # noqa: E731
        X_train, y_train, _ = dataset_loader.load_split(dataset_id, task, input_shape, labels, "train", on_skip=skip)
        X_val, y_val, _ = dataset_loader.load_split(dataset_id, task, input_shape, labels, "val", on_skip=skip)
        if len(X_train) == 0:
            raise ValueError("No readable training samples.")
        self._log(training_id, f"Loaded {len(X_train)} train / {len(X_val)} val samples.")

        device_str, device_used = self._resolve_device(session.get("device", "auto"))
        with self.lock:
            session["device_used"] = device_used
        self._log(training_id, f"Compute device: {device_used} (requested: {session.get('device', 'auto')})")
        policy = keras.mixed_precision.global_policy().name
        if device_used == "cpu" and policy == "mixed_float16":
            keras.mixed_precision.set_global_policy("float32")
        mixed = keras.mixed_precision.global_policy().name == "mixed_float16"

        build_kwargs = dict(
            task=task, num_classes=len(labels), base_model=session["base_model"], input_shape=input_shape,
            dropout_rate=session.get("dropout_rate", 0.3), l2_reg=session.get("l2_reg", 0.0),
            trainable_layers=session.get("trainable_layers", 0),
        )

        device_ctx = tf.device(device_str) if device_str else contextlib.nullcontext()
        with device_ctx:
            core = ModelFactory.create_model(**build_kwargs)
            inp = keras.Input(shape=input_shape)
            x = inp
            aug = None if preprocessing.is_audio_task(task) or is_od else ModelFactory.build_augmentation(session.get("augmentation"))
            if aug is not None:
                x = aug(x)
                self._log(training_id, f"Augmentation: {[l.name for l in aug.layers]}")
            # Train through a thin wrapper so the saved `core` model never
            # carries augmentation layers or optimizer state.
            train_model = keras.Model(inp, core(x))

            loss = make_fomo_loss() if is_od else "sparse_categorical_crossentropy"
            metrics = [] if is_od else ["accuracy"]

            od_eval = val_eval = None
            if is_od:
                od_eval = {"num_classes": len(labels), "labels": labels, "X_train": X_train[:200],
                           "Y_train": y_train[:200], "X_val": X_val, "Y_val": y_val}
            else:
                val_eval = {"X": X_val[:LIVE_EVAL_MAX_SAMPLES], "y": y_val[:LIVE_EVAL_MAX_SAMPLES], "labels": labels}
            progress_cb = TrainingCallback(session, self, od_eval, val_eval, num_train=len(X_train))
            callbacks: list = [progress_cb]
            if session.get("early_stopping"):
                monitor = session.get("early_stopping_monitor", "val_loss")
                if is_od and "accuracy" in monitor:
                    monitor = "val_loss"
                callbacks.append(keras.callbacks.EarlyStopping(
                    monitor=monitor, patience=session.get("early_stopping_patience", 5),
                    restore_best_weights=True, verbose=0,
                ))

            class_weight = None
            if session.get("class_weighting") and not is_od:
                counts = np.bincount(y_train, minlength=len(labels)).astype(float)
                class_weight = {i: float(len(y_train) / (len(labels) * c)) for i, c in enumerate(counts) if c > 0}
                self._log(training_id, f"Class weights: {class_weight}")

            fit_kwargs = dict(
                x=X_train, y=y_train, batch_size=session["batch_size"],
                validation_data=(X_val, y_val) if len(X_val) else None,
                callbacks=callbacks, verbose=0, shuffle=True, class_weight=class_weight,
            )
            total_epochs = int(session["epochs"])
            freeze_epochs = min(int(session.get("freeze_encoder_epochs", 0)), total_epochs)
            backbone = ModelFactory.get_backbone(core)
            lr = float(session["learning_rate"])

            batch_size = int(session["batch_size"])
            with self.lock:
                session["run_info"] = {
                    "num_train": int(len(X_train)), "num_val": int(len(X_val)), "num_classes": len(labels),
                    "labels": labels, "steps_per_epoch": int(np.ceil(len(X_train) / batch_size)),
                    "params_total": int(core.count_params()),
                    "params_trainable": int(sum(int(np.prod(w.shape)) for w in core.trainable_weights)),
                    "train_class_counts": (
                        None if is_od else {labels[i]: int(c) for i, c in
                                            enumerate(np.bincount(y_train, minlength=len(labels)))}),
                }
                session["batch_history"] = []

            if freeze_epochs > 0 and backbone is not None:
                self._log(training_id, f"Phase 1: training the head for {freeze_epochs} epochs (backbone frozen).")
                backbone.trainable = False
                progress_cb.phase = "head"
                train_model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=metrics)
                train_model.fit(epochs=freeze_epochs, **fit_kwargs)
                if not session.get("stop_requested") and total_epochs > freeze_epochs:
                    self._log(training_id, "Phase 2: fine-tuning the backbone at lr x0.1.")
                    set_backbone_trainable(backbone, int(session.get("trainable_layers", 0)))
                    progress_cb.phase = "fine-tune"
                    train_model.compile(optimizer=keras.optimizers.Adam(lr * 0.1), loss=loss, metrics=metrics)
                    # initial_epoch keeps epoch numbering / progress continuous across phases.
                    train_model.fit(epochs=total_epochs, initial_epoch=freeze_epochs, **fit_kwargs)
            else:
                train_model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=metrics)
                train_model.fit(epochs=total_epochs, **fit_kwargs)

        with self.lock:
            session.pop("live", None)
        was_cancelled = bool(session.get("stop_requested"))
        if was_cancelled and not session["metrics"]:
            with self.lock:
                session["status"] = "cancelled"
                session["completed_at"] = time.time()
            self._save_to_disk()
            self._log(training_id, "Cancelled before any epoch completed - nothing saved.", "warning")
            return

        # Mixed-precision layers don't convert cleanly to INT8 TFLite; save a
        # float32 copy of the network instead.
        if mixed:
            keras.mixed_precision.set_global_policy("float32")
            fp32 = ModelFactory.create_model(**build_kwargs)
            fp32.set_weights(core.get_weights())
            core = fp32

        save_path = os.path.join(self.storage_dir, f"{training_id}.keras")
        core.save(save_path)

        evaluation = None
        try:
            evaluation = self._evaluate_holdout(core, task, dataset_id, input_shape, labels)
            if evaluation:
                headline = evaluation.get("f1") if is_od else evaluation.get("accuracy")
                self._log(training_id, f"Held-out {evaluation['split']} {'F1' if is_od else 'accuracy'}: {headline}")
        except Exception as exc:
            self._log(training_id, f"Held-out evaluation skipped: {exc}", "warning")

        last = session["metrics"][-1] if session["metrics"] else {}
        dataset_name = (data_manager.get_dataset(dataset_id) or {}).get("name")
        model_id = str(uuid.uuid4())
        with self.lock:
            session["status"] = "cancelled" if was_cancelled else "completed"
            session["completed_at"] = time.time()
            session["labels"] = labels
            session["evaluation"] = evaluation
            self.trained_models[model_id] = {
                "id": model_id,
                "name": (session.get("name") or f"{session['base_model']} Trained")
                + (" (cancelled - partial)" if was_cancelled else ""),
                "training_id": training_id,
                "task": task,
                "base_model": session["base_model"],
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "device_used": session.get("device_used"),
                "created_at": time.time(),
                "accuracy": last.get("accuracy", 0.0),
                "val_accuracy": last.get("val_accuracy", 0.0),
                "loss": last.get("loss", 0.0),
                "val_loss": last.get("val_loss", 0.0),
                "test_accuracy": (evaluation or {}).get("f1" if is_od else "accuracy"),
                "size_bytes": os.path.getsize(save_path),
                "params": int(core.count_params()),
                "optimized": False,
                "path": save_path,
                "type": "audio" if preprocessing.is_audio_task(task) else "image",
                "labels": labels,
                "input_shape": list(input_shape),
            }
        self._save_to_disk()
        if was_cancelled:
            self._log(training_id, f"Cancelled after {len(session['metrics'])} epoch(s). Partial model saved.", "warning")
        else:
            self._log(training_id, f"Training completed. Model saved to {save_path}")

    @staticmethod
    def _evaluate_holdout(model, task, dataset_id, input_shape, labels) -> Optional[Dict]:
        split = dataset_loader.evaluation_split(dataset_id)
        X, y, _ = dataset_loader.load_split(dataset_id, task, input_shape, labels, split, max_samples=1000)
        if len(X) == 0:
            return None
        probs = model.predict(X, verbose=0, batch_size=64)
        if task == "OBJECT_DETECTION":
            result = score_detections(list(y), list(probs), len(labels))
            result["labels"] = labels
        else:
            result = classification_metrics(y, probs.argmax(axis=-1), labels)
        result["split"] = split
        result["num_samples"] = int(len(X))
        return result

    # ------------------------------------------------------------------
    # Queries / management
    # ------------------------------------------------------------------

    def get_training_status(self, training_id: str) -> dict:
        s = self.training_sessions.get(training_id, {})
        if s and s.get("status") == "queued":
            from app.services.job_queue import job_queue

            return {**s, "queue_position": job_queue.position(training_id)}
        return s

    def get_training_metrics(self, training_id: str) -> List[dict]:
        return self.training_sessions.get(training_id, {}).get("metrics", [])

    def get_all_sessions(self, include_archived: bool = False) -> List[dict]:
        with self.lock:
            sessions = list(self.training_sessions.values())
        if not include_archived:
            sessions = [s for s in sessions if not s.get("archived", False)]
        return sorted(sessions, key=lambda s: s.get("created_at", 0), reverse=True)

    def cancel_training(self, training_id: str) -> bool:
        """Request cancellation. A queued job is skipped when its turn comes;
        a running job stops at the end of the current epoch."""
        session = self.training_sessions.get(training_id)
        if not session or session.get("status") not in ACTIVE_STATUSES:
            return False
        with self.lock:
            session["stop_requested"] = True
            if session.get("status") == "queued":
                session["status"] = "cancelled"
                session["completed_at"] = time.time()
        self._save_to_disk()
        return True

    def archive_training(self, training_id: str) -> bool:
        return self._set_archived(training_id, True)

    def unarchive_training(self, training_id: str) -> bool:
        return self._set_archived(training_id, False)

    def _set_archived(self, training_id: str, value: bool) -> bool:
        if training_id not in self.training_sessions:
            return False
        with self.lock:
            self.training_sessions[training_id]["archived"] = value
        self._save_to_disk()
        return True

    def rename_training(self, training_id: str, name: str) -> bool:
        if training_id not in self.training_sessions:
            return False
        with self.lock:
            self.training_sessions[training_id]["name"] = name
            for m in self.trained_models.values():
                if m.get("training_id") == training_id:
                    m["name"] = name
        self._save_to_disk()
        return True

    def delete_training_session(self, training_id: str) -> bool:
        if training_id not in self.training_sessions:
            return False
        if self.training_sessions[training_id].get("status") in ("initialized", "running"):
            raise ValueError("Cannot delete a training session that is still running - cancel it first.")
        model_path = os.path.join(self.storage_dir, f"{training_id}.keras")
        with contextlib.suppress(OSError):
            os.remove(model_path)
        with self.lock:
            for mid in [m for m, v in self.trained_models.items() if v.get("training_id") == training_id]:
                del self.trained_models[mid]
            del self.training_sessions[training_id]
        self._save_to_disk()
        return True

    def get_trained_models(self, include_archived: bool = False) -> List[dict]:
        with self.lock:
            models = list(self.trained_models.values())
        if include_archived:
            return models
        return [m for m in models
                if not self.training_sessions.get(m.get("training_id"), {}).get("archived", False)]

    def delete_trained_model(self, model_id: str) -> bool:
        if model_id not in self.trained_models:
            return False
        with self.lock:
            del self.trained_models[model_id]
        self._save_to_disk()
        return True
