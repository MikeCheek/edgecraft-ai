import os
import json
import uuid
import time
import logging
import contextlib
from typing import Dict, List, Optional

import tensorflow as tf
import numpy as np
import shutil

from app.services.model_factory import ModelFactory
from app.utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)

# tf.data pipeline constant (audio pipeline only)


class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, session: dict, trainer):
        super().__init__()
        self.session = session
        self.trainer = trainer
        self.epoch_start_time = 0
        self.training_start_time = 0

    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()
        self.session["started_at"] = self.training_start_time

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        now = time.time()
        epoch_duration = now - self.epoch_start_time
        elapsed = now - self.training_start_time
        total_epochs = self.session.get("total_epochs", 1)
        epochs_done = epoch + 1
        avg_epoch_time = elapsed / epochs_done
        remaining = avg_epoch_time * (total_epochs - epochs_done)

        self.session["current_epoch"] = epochs_done
        self.session["progress"] = int((epochs_done / total_epochs) * 100)
        self.session["elapsed_seconds"] = elapsed
        self.session["remaining_seconds"] = max(0, remaining)

        metric_entry = {
            "epoch": epochs_done,
            # mixed_precision wraps metric names; fall back gracefully
            "accuracy":     float(logs.get("accuracy",     logs.get("acc",     0.0))),
            "val_accuracy": float(logs.get("val_accuracy", logs.get("val_acc", 0.0))),
            "loss":         float(logs.get("loss",     0.0)),
            "val_loss":     float(logs.get("val_loss", 0.0)),
            "time_ms":      epoch_duration * 1000,
        }
        self.session["metrics"].append(metric_entry)

        if self.session.get("stop_requested", False):
            self.model.stop_training = True

        self.trainer._save_to_disk()


class Trainer:
    def __init__(self, storage_dir: str = "data_storage"):
        self.storage_dir = storage_dir
        self.db_file = os.path.join(storage_dir, "training_db.json")

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        self.training_sessions: Dict[str, dict] = {}
        self.trained_models: Dict[str, dict] = {}
        self.active_training: Dict[str, bool] = {}

        self._load_from_disk()

    def _save_to_disk(self):
        with open(self.db_file, "w") as f:
            json.dump({
                "training_sessions": self.training_sessions,
                "trained_models":    self.trained_models,
            }, f, indent=2)

    def _load_from_disk(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                data = json.load(f)
                self.training_sessions = data.get("training_sessions", {})
                self.trained_models    = data.get("trained_models",    {})

    def create_training_session(
        self,
        task,
        dataset_id,
        epochs,
        batch_size,
        learning_rate,
        base_model,
        input_shape,
        early_stopping: bool = False,
        early_stopping_patience: int = 5,
        early_stopping_monitor: str = "val_loss",
        dropout_rate: float = 0.5,
        l2_reg: float = 0.0,
        trainable_layers: int = 0,
        freeze_encoder_epochs: int = 0,
        augmentation: dict = None,
        device: str = "auto",
    ):
        if augmentation is None:
            augmentation = {}
        if device not in ("auto", "cpu", "gpu"):
            device = "auto"

        session_id = str(uuid.uuid4())
        self.training_sessions[session_id] = {
            "id":                       session_id,
            "task":                     task,
            "dataset_id":               dataset_id,
            "epochs":                   epochs,
            "total_epochs":             epochs,
            "current_epoch":            0,
            "progress":                 0,
            "batch_size":               batch_size,
            "learning_rate":            learning_rate,
            "base_model":               base_model,
            "input_shape":              input_shape,
            "dropout_rate":             dropout_rate,
            "l2_reg":                   l2_reg,
            "status":                   "initialized",
            "created_at":               time.time(),
            "elapsed_seconds":          0,
            "remaining_seconds":        0,
            "metrics":                  [],
            "stop_requested":           False,
            "early_stopping":           early_stopping,
            "early_stopping_patience":  early_stopping_patience,
            "early_stopping_monitor":   early_stopping_monitor,
            "trainable_layers":         trainable_layers,
            "freeze_encoder_epochs":    freeze_encoder_epochs,
            "augmentation":             augmentation,
            "device":                   device,       # "auto" | "cpu" | "gpu" - user's requested compute target
            "device_used":              None,          # filled in once train() resolves it (e.g. GPU requested but unavailable)
            "archived":                 False,
        }
        self.active_training[session_id] = False
        self._save_to_disk()
        return session_id

    # -------------------------------------------------------------------------
    # Compute device selection
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_device(preference: str) -> "tuple[Optional[str], str]":
        """
        Resolve a user's device preference ("auto" | "cpu" | "gpu") into a
        concrete tf.device() string (or None to let TF place automatically)
        plus the device actually used, so it can be recorded on the session.
        Falls back to CPU with a warning if GPU was requested but none is
        visible to TensorFlow.
        """
        if preference == "cpu":
            return "/CPU:0", "cpu"

        gpus = tf.config.list_physical_devices("GPU")
        if preference == "gpu":
            if gpus:
                return "/GPU:0", "gpu"
            logger.warning("GPU training requested but no GPU is visible to TensorFlow; using CPU instead.")
            return "/CPU:0", "cpu"

        # "auto": don't force placement, let TF prefer GPU if present
        return None, ("gpu" if gpus else "cpu")

    @staticmethod
    def get_available_devices() -> Dict:
        """Used by GET /api/training/devices so the frontend can show which
        options are actually usable (and disable "GPU" if none exists)."""
        gpus = tf.config.list_physical_devices("GPU")
        gpu_details = []
        for gpu in gpus:
            try:
                details = tf.config.experimental.get_device_details(gpu)
                gpu_details.append({
                    "name": details.get("device_name", gpu.name),
                    "compute_capability": details.get("compute_capability"),
                })
            except Exception:
                gpu_details.append({"name": gpu.name, "compute_capability": None})

        return {
            "cpu_available": True,
            "gpu_available": len(gpus) > 0,
            "gpus": gpu_details,
        }

    # -------------------------------------------------------------------------
    # Dataset export
    # -------------------------------------------------------------------------

    def _export_dataset_to_temp_dir(self, dataset_id: str, task: str) -> tuple:
        """Export samples into temp_dir/train/<label>/… and temp_dir/val/<label>/…"""
        from app.services.shared_state import data_manager
        from PIL import Image
        import io

        temp_dir  = os.path.join(self.storage_dir, f"temp_{dataset_id}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        train_dir = os.path.join(temp_dir, "train")
        val_dir   = os.path.join(temp_dir, "val")
        os.makedirs(train_dir)
        os.makedirs(val_dir)

        samples = data_manager.get_samples(dataset_id)
        labels  = data_manager.get_dataset_labels(dataset_id)

        # Pre-create every class folder on both sides so image_dataset_from_directory
        # sees identical, consistently-ordered class lists even if a class has no
        # examples in one of the two splits.
        for split_dir in (train_dir, val_dir):
            for label in labels:
                os.makedirs(os.path.join(split_dir, label), exist_ok=True)

        name_counters: Dict[tuple, Dict[str, int]] = {}
        ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "GIF", "BMP"}

        for sample in samples:
            split = sample.get("split", "unassigned")
            if split not in ("train", "val"):
                continue

            label     = sample["label"]
            split_dir = train_dir if split == "train" else val_dir
            label_dir = os.path.join(split_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            data = data_manager.get_sample_data(sample["id"])
            if not data:
                continue

            if task in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "VISUAL_WAKE_WORDS"]:
                try:
                    with Image.open(io.BytesIO(data)) as img:
                        img.verify()
                    with Image.open(io.BytesIO(data)) as img:
                        if img.format not in ALLOWED_IMAGE_FORMATS:
                            rgb_im = img.convert("RGB")
                            buf = io.BytesIO()
                            rgb_im.save(buf, format="JPEG")
                            data = buf.getvalue()
                            stem, _ = os.path.splitext(sample["filename"])
                            sample["filename"] = f"{stem}.jpg"
                except Exception as e:
                    print(f"Skipping corrupt image: {sample.get('filename', 'unknown')} — {e}")
                    continue

            safe_name   = os.path.basename(sample["filename"])
            counter_key = (split, label)
            if counter_key not in name_counters:
                name_counters[counter_key] = {}

            if safe_name in name_counters[counter_key]:
                name_counters[counter_key][safe_name] += 1
                stem, ext = os.path.splitext(safe_name)
                safe_name = f"{stem}_{name_counters[counter_key][safe_name]}{ext}"
            else:
                name_counters[counter_key][safe_name] = 0

            with open(os.path.join(label_dir, safe_name), "wb") as f:
                f.write(data)

        return train_dir, val_dir

    # -------------------------------------------------------------------------
    # tf.data pipelines
    # -------------------------------------------------------------------------

    def _build_image_pipeline(
        self, train_dir: str, val_dir: str, session: dict, class_names: list
    ):
        """
        Load all images into numpy arrays then wrap in tf.data.

        image_dataset_from_directory (and any tf.data source backed by file I/O)
        spins up internal reader threads bound to the TF session of the calling
        OS thread. When model.fit() later runs on FastAPI's BackgroundTasks thread
        the sessions don't match and TF raises:
            "stream cannot wait for other [[node IteratorGetNext/_4]]"

        Pre-loading into numpy arrays (same approach as _build_audio_pipeline)
        sidesteps this entirely: from_tensor_slices has no background threads.
        """
        from PIL import Image as PILImage
        import io as _io

        img_h      = session["input_shape"][0]
        img_w      = session["input_shape"][1]
        channels   = session["input_shape"][2]
        batch_size = session["batch_size"]
        class_map  = {name: idx for idx, name in enumerate(class_names)}

        def _load(split_dir):
            images, labs = [], []
            for class_name in class_names:
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    fpath = os.path.join(class_dir, fname)
                    try:
                        with PILImage.open(fpath) as img:
                            img = img.convert("L" if channels == 1 else "RGB")
                            img = img.resize((img_w, img_h), PILImage.BILINEAR)
                            arr = np.array(img, dtype=np.float32) / 255.0
                            if channels == 1:
                                arr = arr[:, :, np.newaxis]
                    except Exception as e:
                        print(f"Skipping {fpath}: {e}")
                        continue
                    images.append(arr)
                    labs.append(class_map[class_name])
            return np.array(images, dtype=np.float32), np.array(labs, dtype=np.int32)

        X_train, y_train = _load(train_dir)
        X_val,   y_val   = _load(val_dir)

        if len(X_train) > 0:
            idx = np.random.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .batch(batch_size)
            .prefetch(1)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(batch_size)
            .prefetch(1)
        )

        return train_ds, val_ds

    def _build_audio_pipeline(
        self,
        train_dir: str,
        val_dir: str,
        batch_size: int,
        task: str,
        class_names: list,
        input_shape: tuple,
    ):
        """Audio pre-loaded fully into numpy arrays — no threading issues."""
        class_map       = {name: idx for idx, name in enumerate(class_names)}
        expected_n_mfcc = input_shape[0]
        max_time_steps  = input_shape[1]

        def _load(split_dir):
            features, labels = [], []
            for class_name in class_names:
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    if not fname.endswith((".wav", ".mp3")):
                        continue
                    with open(os.path.join(class_dir, fname), "rb") as f:
                        audio_data = f.read()

                    mfcc = DataProcessor.preprocess_audio(audio_data, task)
                    mfcc = np.squeeze(mfcc)  # collapse any extra dims → (n_mfcc, time)

                    if mfcc.ndim != 2:
                        print(f"Skipping {fname}: unexpected MFCC shape {mfcc.shape}")
                        continue

                    actual_n_mfcc, actual_time = mfcc.shape

                    if actual_n_mfcc != expected_n_mfcc:
                        print(f"Skipping {fname}: expected {expected_n_mfcc} MFCCs, got {actual_n_mfcc}")
                        continue

                    # Pad or truncate time axis
                    if actual_time > max_time_steps:
                        mfcc = mfcc[:, :max_time_steps]
                    elif actual_time < max_time_steps:
                        mfcc = np.pad(mfcc, ((0, 0), (0, max_time_steps - actual_time)), mode="constant")

                    features.append(np.expand_dims(mfcc, axis=-1))  # (n_mfcc, time, 1)
                    labels.append(class_map[class_name])

            return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)

        X_train, y_train = _load(train_dir)
        X_val,   y_val   = _load(val_dir)

        if len(X_train) > 0:
            idx = np.random.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .batch(batch_size)
            .prefetch(1)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(batch_size)
            .prefetch(1)
        )
        return train_ds, val_ds

    # -------------------------------------------------------------------------
    # Main training entry point
    # -------------------------------------------------------------------------

    def train(self, training_id: str):
        try:
            session = self.training_sessions[training_id]
            self.active_training[training_id] = True
            session["status"] = "running"
            self._save_to_disk()

            task       = session["task"]
            dataset_id = session["dataset_id"]

            from app.services.shared_state import data_manager
            if not data_manager.is_split_ready(dataset_id):
                raise ValueError(
                    "Dataset has no complete train/val split. "
                    "Split it via Auto Split before starting training."
                )

            classes     = data_manager.get_dataset_labels(dataset_id)
            num_classes = len(classes)

            train_dir, val_dir = self._export_dataset_to_temp_dir(dataset_id, task)
            temp_dir = os.path.dirname(train_dir)

            if task in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "VISUAL_WAKE_WORDS"]:
                train_ds, val_ds = self._build_image_pipeline(train_dir, val_dir, session, classes)
                model_input_shape = tuple(session["input_shape"])

            elif task in ["AUDIO_CLASSIFICATION", "KEYWORD_SPOTTING"]:
                audio_params = DataProcessor.AUDIO_PARAMS.get(
                    task, DataProcessor.AUDIO_PARAMS["KEYWORD_SPOTTING"]
                )
                if session.get("input_shape"):
                    raw_shape         = tuple(session["input_shape"])
                    model_input_shape = raw_shape[:2]  # (n_mfcc, time_steps)
                else:
                    model_input_shape = (audio_params["n_mfcc"], 101)

                train_ds, val_ds = self._build_audio_pipeline(
                    train_dir, val_dir, session["batch_size"], task, classes, model_input_shape
                )
            else:
                raise ValueError(f"Unknown task: {task}")

            # --- Resolve compute device -------------------------------------
            # For small models there's often no benefit (or a net loss, due to
            # host<->device transfer overhead) to training on GPU, so the user
            # can force CPU explicitly. "auto" leaves TensorFlow's normal
            # placement behavior untouched (prefers GPU if one is visible).
            device_pref = session.get("device", "auto")
            device_str, device_used = self._resolve_device(device_pref)
            session["device_used"] = device_used
            self._save_to_disk()

            # mixed_float16 is a GPU-oriented optimization (it relies on
            # Tensor Cores for the speedup); forcing CPU while that policy is
            # still active just adds pointless cast ops. Temporarily switch
            # to float32 for the duration of this training run when the user
            # asked for CPU explicitly, and restore whatever policy was
            # active afterwards.
            original_policy = tf.keras.mixed_precision.global_policy()
            switched_policy = False
            if device_used == "cpu" and original_policy.name == "mixed_float16":
                tf.keras.mixed_precision.set_global_policy("float32")
                switched_policy = True

            device_ctx = tf.device(device_str) if device_str else contextlib.nullcontext()

            try:
                with device_ctx:
                    model = ModelFactory.create_model(
                        task=task,
                        num_classes=num_classes,
                        base_model=session["base_model"],
                        input_shape=model_input_shape,
                        dropout_rate=session.get("dropout_rate",     0.5),
                        l2_reg=session.get("l2_reg",                 0.0),
                        trainable_layers=session.get("trainable_layers", 0),
                        augmentation=session.get("augmentation",      {}),
                    )

                    # When mixed_precision is active cast the output to float32 to
                    # avoid numerical instability with softmax in float16.
                    policy = tf.keras.mixed_precision.global_policy()
                    if policy.name == "mixed_float16":
                        inputs  = model.input
                        outputs = tf.cast(model.output, tf.float32)
                        model   = tf.keras.Model(inputs, outputs)

                    loss_fn = (
                        "binary_crossentropy"
                        if task == "VISUAL_WAKE_WORDS"
                        else "sparse_categorical_crossentropy"
                    )

                    callbacks: list = [TrainingCallback(session, self)]
                    if session.get("early_stopping", False):
                        callbacks.append(
                            tf.keras.callbacks.EarlyStopping(
                                monitor=session.get("early_stopping_monitor", "val_loss"),
                                patience=session.get("early_stopping_patience", 5),
                                restore_best_weights=True,
                                verbose=0,
                            )
                        )

                    freeze_epochs    = session.get("freeze_encoder_epochs", 0)
                    trainable_layers = session.get("trainable_layers",       0)
                    total_epochs     = session.get("epochs",                 50)
                    lr               = session["learning_rate"]

                    # Find the pretrained backbone by type instead of assuming
                    # a fixed layer index - ModelFactory may insert an
                    # optional augmentation layer AND an optional channel-
                    # adapter layer before the backbone, so "index 1" isn't
                    # reliable. The backbone is itself a nested Keras Model
                    # (e.g. MobileNetV2/V3Small/EfficientNet/ResNet50V2),
                    # which is how we identify it regardless of position.
                    base_model_layer = next(
                        (l for l in model.layers if isinstance(l, tf.keras.Model)), None
                    )

                    if freeze_epochs > 0 and base_model_layer:
                        # Phase 1: train head only
                        base_model_layer.trainable = False
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(lr),
                            loss=loss_fn, metrics=["accuracy"],
                        )
                        model.fit(
                            train_ds, validation_data=val_ds,
                            epochs=freeze_epochs, callbacks=callbacks,
                        )

                        # Phase 2: fine-tune backbone
                        base_model_layer.trainable = True
                        if trainable_layers > 0:
                            for layer in base_model_layer.layers[:-trainable_layers]:
                                layer.trainable = False
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(lr * 0.1),
                            loss=loss_fn, metrics=["accuracy"],
                        )
                        remaining = total_epochs - freeze_epochs
                        if remaining > 0:
                            model.fit(
                                train_ds, validation_data=val_ds,
                                epochs=remaining, callbacks=callbacks,
                            )
                    else:
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(lr),
                            loss=loss_fn, metrics=["accuracy"],
                        )
                        model.fit(
                            train_ds, validation_data=val_ds,
                            epochs=total_epochs, callbacks=callbacks,
                        )
            finally:
                if switched_policy:
                    tf.keras.mixed_precision.set_global_policy(original_policy)

            # Save in modern .keras format
            save_path = os.path.join(self.storage_dir, f"{training_id}.keras")

            was_cancelled = session.get("stop_requested", False)
            has_any_metrics = len(session.get("metrics", [])) > 0

            if was_cancelled and not has_any_metrics:
                # Cancelled before completing even one epoch - nothing
                # meaningful to save or register as a usable model.
                shutil.rmtree(temp_dir, ignore_errors=True)
                session["status"] = "cancelled"
                session["completed_at"] = time.time()
                self._save_to_disk()
                return

            model.save(save_path)
            shutil.rmtree(temp_dir)

            # Previously this was set unconditionally to "completed", so a
            # training run stopped via cancel_training() (which only takes
            # effect at the next epoch boundary) was silently reported as
            # having finished successfully instead of as cancelled.
            session["status"]       = "cancelled" if was_cancelled else "completed"
            session["completed_at"] = time.time()

            model_id     = str(uuid.uuid4())
            last_metrics = session["metrics"][-1] if session["metrics"] else {}

            if task in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "VISUAL_WAKE_WORDS"]:
                model_type = "image"
            elif task in ["AUDIO_CLASSIFICATION", "KEYWORD_SPOTTING"]:
                model_type = "audio"
            elif task == "TABULAR_CLASSIFICATION":
                model_type = "tabular"
            else:
                model_type = "text"

            # Look up the dataset's display name for convenience - dataset_id
            # alone isn't very readable in the UI's model picker.
            dataset_name = None
            try:
                from app.services.shared_state import data_manager as _dm
                ds_info = _dm.get_dataset(dataset_id)
                dataset_name = ds_info.get("name") if ds_info else None
            except Exception:
                pass

            self.trained_models[model_id] = {
                "id":           model_id,
                "name":         f"{session['base_model']} Trained" + (" (cancelled - partial)" if was_cancelled else ""),
                "training_id":  training_id,
                "task":         task,
                "dataset_id":   dataset_id,
                "dataset_name": dataset_name,
                "device_used":  session.get("device_used"),
                "accuracy":     last_metrics.get("accuracy",     0.0),
                "val_accuracy": last_metrics.get("val_accuracy", 0.0),
                "loss":         last_metrics.get("loss",         0.0),
                "val_loss":     last_metrics.get("val_loss",     0.0),
                "size_bytes":   os.path.getsize(save_path),
                "optimized":    False,
                "path":         save_path,
                "type":         model_type,
                "labels":       classes,
            }
            self._save_to_disk()

        except Exception as e:
            session = self.training_sessions.get(training_id, {})
            session["status"] = "failed"
            session["error"]  = str(e)
            self._save_to_disk()
        finally:
            self.active_training[training_id] = False

    # -------------------------------------------------------------------------
    # Public helpers
    # -------------------------------------------------------------------------

    def get_training_status(self, training_id: str) -> dict:
        return self.training_sessions.get(training_id, {})

    def get_training_metrics(self, training_id: str) -> List[dict]:
        return self.training_sessions.get(training_id, {}).get("metrics", [])

    def get_all_sessions(self, include_archived: bool = False) -> List[dict]:
        sessions = list(self.training_sessions.values())
        if not include_archived:
            sessions = [s for s in sessions if not s.get("archived", False)]
        return sorted(sessions, key=lambda s: s.get("created_at", 0), reverse=True)

    def cancel_training(self, training_id: str) -> bool:
        """
        Request cancellation of a running/queued training session.

        NOTE: this only *requests* a stop - TrainingCallback checks
        stop_requested at the END of the current epoch (Keras has no clean
        way to interrupt mid-epoch), so the session will show status
        "running" for a little longer before actually transitioning to
        "cancelled". Previously this also force-set active_training=False
        immediately, which made the UI think training had stopped even
        though the background thread was still executing the current epoch.
        """
        session = self.training_sessions.get(training_id)
        if not session:
            return False
        if session.get("status") not in ("initialized", "running"):
            return False  # already finished/failed/cancelled - nothing to do
        session["stop_requested"] = True
        self._save_to_disk()
        return True

    def archive_training(self, training_id: str) -> bool:
        """Hide a session from the default history view without deleting
        its data - it can still be found via include_archived=True."""
        if training_id in self.training_sessions:
            self.training_sessions[training_id]["archived"] = True
            self._save_to_disk()
            return True
        return False

    def unarchive_training(self, training_id: str) -> bool:
        if training_id in self.training_sessions:
            self.training_sessions[training_id]["archived"] = False
            self._save_to_disk()
            return True
        return False

    def delete_training_session(self, training_id: str) -> bool:
        """Permanently remove a training session record (and its saved
        .keras file, if any). Does not touch other training_ids' models."""
        if training_id not in self.training_sessions:
            return False
        if self.active_training.get(training_id):
            raise ValueError("Cannot delete a training session that is still running - cancel it first.")

        model_path = os.path.join(self.storage_dir, f"{training_id}.keras")
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except OSError:
                pass

        # Also remove any trained_models entries pointing at this training_id
        stale_model_ids = [
            mid for mid, m in self.trained_models.items() if m.get("training_id") == training_id
        ]
        for mid in stale_model_ids:
            del self.trained_models[mid]

        del self.training_sessions[training_id]
        self.active_training.pop(training_id, None)
        self._save_to_disk()
        return True

    def get_trained_models(self) -> List[dict]:
        return list(self.trained_models.values())

    def delete_trained_model(self, model_id: str) -> bool:
        if model_id in self.trained_models:
            del self.trained_models[model_id]
            self._save_to_disk()
            return True
        return False