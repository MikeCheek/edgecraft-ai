import os
import json
import uuid
import time
from typing import Dict, List, Optional

import tensorflow as tf
import numpy as np
import shutil

from app.services.model_factory import ModelFactory
from app.utils.data_processor import DataProcessor

# tf.data pipeline tuning constant
_AUTOTUNE = tf.data.AUTOTUNE


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
            "accuracy": float(logs.get("accuracy", logs.get("acc", 0.0))),
            "val_accuracy": float(logs.get("val_accuracy", logs.get("val_acc", 0.0))),
            "loss": float(logs.get("loss", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "time_ms": epoch_duration * 1000,
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
                "trained_models": self.trained_models,
            }, f, indent=2)

    def _load_from_disk(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                data = json.load(f)
                self.training_sessions = data.get("training_sessions", {})
                self.trained_models = data.get("trained_models", {})

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
    ):
        session_id = str(uuid.uuid4())
        self.training_sessions[session_id] = {
            "id": session_id,
            "task": task,
            "dataset_id": dataset_id,
            "epochs": epochs,
            "total_epochs": epochs,
            "current_epoch": 0,
            "progress": 0,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "base_model": base_model,
            "input_shape": input_shape,
            "dropout_rate": dropout_rate,
            "l2_reg": l2_reg,
            "status": "initialized",
            "created_at": time.time(),
            "elapsed_seconds": 0,
            "remaining_seconds": 0,
            "metrics": [],
            "stop_requested": False,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_monitor": early_stopping_monitor,
        }
        self.active_training[session_id] = False
        self._save_to_disk()
        return session_id

    def _export_dataset_to_temp_dir(self, dataset_id: str, task: str) -> tuple:
        """Export samples into temp_dir/train/<label>/... and temp_dir/val/<label>/...
        using the dataset's precomputed split (samples NOT marked train/val, i.e.
        test/unassigned, are skipped here — only train+val feed model.fit)."""
        from app.services.shared_state import data_manager
        from PIL import Image
        import io

        temp_dir = os.path.join(self.storage_dir, f"temp_{dataset_id}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        train_dir = os.path.join(temp_dir, "train")
        val_dir = os.path.join(temp_dir, "val")
        os.makedirs(train_dir)
        os.makedirs(val_dir)

        samples = data_manager.get_samples(dataset_id)
        labels = data_manager.get_dataset_labels(dataset_id)
        # Pre-create every class folder on both sides so image_dataset_from_directory
        # sees identical, consistently-ordered class lists even if a class has no
        # examples in one of the two splits.
        for split_dir in (train_dir, val_dir):
            for label in labels:
                os.makedirs(os.path.join(split_dir, label), exist_ok=True)

        name_counters: Dict[tuple, Dict[str, int]] = {}

        # Define allowed image formats for TF's decode_image
        ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "GIF", "BMP"}

        for sample in samples:
            split = sample.get("split", "unassigned")
            if split not in ("train", "val"):
                continue  # test/unassigned are not used to fit the model

            label = sample["label"]
            split_dir = train_dir if split == "train" else val_dir
            label_dir = os.path.join(split_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            data = data_manager.get_sample_data(sample["id"])
            if not data:
                continue
            
            # --- START VALIDATION ---
            if task in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "VISUAL_WAKE_WORDS"]:
                try:
                    with Image.open(io.BytesIO(data)) as img:
                        img.verify() # Verify integrity first
                        
                    # Re-open for conversion (verify() closes the file pointer)
                    with Image.open(io.BytesIO(data)) as img:
                        if img.format not in ALLOWED_IMAGE_FORMATS:
                            print(f"Converting {sample.get('filename')} from {img.format} to JPEG...")
                            # Convert to RGB (drops alpha channels from WEBP/PNG if present)
                            rgb_im = img.convert('RGB')
                            
                            # Save to a new byte buffer as JPEG
                            img_byte_arr = io.BytesIO()
                            rgb_im.save(img_byte_arr, format='JPEG')
                            data = img_byte_arr.getvalue() # Overwrite original data with new JPEG bytes
                            
                            # Update the filename to end in .jpg so we don't write a .webp extension
                            stem, _ = os.path.splitext(sample["filename"])
                            sample["filename"] = f"{stem}.jpg"
                except Exception as e:
                    print(f"Skipping corrupt image file: {sample.get('filename', 'unknown')}. Error: {e}")
                    continue
            # --- END VALIDATION ---

            safe_name = os.path.basename(sample["filename"])
            counter_key = (split, label)
            if counter_key not in name_counters:
                name_counters[counter_key] = {}
            
            if safe_name in name_counters[counter_key]:
                name_counters[counter_key][safe_name] += 1
                stem, ext = os.path.splitext(safe_name)
                safe_name = f"{stem}_{name_counters[counter_key][safe_name]}{ext}"
            else:
                name_counters[counter_key][safe_name] = 0

            file_path = os.path.join(label_dir, safe_name)
            with open(file_path, "wb") as f:
                f.write(data)

        return train_dir, val_dir

    def _build_image_pipeline(
        self, train_dir: str, val_dir: str, session: dict, class_names: list
    ):
        """
        Build a GPU-optimised tf.data image pipeline with:
          • cache()    – keeps decoded images in RAM after the first epoch
          • shuffle()  – randomises order each epoch
          • prefetch() – overlaps CPU preprocessing with GPU compute so the
                         GPU is never idle waiting for the next batch
        """
        img_height = session["input_shape"][0]
        img_width = session["input_shape"][1]
        color_mode = "grayscale" if session["input_shape"][2] == 1 else "rgb"
        batch_size = session["batch_size"]

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            class_names=class_names,
            seed=123,
            color_mode=color_mode,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=True,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            class_names=class_names,
            color_mode=color_mode,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False,
        )

        norm = tf.keras.layers.Rescaling(1.0 / 255)

        # cache() before shuffle so we only decode JPEG once;
        # shuffle() + prefetch() after norm so every epoch sees a different order.
        train_ds = (
            train_ds
            .map(lambda x, y: (norm(x), y), num_parallel_calls=_AUTOTUNE)
            .cache()
            .shuffle(buffer_size=1000)
            .prefetch(_AUTOTUNE)
        )
        val_ds = (
            val_ds
            .map(lambda x, y: (norm(x), y), num_parallel_calls=_AUTOTUNE)
            .cache()
            .prefetch(_AUTOTUNE)
        )

        return train_ds, val_ds

    def _build_audio_pipeline(
        self, train_dir: str, val_dir: str, batch_size: int, task: str, class_names: list
    ):
        """Audio is small enough to load fully into RAM; still prefetch."""
        class_map = {name: idx for idx, name in enumerate(class_names)}

        def _load(split_dir):
            features, labels = [], []
            for class_name in class_names:
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for file in os.listdir(class_dir):
                    if file.endswith((".wav", ".mp3")):
                        with open(os.path.join(class_dir, file), "rb") as f:
                            audio_data = f.read()
                        mfcc = DataProcessor.preprocess_audio(audio_data, task)
                        features.append(np.expand_dims(mfcc, axis=-1))
                        labels.append(class_map[class_name])
            return np.array(features), np.array(labels)

        X_train, y_train = _load(train_dir)
        X_val, y_val = _load(val_dir)

        train_idx = np.arange(len(X_train))
        np.random.shuffle(train_idx)
        X_train, y_train = X_train[train_idx], y_train[train_idx]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .batch(batch_size)
            .prefetch(_AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(batch_size)
            .prefetch(_AUTOTUNE)
        )
        return train_ds, val_ds

    def train(self, training_id: str):
        try:
            session = self.training_sessions[training_id]
            self.active_training[training_id] = True
            session["status"] = "running"
            self._save_to_disk()

            task = session["task"]
            dataset_id = session["dataset_id"]

            from app.services.shared_state import data_manager
            if not data_manager.is_split_ready(dataset_id):
                raise ValueError(
                    "Dataset has no complete precomputed train/val split. "
                    "Split it into train/val/test before starting training."
                )

            classes = data_manager.get_dataset_labels(dataset_id)
            num_classes = len(classes)

            train_dir, val_dir = self._export_dataset_to_temp_dir(dataset_id, task)
            temp_dir = os.path.dirname(train_dir)

            if task in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "VISUAL_WAKE_WORDS"]:
                train_ds, val_ds = self._build_image_pipeline(train_dir, val_dir, session, classes)
                model_input_shape = tuple(session["input_shape"])

            elif task in ["AUDIO_CLASSIFICATION", "KEYWORD_SPOTTING"]:
                train_ds, val_ds = self._build_audio_pipeline(
                    train_dir, val_dir, session["batch_size"], task, classes
                )
                audio_params = DataProcessor.AUDIO_PARAMS.get(
                    task, DataProcessor.AUDIO_PARAMS["KEYWORD_SPOTTING"]
                )
                model_input_shape = (audio_params["n_mfcc"], 101)
            else:
                raise ValueError(f"Unknown task: {task}")

            model = ModelFactory.create_model(
                task=task,
                num_classes=num_classes,
                base_model=session["base_model"],
                input_shape=model_input_shape,
                dropout_rate=session.get("dropout_rate", 0.5),
                l2_reg=session.get("l2_reg", 0.0),
            )

            # When mixed_precision is active the final Dense softmax layer must
            # output float32 to avoid numerical instability. ModelFactory already
            # uses softmax, but we cast to be safe.
            policy = tf.keras.mixed_precision.global_policy()
            if policy.name == "mixed_float16":
                # Wrap the model output in a float32 cast layer
                inputs = model.input
                outputs = tf.cast(model.output, tf.float32)
                model = tf.keras.Model(inputs, outputs)

            optimizer = tf.keras.optimizers.Adam(learning_rate=session["learning_rate"])
            loss_fn = (
                "binary_crossentropy"
                if task == "VISUAL_WAKE_WORDS"
                else "sparse_categorical_crossentropy"
            )
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

            callbacks: list = [TrainingCallback(session, self)]
            if session.get("early_stopping", False):
                es = tf.keras.callbacks.EarlyStopping(
                    monitor=session.get("early_stopping_monitor", "val_loss"),
                    patience=session.get("early_stopping_patience", 5),
                    restore_best_weights=True,
                    verbose=0,
                )
                callbacks.append(es)

            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=session["total_epochs"],
                callbacks=callbacks,
            )

            # Save in modern .keras format (replaces deprecated .h5)
            save_path = os.path.join(self.storage_dir, f"{training_id}.keras")
            model.save(save_path)
            shutil.rmtree(temp_dir)

            session["status"] = "completed"
            session["completed_at"] = time.time()

            model_id = str(uuid.uuid4())
            last_metrics = session["metrics"][-1] if session["metrics"] else {}
            self.trained_models[model_id] = {
                "id": model_id,
                "name": f"{session['base_model']} Trained",
                "training_id": training_id,
                "task": task,
                "accuracy": last_metrics.get("accuracy", 0.0),
                "val_accuracy": last_metrics.get("val_accuracy", 0.0),
                "loss": last_metrics.get("loss", 0.0),
                "val_loss": last_metrics.get("val_loss", 0.0),
                "size_bytes": os.path.getsize(save_path),
                "optimized": False,
                "path": save_path,
            }
            self._save_to_disk()

        except Exception as e:
            session = self.training_sessions.get(training_id, {})
            session["status"] = "failed"
            session["error"] = str(e)
            self._save_to_disk()
        finally:
            self.active_training[training_id] = False

    def get_training_status(self, training_id: str) -> dict:
        return self.training_sessions.get(training_id, {})

    def get_training_metrics(self, training_id: str) -> List[dict]:
        session = self.training_sessions.get(training_id, {})
        return session.get("metrics", [])

    def get_all_sessions(self) -> List[dict]:
        """Return all training sessions sorted newest-first."""
        sessions = list(self.training_sessions.values())
        return sorted(sessions, key=lambda s: s.get("created_at", 0), reverse=True)

    def cancel_training(self, training_id: str):
        if training_id in self.training_sessions:
            self.training_sessions[training_id]["stop_requested"] = True
            self.active_training[training_id] = False

    def get_trained_models(self) -> List[dict]:
        return list(self.trained_models.values())

    def delete_trained_model(self, model_id: str) -> bool:
        if model_id in self.trained_models:
            del self.trained_models[model_id]
            self._save_to_disk()
            return True
        return False