import os
import json
import uuid
import time
from typing import Dict, List

from app.services.ml_engine import get_model, TrainingCallback
import tensorflow as tf

import numpy as np

import os
import shutil
import tensorflow as tf
from app.services.model_factory import ModelFactory
from app.utils.data_processor import DataProcessor

class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, session: dict, trainer):
        super().__init__()
        self.session = session
        self.trainer = trainer
        self.epoch_start_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Calculate time taken to help with ETA
        epoch_duration = time.time() - self.epoch_start_time
        
        # Update progress for the frontend
        self.session["current_epoch"] = epoch + 1
        self.session["progress"] = int(((epoch + 1) / self.session.get("total_epochs", 1)) * 100)
        
        # Extract and format metrics safely
        metric_entry = {
            "epoch": epoch + 1,
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
            "loss": float(logs.get("loss", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "time_ms": epoch_duration * 1000
        }
        
        self.session["metrics"].append(metric_entry)
        
        print(f"DEBUG: Epoch {epoch} logs: {logs}")
        
        # Save to disk so the FastAPI polling endpoint serves the fresh data
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
        """Persist training sessions and finished models to JSON"""
        with open(self.db_file, "w") as f:
            json.dump({
                "training_sessions": self.training_sessions,
                "trained_models": self.trained_models
            }, f, indent=2)

    def _load_from_disk(self):
        """Load state from disk on startup"""
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                data = json.load(f)
                self.training_sessions = data.get("training_sessions", {})
                self.trained_models = data.get("trained_models", {})

    def create_training_session(self, task, dataset_id, epochs, batch_size,
                                learning_rate, base_model, validation_split, input_shape):
        session_id = str(uuid.uuid4())
        self.training_sessions[session_id] = {
            "id": session_id,
            "task": task,
            "dataset_id": dataset_id,
            "epochs": epochs,
            "total_epochs": epochs,    # <-- FIXED: Prevents NaN ETA
            "current_epoch": 0,        # <-- FIXED: Tells UI we are starting
            "progress": 0,             # <-- FIXED: Initial progress bar state
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "base_model": base_model,
            "validation_split": validation_split,
            "input_shape": input_shape,
            "status": "initialized",
            "created_at": time.time(),
            "metrics": []
        }
        self.active_training[session_id] = False
        self._save_to_disk()
        return session_id

    def _export_dataset_to_temp_dir(self, dataset_id: str) -> str:
        from app.services.shared_state import data_manager

        """Extracts samples from DataManager into a standard Keras directory structure."""
        temp_dir = os.path.join(self.storage_dir, f"temp_{dataset_id}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        samples = data_manager.get_samples(dataset_id)
        for sample in samples:
            label_dir = os.path.join(temp_dir, sample["label"])
            os.makedirs(label_dir, exist_ok=True)

            data = data_manager.get_sample_data(sample["id"])
            if data:
                file_path = os.path.join(label_dir, sample["filename"])
                with open(file_path, "wb") as f:
                    f.write(data)
        return temp_dir

    def train(self, training_id: str):
        try:
            session = self.training_sessions[training_id]
            self.active_training[training_id] = True
            session["status"] = "running"
            session["started_at"] = time.time()

            # 1. Prepare Data Directory
            dataset_dir = self._export_dataset_to_temp_dir(session["dataset_id"])
            classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            num_classes = len(classes)

            # 2. Route Data Pipeline based on Task
            if session["task"] in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "VISUAL_WAKE_WORDS"]:
                img_height, img_width = session["input_shape"][0], session["input_shape"][1]
                color_mode = "grayscale" if session["input_shape"][2] == 1 else "rgb"

                train_ds = tf.keras.utils.image_dataset_from_directory(
                    dataset_dir,
                    validation_split=session["validation_split"],
                    subset="training",
                    seed=123,
                    color_mode=color_mode,
                    image_size=(img_height, img_width),
                    batch_size=session["batch_size"]
                )
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    dataset_dir,
                    validation_split=session["validation_split"],
                    subset="validation",
                    seed=123,
                    color_mode=color_mode,
                    image_size=(img_height, img_width),
                    batch_size=session["batch_size"]
                )
                # Normalize images
                normalization_layer = tf.keras.layers.Rescaling(1./255)
                train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
                val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

            elif session["task"] in ["AUDIO_CLASSIFICATION", "KEYWORD_SPOTTING"]:
                # Custom Audio Loader (Implementation in Phase 4)
                train_ds, val_ds = self._prepare_audio_dataset(
                    dataset_dir, session["batch_size"], session["validation_split"], session["task"]
                )

            # 3. Create Model dynamically using the Factory
            model = ModelFactory.create_model(
                task=session["task"],
                num_classes=num_classes,
                base_model=session["base_model"]
            )

            # Optional: Resize Input layer dynamically if necessary via Factory update
            optimizer = tf.keras.optimizers.Adam(learning_rate=session["learning_rate"])
            loss_fn = 'binary_crossentropy' if num_classes == 2 and session["task"] == "VISUAL_WAKE_WORDS" else 'sparse_categorical_crossentropy'

            model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

            # 4. Train
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=session["total_epochs"],
                callbacks=[TrainingCallback(session, self)] # <-- Injecting the bridge here!
            )

            # 5. Save and Cleanup
            save_path = os.path.join(self.storage_dir, f"{training_id}.h5")
            model.save(save_path)
            shutil.rmtree(dataset_dir) # Cleanup temp data

            session["status"] = "completed"
            session["completed_at"] = time.time()

            # Register in state
            model_id = str(uuid.uuid4())
            self.trained_models[model_id] = {
                "id": model_id,
                "name": f"{session['base_model']} Trained",
                "training_id": training_id,
                "task": session["task"],
                "accuracy": session["metrics"][-1]["val_accuracy"],
                "size_bytes": os.path.getsize(save_path),
                "optimized": False,
                "path": save_path
            }
            self._save_to_disk()

        except Exception as e:
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
    
    def cancel_training(self, training_id: str):
        if training_id in self.active_training:
            self.active_training[training_id] = False
    
    def get_trained_models(self) -> List[dict]:
        return list(self.trained_models.values())
    
    def delete_trained_model(self, model_id: str) -> bool:
        if model_id in self.trained_models:
            del self.trained_models[model_id]
            self._save_to_disk()
            return True
        return False
    
    def _prepare_audio_dataset(self, dataset_dir: str, batch_size: int, val_split: float, task: str):
        """Converts audio files to MFCC features using the DataProcessor."""
        import numpy as np

        features = []
        labels = []
        class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        class_map = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(dataset_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(('.wav', '.mp3')):
                    file_path = os.path.join(class_dir, file)
                    with open(file_path, 'rb') as f:
                        audio_data = f.read()

                    # Utilize your existing DataProcessor
                    mfcc = DataProcessor.preprocess_audio(audio_data, task)
                    # Add channel dimension for CNN (Height, Width, Channels)
                    mfcc = np.expand_dims(mfcc, axis=-1)

                    features.append(mfcc)
                    labels.append(class_map[class_name])

        X = np.array(features)
        y = np.array(labels)

        # Shuffle and Split
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        split_idx = int(len(X) * (1 - val_split))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        return train_ds, val_ds