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
from app.services.job_logs import job_log_broker

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
        job_log_broker.log(self.session["id"], f"Training started - {self.session.get('total_epochs', '?')} epochs planned.")

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

        job_log_broker.log(
            self.session["id"],
            f"Epoch {epochs_done}/{total_epochs} - "
            f"loss: {metric_entry['loss']:.4f} - accuracy: {metric_entry['accuracy']:.4f} - "
            f"val_loss: {metric_entry['val_loss']:.4f} - val_accuracy: {metric_entry['val_accuracy']:.4f} "
            f"({epoch_duration:.1f}s)",
        )

        if self.session.get("stop_requested", False):
            self.model.stop_training = True
            job_log_broker.log(self.session["id"], "Cancellation requested - stopping after this epoch.", level="warning")

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
    # Object Detection pipeline and losses
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_iou_matrix(boxes_a, boxes_b):
        """Compute pairwise IoU between two sets of boxes.

        Both inputs: (N, 4) or (N, M, 4) in cx, cy, w, h normalized format.
        Returns: (N, M) IoU matrix.
        """
        a_x1 = boxes_a[..., 0] - boxes_a[..., 2] / 2
        a_y1 = boxes_a[..., 1] - boxes_a[..., 3] / 2
        a_x2 = boxes_a[..., 0] + boxes_a[..., 2] / 2
        a_y2 = boxes_a[..., 1] + boxes_a[..., 3] / 2

        b_x1 = boxes_b[..., 0] - boxes_b[..., 2] / 2
        b_y1 = boxes_b[..., 1] - boxes_b[..., 3] / 2
        b_x2 = boxes_b[..., 0] + boxes_b[..., 2] / 2
        b_y2 = boxes_b[..., 1] + boxes_b[..., 3] / 2

        inter_x1 = tf.maximum(a_x1, b_x1)
        inter_y1 = tf.maximum(a_y1, b_y1)
        inter_x2 = tf.minimum(a_x2, b_x2)
        inter_y2 = tf.minimum(a_y2, b_y2)

        inter_area = tf.maximum(inter_x2 - inter_x1, 0.0) * tf.maximum(inter_y2 - inter_y1, 0.0)
        a_area = boxes_a[..., 2] * boxes_a[..., 3]
        b_area = boxes_b[..., 2] * boxes_b[..., 3]
        union = a_area + b_area - inter_area
        return inter_area / (union + 1e-7)

    @staticmethod
    def od_loss(y_true_boxes, y_true_classes, y_true_scores,
                y_pred_boxes, y_pred_classes, y_pred_scores,
                num_classes, anchor_count):
        """Multi-task object detection loss.

        y_true_boxes:   (batch, max_det, 4)
        y_true_classes: (batch, max_det, num_classes) one-hot
        y_true_scores:  (batch, max_det) 1=object, 0=no object

        y_pred_boxes:   (batch, total_anchors, 4)
        y_pred_classes: (batch, total_anchors, num_classes)
        y_pred_scores:  (batch, total_anchors)
        """
        obj_mask = tf.squeeze(y_true_scores, axis=-1)  # (batch, max_det)

        # Box loss — only for positive samples
        box_diff = y_pred_boxes - y_true_boxes
        box_loss_raw = tf.reduce_sum(tf.abs(box_diff), axis=-1)  # L1
        pos_mask = tf.cast(obj_mask > 0.5, tf.float32)
        box_loss = tf.reduce_sum(box_loss_raw * pos_mask) / (tf.reduce_sum(pos_mask) + 1e-7)

        # For simplicity, compare pred objectness against IoU with GT
        # This is a simplified assignment — each GT is matched to its
        # highest-IoU anchor (we use the raw pred_boxes vs GT directly
        # since anchor-free head predicts absolute coords).
        iou = tf.map_fn(
            lambda pair: tf.reduce_max(
                tf.map_fn(lambda gt: Trainer._compute_iou_matrix(pair, gt), y_true_boxes, fn_output_signature=tf.float32),
                axis=-1
            ),
            y_pred_boxes,
            fn_output_signature=tf.float32,
        )  # (batch, total_anchors)

        obj_target = tf.stop_gradient(tf.cast(iou > 0.5, tf.float32))
        obj_loss = tf.keras.losses.binary_crossentropy(obj_target, y_pred_scores)
        obj_loss = tf.reduce_mean(obj_loss)

        # Classification loss — focal-style weighting
        cls_target = tf.tile(y_true_classes, [1, anchor_count // tf.shape(y_true_classes)[1] + 1, 1])[:, :tf.shape(y_pred_classes)[1], :]
        cls_loss = tf.keras.losses.categorical_crossentropy(cls_target, y_pred_classes)
        cls_loss = tf.reduce_mean(cls_loss)

        total = box_loss * 5.0 + obj_loss * 1.0 + cls_loss * 1.0
        return total

    @staticmethod
    def compute_mAP(y_true_boxes_list, y_true_classes_list, y_true_scores_list,
                    y_pred_boxes_list, y_pred_classes_list, y_pred_scores_list,
                    iou_threshold=0.5):
        """Simple mAP@0.5 computation across a batch.

        Returns a Python float (mean average precision).
        """
        total_tp = 0
        total_fp = 0
        total_gt = 0

        for i in range(len(y_pred_boxes_list)):
            pred_boxes = y_pred_boxes_list[i]
            pred_scores = y_pred_scores_list[i]
            pred_classes = y_pred_classes_list[i]
            gt_boxes = y_true_boxes_list[i]
            gt_scores = y_true_scores_list[i]

            gt_mask = gt_scores.flatten() > 0.5
            gt_b = gt_boxes[gt_mask]
            total_gt += int(np.sum(gt_mask))

            if len(gt_b) == 0 or len(pred_boxes) == 0:
                total_fp += len(pred_boxes)
                continue

            order = np.argsort(-pred_scores.flatten())
            pred_boxes = pred_boxes[order]
            pred_classes = pred_classes[order]

            matched = np.zeros(len(gt_b), dtype=bool)
            for pb, pc in zip(pred_boxes, pred_classes):
                ious = np.array([float(Trainer._compute_iou_matrix(
                    tf.constant(pb, dtype=tf.float32),
                    tf.constant(gb, dtype=tf.float32),
                )) for gb in gt_b])
                best_iou = np.max(ious) if len(ious) > 0 else 0
                best_idx = np.argmax(ious) if len(ious) > 0 else -1
                if best_iou >= iou_threshold and not matched[best_idx]:
                    total_tp += 1
                    matched[best_idx] = True
                else:
                    total_fp += 1

        precision = total_tp / (total_tp + total_fp + 1e-7)
        return float(precision)

    def _build_od_pipeline(
        self,
        dataset_id: str,
        class_names: list,
        input_shape: tuple,
        batch_size: int,
    ):
        """Build tf.data pipeline for object detection.

        Returns (train_ds, val_ds) where each element is:
          (images, {"boxes": ..., "classes": ..., "scores": ...})
        """
        from app.services.shared_state import data_manager
        from PIL import Image as PILImage
        import io as _io

        img_h, img_w = input_shape[0], input_shape[1]
        class_map = {name: idx for idx, name in enumerate(class_names)}
        num_classes = len(class_names)
        MAX_DET = 20  # max detections per image

        samples = data_manager.get_samples(dataset_id)

        def _load_split(split_name):
            images, boxes_list, classes_list, scores_list = [], [], [], []
            for sample in samples:
                if sample.get("split") != split_name:
                    continue
                data = data_manager.get_sample_data(sample["id"])
                if not data:
                    continue
                try:
                    with PILImage.open(_io.BytesIO(data)) as img:
                        img = img.convert("RGB")
                        img = img.resize((img_w, img_h), PILImage.BILINEAR)
                        arr = np.array(img, dtype=np.float32) / 255.0
                except Exception:
                    continue

                raw_anns = sample.get("annotations") or []
                bboxes, cls_ohs = [], []
                for ann in raw_anns:
                    if isinstance(ann, dict):
                        cx = ann.get("cx", 0)
                        cy = ann.get("cy", 0)
                        bw = ann.get("w", 0)
                        bh = ann.get("h", 0)
                        cn = ann.get("class_name", "object")
                    else:
                        cx, cy, bw, bh = getattr(ann, "cx", 0), getattr(ann, "cy", 0), getattr(ann, "w", 0), getattr(ann, "h", 0)
                        cn = getattr(ann, "class_name", "object")
                    cls_idx = class_map.get(cn, 0)
                    oh = np.zeros(num_classes, dtype=np.float32)
                    oh[cls_idx] = 1.0
                    bboxes.append([cx, cy, bw, bh])
                    cls_ohs.append(oh)

                # Pad to MAX_DET
                pad_count = MAX_DET - len(bboxes)
                if pad_count > 0:
                    bboxes += [[0, 0, 0, 0]] * pad_count
                    cls_ohs += [[0] * num_classes] * pad_count
                bboxes = bboxes[:MAX_DET]
                cls_ohs = cls_ohs[:MAX_DET]

                scores = [1.0] * min(len(raw_anns), MAX_DET) + [0.0] * max(0, MAX_DET - len(raw_anns))
                scores = scores[:MAX_DET]

                images.append(arr)
                boxes_list.append(np.array(bboxes, dtype=np.float32))
                classes_list.append(np.array(cls_ohs, dtype=np.float32))
                scores_list.append(np.array(scores, dtype=np.float32))

            if not images:
                empty_img = np.zeros((1, img_h, img_w, 3), dtype=np.float32)
                empty_box = np.zeros((1, MAX_DET, 4), dtype=np.float32)
                empty_cls = np.zeros((1, MAX_DET, num_classes), dtype=np.float32)
                empty_sc  = np.zeros((1, MAX_DET), dtype=np.float32)
                return tf.data.Dataset.from_tensor_slices((
                    empty_img, {"boxes": empty_box, "classes": empty_cls, "scores": empty_sc}
                )).batch(1)

            X = np.array(images, dtype=np.float32)
            Y_boxes = np.array(boxes_list, dtype=np.float32)
            Y_cls = np.array(classes_list, dtype=np.float32)
            Y_sc  = np.array(scores_list, dtype=np.float32)

            idx = np.random.permutation(len(X))
            X, Y_boxes, Y_cls, Y_sc = X[idx], Y_boxes[idx], Y_cls[idx], Y_sc[idx]

            return tf.data.Dataset.from_tensor_slices((
                X, {"boxes": Y_boxes, "classes": Y_cls, "scores": Y_sc}
            )).batch(batch_size).prefetch(1)

        train_ds = _load_split("train")
        val_ds = _load_split("val")
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

            job_log_broker.log(training_id, f"Training session {training_id} starting.")
            job_log_broker.log(training_id, f"Task: {task}  |  Base model: {session.get('base_model')}  |  Input shape: {session.get('input_shape')}")

            from app.services.shared_state import data_manager
            if not data_manager.is_split_ready(dataset_id):
                raise ValueError(
                    "Dataset has no complete train/val split. "
                    "Split it via Auto Split before starting training."
                )

            classes     = data_manager.get_dataset_labels(dataset_id)
            num_classes = len(classes)
            job_log_broker.log(training_id, f"Dataset {dataset_id}: {num_classes} classes -> {classes}")

            train_dir, val_dir = self._export_dataset_to_temp_dir(dataset_id, task)
            temp_dir = os.path.dirname(train_dir)
            job_log_broker.log(training_id, "Dataset exported to temp directory - building pipeline...")

            if task == "OBJECT_DETECTION":
                # For OD, we need the annotation classes from the dataset,
                # not the file-label classes used for classification.
                od_classes = data_manager.get_dataset_labels(dataset_id)
                ann_meta = data_manager.datasets.get(dataset_id, {}).get("metadata", {})
                ann_classes = ann_meta.get("annotation_classes")
                if ann_classes:
                    od_classes = ann_classes
                    num_classes = len(od_classes)
                    job_log_broker.log(training_id, f"OD annotation classes: {od_classes}")

                raw_shape = tuple(session["input_shape"])
                model_input_shape = (raw_shape[0], raw_shape[1], 3)
                train_ds, val_ds = self._build_od_pipeline(
                    dataset_id, od_classes, model_input_shape, session["batch_size"]
                )
                job_log_broker.log(training_id, f"OD pipeline built with {len(od_classes)} classes, input {model_input_shape}")

            elif task in ["IMAGE_CLASSIFICATION", "VISUAL_WAKE_WORDS"]:
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
            job_log_broker.log(training_id, f"Compute device: {device_used} (requested: {device_pref})")
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
                    if policy.name == "mixed_float16" and task != "OBJECT_DETECTION":
                        inputs  = model.input
                        outputs = tf.cast(model.output, tf.float32)
                        model   = tf.keras.Model(inputs, outputs)

                    is_od = task == "OBJECT_DETECTION"

                    if is_od:
                        # OD uses custom loss via train_step override
                        od_num_classes = num_classes
                        od_classes_list = data_manager.get_dataset_labels(dataset_id)
                        ann_meta = data_manager.datasets.get(dataset_id, {}).get("metadata", {})
                        ann_classes = ann_meta.get("annotation_classes")
                        if ann_classes:
                            od_classes_list = ann_classes

                        def _od_loss(y_true, y_pred):
                            if isinstance(y_true, dict):
                                gt_boxes = y_true["boxes"]    # (batch, MAX_DET, 4)
                                gt_cls = y_true["classes"]    # (batch, MAX_DET, C)
                                gt_sc = y_true["scores"]      # (batch, MAX_DET)
                            else:
                                gt_boxes = y_true
                                gt_cls = tf.zeros_like(y_pred["classes"][:, :tf.shape(y_true)[1], :])
                                gt_sc = tf.ones(tf.shape(y_true)[0:1], dtype=tf.float32)

                            pred_boxes = y_pred["boxes"]     # (batch, total_anchors, 4)
                            pred_cls = y_pred["classes"]     # (batch, total_anchors, C)
                            pred_sc = y_pred["scores"]       # (batch, total_anchors)

                            num_gt = tf.shape(gt_boxes)[1]

                            # Positive mask: GT entries with score > 0.5 are real objects
                            pos_mask = tf.cast(gt_sc > 0.5, tf.float32)  # (batch, MAX_DET)

                            # ── Box loss: compare each GT with its best-matching pred anchor
                            # For simplicity, compare the first num_gt preds directly against GTs
                            # (the model learns to predict GT-like boxes in early anchor slots).
                            pred_trunc = pred_boxes[:, :num_gt, :]  # (batch, num_gt, 4)
                            box_diff = tf.abs(pred_trunc - gt_boxes)
                            box_loss_per = tf.reduce_sum(box_diff, axis=-1)  # (batch, num_gt)
                            box_loss = tf.reduce_sum(box_loss_per * pos_mask) / (tf.reduce_sum(pos_mask) + 1e-7)

                            # ── Objectness loss: target = 1 for best-matching anchors
                            # Compute IoU between each pred anchor and each GT
                            pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
                            pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
                            pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
                            pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

                            gt_x1 = gt_boxes[..., 0] - gt_boxes[..., 2] / 2
                            gt_y1 = gt_boxes[..., 1] - gt_boxes[..., 3] / 2
                            gt_x2 = gt_boxes[..., 0] + gt_boxes[..., 2] / 2
                            gt_y2 = gt_boxes[..., 1] + gt_boxes[..., 3] / 2

                            # Expand for broadcasting: (batch, total_anchors, 1, 4) vs (batch, 1, MAX_DET, 4)
                            inter_x1 = tf.maximum(pred_x1[..., tf.newaxis], gt_x1[:, tf.newaxis, :])
                            inter_y1 = tf.maximum(pred_y1[..., tf.newaxis], gt_y1[:, tf.newaxis, :])
                            inter_x2 = tf.minimum(pred_x2[..., tf.newaxis], gt_x2[:, tf.newaxis, :])
                            inter_y2 = tf.minimum(pred_y2[..., tf.newaxis], gt_y2[:, tf.newaxis, :])
                            inter_area = tf.maximum(inter_x2 - inter_x1, 0.0) * tf.maximum(inter_y2 - inter_y1, 0.0)

                            pred_area = pred_boxes[..., 2] * pred_boxes[..., 3]
                            gt_area = gt_boxes[..., 2] * gt_boxes[..., 3]
                            union = pred_area[..., tf.newaxis] + gt_area[:, tf.newaxis, :] - inter_area
                            ious = inter_area / (union + 1e-7)  # (batch, total_anchors, MAX_DET)
                            max_iou = tf.reduce_max(ious, axis=-1)  # (batch, total_anchors)

                            obj_target = tf.stop_gradient(tf.cast(max_iou > 0.5, tf.float32))
                            obj_loss = tf.reduce_mean(
                                tf.keras.losses.binary_crossentropy(obj_target, pred_sc)
                            )

                            # ── Classification loss: only for matched anchors
                            # Match each GT to its best pred anchor
                            best_pred_idx = tf.cast(tf.argmax(max_iou, axis=1), tf.int32)  # (batch,)
                            batch_idx = tf.range(tf.shape(pred_boxes)[0])
                            best_cls_pred = tf.gather_nd(pred_cls, tf.stack([batch_idx, best_pred_idx], axis=1))  # (batch, C)
                            best_gt_cls = tf.reduce_sum(gt_cls * pos_mask[:, :, tf.newaxis], axis=1)  # (batch, C) — sum of positive GT one-hots
                            cls_loss = tf.reduce_mean(
                                tf.keras.losses.categorical_crossentropy(best_gt_cls, best_cls_pred)
                            )

                            return box_loss * 5.0 + obj_loss * 1.0 + cls_loss * 1.0

                        loss_fn = _od_loss
                        loss_fn.__name__ = "od_multi_task_loss"
                    else:
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
                    trainable_layers_cfg = session.get("trainable_layers",       0)
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

                    if is_od:
                        # OD training: custom training loop via GradientTape
                        # because the model dict output (total_anchors) differs
                        # from the target dict (MAX_DET), requiring IoU-based
                        # assignment inside the loss.
                        od_optimizer = tf.keras.optimizers.Adam(lr)
                        train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")

                        @tf.function
                        def _od_train_step(images, targets):
                            with tf.GradientTape() as tape:
                                preds = model(images, training=True)
                                loss = _od_loss(targets, preds)
                            grads = tape.gradient(loss, model.trainable_variables)
                            od_optimizer.apply_gradients(zip(grads, model.trainable_variables))
                            train_loss_tracker.update_state(loss)
                            return loss

                        @tf.function
                        def _od_val_step(images, targets):
                            preds = model(images, training=False)
                            return _od_loss(targets, preds)

                        total_epochs = session.get("epochs", 50)
                        for epoch in range(total_epochs):
                            if session.get("stop_requested", False):
                                break

                            epoch_start = time.time()
                            train_loss_tracker.reset_state()

                            for batch_images, batch_targets in train_ds:
                                _od_train_step(batch_images, batch_targets)

                            # Validation
                            val_losses = []
                            for batch_images, batch_targets in val_ds:
                                val_losses.append(_od_val_step(batch_images, batch_targets))
                            val_loss = tf.reduce_mean(val_losses).numpy() if val_losses else 0.0

                            epoch_duration = time.time() - epoch_start
                            train_loss = float(train_loss_tracker.result())

                            # Feed metrics to TrainingCallback-style logging
                            metric_entry = {
                                "epoch": epoch + 1,
                                "accuracy": 0.0,
                                "val_accuracy": 0.0,
                                "loss": train_loss,
                                "val_loss": float(val_loss),
                                "time_ms": epoch_duration * 1000,
                            }
                            session["metrics"].append(metric_entry)
                            session["current_epoch"] = epoch + 1
                            session["progress"] = int(((epoch + 1) / total_epochs) * 100)
                            session["elapsed_seconds"] = time.time() - session.get("started_at", time.time())
                            avg_epoch = session["elapsed_seconds"] / (epoch + 1)
                            session["remaining_seconds"] = avg_epoch * (total_epochs - epoch - 1)

                            job_log_broker.log(
                                training_id,
                                f"Epoch {epoch+1}/{total_epochs} - "
                                f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f} "
                                f"({epoch_duration:.1f}s)",
                            )
                            self._save_to_disk()

                    elif freeze_epochs > 0 and base_model_layer:
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
                        if trainable_layers_cfg > 0:
                            for layer in base_model_layer.layers[:-trainable_layers_cfg]:
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
                job_log_broker.log(training_id, "Training cancelled before any epoch completed - nothing saved.", level="warning")
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

            if was_cancelled:
                job_log_broker.log(training_id, f"Training cancelled after {len(session['metrics'])} epoch(s). Partial model saved.", level="warning")
            else:
                job_log_broker.log(training_id, f"Training completed successfully. Model saved to {save_path}")

        except Exception as e:
            import traceback
            session = self.training_sessions.get(training_id, {})
            session["status"] = "failed"
            session["error"]  = str(e)
            self._save_to_disk()
            # Log the FULL traceback to the job console, not just str(e) -
            # this is exactly what was missing when a training crash (e.g.
            # the Lambda-layer deserialization error) only ever showed up
            # in the backend's own terminal and never reached the frontend.
            job_log_broker.log(training_id, f"Training FAILED: {e}", level="error")
            job_log_broker.log(training_id, traceback.format_exc(), level="error")
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

    def get_trained_models(self, include_archived: bool = False) -> List[dict]:
        """
        By default, excludes models whose underlying training session has
        been archived - previously archiving a training run only hid it
        from the training history list, but the model it produced kept
        showing up everywhere else (Optimization Studio's model picker,
        the Dashboard, etc), which defeated the point of archiving it.
        """
        models = list(self.trained_models.values())
        if include_archived:
            return models
        return [
            m for m in models
            if not self.training_sessions.get(m.get("training_id"), {}).get("archived", False)
        ]

    def delete_trained_model(self, model_id: str) -> bool:
        if model_id in self.trained_models:
            del self.trained_models[model_id]
            self._save_to_disk()
            return True
        return False