# app/services/ml_engine.py
import tensorflow as tf
import numpy as np
import os
import time

class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, session, trainer_instance):
        self.session = session
        self.trainer = trainer_instance

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Update metrics in real-time
        self.session["metrics"].append({
            "epoch": epoch + 1,
            "loss": float(logs.get("loss")),
            "accuracy": float(logs.get("accuracy")),
            "val_loss": float(logs.get("val_loss")),
            "val_accuracy": float(logs.get("val_accuracy")),
            "timestamp": time.time()
        })
        self.session["current_epoch"] = epoch + 1
        self.trainer._save_to_disk()

def get_model(base_model_name: str, num_classes: int):
    # Factory to swap model architectures easily
    if base_model_name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights='imagenet'
        )
    else:
        # Fallback to a simple CNN if MobileNetV2 is not selected
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    base.trainable = False # Use transfer learning
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model