# FIX: TrainingCallback is now defined only in trainer.py.
# This file retains only the get_model() factory for backwards compatibility.
# Importing TrainingCallback from here caused a duplicate-class conflict with
# the richer version in trainer.py (which persists metrics to disk, etc.).

import tensorflow as tf


def get_model(base_model_name: str, num_classes: int, input_shape: tuple = (224, 224, 3)):
    """Legacy model factory — prefer ModelFactory.create_model() for new code."""
    if base_model_name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights='imagenet'
        )
        base.trainable = False
        return tf.keras.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    # Fallback simple CNN
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])