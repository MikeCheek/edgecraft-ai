import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple

class ModelFactory:
    """Factory for creating TinyML models for different tasks"""

    @staticmethod
    def create_image_classification_model(
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 10,
        base_model_name: str = "MobileNetV2"
    ) -> keras.Model:
        """Create image classification model"""
        if base_model_name == "MobileNetV2":
            base_model = keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
        else:
            # Custom 3-layer CNN for ultra-low memory
            return keras.Sequential([
                layers.Conv2D(16, 3, activation='relu', input_shape=input_shape),
                layers.MaxPooling2D(2),
                layers.Conv2D(32, 3, activation='relu'),
                layers.MaxPooling2D(2),
                layers.Conv2D(64, 3, activation='relu'),
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation='softmax')
            ])

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    @staticmethod
    def create_visual_wake_words_model(
        input_shape: Tuple[int, int, int] = (96, 96, 1)
    ) -> keras.Model:
        """Create Visual Wake Words binary classifier"""
        base_model = keras.applications.MobileNetV2(
            input_shape=(96, 96, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = keras.Sequential([
            layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x), input_shape=input_shape),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    @staticmethod
    def create_audio_classification_model(
        input_shape: Tuple[int, int] = (40, 101),
        num_classes: int = 4
    ) -> keras.Model:
        """Create audio classification model using MFCC spectrograms"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*input_shape, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    @staticmethod
    def create_keyword_spotting_model(
        input_shape: Tuple[int, int] = (40, 101),
        num_classes: int = 2
    ) -> keras.Model:
        """Create keyword spotting model (lightweight audio)"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*input_shape, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    @staticmethod
    def create_model(
        task: str,
        num_classes: int = 10,
        base_model: str = "MobileNetV2"
    ) -> keras.Model:
        """Factory method to create appropriate model for task"""
        models = {
            "IMAGE_CLASSIFICATION": lambda: ModelFactory.create_image_classification_model(
                num_classes=num_classes,
                base_model_name=base_model
            ),
            "OBJECT_DETECTION": lambda: ModelFactory.create_image_classification_model(
                num_classes=num_classes,
                base_model_name=base_model
            ),
            "VISUAL_WAKE_WORDS": lambda: ModelFactory.create_visual_wake_words_model(),
            "KEYWORD_SPOTTING": lambda: ModelFactory.create_keyword_spotting_model(
                num_classes=num_classes
            ),
            "AUDIO_CLASSIFICATION": lambda: ModelFactory.create_audio_classification_model(
                num_classes=num_classes
            ),
        }

        if task not in models:
            raise ValueError(f"Unknown task: {task}")

        return models[task]()
