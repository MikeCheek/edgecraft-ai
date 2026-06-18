import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Tuple

class ModelFactory:
    """Factory for creating TinyML models for different tasks.

    All models accept optional `dropout_rate` and `l2_reg` arguments so the
    LLM-advisor suggestions can actually be applied without editing code.
    """

    # ------------------------------------------------------------------ image

    @staticmethod
    def create_image_classification_model(
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 10,
        base_model_name: str = "MobileNetV2",
        dropout_rate: float = 0.5,
        l2_reg: float = 0.0,
        trainable_layers: int = 0,  # 0 = unfreeze all, >0 = unfreeze last N layers
        augmentation: dict = {},
    ) -> keras.Model:
        """Create image classification model with optional regularisation."""
        # 1. Prepare Data Augmentation Block (Runs on GPU during training, zero overhead on Edge)
        aug_layers = []
        if augmentation:
            if augmentation.get("horizontal_flip"):
                aug_layers.append(layers.RandomFlip("horizontal"))
            if augmentation.get("random_rotation"):
                # 0.2 means ±20% of 2Pi
                aug_layers.append(layers.RandomRotation(augmentation.get("random_rotation")))
            if augmentation.get("random_crop"):
                aug_layers.append(layers.RandomZoom(0.2)) 
        
        data_augmentation = keras.Sequential(aug_layers, name="data_augmentation")

        # 2. Prepare Regularization
        reg = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

        # 3. Build Model Pipeline
        if base_model_name in ["MobileNetV2", "MobileNetV3Small"]:
            if base_model_name == "MobileNetV2":
                base = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
            else:
                base = keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False, weights="imagenet")
            
            # Apply Trainable Layers Logic
            if trainable_layers > 0:
                base.trainable = True
                for layer in base.layers[:-trainable_layers]:
                    layer.trainable = False
            else:
                base.trainable = True

            model = keras.Sequential([
                keras.Input(shape=input_shape),
                data_augmentation,
                base,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax", kernel_regularizer=reg)
            ])

        elif base_model_name == "EfficientNet":
            base = keras.applications.EfficientNetB0(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet",
            )
            base.trainable = False
            model = keras.Sequential([
                data_augmentation,
                base,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu", kernel_regularizer=reg),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax"),
            ])
            
        elif base_model_name == "ResNet50V2":
            base = keras.applications.ResNet50V2(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet",
            )
            base.trainable = False
            model = keras.Sequential([
                data_augmentation,
                base,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu", kernel_regularizer=reg),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax"),
            ])

        else:
            # Custom3LayerCNN - ultra-low memory
            model = keras.Sequential([
                data_augmentation,
                layers.Conv2D(16, 3, activation="relu", input_shape=input_shape,
                              kernel_regularizer=reg),
                layers.MaxPooling2D(2),
                layers.Conv2D(32, 3, activation="relu", kernel_regularizer=reg),
                layers.MaxPooling2D(2),
                layers.Conv2D(64, 3, activation="relu", kernel_regularizer=reg),
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu", kernel_regularizer=reg),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax"),
            ])

        return model

    # ------------------------------------------------------------------- VWW

    @staticmethod
    def create_visual_wake_words_model(
        input_shape: Tuple[int, int, int] = (96, 96, 1),
        dropout_rate: float = 0.3,
        l2_reg: float = 0.0,
    ) -> keras.Model:
        """Binary classifier for Visual Wake Words."""
        reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
        base = keras.applications.MobileNetV2(
            input_shape=(96, 96, 3),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False
        model = keras.Sequential([
            layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x), input_shape=input_shape),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu", kernel_regularizer=reg),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="sigmoid"),
        ])
        return model

    # ------------------------------------------------------------------ audio

    @staticmethod
    def create_audio_classification_model(
        input_shape: Tuple[int, int] = (40, 101),
        num_classes: int = 4,
        base_model_name: str = "MFCC_CNN",
        dropout_rate: float = 0.5,
        l2_reg: float = 0.0,
    ) -> keras.Model:
        """Audio classification model using MFCC spectrograms."""
        reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
        
        if base_model_name == "AudioLSTM":
            model = keras.Sequential([
                layers.Reshape((input_shape[1], input_shape[0]), input_shape=(*input_shape, 1)),
                layers.LSTM(64, return_sequences=True, kernel_regularizer=reg),
                layers.LSTM(64, kernel_regularizer=reg),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax")
            ])
        elif base_model_name == "AudioGRU":
            model = keras.Sequential([
                layers.Reshape((input_shape[1], input_shape[0]), input_shape=(*input_shape, 1)),
                layers.GRU(64, return_sequences=True, kernel_regularizer=reg),
                layers.GRU(64, kernel_regularizer=reg),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax")
            ])
        else:
            model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*input_shape, 1),
                              kernel_regularizer=reg),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=reg),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=reg),
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu", kernel_regularizer=reg),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax"),
            ])
        return model

    @staticmethod
    def create_keyword_spotting_model(
        input_shape: Tuple[int, int] = (40, 101),
        num_classes: int = 2,
        base_model_name: str = "MFCC_CNN",
        dropout_rate: float = 0.3,
        l2_reg: float = 0.0,
    ) -> keras.Model:
        """Lightweight keyword spotting model."""
        reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
        
        if base_model_name == "AudioLSTM":
            model = keras.Sequential([
                layers.Reshape((input_shape[1], input_shape[0]), input_shape=(*input_shape, 1)),
                layers.LSTM(32, kernel_regularizer=reg),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax")
            ])
        elif base_model_name == "AudioGRU":
            model = keras.Sequential([
                layers.Reshape((input_shape[1], input_shape[0]), input_shape=(*input_shape, 1)),
                layers.GRU(32, kernel_regularizer=reg),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax")
            ])
        else:
            model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*input_shape, 1),
                              kernel_regularizer=reg),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=reg),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu", kernel_regularizer=reg),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax"),
            ])
        return model

    # ----------------------------------------------------------------- router

    @staticmethod
    def create_model(
        task: str,
        num_classes: int = 10,
        base_model: str = "MobileNetV2",
        input_shape: tuple = (224, 224, 3),
        dropout_rate: float = 0.5,
        l2_reg: float = 0.0,
        trainable_layers: int = 0,
        augmentation: dict = None,
    ) -> keras.Model:
        if augmentation is None:
            augmentation = {}
            
        """Dispatch to the correct model builder."""
        factories = {
            "IMAGE_CLASSIFICATION": lambda: ModelFactory.create_image_classification_model(
                input_shape=input_shape, num_classes=num_classes,
                base_model_name=base_model, dropout_rate=dropout_rate, l2_reg=l2_reg,
                trainable_layers=trainable_layers, augmentation=augmentation, # <-- Passed down
            ),
            "OBJECT_DETECTION": lambda: ModelFactory.create_image_classification_model(
                input_shape=input_shape, num_classes=num_classes,
                base_model_name=base_model, dropout_rate=dropout_rate, l2_reg=l2_reg,
                trainable_layers=trainable_layers, augmentation=augmentation, # <-- Passed down
            ),
            "VISUAL_WAKE_WORDS": lambda: ModelFactory.create_visual_wake_words_model(
                input_shape=input_shape, dropout_rate=dropout_rate, l2_reg=l2_reg,
                trainable_layers=trainable_layers, augmentation=augmentation, # <-- Passed down
            ),
            "KEYWORD_SPOTTING": lambda: ModelFactory.create_keyword_spotting_model(
                input_shape=input_shape, num_classes=num_classes,
                base_model_name=base_model, dropout_rate=dropout_rate, l2_reg=l2_reg,
                trainable_layers=trainable_layers, augmentation=augmentation, # <-- Passed down
            ),
            "AUDIO_CLASSIFICATION": lambda: ModelFactory.create_audio_classification_model(
                input_shape=input_shape, num_classes=num_classes,
                base_model_name=base_model, dropout_rate=dropout_rate, l2_reg=l2_reg,
                trainable_layers=trainable_layers, augmentation=augmentation, # <-- Passed down
            ),
        }
        if task not in factories:
            raise ValueError(f"Unknown task: {task}")
        return factories[task]()