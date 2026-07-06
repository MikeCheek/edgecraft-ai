import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras as _keras_top  # needed for keras.saving.register_keras_serializable below
from tensorflow.keras import layers, regularizers
from typing import Tuple


# ---------------------------------------------------------------------------
# Custom layers (replacing layers.Lambda(lambda ...) - see note below)
# ---------------------------------------------------------------------------
# layers.Lambda wrapping a raw Python lambda (e.g. `lambda t: tf.repeat(...)`)
# looks fine at first: the model builds, trains, and even reloads with
# safe_mode=False. But the reconstructed lambda's __globals__ doesn't
# reliably carry over the enclosing module's imports after
# deserialization, so the FIRST time the model needs to be retraced -
# which TFLite conversion always does - it can fail with
# `NameError: name 'tf' is not defined` deep inside TensorFlow's tracing
# machinery. A properly registered Layer subclass has a real `call()`
# method (no serialized bytecode closures at all), so it doesn't have
# this problem, and doesn't even need Lambda's safe_mode=False escape
# hatch since the Keras-3 "unsafe Lambda deserialization" guard only
# applies to actual Lambda layers.
@_keras_top.saving.register_keras_serializable(package="EdgeCraftAI")
class ChannelTile3(keras.layers.Layer):
    """Tiles a 1-channel (grayscale) input to 3 channels, e.g. so it can
    feed an ImageNet-pretrained (RGB-only) backbone."""

    def call(self, inputs):
        return tf.repeat(inputs, 3, axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (3,)


@_keras_top.saving.register_keras_serializable(package="EdgeCraftAI")
class GrayscaleToRGB(keras.layers.Layer):
    """Converts a 1-channel image to 3-channel RGB via
    tf.image.grayscale_to_rgb (used by the Visual Wake Words model)."""

    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (3,)


@_keras_top.saving.register_keras_serializable(package="EdgeCraftAI")
class AudioToRGB(keras.layers.Layer):
    """Tiles a 1-channel MFCC/spectrogram input to 3 channels and resizes
    it to a fixed spatial size, so it can feed an ImageNet-pretrained
    backbone (used by the transfer-learning optimization path)."""

    def __init__(self, target_size=(96, 96), **kwargs):
        super().__init__(**kwargs)
        self.target_size = tuple(target_size)

    def call(self, inputs):
        return tf.image.resize(tf.repeat(inputs, 3, axis=-1), list(self.target_size))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_size[0], self.target_size[1], 3)

    def get_config(self):
        config = super().get_config()
        config.update({"target_size": self.target_size})
        return config


class ModelFactory:
    """Factory for creating TinyML models for different tasks.

    All models accept optional `dropout_rate` and `l2_reg` arguments so the
    LLM-advisor suggestions can actually be applied without editing code.
    """

    # Rough edge-suitability tiers for the image base models, used by the
    # LLM advisor and MCU advisor to steer recommendations away from models
    # that are unrealistic on MCU-class hardware. Params are approximate
    # (ImageNet-pretrained, include_top=False) and vary with input_shape.
    EDGE_SUITABILITY = {
        "Custom3LayerCNN":  {"tier": "tiny",   "approx_params_m": 0.05, "note": "Best for the tightest MCUs (Nano 33 BLE, Pico)"},
        "MobileNetV1_0.25": {"tier": "tiny",   "approx_params_m": 0.47, "note": "Smallest ImageNet-pretrained option; great default for ESP32-class boards"},
        "MobileNetV3Small": {"tier": "small",  "approx_params_m": 1.5,  "note": "Good default for ESP32-S3 with PSRAM"},
        "MobileNetV2":      {"tier": "medium", "approx_params_m": 2.3,  "note": "Fine for ESP32-S3 with PSRAM; heavier than V3Small for similar accuracy"},
        "EfficientNet":     {"tier": "large",  "approx_params_m": 4.0,  "note": "Not recommended for MCU deployment without aggressive pruning/quantization"},
        "ResNet50V2":       {"tier": "very_large", "approx_params_m": 23.5, "note": "Not suitable for MCU deployment - use for desktop/server baselines only"},
    }

    # --- Image Models ---

    @staticmethod
    def create_image_classification_model(
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 10,
        base_model_name: str = "MobileNetV2",
        dropout_rate: float = 0.5,
        l2_reg: float = 0.0,
        trainable_layers: int = 0,  # 0 = unfreeze all, >0 = unfreeze last N layers
        augmentation: dict = None,
    ) -> keras.Model:
        """Create image classification model with optional regularisation."""
        if augmentation is None:
            augmentation = {}

        # 1. Prepare Data Augmentation Block
        # FIX: only build the Sequential when there are actual aug layers;
        #      keras.Sequential([]) raises an error on build.
        aug_layers = []
        if augmentation:
            if augmentation.get("horizontal_flip"):
                aug_layers.append(layers.RandomFlip("horizontal"))
            if augmentation.get("random_rotation"):
                aug_layers.append(layers.RandomRotation(augmentation["random_rotation"]))
            if augmentation.get("random_crop"):
                aug_layers.append(layers.RandomZoom(0.2))

        # Only create the augmentation block when it has at least one layer
        data_augmentation = keras.Sequential(aug_layers, name="data_augmentation") if aug_layers else None

        # 2. Prepare Regularization
        reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

        # Helper: prepend augmentation layer to a list only when it exists
        def _with_aug(layer_list):
            return ([data_augmentation] + layer_list) if data_augmentation else layer_list

        # 3. Build Model Pipeline
        if base_model_name in ["MobileNetV2", "MobileNetV3Small", "MobileNetV1_0.25"]:
            channels = input_shape[2] if len(input_shape) >= 3 else 3
            # ImageNet-pretrained weights are only defined for 3-channel RGB
            # input. Previously, choosing one of these backbones together
            # with a 1-channel (grayscale) input_shape crashed with a
            # confusing Keras error at weight-load time ("Weight expects
            # shape (3, 3, 1, 16). Received saved weight with shape
            # (3, 3, 3, 16)"). We now build the backbone at (H, W, 3) and
            # transparently tile a 1-channel input to 3 channels beforehand,
            # the same pattern already used in create_visual_wake_words_model.
            backbone_input_shape = (input_shape[0], input_shape[1], 3)

            if base_model_name == "MobileNetV2":
                base = keras.applications.MobileNetV2(
                    input_shape=backbone_input_shape, include_top=False, weights="imagenet"
                )
            elif base_model_name == "MobileNetV1_0.25":
                # Width multiplier 0.25x - by far the smallest ImageNet-pretrained
                # option available in keras.applications, purpose-built for
                # microcontroller-class RAM/flash budgets (ESP32, Nano 33 BLE).
                base = keras.applications.MobileNet(
                    input_shape=backbone_input_shape, alpha=0.25, include_top=False, weights="imagenet"
                )
            else:
                base = keras.applications.MobileNetV3Small(
                    input_shape=backbone_input_shape, include_top=False, weights="imagenet"
                )

            if trainable_layers > 0:
                base.trainable = True
                for layer in base.layers[:-trainable_layers]:
                    layer.trainable = False
            else:
                base.trainable = True

            inp = keras.Input(shape=input_shape)
            x = data_augmentation(inp) if data_augmentation else inp
            if channels != 3:
                x = ChannelTile3(name="channel_adapter")(x)
            x = base(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(dropout_rate)(x)
            out = layers.Dense(num_classes, activation="softmax", kernel_regularizer=reg)(x)
            model = keras.Model(inp, out)

        elif base_model_name == "EfficientNet":
            channels = input_shape[2] if len(input_shape) >= 3 else 3
            backbone_input_shape = (input_shape[0], input_shape[1], 3)
            base = keras.applications.EfficientNetB0(
                input_shape=backbone_input_shape,
                include_top=False,
                weights="imagenet",
            )
            base.trainable = False

            inp = keras.Input(shape=input_shape)
            x = data_augmentation(inp) if data_augmentation else inp
            if channels != 3:
                x = ChannelTile3(name="channel_adapter")(x)
            x = base(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
            x = layers.Dropout(dropout_rate)(x)
            out = layers.Dense(num_classes, activation="softmax")(x)
            model = keras.Model(inp, out)

        elif base_model_name == "ResNet50V2":
            channels = input_shape[2] if len(input_shape) >= 3 else 3
            backbone_input_shape = (input_shape[0], input_shape[1], 3)
            base = keras.applications.ResNet50V2(
                input_shape=backbone_input_shape,
                include_top=False,
                weights="imagenet",
            )
            base.trainable = False

            inp = keras.Input(shape=input_shape)
            x = data_augmentation(inp) if data_augmentation else inp
            if channels != 3:
                x = ChannelTile3(name="channel_adapter")(x)
            x = base(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
            x = layers.Dropout(dropout_rate)(x)
            out = layers.Dense(num_classes, activation="softmax")(x)
            model = keras.Model(inp, out)

        else:
            # Custom3LayerCNN — ultra-low memory
            model = keras.Sequential(
                _with_aug([
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
            )

        return model

    # --- Visual Wake Words ---

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
            GrayscaleToRGB(input_shape=input_shape, name="grayscale_to_rgb"),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu", kernel_regularizer=reg),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="sigmoid"),
        ])
        return model

    # --- Audio Models ---

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
                layers.Dense(num_classes, activation="softmax"),
            ])
        elif base_model_name == "AudioGRU":
            model = keras.Sequential([
                layers.Reshape((input_shape[1], input_shape[0]), input_shape=(*input_shape, 1)),
                layers.GRU(64, return_sequences=True, kernel_regularizer=reg),
                layers.GRU(64, kernel_regularizer=reg),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax"),
            ])
        else:  # MFCC_CNN / WaveNet fallback
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
                layers.Dense(num_classes, activation="softmax"),
            ])
        elif base_model_name == "AudioGRU":
            model = keras.Sequential([
                layers.Reshape((input_shape[1], input_shape[0]), input_shape=(*input_shape, 1)),
                layers.GRU(32, kernel_regularizer=reg),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax"),
            ])
        else:  # MFCC_CNN / WaveNet fallback
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

    # --- Router ---

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
        """Dispatch to the correct model builder."""
        if augmentation is None:
            augmentation = {}

        factories = {
            "IMAGE_CLASSIFICATION": lambda: ModelFactory.create_image_classification_model(
                input_shape=input_shape, num_classes=num_classes,
                base_model_name=base_model, dropout_rate=dropout_rate, l2_reg=l2_reg,
                trainable_layers=trainable_layers, augmentation=augmentation,
            ),
            "OBJECT_DETECTION": lambda: ModelFactory.create_image_classification_model(
                input_shape=input_shape, num_classes=num_classes,
                base_model_name=base_model, dropout_rate=dropout_rate, l2_reg=l2_reg,
                trainable_layers=trainable_layers, augmentation=augmentation,
            ),
            "VISUAL_WAKE_WORDS": lambda: ModelFactory.create_visual_wake_words_model(
                input_shape=input_shape, dropout_rate=dropout_rate, l2_reg=l2_reg,
            ),
            "KEYWORD_SPOTTING": lambda: ModelFactory.create_keyword_spotting_model(
                input_shape=input_shape, num_classes=num_classes,
                base_model_name=base_model, dropout_rate=dropout_rate, l2_reg=l2_reg,
            ),
            "AUDIO_CLASSIFICATION": lambda: ModelFactory.create_audio_classification_model(
                input_shape=input_shape, num_classes=num_classes,
                base_model_name=base_model, dropout_rate=dropout_rate, l2_reg=l2_reg,
            ),
        }
        if task not in factories:
            raise ValueError(f"Unknown task: {task}")
        return factories[task]()
