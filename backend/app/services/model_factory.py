"""
Model builders for every supported task.

Design rules (all models):
* Inputs are float32 in [0, 1] (images) or z-scored MFCCs (audio) - exactly
  what app.services.preprocessing produces and what the exported sketch
  feeds on-device. Backbone-specific normalisation is a Rescaling layer
  inside the model, so it is baked into the .tflite file too.
* Pretrained backbones keep their BatchNorm layers frozen (inference mode)
  during fine-tuning - otherwise small batches wreck the ImageNet features.
  Keras 3 ignores the construction-time training=False once the backbone
  is made trainable, so set_backbone_trainable() freezes BN explicitly.
* The final layer always computes in float32, so mixed-precision GPU
  training stays numerically stable without wrapping the model in raw
  tf ops (which Keras 3 rejects).
* Data augmentation is NOT part of these models. The trainer wraps the
  model with an augmentation front-end for training only and saves the
  bare model, so augmentation layers never reach TFLite.
"""

from typing import Dict, Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow import keras
import keras as _keras_top
from tensorflow.keras import layers, regularizers


@_keras_top.saving.register_keras_serializable(package="EdgeCraftAI")
class ChannelTile3(keras.layers.Layer):
    """Tiles a 1-channel input to 3 channels for an RGB-only pretrained backbone.

    Implemented as a concatenation: tf.repeat lowers to TILE/SHAPE ops that
    TensorFlow Lite Micro doesn't support, which made grayscale models on
    pretrained backbones impossible to run on a microcontroller."""

    def call(self, inputs):
        return tf.concat([inputs, inputs, inputs], axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (3,)


@_keras_top.saving.register_keras_serializable(package="EdgeCraftAI")
class GrayscaleToRGB(keras.layers.Layer):
    """Kept only so models saved by earlier versions still deserialise."""

    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (3,)


@_keras_top.saving.register_keras_serializable(package="EdgeCraftAI")
class AudioToRGB(keras.layers.Layer):
    """Kept only so models saved by earlier versions still deserialise."""

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


# Input scaling each pretrained backbone expects, applied to [0, 1] pixels:
# (scale, offset) for layers.Rescaling.
_BACKBONE_INPUT_SCALING = {
    "MobileNetV2": (2.0, -1.0),        # [-1, 1]
    "MobileNetV1_0.25": (2.0, -1.0),   # [-1, 1]
    "MobileNetV3Small": (2.0, -1.0),   # [-1, 1] with include_preprocessing=False
    "EfficientNet": (255.0, 0.0),      # [0, 255], normalisation built in
    "ResNet50V2": (2.0, -1.0),         # [-1, 1]
}

IMAGE_BACKBONES = list(_BACKBONE_INPUT_SCALING) + ["Custom3LayerCNN"]
AUDIO_MODELS = ["DS_CNN", "MFCC_CNN", "AudioGRU", "AudioLSTM"]
OD_MODELS = ["FOMO_MobileNetV2", "FOMO_Tiny"]

# Names used by earlier versions -> current builder.
MODEL_ALIASES = {
    "WaveNet": "DS_CNN",
    "SSD_MobileNetV2": "FOMO_MobileNetV2",
    "EfficientDet_Lite": "FOMO_MobileNetV2",
    "YOLO_Nano": "FOMO_Tiny",
    "NanoDet": "FOMO_Tiny",
}

# FOMO output grid is input / 8 on each axis.
FOMO_STRIDE = 8


def _final_dense(num_classes: int, reg=None, name: str = "predictions"):
    return layers.Dense(num_classes, activation="softmax", kernel_regularizer=reg,
                        dtype="float32", name=name)


def set_backbone_trainable(base: keras.Model, trainable_layers: int = 0) -> None:
    """Unfreeze a pretrained backbone (only its last `trainable_layers`
    layers if > 0) while keeping every BatchNormalization layer frozen: a
    non-trainable BN layer runs in inference mode with its ImageNet
    statistics, which fine-tuning on small batches would otherwise destroy."""
    base.trainable = True
    if trainable_layers > 0:
        for layer in base.layers[:-trainable_layers]:
            layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


class ModelFactory:
    """Factory for creating TinyML models for different tasks."""

    EDGE_SUITABILITY: Dict[str, Dict] = {
        "Custom3LayerCNN":  {"tier": "tiny",   "approx_params_m": 0.03, "note": "Best for the tightest MCUs (Nano 33 BLE, Pico)"},
        "MobileNetV1_0.25": {"tier": "tiny",   "approx_params_m": 0.22, "note": "Smallest ImageNet-pretrained option; great default for ESP32-class boards"},
        "MobileNetV3Small": {"tier": "small",  "approx_params_m": 0.94, "note": "Good default for ESP32-S3 with PSRAM"},
        "MobileNetV2":      {"tier": "medium", "approx_params_m": 2.3,  "note": "Fine for ESP32-S3 with PSRAM; heavier than V3Small for similar accuracy"},
        "EfficientNet":     {"tier": "large",  "approx_params_m": 4.0,  "note": "Not recommended for MCU deployment"},
        "ResNet50V2":       {"tier": "very_large", "approx_params_m": 23.5, "note": "Desktop/server baseline only"},
        "DS_CNN":           {"tier": "tiny",   "approx_params_m": 0.02, "note": "Depthwise-separable CNN, the standard MCU keyword-spotting model"},
        "MFCC_CNN":         {"tier": "small",  "approx_params_m": 0.1,  "note": "Plain CNN over MFCCs"},
        "AudioGRU":         {"tier": "small",  "approx_params_m": 0.05, "note": "Recurrent; check TFLite Micro op support for your board"},
        "AudioLSTM":        {"tier": "small",  "approx_params_m": 0.06, "note": "Recurrent; check TFLite Micro op support for your board"},
        "FOMO_MobileNetV2": {"tier": "small",  "approx_params_m": 0.1,  "note": "Centroid detector on a MobileNetV2-0.35 trunk; fits ESP32-S3"},
        "FOMO_Tiny":        {"tier": "tiny",   "approx_params_m": 0.02, "note": "From-scratch centroid detector for very small MCUs"},
    }

    @staticmethod
    def resolve_name(name: str) -> str:
        return MODEL_ALIASES.get(name, name)

    # ------------------------------------------------------------------
    # Augmentation (training-only front-end, see Trainer)
    # ------------------------------------------------------------------

    @staticmethod
    def build_augmentation(augmentation: Optional[dict]) -> Optional[keras.Sequential]:
        augmentation = augmentation or {}
        aug = []
        if augmentation.get("horizontal_flip"):
            aug.append(layers.RandomFlip("horizontal"))
        if augmentation.get("vertical_flip"):
            aug.append(layers.RandomFlip("vertical"))
        if augmentation.get("random_rotation"):
            aug.append(layers.RandomRotation(float(augmentation["random_rotation"])))
        if augmentation.get("random_crop") or augmentation.get("random_zoom"):
            aug.append(layers.RandomZoom(float(augmentation.get("random_zoom") or 0.2)))
        if augmentation.get("random_translation"):
            t = float(augmentation["random_translation"])
            aug.append(layers.RandomTranslation(t, t))
        if augmentation.get("random_brightness"):
            aug.append(layers.RandomBrightness(float(augmentation["random_brightness"]), value_range=(0.0, 1.0)))
        if augmentation.get("random_contrast"):
            aug.append(layers.RandomContrast(float(augmentation["random_contrast"])))
        return keras.Sequential(aug, name="augmentation") if aug else None

    # ------------------------------------------------------------------
    # Image classification (also used for Visual Wake Words)
    # ------------------------------------------------------------------

    @staticmethod
    def _pretrained_backbone(name: str, input_hw: Tuple[int, int]):
        shape = (input_hw[0], input_hw[1], 3)
        if name == "MobileNetV2":
            return keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights="imagenet")
        if name == "MobileNetV1_0.25":
            return keras.applications.MobileNet(input_shape=shape, alpha=0.25, include_top=False, weights="imagenet")
        if name == "MobileNetV3Small":
            return keras.applications.MobileNetV3Small(
                input_shape=shape, include_top=False, weights="imagenet",
                include_preprocessing=False, minimalistic=False,
            )
        if name == "EfficientNet":
            return keras.applications.EfficientNetB0(input_shape=shape, include_top=False, weights="imagenet")
        if name == "ResNet50V2":
            return keras.applications.ResNet50V2(input_shape=shape, include_top=False, weights="imagenet")
        raise ValueError(f"Unknown backbone {name}")

    @staticmethod
    def create_image_classification_model(
        input_shape: Tuple[int, int, int] = (96, 96, 3),
        num_classes: int = 2,
        base_model_name: str = "MobileNetV3Small",
        dropout_rate: float = 0.3,
        l2_reg: float = 0.0,
        trainable_layers: int = 0,
    ) -> keras.Model:
        reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
        channels = input_shape[2] if len(input_shape) >= 3 else 3
        inp = keras.Input(shape=tuple(input_shape), name="input")

        if base_model_name in _BACKBONE_INPUT_SCALING:
            scale, offset = _BACKBONE_INPUT_SCALING[base_model_name]
            x = inp
            if channels != 3:
                x = ChannelTile3(name="channel_adapter")(x)
            x = layers.Rescaling(scale, offset, name="backbone_scaling")(x)
            base = ModelFactory._pretrained_backbone(base_model_name, (input_shape[0], input_shape[1]))
            base._name = "backbone"
            set_backbone_trainable(base, trainable_layers)
            x = base(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            if base_model_name in ("EfficientNet", "ResNet50V2"):
                x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
            x = layers.Dropout(dropout_rate)(x)
            out = _final_dense(num_classes, reg)(x)
            return keras.Model(inp, out, name=f"IC_{base_model_name}")

        # Custom3LayerCNN - from scratch, ultra-low memory.
        x = inp
        for filters in (8, 16, 32):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
            x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        out = _final_dense(num_classes, reg)(x)
        return keras.Model(inp, out, name="IC_Custom3LayerCNN")

    # ------------------------------------------------------------------
    # Audio (keyword spotting / audio classification)
    # ------------------------------------------------------------------

    @staticmethod
    def create_audio_model(
        input_shape: Sequence[int],
        num_classes: int,
        base_model_name: str = "DS_CNN",
        dropout_rate: float = 0.3,
        l2_reg: float = 0.0,
    ) -> keras.Model:
        reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
        n_mfcc, frames = int(input_shape[0]), int(input_shape[1])
        inp = keras.Input(shape=(n_mfcc, frames, 1), name="input")

        if base_model_name in ("AudioLSTM", "AudioGRU"):
            # (n_mfcc, frames, 1) -> (frames, n_mfcc): a real transpose, not a
            # reshape, so each timestep is one MFCC frame.
            x = layers.Permute((2, 1, 3))(inp)
            x = layers.Reshape((frames, n_mfcc))(x)
            rnn = layers.LSTM if base_model_name == "AudioLSTM" else layers.GRU
            # unroll=True: a static graph converts to plain TFLite ops
            # (TensorList-based loops can't be INT8-quantized or run on TFLM).
            x = rnn(48, return_sequences=True, unroll=True, kernel_regularizer=reg)(x)
            x = rnn(48, unroll=True, kernel_regularizer=reg)(x)
        elif base_model_name == "MFCC_CNN":
            x = inp
            for filters in (32, 64):
                x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
                x = layers.MaxPooling2D(2)(x)
            x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
        else:
            # DS-CNN (Zhang et al., "Hello Edge"), small variant. BatchNorm
            # momentum 0.9 (instead of Keras' 0.99) so the moving statistics
            # keep up on the few hundred clips typical of a custom dataset.
            x = layers.Conv2D(64, (10, 4), strides=(2, 2), padding="same", use_bias=False,
                              kernel_regularizer=reg)(inp)
            x = layers.BatchNormalization(momentum=0.9)(x)
            x = layers.ReLU()(x)
            for _ in range(4):
                x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
                x = layers.BatchNormalization(momentum=0.9)(x)
                x = layers.ReLU()(x)
                x = layers.Conv2D(64, 1, use_bias=False, kernel_regularizer=reg)(x)
                x = layers.BatchNormalization(momentum=0.9)(x)
                x = layers.ReLU()(x)
            x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        out = _final_dense(num_classes, reg)(x)
        return keras.Model(inp, out, name=f"AUDIO_{base_model_name}")

    # Back-compat names
    @staticmethod
    def create_keyword_spotting_model(input_shape=(40, 49), num_classes=2, base_model_name="DS_CNN",
                                      dropout_rate=0.3, l2_reg=0.0):
        return ModelFactory.create_audio_model(input_shape, num_classes, base_model_name, dropout_rate, l2_reg)

    @staticmethod
    def create_audio_classification_model(input_shape=(40, 99), num_classes=4, base_model_name="DS_CNN",
                                          dropout_rate=0.3, l2_reg=0.0):
        return ModelFactory.create_audio_model(input_shape, num_classes, base_model_name, dropout_rate, l2_reg)

    # ------------------------------------------------------------------
    # Object detection: FOMO-style centroid detection
    # ------------------------------------------------------------------

    @staticmethod
    def create_object_detection_model(
        input_shape: Tuple[int, int, int] = (96, 96, 3),
        num_classes: int = 1,
        base_model_name: str = "FOMO_MobileNetV2",
        dropout_rate: float = 0.1,
        l2_reg: float = 0.0,
        trainable_layers: int = 0,
    ) -> keras.Model:
        """Faster-Objects-More-Objects style detector.

        Output: (H/8, W/8, num_classes + 1) softmax grid. Channel 0 is
        background; a cell whose argmax is k>0 contains the centre of an
        object of class k-1. Small enough for MCUs, trains with a plain
        weighted cross-entropy, and post-processing on-device is just an
        argmax per cell.
        """
        reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
        h, w = int(input_shape[0]), int(input_shape[1])
        if h % FOMO_STRIDE or w % FOMO_STRIDE:
            raise ValueError(f"Object detection input height/width must be multiples of {FOMO_STRIDE}.")
        channels = input_shape[2] if len(input_shape) >= 3 else 3
        inp = keras.Input(shape=(h, w, channels), name="input")
        x = inp

        if base_model_name == "FOMO_MobileNetV2":
            if channels != 3:
                x = ChannelTile3(name="channel_adapter")(x)
            x = layers.Rescaling(2.0, -1.0, name="backbone_scaling")(x)
            base = keras.applications.MobileNetV2(
                input_shape=(h, w, 3), alpha=0.35, include_top=False, weights="imagenet"
            )
            trunk = keras.Model(base.input, base.get_layer("block_6_expand_relu").output, name="backbone")
            set_backbone_trainable(trunk, trainable_layers)
            x = trunk(x, training=False)
        else:
            for i, filters in enumerate((8, 16, 32)):
                x = layers.Conv2D(filters, 3, strides=2, padding="same", use_bias=False,
                                  kernel_regularizer=reg, name=f"fomo_s{i}")(x)
                x = layers.BatchNormalization(momentum=0.9)(x)
                x = layers.ReLU()(x)
                x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=reg)(x)
                x = layers.BatchNormalization(momentum=0.9)(x)
                x = layers.ReLU()(x)

        x = layers.Conv2D(32, 1, activation="relu", kernel_regularizer=reg, name="head_conv")(x)
        x = layers.Dropout(dropout_rate)(x)
        out = layers.Conv2D(num_classes + 1, 1, activation="softmax", dtype="float32", name="head_logits")(x)
        return keras.Model(inp, out, name=f"OD_{base_model_name}")

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    @staticmethod
    def create_model(
        task: str,
        num_classes: int = 2,
        base_model: str = "MobileNetV3Small",
        input_shape: tuple = (96, 96, 3),
        dropout_rate: float = 0.3,
        l2_reg: float = 0.0,
        trainable_layers: int = 0,
        augmentation: dict = None,  # accepted for back-compat; see build_augmentation
    ) -> keras.Model:
        base_model = ModelFactory.resolve_name(base_model)
        if task in ("IMAGE_CLASSIFICATION", "VISUAL_WAKE_WORDS"):
            if base_model not in IMAGE_BACKBONES:
                base_model = "Custom3LayerCNN"
            return ModelFactory.create_image_classification_model(
                input_shape=tuple(input_shape), num_classes=num_classes, base_model_name=base_model,
                dropout_rate=dropout_rate, l2_reg=l2_reg, trainable_layers=trainable_layers,
            )
        if task in ("KEYWORD_SPOTTING", "AUDIO_CLASSIFICATION"):
            if base_model not in AUDIO_MODELS:
                base_model = "DS_CNN"
            return ModelFactory.create_audio_model(
                input_shape=input_shape, num_classes=num_classes, base_model_name=base_model,
                dropout_rate=dropout_rate, l2_reg=l2_reg,
            )
        if task == "OBJECT_DETECTION":
            if base_model not in OD_MODELS:
                base_model = "FOMO_MobileNetV2"
            return ModelFactory.create_object_detection_model(
                input_shape=tuple(input_shape), num_classes=num_classes, base_model_name=base_model,
                dropout_rate=dropout_rate, l2_reg=l2_reg, trainable_layers=trainable_layers,
            )
        raise ValueError(f"Unknown task: {task}")

    @staticmethod
    def get_backbone(model: keras.Model) -> Optional[keras.Model]:
        """The nested pretrained backbone (for freeze / fine-tune phases), if any."""
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                return layer
        return None
