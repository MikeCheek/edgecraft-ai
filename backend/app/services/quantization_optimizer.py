import tensorflow as tf
import numpy as np
from typing import Tuple, Optional


class QuantizationOptimizer:
    """Handle model quantization and pruning for TinyML"""

    @staticmethod
    def apply_int8_quantization(model_path: str, representative_data: np.ndarray) -> bytes:
        """Apply INT8 post-training quantization"""
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        def representative_dataset_gen():
            for i in range(min(100, len(representative_data))):
                yield [representative_data[i:i+1].astype(np.float32)]

        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()
        return tflite_model

    @staticmethod
    def apply_float16_quantization(model_path: str) -> bytes:
        """Apply FLOAT16 quantization for balanced accuracy/size"""
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        return tflite_model

    @staticmethod
    def apply_float16_quantization_from_keras(model: tf.keras.Model) -> bytes:
        """Apply FLOAT16 quantization directly from a Keras model (no saved path needed)."""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        return converter.convert()

    @staticmethod
    def apply_pruning(model: tf.keras.Model, sparsity: float = 0.5) -> tf.keras.Model:
        """Apply weight pruning to reduce model size"""
        import tensorflow_model_optimization as tfmot

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=sparsity,
                begin_step=0,
                end_step=1000
            )
        }

        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        return pruned_model

    @staticmethod
    def convert_to_tflite(model: tf.keras.Model) -> bytes:
        """Convert Keras model to TFLite format"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        return tflite_model

    @staticmethod
    def estimate_model_size(model: tf.keras.Model) -> dict:
        """Estimate model size before and after quantization (from live Keras model)."""
        tflite_full = QuantizationOptimizer.convert_to_tflite(model)
        tflite_fp16 = QuantizationOptimizer.apply_float16_quantization_from_keras(model)

        full_size = len(tflite_full)
        fp16_size = len(tflite_fp16)
        # INT8 is roughly 25% of the float32 size — exact value requires
        # a representative dataset, so we give an estimate here.
        int8_size_est = int(full_size * 0.25)

        return {
            "full_precision": full_size,
            "int8_quantized": int8_size_est,
            "float16_quantized": fp16_size,
            "compression_ratio_int8": int8_size_est / full_size if full_size > 0 else 0,
            "compression_ratio_float16": fp16_size / full_size if full_size > 0 else 0,
        }