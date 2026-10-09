"""
Process-wide singletons. GPU configuration has to happen here, before any
model is built, because TensorFlow locks device settings on first use.
"""

import logging
import os

import tensorflow as tf

logger = logging.getLogger(__name__)


def _configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("No GPU visible to TensorFlow; training runs on CPU.")
        return

    for gpu in gpus:
        try:
            # Don't grab all VRAM up front (display drivers / other processes).
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"Could not set memory growth for {gpu}: {e}")

    # float16 compute on Volta+ GPUs roughly doubles throughput. Models keep
    # their final layer in float32 and are re-saved in float32 after
    # training, so this never leaks into exported TFLite files.
    if os.environ.get("MIXED_PRECISION", "true").strip().lower() in ("1", "true", "yes"):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision policy set to mixed_float16.")
        except Exception as e:
            logger.warning(f"Could not enable mixed precision: {e}")

    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        cc = details.get("compute_capability", ("?", "?"))
        logger.info(f"GPU found: {details.get('device_name', gpu.name)} (compute capability {cc[0]}.{cc[1]})")


_configure_gpu()

from app.services.data_manager import DataManager  # noqa: E402
from app.services.trainer import Trainer  # noqa: E402

data_manager = DataManager()
trainer = Trainer()
