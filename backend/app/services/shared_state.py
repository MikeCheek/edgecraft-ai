# app/services/shared_state.py
#
# This module is imported first by FastAPI at startup (via app/main.py).
# ALL GPU configuration MUST happen here, before any tf.keras / model call,
# because TensorFlow locks device settings on first use.

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


def _configure_gpu() -> None:
    """
    1. Enable memory growth  – prevents TF from gobbling all VRAM at startup,
       which would crash if a display driver or another process is using the GPU.
    2. Enable mixed precision – on Ampere/Turing/Ada GPUs (RTX 20xx/30xx/40xx)
       this roughly doubles throughput with no accuracy loss by running matrix
       ops in float16 while keeping accumulators in float32.
    3. Log what was found so startup is transparent.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        logger.warning(
            "No GPU detected by TensorFlow. Training will run on CPU.\n"
            "  • If you installed tensorflow[and-cuda]==2.17.1 on Linux/WSL2,\n"
            "    check that your NVIDIA driver is ≥ 525 and run:\n"
            "      python -c \"import tensorflow as tf; print(tf.config.list_physical_devices())\"\n"
            "  • On Windows native, GPU is not supported from TF 2.11+; use WSL2."
        )
        return

    # ── 1. Memory growth ────────────────────────────────────────────────────
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Can happen if TF was already initialised (e.g. during hot-reload)
            logger.warning(f"Could not set memory growth for {gpu}: {e}")

    # ── 2. Mixed precision (float16 compute, float32 variables) ─────────────
    # Only beneficial on compute-capability ≥ 7.0 (Volta / Turing / Ampere / Ada).
    # On older GPUs it may actually slow things down; TF handles this gracefully.
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision policy set to mixed_float16.")
    except Exception as e:
        logger.warning(f"Could not enable mixed precision: {e}")

    # ── 3. Log detected GPUs ────────────────────────────────────────────────
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get("device_name", gpu.name)
        cc = details.get("compute_capability", ("?", "?"))
        logger.info(f"GPU found: {name}  (compute capability {cc[0]}.{cc[1]})")


_configure_gpu()

# ── Singleton services ───────────────────────────────────────────────────────
from app.services.data_manager import DataManager  # noqa: E402
from app.services.trainer import Trainer            # noqa: E402
from app.services.converter import Converter        # noqa: E402

data_manager = DataManager()
trainer = Trainer()
converter = Converter()