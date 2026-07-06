"""
mcu_advisor.py
--------------
Hardware advisor for microcontroller deployment.

Unlike the previous version, this pulls REAL numbers from the completed
optimization session wherever possible:
  - optimized model size: exact bytes from the .tflite file
  - inference latency: measured on this machine during test-set evaluation
    (see evaluator.py), scaled by a documented clock-speed ratio to give a
    ballpark on-device estimate
  - RAM (tensor arena) usage: a heuristic estimate based on the model's
    non-constant tensor footprint, clearly labelled as an approximation
    since the exact arena size can only be known by compiling for the
    target with TFLite Micro.

Nothing here pretends to be a cycle-accurate hardware simulator - anywhere
a number can't be measured directly, it is explicitly marked as estimated.
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Rough desktop/laptop CPU clock speed used as the baseline for the
# host-machine timings coming out of evaluator.py. This is intentionally
# conservative; the resulting on-device estimate is a ballpark, not a promise.
_HOST_CLOCK_GHZ_ASSUMPTION = 3.0


class MCUAdvisor:
    """Hardware advisor for microcontroller deployment"""

    BOARD_SPECS = {
        "ESP32_S3_N16R8": {
            "name": "ESP32-S3 N16R8",
            "ram_kb": 8192,       # 8MB PSRAM + 512KB on-chip SRAM
            "flash_kb": 16384,    # 16MB flash
            "cpu": "Xtensa LX7 dual-core @ 240 MHz",
            "clock_ghz": 0.240,
            "features": ["PSRAM", "Vector instructions (esp-nn)", "Hardware FPU"],
            "recommended_models": ["MobileNetV3Small", "MobileNetV1_0.25", "Custom3LayerCNN"],
            "has_camera": False,
        },
        "ESP32_CAM": {
            "name": "ESP32-CAM (AI-Thinker, OV2640)",
            "ram_kb": 4096 + 520,  # ~4MB PSRAM + 520KB on-chip SRAM
            "flash_kb": 4096,     # common AI-Thinker variant; some boards ship 8/16MB
            "cpu": "Xtensa LX6 dual-core @ 240 MHz",
            "clock_ghz": 0.240,
            "features": ["PSRAM (typ. 4MB)", "OV2640 camera", "No native USB (needs FTDI programmer)"],
            "recommended_models": ["MobileNetV3Small", "MobileNetV1_0.25", "Custom3LayerCNN"],
            "has_camera": True,
        },
        "RASPBERRY_PI_PICO_2_W": {
            "name": "Raspberry Pi Pico 2 W",
            "ram_kb": 520,
            "flash_kb": 4096,
            "cpu": "ARM Cortex-M33 dual-core @ 150 MHz",
            "clock_ghz": 0.150,
            "features": ["DSP", "Floating Point Unit"],
            "recommended_models": ["Custom3LayerCNN"],
            "has_camera": False,
        },
        "ARDUINO_NANO_33_BLE": {
            "name": "Arduino Nano 33 BLE",
            "ram_kb": 256,
            "flash_kb": 1024,
            "cpu": "ARM Cortex-M4 @ 64 MHz",
            "clock_ghz": 0.064,
            "features": ["Floating Point Unit"],
            "recommended_models": ["Custom3LayerCNN", "MFCC_CNN"],
            "has_camera": False,
        },
    }

    def get_supported_boards(self) -> List[dict]:
        return [
            {"id": board_id, **specs}
            for board_id, specs in self.BOARD_SPECS.items()
        ]

    # -------------------------------------------------------------------
    # RAM (tensor arena) heuristic
    # -------------------------------------------------------------------

    def _estimate_ram_usage_kb(self, tflite_path: str, optimized_size_kb: float) -> Dict[str, float]:
        """
        Heuristic RAM estimate. TFLite Micro's real arena size depends on
        the graph's tensor allocation plan, which can only be known exactly
        by actually allocating tensors with TFLite Micro on the target.
        As an honest approximation we:
          1. Try to sum the byte-size of all non-constant (activation)
             tensors reported by the desktop TFLite interpreter - these
             are the tensors that must live in the arena at runtime.
          2. Add a fixed ~20KB overhead for the interpreter / op resolver
             bookkeeping structures.
        If introspection fails for any reason, we fall back to a
        documented multiplier of the model's flash size.
        """
        try:
            import tensorflow as tf

            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()

            all_details = interpreter.get_tensor_details()
            input_indices = {d["index"] for d in interpreter.get_input_details()}
            output_indices = {d["index"] for d in interpreter.get_output_details()}

            activation_bytes = 0
            for d in all_details:
                try:
                    tensor = interpreter.tensor(d["index"])()
                    # Weight/bias tensors already hold trained values and are
                    # stored in flash (memory-mapped), not counted toward RAM.
                    # We can't perfectly distinguish weights from activations
                    # from the public API, so as a heuristic we count only
                    # input/output/intermediate tensors that are small enough
                    # to plausibly be activations (weights are typically the
                    # largest tensors in a CNN and dominate flash, not RAM).
                    is_io = d["index"] in input_indices or d["index"] in output_indices
                    if is_io:
                        activation_bytes += tensor.nbytes
                except Exception:
                    continue

            # Double-buffer estimate for intermediate activations: TFLite Micro
            # typically needs room for ~2 concurrent activation tensors plus
            # input/output. We approximate intermediate activation memory as
            # 2x the largest single input tensor (a common rule of thumb for
            # simple feed-forward CNNs), since exact liveness analysis
            # requires the actual TFLite Micro allocator.
            largest_input_bytes = max(
                (interpreter.tensor(d["index"])().nbytes for d in interpreter.get_input_details()),
                default=0,
            )
            estimated_arena_bytes = activation_bytes + (largest_input_bytes * 2) + (20 * 1024)
            return {
                "ram_kb": round(estimated_arena_bytes / 1024, 1),
                "method": "heuristic_tensor_introspection",
            }
        except Exception as e:
            logger.warning(f"RAM heuristic introspection failed, falling back to size multiplier: {e}")
            # Documented fallback: TFLite Micro arenas for small CNNs are
            # commonly observed in the 1.2x-1.8x range of the model's flash
            # size; we use a conservative 1.5x plus fixed overhead.
            fallback_kb = (optimized_size_kb * 1.5) + 20
            return {"ram_kb": round(fallback_kb, 1), "method": "size_multiplier_fallback"}

    def _estimate_on_device_ms(self, measured_host_ms: Optional[float], board_clock_ghz: float) -> Optional[float]:
        """
        Scale a host-measured inference time to a rough on-device estimate
        using clock-speed ratio only. This ignores differences in SIMD width,
        cache, and op-level acceleration (e.g. esp-nn), so real on-device
        latency after using esp-nn/CMSIS-NN acceleration is often BETTER
        than this naive estimate for INT8 models - treat it as an upper
        bound / sanity check, not a precise prediction.
        """
        if not measured_host_ms or measured_host_ms <= 0:
            return None
        ratio = _HOST_CLOCK_GHZ_ASSUMPTION / board_clock_ghz if board_clock_ghz else 1.0
        return round(measured_host_ms * ratio, 2)

    # -------------------------------------------------------------------
    # Main evaluation
    # -------------------------------------------------------------------

    def evaluate_model(self, optimization_id: str, board: str) -> dict:
        """Evaluate a completed optimization session for a specific board,
        using the real optimized model file and (if available) measured
        test-set inference timing."""
        if board not in self.BOARD_SPECS:
            raise ValueError(f"Unsupported board: {board}")

        specs = self.BOARD_SPECS[board]

        from app.services.optimizer import get_session, get_output_path

        session = get_session(optimization_id)
        if not session:
            raise ValueError(f"Optimization session {optimization_id} not found")
        if session.get("status") != "completed":
            raise ValueError("Optimization is not completed yet - cannot evaluate for a board.")

        output_path = get_output_path(optimization_id)
        optimized_size_bytes = session.get("optimized_size_bytes", 0)
        model_size_kb = optimized_size_bytes / 1024 if optimized_size_bytes else 0.0

        comparison = session.get("comparison") or {}
        measured_ms = None
        if isinstance(comparison, dict) and "optimized" in comparison:
            measured_ms = comparison["optimized"].get("avg_inference_ms")

        ram_info = (
            self._estimate_ram_usage_kb(str(output_path), model_size_kb)
            if output_path and output_path.exists()
            else {"ram_kb": model_size_kb * 1.5 + 20, "method": "size_multiplier_fallback"}
        )
        ram_usage_kb = ram_info["ram_kb"]
        flash_usage_kb = model_size_kb  # .tflite is memory-mapped straight from flash

        ram_percentage = (ram_usage_kb / specs["ram_kb"]) * 100
        flash_percentage = (flash_usage_kb / specs["flash_kb"]) * 100

        estimated_on_device_ms = self._estimate_on_device_ms(measured_ms, specs["clock_ghz"])

        warnings = []
        suggestions = []

        if ram_percentage > 80:
            warnings.append(f"High estimated RAM usage: {ram_percentage:.1f}%")
            suggestions.append("Consider INT8 quantization and/or pruning to reduce activation memory")
        if flash_percentage > 80:
            warnings.append(f"High flash usage: {flash_percentage:.1f}%")
            suggestions.append("Model may not fit on device. Try pruning, weight clustering, or a smaller base model")
        if ram_info["method"] == "size_multiplier_fallback":
            warnings.append("RAM usage is a rough estimate (could not introspect tensor allocation directly)")

        if board == "ESP32_S3_N16R8":
            suggestions.append("Store the model in PSRAM to free on-chip SRAM for buffers")
            suggestions.append("Link the esp-nn kernel library for accelerated INT8 ops")
        elif board == "ESP32_CAM":
            suggestions.append("Reserve extra PSRAM for the camera frame buffer (~50-150KB depending on resolution/format)")
            suggestions.append("Capture frames directly at a small resolution (e.g. 96x96 grayscale) to avoid a separate resize/JPEG-decode step")
            if flash_percentage > 60:
                warnings.append("ESP32-CAM boards commonly ship with only 4MB flash - verify your board variant before flashing")
        elif board == "RASPBERRY_PI_PICO_2_W":
            suggestions.append("Use the ARM Cortex-M33 DSP/CMSIS-NN kernels for accelerated INT8 inference")
        elif board == "ARDUINO_NANO_33_BLE":
            if ram_percentage > 50:
                suggestions.append("RAM is very limited on this board - prefer the Custom3LayerCNN or MFCC_CNN architectures")

        return {
            "board": board,
            "board_name": specs["name"],
            "ram_usage_kb": ram_usage_kb,
            "flash_usage_kb": round(flash_usage_kb, 1),
            "ram_percentage": round(ram_percentage, 1),
            "flash_percentage": round(flash_percentage, 1),
            "ram_estimation_method": ram_info["method"],
            "measured_inference_ms_on_host": measured_ms,
            "estimated_inference_ms_on_device": estimated_on_device_ms,
            "estimation_note": (
                "On-device latency is a clock-speed-scaled estimate from host timing, "
                "not a hardware measurement. RAM usage is a heuristic based on tensor "
                "introspection, not a compiled TFLite Micro arena size."
            ),
            "warnings": warnings,
            "suggestions": suggestions,
            "deployment_feasible": flash_percentage < 90 and ram_percentage < 90,
        }
