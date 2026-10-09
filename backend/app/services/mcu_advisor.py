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

from pathlib import Path
from typing import Dict, List, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Rough desktop/laptop CPU clock speed used as the baseline for the
# host-machine timings coming out of evaluator.py. This is intentionally
# conservative; the resulting on-device estimate is a ballpark, not a promise.
_HOST_CLOCK_GHZ_ASSUMPTION = 3.0


TFLM_FIXED_OVERHEAD_BYTES = 8 * 1024
TFLM_PER_OP_OVERHEAD_BYTES = 512


def graph_ops(tflite_bytes: bytes) -> List[dict]:
    from app.services.evaluator import make_interpreter

    interpreter = make_interpreter(tflite_bytes, no_delegate=True)
    return interpreter._get_ops_details()  # stable since TF 2.6


def estimate_arena_bytes(tflite_bytes: bytes) -> Dict[str, int]:
    """Peak simultaneous activation memory + TFLM bookkeeping, in bytes."""
    from app.services.evaluator import make_interpreter

    interpreter = make_interpreter(tflite_bytes, no_delegate=True)
    ops = interpreter._get_ops_details()
    details = {d["index"]: d for d in interpreter.get_tensor_details()}

    def nbytes(idx: int) -> int:
        d = details.get(idx)
        if d is None:
            return 0
        return int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize if len(d["shape"]) else np.dtype(d["dtype"]).itemsize

    inputs = [d["index"] for d in interpreter.get_input_details()]
    outputs = [d["index"] for d in interpreter.get_output_details()]
    first_def: Dict[int, int] = {i: -1 for i in inputs}
    last_use: Dict[int, int] = {}
    for step, op in enumerate(ops):
        for t in op["outputs"]:
            if t >= 0:
                first_def.setdefault(t, step)
        for t in op["inputs"]:
            if t in first_def:  # only activations (weights are never produced by an op)
                last_use[t] = step
    for t in outputs:
        last_use[t] = len(ops)
    for t, s in first_def.items():
        last_use.setdefault(t, s)

    peak = 0
    for step in range(-1, len(ops) + 1):
        live = sum(nbytes(t) for t, s in first_def.items() if s <= step <= last_use[t])
        peak = max(peak, live)
    arena = peak + TFLM_FIXED_OVERHEAD_BYTES + TFLM_PER_OP_OVERHEAD_BYTES * len(ops)
    return {"arena_bytes": int(arena), "peak_activation_bytes": int(peak), "ops": len(ops)}


# Builtin ops TensorFlow Lite Micro implements (used when the optional
# tflite-micro package isn't installed to flag models that can't run).
TFLM_SUPPORTED_OPS = {
    "ABS", "ADD", "ADD_N", "ARG_MAX", "ARG_MIN", "AVERAGE_POOL_2D", "BATCH_MATMUL", "BATCH_TO_SPACE_ND",
    "BROADCAST_ARGS", "BROADCAST_TO", "CAST", "CEIL", "CONCATENATION", "CONV_2D", "COS", "CUMSUM",
    "DEPTH_TO_SPACE", "DEPTHWISE_CONV_2D", "DEQUANTIZE", "DIV", "ELU", "EQUAL", "EXP", "EXPAND_DIMS",
    "FILL", "FLOOR", "FLOOR_DIV", "FLOOR_MOD", "FULLY_CONNECTED", "GATHER", "GATHER_ND", "GREATER",
    "GREATER_EQUAL", "HARD_SWISH", "L2_NORMALIZATION", "L2_POOL_2D", "LEAKY_RELU", "LESS", "LESS_EQUAL",
    "LOG", "LOG_SOFTMAX", "LOGICAL_AND", "LOGICAL_NOT", "LOGICAL_OR", "LOGISTIC", "MAX_POOL_2D", "MAXIMUM",
    "MEAN", "MINIMUM", "MIRROR_PAD", "MUL", "NEG", "NOT_EQUAL", "PACK", "PAD", "PADV2", "PRELU", "QUANTIZE",
    "REDUCE_MAX", "REDUCE_MIN", "RELU", "RELU6", "RESHAPE", "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR",
    "ROUND", "RSQRT", "SELECT_V2", "SHAPE", "SIN", "SLICE", "SOFTMAX", "SPACE_TO_BATCH_ND", "SPACE_TO_DEPTH",
    "SPLIT", "SPLIT_V", "SQRT", "SQUARE", "SQUARED_DIFFERENCE", "SQUEEZE", "STRIDED_SLICE", "SUB", "SUM",
    "SVDF", "TANH", "TRANSPOSE", "TRANSPOSE_CONV", "UNIDIRECTIONAL_SEQUENCE_LSTM", "UNPACK", "ZEROS_LIKE",
}


def tflm_verify(tflite_bytes: bytes, max_arena: int = 16 * 1024 * 1024) -> Dict:
    """Check a model against TensorFlow Lite Micro.

    With the optional `tflite-micro` package installed this runs the real
    TFLM interpreter: it confirms every op is supported, finds the minimum
    tensor arena by bisection, and compares TFLM output with the desktop
    interpreter on a random input. Without it, ops are checked against
    TFLM_SUPPORTED_OPS.
    """
    try:
        ops = sorted({o["op_name"] for o in graph_ops(tflite_bytes)})
    except Exception as exc:
        return {"available": False, "error": f"Could not read model ops: {exc}", "ops": []}
    unsupported = [o for o in ops if o not in TFLM_SUPPORTED_OPS]
    result: Dict = {"ops": ops, "unsupported_ops": unsupported}
    try:
        from tflite_micro.python.tflite_micro import runtime as tflm
    except Exception:
        result.update({"available": False, "supported": not unsupported})
        return result

    result["available"] = True

    def _fits(size: int) -> bool:
        try:
            tflm.Interpreter.from_bytes(tflite_bytes, arena_size=size)
            return True
        except Exception:
            return False

    if not _fits(max_arena):
        result.update({"supported": False,
                       "error": f"TFLM failed to allocate (unsupported ops: {unsupported or 'unknown'})"})
        return result
    lo, hi = 1024, max_arena
    while hi - lo > 512:
        mid = (lo + hi) // 2
        if _fits(mid):
            hi = mid
        else:
            lo = mid
    result.update({"supported": True, "arena_bytes": int(hi)})

    try:
        from app.services.evaluator import make_interpreter

        ref = make_interpreter(tflite_bytes, no_delegate=True)
        inp, out = ref.get_input_details()[0], ref.get_output_details()[0]
        x = np.random.default_rng(0).random(inp["shape"]).astype(np.float32)
        if inp["dtype"] in (np.int8, np.uint8):
            from app.services.preprocessing import quantize_input

            x = quantize_input(x, *inp["quantization"], inp["dtype"])
        ref.set_tensor(inp["index"], x)
        ref.invoke()
        micro = tflm.Interpreter.from_bytes(tflite_bytes, arena_size=hi + 4096)
        micro.set_input(x, 0)
        micro.invoke()
        diff = np.abs(micro.get_output(0).astype(np.float32) - ref.get_tensor(out["index"]).astype(np.float32))
        result["max_abs_diff"] = float(diff.max())
    except Exception as exc:
        result["comparison_error"] = str(exc)
    return result


class MCUAdvisor:
    """Hardware advisor for microcontroller deployment"""

    BOARD_SPECS = {
        "ESP32_S3_N16R8": {
            "name": "ESP32-S3 N16R8",
            "ram_kb": 8192,       # 8MB PSRAM + 512KB on-chip SRAM
            "internal_sram_kb": 320,  # usable for static buffers
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
            "internal_sram_kb": 160,
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
        """Tensor-arena estimate from a liveness analysis of the TFLite graph.

        Walks the ops in execution order, tracks when each activation tensor
        is produced and last consumed, and takes the peak total size of
        tensors alive at the same time - the same quantity TFLite Micro's
        greedy memory planner minimises. Weights stay in flash and are not
        counted. A per-op allowance covers TFLM's persistent buffers
        (quantization params, op state). Falls back to a size multiplier if
        the graph can't be introspected.
        """
        try:
            est = estimate_arena_bytes(Path(tflite_path).read_bytes())
            return {"ram_kb": round(est["arena_bytes"] / 1024, 1), "method": "liveness_analysis",
                    "peak_activation_kb": round(est["peak_activation_bytes"] / 1024, 1),
                    "ops": est["ops"]}
        except Exception as e:
            logger.warning(f"Arena liveness analysis failed, falling back to size multiplier: {e}")
            return {"ram_kb": round(optimized_size_kb * 1.5 + 20, 1), "method": "size_multiplier_fallback"}

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

        tflm = (session.get("metrics") or {}).get("tflm") or {}
        ram_info = (
            {"ram_kb": round(tflm["arena_bytes"] / 1024, 1), "method": "tflite_micro_measured"}
            if tflm.get("supported") and tflm.get("arena_bytes") else None
        ) or (
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
        tflm_blocked = bool(tflm.get("unsupported_ops")) or (tflm.get("available") and not tflm.get("supported"))
        if tflm_blocked:
            warnings.append(
                "TensorFlow Lite Micro cannot run this model"
                + (f" (unsupported ops: {', '.join(tflm['unsupported_ops'])})" if tflm.get("unsupported_ops") else "")
            )
            suggestions.append("Re-train with a TFLM-friendly architecture or re-run the optimization with this version")
        if specs.get("internal_sram_kb") and ram_usage_kb > specs["internal_sram_kb"] and board.startswith("ESP32"):
            suggestions.append("Tensor arena exceeds internal SRAM - the exported sketch allocates it in PSRAM automatically")

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
                "not a hardware measurement. RAM usage is a liveness-analysis estimate of the "
                "TFLite Micro tensor arena (peak simultaneous activations plus bookkeeping)."
            ),
            "warnings": warnings,
            "suggestions": suggestions,
            "deployment_feasible": flash_percentage < 90 and ram_percentage < 90 and not tflm_blocked,
            "tflm": tflm or None,
        }
