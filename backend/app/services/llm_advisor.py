import json
import logging
import os

from app import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama configuration (backend .env driven)
# ---------------------------------------------------------------------------
# OLLAMA_ENABLED=true   -> unlocks provider="ollama" for the LLM endpoints
# OLLAMA_MODEL=<name>   -> default local model to use (falls back to "phi3")
# OLLAMA_HOST=<url>     -> base URL of the Ollama server (falls back to
#                          http://localhost:11434)
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "phi3"


def _ollama_enabled() -> bool:
    return os.environ.get("OLLAMA_ENABLED", "false").strip().lower() in ("1", "true", "yes")


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


def _ollama_default_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def get_provider_config() -> dict:
    """
    Server-side, .env-driven view of which LLM providers are actually
    usable right now. The frontend calls GET /api/optimization/llm-config
    to read this at startup so it can:
      - auto-select the only available provider, or
      - offer the user a choice (remembered client-side) when both
        OPENROUTER_API_KEY and OLLAMA_ENABLED=true are present.
    """
    return {
        "openrouter_available": bool(os.environ.get("OPENROUTER_API_KEY")),
        "ollama_available": _ollama_enabled(),
        "ollama_model": _ollama_default_model(),
        "ollama_host": _ollama_host(),
    }


class LLMAdvisor:
    # Post-training review lives in app.services.training_review (deterministic
    # analysis + grounded AI suggestions). These helpers back the other LLM
    # features and go through llm_client.chat_json (retries, time budget,
    # tolerant JSON parsing).

    async def _call_openrouter(self, messages, model_name):
        from app.services.llm_client import chat_json

        return await chat_json(messages, "openrouter", model_name or config.DEFAULT_OPENROUTER_MODEL)

    async def _call_ollama(self, messages, model_name):
        from app.services.llm_client import chat_json

        if not model_name or "/" in model_name:
            model_name = _ollama_default_model()
        return await chat_json(messages, "ollama", model_name)

    # ------------------------------------------------------------------
    # Board-specific optimization advice (used by /optimization/llm-optimize)
    # ------------------------------------------------------------------

    async def get_optimization_advice(
        self, optimization_id: str, board: str,
        provider: str = None, model_name: str = None,
    ) -> dict:
        """
        Board-specific deployment advice grounded in the real optimization
        session (size, compression ratio, measured accuracy/latency). Uses the
        chosen LLM provider when given, and always falls back to a
        rule-based answer (marked source="rules") if the LLM is unavailable.
        """
        from app.services.optimizer import get_session
        from app.services.mcu_advisor import MCUAdvisor

        session = get_session(optimization_id)
        if not session or session.get("status") != "completed":
            raise ValueError("Optimization session not found or not completed yet.")

        optimized_kb = round(session.get("optimized_size_bytes", 0) / 1024, 1)
        original_kb = round(session.get("original_size_bytes", 0) / 1024, 1)
        ratio = session.get("compression_ratio", 0.0)
        method = session.get("frontend_method", session.get("method"))
        comparison = session.get("comparison") or {}
        rules = self._rule_based_optimization_advice(board, optimized_kb, original_kb, ratio, method, comparison)

        if provider not in ("openrouter", "ollama"):
            return {**rules, "source": "rules"}

        specs = MCUAdvisor.BOARD_SPECS.get(board, {})
        facts = {
            "board": board, "board_specs": {k: specs.get(k) for k in ("name", "ram_kb", "flash_kb", "cpu")},
            "method": method, "original_float32_tflite_kb": original_kb, "optimized_kb": optimized_kb,
            "compression_ratio": ratio,
            "evaluation": {k: v for k, v in comparison.items() if k in ("deltas", "test_split_used", "num_samples_evaluated")},
            "optimized_metrics": (comparison.get("optimized") or {}).get("metrics", {}).get("per_class"),
        }
        messages = [
            {"role": "system", "content": (
                "You are an embedded ML deployment expert inside EdgeCraft AI. Respond ONLY with JSON: "
                '{"summary": str, "strategies": [str], "challenges": [str], "testing": [str]}. '
                "Ground every point in the numbers provided; 3-5 items per list."
            )},
            {"role": "user", "content": json.dumps(facts, indent=2)},
        ]
        try:
            if provider == "openrouter":
                result = await self._call_openrouter(messages, model_name or config.DEFAULT_OPENROUTER_MODEL)
            else:
                if not _ollama_enabled():
                    raise ValueError("Ollama is not enabled (OLLAMA_ENABLED=true in backend .env).")
                result = await self._call_ollama(messages, model_name)
            if isinstance(result, list) and result:
                result = result[0]
            if not isinstance(result, dict) or "summary" not in result:
                raise RuntimeError("LLM response missing 'summary'.")
            for key in ("strategies", "challenges", "testing"):
                if not isinstance(result.get(key), list):
                    result[key] = rules.get(key, [])
            return {**result, "source": provider}
        except Exception as e:
            logger.warning(f"LLM optimization advice failed, using rules: {e}")
            return {**rules, "source": "rules", "llm_error": str(e)}

    @staticmethod
    def _rule_based_optimization_advice(
        board: str, optimized_kb: float, original_kb: float,
        ratio: float, method: str, comparison: dict,
    ) -> dict:
        board_specs = {
            "ESP32_S3_N16R8": {
                "strategies": [
                    "Store the model in PSRAM (8MB available) to free on-chip SRAM for buffers",
                    "Link the esp-nn kernel library for accelerated INT8 ops",
                    "Use the Huge APP partition scheme to leave room for the TFLite Micro runtime",
                ],
                "challenges": [
                    "Thermal throttling under sustained continuous inference",
                    "Power budget if Wi-Fi/BLE radio is active alongside inference",
                ],
                "testing": [
                    "Measure on-device latency with millis() around interpreter->Invoke()",
                    "Monitor heap with ESP.getFreeHeap() during inference",
                ],
            },
            "ESP32_CAM": {
                "strategies": [
                    "Capture frames at a small resolution close to the model's input to skip a separate resize step",
                    "Use grayscale (PIXFORMAT_GRAYSCALE) if the model was trained on 1-channel input to halve frame buffer size",
                    "Reserve headroom in PSRAM for the camera frame buffer in addition to the tensor arena",
                ],
                "challenges": [
                    "No native USB - you'll need an FTDI adapter to flash and monitor",
                    "Camera + PSRAM + model must all coexist in a relatively small RAM budget",
                ],
                "testing": [
                    "Confirm esp_camera_init() succeeds before testing inference",
                    "Log frame capture time separately from inference time to isolate bottlenecks",
                ],
            },
            "RASPBERRY_PI_PICO_2_W": {
                "strategies": [
                    "INT8 quantisation is close to mandatory given the 520KB RAM budget",
                    "Use CMSIS-NN kernels via the Cortex-M33 DSP unit for faster INT8 ops",
                ],
                "challenges": [
                    "520KB RAM is tight if using PSRAM-free buffers for image tasks",
                    "Verify the TFLite Micro Pico port you're using supports your ops",
                ],
                "testing": [
                    "Check available heap before AllocateTensors() to catch OOM early",
                    "Benchmark inference latency with the SDK's timer functions",
                ],
            },
            "ARDUINO_NANO_33_BLE": {
                "strategies": [
                    "Maximum INT8 quantisation plus pruning/clustering is strongly recommended",
                    "Prefer the Custom3LayerCNN (image) or DS_CNN (audio) architectures over any MobileNet variant",
                ],
                "challenges": [
                    "256KB RAM is extremely limited for anything beyond tiny models",
                    "Flash is only 1MB - even a well-compressed image model may not fit",
                ],
                "testing": [
                    "Confirm the compiled sketch's flash usage with arm-none-eabi-size before flashing",
                    "Watch for AllocateTensors() failures - the first sign RAM is too tight",
                ],
            },
        }
        base = board_specs.get(board, board_specs["ESP32_S3_N16R8"])

        summary = (
            f"Applying {method} took the model from {original_kb}KB to {optimized_kb}KB "
            f"({ratio*100:.1f}% of original size)."
        )
        if isinstance(comparison, dict) and "deltas" in comparison:
            deltas = comparison["deltas"]
            summary += (
                f" On the test set, accuracy changed by {deltas.get('accuracy_delta', 0)*100:+.2f} "
                f"percentage points, with a measured {deltas.get('speedup_factor', 1)}x speedup."
            )

        return {"summary": summary, **base}

    # ------------------------------------------------------------------
    # Pre-training recommendations (used to help pick base_model / hyperparameters
    # BEFORE a training run, not just tune an already-running one)
    # ------------------------------------------------------------------

    async def recommend_training_params(
        self, task: str, dataset_stats: dict, target_board: str = "ESP32_S3_N16R8",
        provider: str = "openrouter", model_name: str = config.DEFAULT_OPENROUTER_MODEL,
    ) -> dict:
        """
        Suggest a starting base_model + hyperparameters for a NEW training
        run, given the task, dataset shape, and the board the user intends
        to deploy to.
        """
        from app.services.mcu_advisor import MCUAdvisor

        board_specs = MCUAdvisor.BOARD_SPECS.get(target_board, {})

        system_prompt = (
            "You are an expert Edge AI Architect embedded inside 'EdgeCraft AI', a local "
            "TinyML Studio. A user is about to start a NEW training run and wants your "
            "advice on the best starting configuration for their target microcontroller.\n\n"
            "You MUST respond STRICTLY in valid JSON with this schema, no markdown fences:\n"
            "{\n"
            '  "base_model": "MobileNetV3Small",\n'
            '  "input_shape": [96, 96, 1],\n'
            '  "batch_size": 16,\n'
            '  "epochs": 50,\n'
            '  "learning_rate": 0.001,\n'
            '  "dropout_rate": 0.3,\n'
            '  "augmentation": {"horizontal_flip": true, "random_rotation": 0.1},\n'
            '  "reasoning": "Why this configuration suits the dataset and target board."\n'
            "}"
        )
        user_prompt = f"""
        Task: {task}
        Target board: {target_board} (RAM: {board_specs.get('ram_kb', 'unknown')}KB,
        Flash: {board_specs.get('flash_kb', 'unknown')}KB, {board_specs.get('cpu', '')})
        Dataset: {json.dumps(dataset_stats, indent=2)}

        Recommend a base_model (from: MobileNetV2, MobileNetV3Small, MobileNetV1_0.25,
        EfficientNet, ResNet50V2, Custom3LayerCNN for image classification / visual wake words;
        FOMO_MobileNetV2, FOMO_Tiny for object detection; DS_CNN, MFCC_CNN, AudioGRU,
        AudioLSTM for audio tasks) and full hyperparameters optimised for this specific
        microcontroller's memory constraints, not just for accuracy. Favor smaller,
        efficient architectures when RAM/Flash are tight. If the dataset's image_stats
        show non-uniform aspect ratios or resolutions much larger/smaller than your
        recommended input_shape, mention that in your reasoning and factor it into the
        augmentation/preprocessing advice.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            if provider == "openrouter":
                result = await self._call_openrouter(messages, model_name)
            elif provider == "ollama":
                if not _ollama_enabled():
                    raise ValueError(
                        "Ollama is not enabled on this backend. Set OLLAMA_ENABLED=true "
                        "in the backend .env (optionally with OLLAMA_MODEL / OLLAMA_HOST)."
                    )
                result = await self._call_ollama(messages, model_name)
            else:
                raise ValueError(f"Unknown provider '{provider}'. Expected 'openrouter' or 'ollama'.")
            if isinstance(result, list) and result:
                result = result[0]
            if not isinstance(result, dict) or "base_model" not in result:
                raise RuntimeError(f"LLM response missing 'base_model': {str(result)[:300]}")
            recommendation = self.validate_training_recommendation(task, result)
            recommendation["source"] = provider
            return recommendation
        except Exception as e:
            logger.warning(f"LLM training recommendation failed, using rules: {e}")
            rec = self._rule_based_training_recommendation(task, dataset_stats, target_board)
            rec["source"] = "rules"
            rec["llm_error"] = str(e)
            return rec

    @staticmethod
    def validate_training_recommendation(task: str, rec: dict) -> dict:
        """Clamp an LLM recommendation to values the trainer actually supports,
        so an invented model name or a nonsensical shape can't reach training."""
        from app.services.model_factory import AUDIO_MODELS, IMAGE_BACKBONES, OD_MODELS, ModelFactory
        from app.services import preprocessing

        if task in preprocessing.AUDIO_TASKS:
            allowed = AUDIO_MODELS
        elif task == "OBJECT_DETECTION":
            allowed = OD_MODELS
        else:
            allowed = IMAGE_BACKBONES
        out = dict(rec)
        notes = []
        base = ModelFactory.resolve_name(str(rec.get("base_model", "")))
        if base not in allowed:
            notes.append(f"'{rec.get('base_model')}' is not available; using {allowed[0]}.")
            base = allowed[0]
        out["base_model"] = base

        default_shape = list(preprocessing.default_input_shape(task))
        shape = rec.get("input_shape")
        if preprocessing.is_audio_task(task):
            out["input_shape"] = default_shape  # fixed by the audio front-end
        else:
            try:
                shape = [int(v) for v in shape]
                if len(shape) != 3 or shape[2] not in (1, 3) or not (16 <= shape[0] <= 320 and 16 <= shape[1] <= 320):
                    raise ValueError
                if task == "OBJECT_DETECTION":
                    shape = [shape[0] - shape[0] % 8, shape[1] - shape[1] % 8, shape[2]]
                out["input_shape"] = shape
            except Exception:
                notes.append(f"Invalid input_shape {shape}; using {default_shape}.")
                out["input_shape"] = default_shape

        def _num(key, lo, hi, default, cast=float):
            try:
                v = cast(rec.get(key, default))
                return min(hi, max(lo, v))
            except Exception:
                return default

        out["batch_size"] = _num("batch_size", 1, 256, 16, int)
        out["epochs"] = _num("epochs", 1, 300, 30, int)
        out["learning_rate"] = _num("learning_rate", 1e-6, 0.1, 0.001)
        out["dropout_rate"] = _num("dropout_rate", 0.0, 0.9, 0.3)
        aug = rec.get("augmentation") if isinstance(rec.get("augmentation"), dict) else {}
        out["augmentation"] = {k: v for k, v in aug.items() if k in (
            "horizontal_flip", "vertical_flip", "random_rotation", "random_zoom",
            "random_translation", "random_brightness", "random_contrast")}
        if notes:
            out["reasoning"] = (str(rec.get("reasoning", "")) + " [Adjusted: " + " ".join(notes) + "]").strip()
        return out

    @staticmethod
    def _rule_based_training_recommendation(task: str, dataset_stats: dict, target_board: str) -> dict:
        from app.services.mcu_advisor import MCUAdvisor

        specs = MCUAdvisor.BOARD_SPECS.get(target_board, {})
        ram_kb = specs.get("ram_kb", 512)
        is_tiny_board = ram_kb <= 1024  # Nano 33 BLE / Pico-class RAM budgets

        is_audio = task in ("AUDIO_CLASSIFICATION", "KEYWORD_SPOTTING")
        sample_count = dataset_stats.get("sample_count", 0)

        from app.services import preprocessing

        if is_audio:
            base_model = "DS_CNN"
            input_shape = list(preprocessing.default_input_shape(task))
        elif task == "OBJECT_DETECTION":
            base_model = "FOMO_Tiny" if is_tiny_board else "FOMO_MobileNetV2"
            input_shape = [96, 96, 1] if is_tiny_board else [96, 96, 3]
        elif task == "VISUAL_WAKE_WORDS":
            base_model = "Custom3LayerCNN" if is_tiny_board else "MobileNetV3Small"
            input_shape = [96, 96, 1]
        else:
            if is_tiny_board:
                base_model = "Custom3LayerCNN"
                input_shape = [64, 64, 1]
            else:
                base_model = "MobileNetV3Small"
                input_shape = [96, 96, 3]

        # Smaller datasets need more regularisation and fewer epochs to avoid overfitting.
        if sample_count and sample_count < 200:
            epochs, dropout_rate = 30, 0.5
        elif sample_count and sample_count < 1000:
            epochs, dropout_rate = 50, 0.4
        else:
            epochs, dropout_rate = 80, 0.3

        return {
            "base_model": base_model,
            "input_shape": input_shape,
            "batch_size": 8 if is_tiny_board else 16,
            "epochs": epochs,
            "learning_rate": 0.001,
            "dropout_rate": dropout_rate,
            "augmentation": {"horizontal_flip": True, "random_rotation": 0.1} if task in ("IMAGE_CLASSIFICATION", "VISUAL_WAKE_WORDS") else {},
            "reasoning": (
                f"{target_board} has ~{ram_kb}KB RAM, so a "
                f"{'very compact custom CNN' if is_tiny_board else 'lightweight MobileNet variant'} "
                f"was chosen at a small input resolution to keep the tensor arena and flash "
                f"footprint deployable. Regularisation ({dropout_rate} dropout) is set based on "
                f"your dataset size ({sample_count or 'unknown'} samples) to reduce overfitting risk."
            ),
        }
