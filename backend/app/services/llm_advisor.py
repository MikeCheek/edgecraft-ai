import os
import aiohttp
import json
import logging

from app.services.shared_state import data_manager

logger = logging.getLogger(__name__)

class LLMAdvisor:
    async def generate_suggestions(self, context, provider="openrouter", model_name="google/gemini-2.0-flash-lite-preview-02-05:free"):
        # 1. Safely Extract Training Context
        # NOTE: `context` here is the raw training session dict returned by
        # Trainer.get_training_status() - a FLAT dict (task, dataset_id,
        # epochs, batch_size, base_model, input_shape, metrics, ...), not
        # nested under a "config" key. Reading `context.get("config", {})`
        # always returned an empty dict, silently breaking every
        # base_model/epochs/batch_size/learning_rate lookup below.
        task = context.get("task", "Unknown Task")
        training_config = context
        metrics_history = context.get("metrics", [])
        dataset_id = training_config.get("dataset_id") or context.get("dataset_id")

        # 2. Safely Extract Dataset Context
        dataset_info = {}
        labels = []
        if dataset_id:
            dataset_info = data_manager.get_dataset(dataset_id)
            # If your datasets store labels differently, adjust this:
            labels = list(dataset_info.get("labels", {}).keys()) if "labels" in dataset_info else "Unknown"

        # 3. Prevent Context Window Overflow (Limit to last 10 epochs)
        if len(metrics_history) > 10:
            metrics_summary = metrics_history[-10:]
        else:
            metrics_summary = metrics_history

        # 4. Construct the TinyML-Specific System Prompt
        system_prompt = (
            "You are an expert Edge AI Architect and Machine Learning Advisor embedded inside 'EdgeCraft AI', "
            "a local TinyML Studio. Your purpose is to help developers train highly efficient neural networks "
            "for extreme edge microcontrollers (e.g., ESP32, Raspberry Pi Pico, Arduino Nano).\n\n"
            "Analyze the provided dataset constraints, hyperparameters, and epoch metrics history. "
            "Identify issues like overfitting, vanishing gradients, under-capacity, or memory bloat.\n\n"
            "You MUST respond STRICTLY in valid JSON format as a list of objects. Do not include markdown formatting like ```json. "
            "Each object must precisely match this schema:\n"
            "[\n"
            "  {\n"
            "    \"suggestion\": \"Short, actionable title\",\n"
            "    \"reasoning\": \"Deep, metric-driven explanation of why this will help.\",\n"
            "    \"parameters_to_adjust\": {\"learning_rate\": 0.0005, \"batch_size\": 16},\n"
            "    \"estimated_improvement\": \"Expected result on accuracy or RAM/Flash.\"\n"
            "  }\n"
            "]"
        )

        # 5. Inject the Real-Time Variables into the User Prompt
        user_prompt = f"""
        Analyze the following TinyML training session and provide 2 to 3 concrete suggestions for improvement:

        [PROJECT & TASK CONTEXT]
        - Platform: EdgeCraft AI (TinyML deployment)
        - Task Type: {task}
        - Base Architecture: {training_config.get('base_model', 'Unknown Base Model')}
        
        [DATASET CONTEXT]
        - Total Samples: {dataset_info.get('sample_count', 'Unknown')}
        - Target Classes: {labels}
        - Validation Split: {training_config.get('validation_split', 'Unknown')}

        [CURRENT HYPERPARAMETERS]
        - Target Epochs: {training_config.get('epochs', 'Unknown')}
        - Batch Size: {training_config.get('batch_size', 'Unknown')}
        - Learning Rate: {training_config.get('learning_rate', 'Unknown')}

        [METRICS HISTORY (Last {len(metrics_summary)} Epochs)]
        {json.dumps(metrics_summary, indent=2)}

        Focus your advice heavily on microcontroller constraints. If validation loss is diverging from training loss, suggest TinyML-friendly regularization (like Dropout or heavier data augmentation). If accuracy is plateauing, suggest LR tuning or architecture changes.
        """

        # 6. Dispatch to your LLM API Wrapper (e.g., OpenRouter, OpenAI, or Ollama)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            if provider == "openrouter":
                return await self._call_openrouter(messages, model_name)
            elif provider == "ollama":
                return await self._call_ollama(messages, model_name)
            else:
                return self._mock_suggestions()
        except Exception as e:
            logger.error(f"LLM Advisor Error: {e}")
            return self._mock_suggestions()

    async def _call_openrouter(self, messages, model_name):
        # SECURE: Pulling directly from the backend environment
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing from the backend .env file.")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_name,
            "messages": messages,
            "response_format": {"type": "json_object"} # OpenRouter strict JSON mode
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload, timeout=45) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]

                        # Clean up markdown code blocks if the LLM ignores the response_format
                        if content.startswith("```json"):
                            content = content.replace("```json", "").replace("```", "").strip()
                        elif content.startswith("```"):
                            content = content.replace("```", "").strip()

                        parsed = json.loads(content)
                        # Handle varied JSON root structures
                        if isinstance(parsed, dict) and "suggestions" in parsed:
                            return parsed["suggestions"]
                        return parsed
                    else:
                        logger.error(f"OpenRouter returned status {response.status}: {await response.text()}")
            except Exception as e:
                logger.error(f"OpenRouter connection error: {e}")
        return self._mock_suggestions()

    async def _call_ollama(self, messages, model_name):
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": messages[-1]["content"] if messages else "",
            "stream": False,
            "format": "json"
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload, timeout=45) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data.get("response", "[]")
                        try:
                            parsed = json.loads(response_text)
                            if isinstance(parsed, dict) and "suggestions" in parsed:
                                return parsed["suggestions"]
                            return parsed
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                logger.error(f"Ollama connection error: {e}")
        return self._mock_suggestions()
        
    # ------------------------------------------------------------------
    # Board-specific optimization advice (used by /optimization/llm-optimize)
    # ------------------------------------------------------------------

    async def get_optimization_advice(
        self, optimization_id: str, board: str,
        use_local_llm: bool = False, model_name: str = "phi3",
    ) -> dict:
        """
        Real, grounded advice about deploying a specific completed
        optimization session to a specific board - pulls the actual
        model size / compression ratio / measured inference time rather
        than making generic statements.
        """
        from app.services.optimizer import get_session

        session = get_session(optimization_id)
        if not session or session.get("status") != "completed":
            raise ValueError("Optimization session not found or not completed yet.")

        optimized_kb = round(session.get("optimized_size_bytes", 0) / 1024, 1)
        original_kb = round(session.get("original_size_bytes", 0) / 1024, 1)
        ratio = session.get("compression_ratio", 0.0)
        method = session.get("frontend_method", session.get("method"))
        comparison = session.get("comparison") or {}

        if use_local_llm:
            try:
                from app.services.local_llm_advisor import LocalLLMAdvisor
                local = LocalLLMAdvisor(model_name=model_name)
                status = local.get_status()
                if status.get("available"):
                    return local.get_optimization_advice(board=board, model_size_kb=optimized_kb)
            except Exception as e:
                logger.warning(f"Local LLM optimization advice failed, falling back to rules: {e}")

        return self._rule_based_optimization_advice(
            board, optimized_kb, original_kb, ratio, method, comparison
        )

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
                    "Prefer the Custom3LayerCNN / MFCC_CNN architectures over any MobileNet variant",
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
        provider: str = "openrouter", model_name: str = "google/gemini-2.0-flash-lite-preview-02-05:free",
    ) -> dict:
        """
        Suggest a starting base_model + hyperparameters for a NEW training
        run, given the task, dataset shape, and the board the user intends
        to deploy to. Falls back to rule-based defaults tuned for edge
        deployment if no LLM is reachable.
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
        EfficientNet, ResNet50V2, Custom3LayerCNN for image tasks, or MFCC_CNN, AudioLSTM,
        AudioGRU for audio tasks) and full hyperparameters optimised for this specific
        microcontroller's memory constraints, not just for accuracy. Favor smaller,
        efficient architectures when RAM/Flash are tight.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            if provider == "openrouter":
                result = await self._call_openrouter(messages, model_name)
            elif provider == "ollama":
                result = await self._call_ollama(messages, model_name)
            else:
                result = None
            if isinstance(result, dict) and "base_model" in result:
                return result
            if isinstance(result, list) and result and "base_model" in result[0]:
                return result[0]
        except Exception as e:
            logger.error(f"recommend_training_params LLM error: {e}")

        return self._rule_based_training_recommendation(task, dataset_stats, target_board)

    @staticmethod
    def _rule_based_training_recommendation(task: str, dataset_stats: dict, target_board: str) -> dict:
        from app.services.mcu_advisor import MCUAdvisor

        specs = MCUAdvisor.BOARD_SPECS.get(target_board, {})
        ram_kb = specs.get("ram_kb", 512)
        is_tiny_board = ram_kb <= 1024  # Nano 33 BLE / Pico-class RAM budgets

        is_audio = task in ("AUDIO_CLASSIFICATION", "KEYWORD_SPOTTING")
        sample_count = dataset_stats.get("sample_count", 0)

        if is_audio:
            base_model = "MFCC_CNN"
            input_shape = [40, 101, 1] if task == "KEYWORD_SPOTTING" else [64, 101, 1]
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
            "augmentation": {"horizontal_flip": True, "random_rotation": 0.1} if not is_audio else {},
            "reasoning": (
                f"{target_board} has ~{ram_kb}KB RAM, so a "
                f"{'very compact custom CNN' if is_tiny_board else 'lightweight MobileNet variant'} "
                f"was chosen at a small input resolution to keep the tensor arena and flash "
                f"footprint deployable. Regularisation ({dropout_rate} dropout) is set based on "
                f"your dataset size ({sample_count or 'unknown'} samples) to reduce overfitting risk."
            ),
        }

    def _mock_suggestions(self):
        return [
            {
                "suggestion": "⚠️ Live LLM API Offline (Tuning Tip: Adjust Learning Rate & Batch Size)",
                "reasoning": (
                    "The live LLM Advisor API is currently down or unable to connect. As an automated "
                    "FAQ fallback: If your training metrics are stagnant, unstable, or your loss is "
                    "exploding, your learning rate is likely misconfigured for your batch structure."
                ),
                "parameters_to_adjust": {
                    "learning_rate": 0.001, 
                    "batch_size": 32
                },
                "estimated_improvement": "Stabilizes gradient descent and ensures reliable, steady loss reduction."
            },
            {
                "suggestion": "FAQ: Optimize Resolution for Target Hardware constraints (Avoid OOM)",
                "reasoning": (
                    "Why is my training slow or crashing on the target edge chip? Massive image dimensions "
                    "exhaust hardware micro-RAM. Downscaling images to traditional TinyML standards (like 96x96) "
                    "allows complex convolutional layers to run comfortably inside tight hardware boundaries."
                ),
                "parameters_to_adjust": {
                    "image_width": 96, 
                    "image_height": 96, 
                    "base_model": "MobileNetV3Small"
                },
                "estimated_improvement": "Drastically slashes model RAM/Flash footprint by ~50% to 70%."
            },
            {
                "suggestion": "FAQ: Mitigate Overfitting (High Train Accuracy vs. Poor Val Accuracy)",
                "reasoning": (
                    "Why does my model score 98% on training but fails completely on validation? The network is "
                    "memorizing your exact assets rather than learning generic visual concepts. Injecting robust "
                    "data augmentation rules (flips, slight shifts) and boosting dropout parameters addresses this."
                ),
                "parameters_to_adjust": {
                    "dropout_rate": 0.3, 
                    "epochs": 50,
                    "validation_split": 0.20
                },
                "estimated_improvement": "Bridges the accuracy generalization gap and stabilizes validation fluctuations."
            }
        ]