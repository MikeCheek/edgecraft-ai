import os
import aiohttp
import json
import logging

from app.services.shared_state import data_manager

logger = logging.getLogger(__name__)

class LLMAdvisor:
    async def generate_suggestions(self, context, provider="openrouter", model_name="google/gemini-2.0-flash-lite-preview-02-05:free"):
        # 1. Safely Extract Training Context
        task = context.get("task", "Unknown Task")
        training_config = context.get("config", {})
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