# from typing import Dict, List
# import json

# class LLMAdvisor:
#     """LLM-powered advisor for model optimization suggestions"""

#     def __init__(self):
#         self.llm_suggestions_db = {}

#     def get_suggestions(self, training_id: str, metrics: dict, context: dict = None) -> List[dict]:
#         """Get LLM-powered suggestions with rich session context injected"""
#         context = context or {}
#         suggestions = []

#         loss = metrics.get("loss", 1.0)
#         accuracy = metrics.get("accuracy", 0.0)
#         val_loss = metrics.get("val_loss", 1.0)
#         val_accuracy = metrics.get("val_accuracy", 0.0)
        
#         base_model = context.get("base_model", "the selected model")
#         lr = context.get("learning_rate", 0.001)
#         dropout = context.get("dropout_rate", 0.5)
#         epochs = context.get("epochs", 50)

#         # Smart Overfitting Check
#         if (accuracy - val_accuracy) > 0.08:
#             new_dropout = min(0.8, dropout + 0.1)
#             new_l2 = context.get("l2_reg", 0.0) + 0.001
#             suggestions.append({
#                 "suggestion": "Model is experiencing significant overfitting.",
#                 "reasoning": f"Training accuracy ({accuracy:.2%}) outpaces validation accuracy ({val_accuracy:.2%}) by over 8%. {base_model} might be memorizing the training data.",
#                 "parameters_to_adjust": {
#                     "dropout_rate": round(new_dropout, 2),
#                     "l2_regularization": round(new_l2, 4),
#                     "early_stopping": "Enable to halt training before overfitting worsens"
#                 },
#                 "estimated_improvement": "Better generalization on unseen edge data (expected +3-5% val acc)."
#             })

#         # Underfitting / Non-convergence Check
#         if accuracy < 0.70 and val_accuracy < 0.70:
#             suggestions.append({
#                 "suggestion": "Model is underfitting and struggling to learn features.",
#                 "reasoning": f"Both train and val accuracy are exceptionally low. The current learning rate ({lr}) might be suboptimal or {base_model} lacks the capacity for this dataset.",
#                 "parameters_to_adjust": {
#                     "learning_rate": lr * 0.1 if loss > 1.5 else lr * 5,
#                     "epochs": int(epochs * 1.5),
#                     "base_model": "Try a more complex model like EfficientNet or ResNet50V2 if RAM permits"
#                 },
#                 "estimated_improvement": "Stronger feature extraction (+10-15% overall accuracy)."
#             })

#         # Convergence Confidence Check
#         if val_loss > 0.6 and accuracy > 0.8:
#             suggestions.append({
#                 "suggestion": "Validation Loss is high despite good accuracy (Confidence Issue).",
#                 "reasoning": "The model makes correct predictions but with low confidence probabilities. This often happens if the model is over-penalized or the learning rate prevents final convergence.",
#                 "parameters_to_adjust": {
#                     "learning_rate": lr * 0.5,
#                     "batch_size": "Increase by 2x for smoother gradient descent"
#                 },
#                 "estimated_improvement": "Lower validation loss (target < 0.4)."
#             })

#         # Production Ready Check
#         if val_accuracy >= 0.90 and abs(accuracy - val_accuracy) <= 0.05:
#             suggestions.append({
#                 "suggestion": f"Excellent {base_model} performance!",
#                 "reasoning": f"High validation accuracy ({val_accuracy:.2%}) with a minimal train/val gap indicates a highly robust model ready for hardware quantization.",
#                 "parameters_to_adjust": {
#                     "next_step": "Proceed to the Optimization & Quantization tab"
#                 },
#                 "estimated_improvement": "Production ready for TinyML edge deployment."
#             })
            
#         # Fallback if nothing specific triggered
#         if not suggestions:
#             suggestions.append({
#                 "suggestion": "Training completed with moderate success.",
#                 "reasoning": "The model learned basic features but plateaued. There is room for parameter tuning.",
#                 "parameters_to_adjust": {
#                     "epochs": epochs + 20,
#                     "learning_rate": lr * 0.5
#                 },
#                 "estimated_improvement": "Gradual refinement (+2-4% accuracy)."
#             })

#         return suggestions

#     def get_optimization_advice(self, optimization_id: str, board: str) -> Dict:
#         """Get LLM-powered advice on optimization for specific board deployment"""
#         advice = {
#             "board": board,
#             "optimization_strategies": [],
#             "deployment_tips": [],
#             "potential_issues": []
#         }

#         # Board-specific optimization advice
#         if board == "ESP32_S3_N16R8":
#             advice["optimization_strategies"] = [
#                 "INT8 quantization is recommended for this powerful board",
#                 "Take advantage of the 8MB PSRAM for larger models",
#                 "Use esp-nn acceleration library components for optimized 8-bit inference"
#             ]
#             advice["deployment_tips"] = [
#                 "Store model in external PSRAM to preserve on-chip RAM",
#                 "Use hardware acceleration for matrix operations",
#                 "Consider using WiFi for OTA model updates"
#             ]
#             advice["potential_issues"] = [
#                 "Thermal management at high inference rates",
#                 "Power consumption with continuous inference"
#             ]

#         elif board == "RASPBERRY_PI_PICO_2_W":
#             advice["optimization_strategies"] = [
#                 "INT8 quantization is mandatory due to 520KB RAM limit",
#                 "Consider FLOAT16 if accuracy is critical",
#                 "Implement quantization-aware training for better post-quantization performance"
#             ]
#             advice["deployment_tips"] = [
#                 "Use the ARM Cortex-M33 DSP unit for inference acceleration",
#                 "Load model from flash during initialization",
#                 "Implement batching for multiple inferences"
#             ]
#             advice["potential_issues"] = [
#                 "Very limited on-chip RAM - streaming inference may be necessary",
#                 "Slow clock speed requires efficient quantization"
#             ]

#         elif board == "ARDUINO_NANO_33_BLE":
#             advice["optimization_strategies"] = [
#                 "Aggressive INT8 quantization required (256KB RAM limit)",
#                 "Consider model pruning to 70%+ sparsity",
#                 "Use lightweight architectures like MobileNet v1"
#             ]
#             advice["deployment_tips"] = [
#                 "Implement block-wise inference to process model in chunks",
#                 "Use BLE for wireless model updates",
#                 "Consider storing very small models (< 50KB) for real-time inference"
#             ]
#             advice["potential_issues"] = [
#                 "Severe memory constraints may require model splitting",
#                 "Slow inference due to limited computational resources",
#                 "Limited model complexity possible"
#             ]

#         advice["next_steps"] = [
#             f"1. Apply INT8 quantization optimization for {board}",
#             "2. Test inference latency and accuracy on actual hardware",
#             "3. Implement error handling and fallback strategies",
#             "4. Consider edge caching for frequently used models"
#         ]

#         return advice

import aiohttp
import json
import logging

logger = logging.getLogger(__name__)

class LLMAdvisor:
    async def generate_suggestions(self, context, provider="ollama", model_name="llama3", api_key=None):
        prompt = self._build_prompt(context)
        
        try:
            if provider == "ollama":
                return await self._call_ollama(prompt, model_name)
            elif provider == "openrouter":
                if not api_key:
                    raise ValueError("OpenRouter API key is required when provider is openrouter.")
                return await self._call_openrouter(prompt, model_name, api_key)
            else:
                return self._mock_suggestions()
        except Exception as e:
            logger.error(f"LLM Advisor Error: {e}")
            return self._mock_suggestions()

    async def _call_ollama(self, prompt, model_name):
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
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

    async def _call_openrouter(self, prompt, model_name, api_key):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_instruction = (
            "You are an AI optimization assistant for TinyML models. "
            "Return ONLY a valid JSON array of objects. "
            "Each object MUST strictly have these keys: "
            "'suggestion' (string), 'reasoning' (string), 'parameters_to_adjust' (object), 'estimated_improvement' (string)."
        )
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "response_format": "json"
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload, timeout=45) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        
                        if content.startswith("```json"):
                            content = content.replace("```json", "").replace("```", "").strip()
                        elif content.startswith("```"):
                            content = content.replace("```", "").strip()
                            
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "suggestions" in parsed:
                            return parsed["suggestions"]
                        return parsed
            except Exception as e:
                logger.error(f"OpenRouter connection error: {e}")
        return self._mock_suggestions()
        
    def _build_prompt(self, context):
        return (
            f"Analyze this training session metrics and configuration and suggest improvements for TinyML deployment:\n"
            f"{context}\n\n"
            "Format your output strictly as a JSON array of objects with the following schema:\n"
            "[\n"
            "  {\n"
            "    \"suggestion\": \"Short title of the suggestion\",\n"
            "    \"reasoning\": \"Why you are suggesting this based on the metrics\",\n"
            "    \"parameters_to_adjust\": {\"param_name\": \"new_value\"},\n"
            "    \"estimated_improvement\": \"What metrics will improve\"\n"
            "  }\n"
            "]"
        )
        
    def _mock_suggestions(self):
        return [
            {
                "suggestion": "Reduce Base Model Complexity",
                "reasoning": "Model might be overfitting or too large for the target hardware constraints.",
                "parameters_to_adjust": {"base_model": "MobileNetV3Small"},
                "estimated_improvement": "Reduced memory footprint by ~40%"
            },
            {
                "suggestion": "Increase Validation Split",
                "reasoning": "Validation metrics are noisy. Increasing the validation set will give a more stable evaluation.",
                "parameters_to_adjust": {"validation_split": 0.25},
                "estimated_improvement": "More reliable generalization estimation"
            }
        ]