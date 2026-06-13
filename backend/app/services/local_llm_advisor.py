import json
import subprocess
from typing import Optional, Dict, List

class LocalLLMAdvisor:
    """Interface with local LLM (Ollama) for intelligent suggestions"""

    def __init__(self, model_name: str = "neural-chat"):
        self.model_name = model_name
        self.available = self._check_ollama_available()

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available locally"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama with a prompt"""
        if not self.available:
            return None

        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout if result.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def get_training_suggestions(self, metrics: Dict) -> List[Dict]:
        """Get LLM suggestions for model training improvements"""
        if not self.available:
            return self._get_rule_based_suggestions(metrics)

        prompt = f"""
Given these training metrics:
- Loss: {metrics.get('loss', 0):.4f}
- Accuracy: {metrics.get('accuracy', 0):.2%}
- Val Loss: {metrics.get('val_loss', 0):.4f}
- Val Accuracy: {metrics.get('val_accuracy', 0):.2%}

Provide 2-3 specific, actionable suggestions to improve the model.
Format as JSON list with 'suggestion', 'reasoning', 'parameters', 'expected_improvement' fields.
Keep suggestions concise for TinyML context.
"""
        response = self._call_ollama(prompt)
        if response:
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass

        return self._get_rule_based_suggestions(metrics)

    def get_optimization_advice(self, board: str, model_size: int) -> Dict:
        """Get LLM advice for deployment optimization"""
        if not self.available:
            return self._get_rule_based_optimization_advice(board, model_size)

        prompt = f"""
For deploying a {model_size}KB TinyML model to {board}:
1. Provide 3 specific optimization strategies
2. List potential deployment challenges
3. Suggest testing procedures

Format as JSON with 'strategies', 'challenges', 'testing' fields.
Be specific to the {board} board constraints.
"""
        response = self._call_ollama(prompt)
        if response:
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass

        return self._get_rule_based_optimization_advice(board, model_size)

    @staticmethod
    def _get_rule_based_suggestions(metrics: Dict) -> List[Dict]:
        """Fallback rule-based suggestions when LLM unavailable"""
        suggestions = []
        accuracy = metrics.get('accuracy', 0)
        val_accuracy = metrics.get('val_accuracy', 0)
        loss = metrics.get('loss', 1.0)

        if accuracy < 0.7:
            suggestions.append({
                "suggestion": "Model accuracy is low. Consider using a larger base model or more training epochs.",
                "reasoning": f"Training accuracy of {accuracy:.2%} suggests underfitting",
                "parameters": {"epochs": "+30", "model_size": "increase"},
                "expected_improvement": "5-15% accuracy gain"
            })

        if (val_accuracy - accuracy) < -0.05:
            suggestions.append({
                "suggestion": "Model is overfitting. Add more regularization or dropout.",
                "reasoning": f"Validation gap of {accuracy - val_accuracy:.2%} indicates overfitting",
                "parameters": {"dropout": 0.4, "l2_reg": 0.0001},
                "expected_improvement": "3-8% validation improvement"
            })

        if loss > 0.5:
            suggestions.append({
                "suggestion": "Loss is high. Try reducing learning rate and training longer.",
                "reasoning": "Model hasn't converged well",
                "parameters": {"learning_rate": 0.0001, "epochs": "+50"},
                "expected_improvement": "Loss reduction to 0.2-0.3"
            })

        return suggestions or [{
            "suggestion": "Model performance is excellent! Ready for optimization and deployment.",
            "reasoning": "Metrics are within acceptable TinyML ranges",
            "parameters": {},
            "expected_improvement": "Proceed to quantization"
        }]

    @staticmethod
    def _get_rule_based_optimization_advice(board: str, model_size: int) -> Dict:
        """Fallback rule-based optimization advice"""
        board_specs = {
            "ESP32_S3_N16R8": {
                "strategies": [
                    "Use INT8 quantization for 4x size reduction",
                    "Leverage 8MB PSRAM for model storage",
                    "Enable esp-nn acceleration library"
                ],
                "challenges": [
                    "Thermal management at high inference rates",
                    "Power consumption monitoring needed"
                ],
                "testing": ["Test on-device inference latency", "Verify memory usage in real-time"]
            },
            "RASPBERRY_PI_PICO_2_W": {
                "strategies": [
                    "Apply aggressive INT8 quantization (required)",
                    "Use weight pruning to 70% sparsity",
                    "Enable ARM DSP unit acceleration"
                ],
                "challenges": [
                    "520KB RAM limit is strict",
                    "May need model splitting or streaming",
                    "Slow inference speed"
                ],
                "testing": ["Verify RAM allocation", "Test streaming inference if needed"]
            },
            "ARDUINO_NANO_33_BLE": {
                "strategies": [
                    "Maximum INT8 quantization and pruning",
                    "Use ultra-lightweight architectures",
                    "Block-wise inference processing"
                ],
                "challenges": [
                    "256KB RAM is extremely limited",
                    "Must use smallest possible models",
                    "Very slow inference"
                ],
                "testing": ["Confirm model fits in SRAM", "Test battery impact"]
            }
        }

        return board_specs.get(board, board_specs["ESP32_S3_N16R8"])
