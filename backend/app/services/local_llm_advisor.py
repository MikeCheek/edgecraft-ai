"""
local_llm_advisor.py
--------------------
Talks to a locally-running Ollama instance (http://localhost:11434).
Falls back to rule-based suggestions when Ollama is unavailable.

Recommended free models (pull one before enabling):
  ollama pull phi3          # ~2 GB, fast, great for structured JSON
  ollama pull mistral       # ~4 GB, strong reasoning
  ollama pull gemma:2b      # ~1.5 GB, very lightweight
  ollama pull llama3:8b     # ~4.7 GB, best quality
"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional, Dict, List

# BUGFIX: these used to be hardcoded, so a backend .env with
# OLLAMA_HOST=http://some-other-host:11434 or OLLAMA_MODEL=mistral was
# silently ignored by this module (llm_advisor.py's own _call_ollama had
# the same bug, fixed separately there). Now both fall back to the same
# localhost:11434 / "phi3" defaults when unset, matching llm_advisor.py.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "phi3")

class LocalLLMAdvisor:
    """Interface with a local Ollama LLM for intelligent ML suggestions."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.available = self._check_ollama_available()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _check_ollama_available(self) -> bool:
        """Check if the Ollama HTTP server is reachable. Real network check -
        this used to be a no-op that always reported 'connected'."""
        try:
            req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_status(self) -> Dict:
        """Return availability status and active model name."""
        return {
            "available": self.available,
            "status": "connected" if self.available else "disconnected",
            "model": self.model_name if self.available else None,
            "ollama_url": OLLAMA_BASE_URL,
        }

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate(self, prompt: str, system: str = "") -> Optional[str]:
        """Send a prompt to Ollama and return the response text."""
        if not self.available:
            return None
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            body = json.dumps({
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 600,
                },
            }).encode()

            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                return data.get("message", {}).get("content", "").strip()
        except Exception:
            return None

    def _parse_json_response(self, text: Optional[str]):
        """Try to extract JSON from a model response (strips markdown fences)."""
        if not text:
            return None
        clean = text.strip()
        for fence in ("```json", "```"):
            if clean.startswith(fence):
                clean = clean[len(fence):]
        if clean.endswith("```"):
            clean = clean[:-3]
        try:
            return json.loads(clean.strip())
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Training suggestions (post-training)
    # ------------------------------------------------------------------

    def get_training_suggestions(self, metrics: Dict) -> List[Dict]:
        """Get improvement suggestions for a completed training run.
        Uses the local LLM when available, rule-based fallback otherwise."""
        if not self.available:
            return self._rule_based_suggestions(metrics)

        system = (
            "You are an expert in TinyML and edge AI. "
            "Respond ONLY with a valid JSON array. No prose, no markdown fences."
        )
        prompt = f"""
A TinyML model finished training with these metrics:
- Training accuracy: {metrics.get('accuracy', 0):.4f}
- Validation accuracy: {metrics.get('val_accuracy', 0):.4f}
- Training loss: {metrics.get('loss', 0):.4f}
- Validation loss: {metrics.get('val_loss', 0):.4f}
- Epochs completed: {metrics.get('epoch', '?')}

Give 2-3 specific, actionable suggestions to improve this model for TinyML deployment.
Return a JSON array where each item has these exact keys:
  "suggestion"            (string: short title)
  "reasoning"             (string: why this suggestion applies to these metrics)
  "parameters_to_adjust"  (object: key-value pairs of hyperparameters to change)
  "estimated_improvement" (string: expected benefit)
"""
        response = self._generate(prompt, system)
        parsed = self._parse_json_response(response)
        if isinstance(parsed, list) and parsed:
            return [
                {
                    "suggestion": s.get("suggestion", ""),
                    "reasoning": s.get("reasoning", ""),
                    "parameters_to_adjust": s.get("parameters_to_adjust") or s.get("parameters", {}),
                    "estimated_improvement": (
                        s.get("estimated_improvement") or s.get("expected_improvement", "")
                    ),
                }
                for s in parsed
            ]
        return self._rule_based_suggestions(metrics)

    # ------------------------------------------------------------------
    # Optimization / deployment advice (called from llm_advisor.get_optimization_advice)
    # ------------------------------------------------------------------

    def get_optimization_advice(self, board: str, model_size_kb: float) -> Dict:
        """Get board-specific deployment advice.
        Uses the local LLM when available, rule-based fallback otherwise."""
        if not self.available:
            return self._rule_based_optimization_advice(board, model_size_kb)

        system = (
            "You are an expert in embedded ML and MCU deployment. "
            "Respond ONLY with a valid JSON object. No prose, no markdown fences."
        )
        prompt = f"""
I need to deploy a {model_size_kb} KB TinyML model to a {board}.

Return a JSON object with exactly these keys:
  "strategies"  (array of 3 strings: specific optimisation steps for this board)
  "challenges"  (array of strings: potential issues to watch out for)
  "testing"     (array of strings: how to verify the deployment works)

Be specific to the constraints of {board}.
"""
        response = self._generate(prompt, system)
        parsed = self._parse_json_response(response)
        if isinstance(parsed, dict):
            return parsed
        return self._rule_based_optimization_advice(board, model_size_kb)

    # ------------------------------------------------------------------
    # Rule-based fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based_suggestions(metrics: Dict) -> List[Dict]:
        suggestions = []
        accuracy = metrics.get("accuracy", 0)
        val_accuracy = metrics.get("val_accuracy", 0)
        loss = metrics.get("loss", 1.0)

        if accuracy < 0.7:
            suggestions.append({
                "suggestion": "Model accuracy is low - possible underfitting",
                "reasoning": f"Training accuracy of {accuracy:.2%} is below acceptable threshold.",
                "parameters_to_adjust": {"epochs": "+30", "base_model": "try larger architecture"},
                "estimated_improvement": "5-15% accuracy gain",
            })

        if (val_accuracy - accuracy) < -0.05:
            suggestions.append({
                "suggestion": "Overfitting detected - add regularisation",
                "reasoning": f"Validation gap of {accuracy - val_accuracy:.2%} indicates overfitting.",
                "parameters_to_adjust": {"dropout": 0.4, "l2_regularization": 0.0001},
                "estimated_improvement": "3-8% validation improvement",
            })

        if loss > 0.5:
            suggestions.append({
                "suggestion": "Loss is high - model has not converged well",
                "reasoning": "Loss should ideally be below 0.3 after training.",
                "parameters_to_adjust": {"learning_rate": 0.0001, "epochs": "+50"},
                "estimated_improvement": "Loss reduction to 0.2-0.3",
            })

        return suggestions or [{
            "suggestion": "Excellent performance - ready for deployment!",
            "reasoning": "All metrics are within acceptable TinyML ranges.",
            "parameters_to_adjust": {},
            "estimated_improvement": "Proceed to quantisation and board optimisation.",
        }]

    @staticmethod
    def _rule_based_optimization_advice(board: str, model_size_kb: float) -> Dict:
        board_specs = {
            "ESP32_S3_N16R8": {
                "strategies": [
                    "Apply INT8 quantisation for ~4x size reduction",
                    "Store model in 8 MB PSRAM to free on-chip SRAM",
                    "Enable esp-nn acceleration library for optimised 8-bit inference",
                ],
                "challenges": [
                    "Thermal management at sustained high inference rates",
                    "Power budget with Wi-Fi + inference running simultaneously",
                ],
                "testing": [
                    "Measure on-device latency with a stopwatch sketch",
                    "Monitor heap usage with ESP.getFreeHeap() during inference",
                ],
            },
            "ESP32_CAM": {
                "strategies": [
                    "Capture frames already close to the model's input resolution to skip resizing",
                    "Use grayscale capture if the model expects 1 channel, halving frame buffer size",
                    "Apply INT8 quantisation - camera + PSRAM already consume a large RAM share",
                ],
                "challenges": [
                    "No native USB - requires an FTDI adapter to flash and monitor",
                    "Camera frame buffer and tensor arena must coexist within limited PSRAM",
                ],
                "testing": [
                    "Confirm esp_camera_init() succeeds before testing inference",
                    "Log capture time separately from inference time to isolate bottlenecks",
                ],
            },
            "RASPBERRY_PI_PICO_2_W": {
                "strategies": [
                    "INT8 quantisation is mandatory (520 KB RAM limit)",
                    "Apply weight pruning to >=70% sparsity",
                    "Use ARM Cortex-M33 DSP unit for CMSIS-NN acceleration",
                ],
                "challenges": [
                    "520 KB RAM is strict - streaming inference may be required",
                    "Slow clock means careful latency budgeting is essential",
                ],
                "testing": [
                    "Verify RAM allocation at runtime",
                    "Benchmark inference latency with the SDK's timer functions",
                ],
            },
            "ARDUINO_NANO_33_BLE": {
                "strategies": [
                    "Maximum INT8 quantisation + aggressive pruning required",
                    "Use ultra-lightweight architectures (e.g. Custom3LayerCNN)",
                    "Implement block-wise inference to fit 256 KB RAM",
                ],
                "challenges": [
                    "256 KB RAM is extremely limited - model splitting may be needed",
                    "Very slow inference; consider duty-cycling the sensor",
                ],
                "testing": [
                    "Confirm model binary fits in flash with arm-none-eabi-size",
                    "Measure battery drain over 1 hour of continuous inference",
                ],
            },
        }
        return board_specs.get(board, board_specs["ESP32_S3_N16R8"])