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
        """Send a prompt to Ollama and return the response text using /api/chat."""
        if not self.available:
            return None
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            # The 'prompt' argument here contains the full user context (e.g., formatted metrics)
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
                f"{OLLAMA_BASE_URL}/api/chat", # Changed endpoint to /api/chat
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                # The chat endpoint returns the content in data['message']['content']
                return data.get("message", {}).get("content", "").strip()
        except Exception:
            return None

    def _parse_json_response(self, text: Optional[str]):
        """Try to extract JSON from a model response (strips markdown fences).