import requests
import json
import os

class LLMIntegrator:
    @staticmethod
    def query_ollama(model: str, prompt: str, base_url: str = "http://localhost:11434"):
        """Queries a local Ollama instance."""
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=30
            )
            return response.json().get("response", "Error: No response from Ollama")
        except Exception as e:
            return f"Ollama Error: {str(e)}"

    @staticmethod
    def query_openrouter(model: str, prompt: str, api_key: str):
        """Queries OpenRouter."""
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:5173", # Required by OR
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenRouter Error: {str(e)}"