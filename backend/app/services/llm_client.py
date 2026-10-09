"""
Robust JSON chat calls to OpenRouter or a local Ollama server.

Free / shared models time out, rate-limit, wrap JSON in prose or markdown,
emit <think> blocks, or reject `response_format`. Every LLM feature goes
through `chat_json`, which:

* retries transient failures (timeouts, connection errors, 408/429/5xx,
  empty or unparseable replies, replies the caller's validator rejects)
  with backoff, honouring Retry-After, inside one overall time budget;
* retries once without `response_format` when a model rejects JSON mode;
* extracts the first JSON object/array from whatever text comes back and
  tolerates trailing commas.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Callable, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def attempt_timeout() -> float:
    """Seconds one LLM request may take (LLM_TIMEOUT_SECONDS, default 90)."""
    return _env_float("LLM_TIMEOUT_SECONDS", 90.0)


def total_budget() -> float:
    """Seconds all attempts together may take (LLM_TOTAL_BUDGET_SECONDS, default 170).
    The frontend waits a little longer than this."""
    return _env_float("LLM_TOTAL_BUDGET_SECONDS", 170.0)


class LLMError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_json(text: str) -> Any:
    """First JSON object or array in `text` (fences, prose, <think> blocks
    and trailing commas tolerated). Raises ValueError if there is none."""
    if not text or not text.strip():
        raise ValueError("empty reply")
    text = _THINK_RE.sub("", text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        depth, in_str, esc = 0, False, False
        for end in range(start, len(text)):
            c = text[end]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
                        try:
                            return json.loads(attempt)
                        except json.JSONDecodeError:
                            continue
                    break
    raise ValueError(f"no JSON found in reply: {text[:200]!r}")


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def _retry_after(headers) -> Optional[float]:
    try:
        return float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


async def _openrouter(session: aiohttp.ClientSession, messages, model: str, json_mode: bool, timeout: float) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY is missing from the backend .env file.")
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "X-Title": "EdgeCraft AI"}
    async with session.post(OPENROUTER_URL, headers=headers, json=payload,
                            timeout=aiohttp.ClientTimeout(total=timeout, sock_connect=15)) as resp:
        body = await resp.text()
        if resp.status != 200:
            msg = f"OpenRouter HTTP {resp.status}: {body[:300]}"
            if resp.status == 400 and json_mode and ("response_format" in body or "json" in body.lower()):
                raise LLMError(msg + " [json-mode-unsupported]", retryable=True)
            raise LLMError(msg, retryable=resp.status in RETRYABLE_STATUS, retry_after=_retry_after(resp.headers))
    data = json.loads(body)
    if "error" in data:
        err = data["error"] if isinstance(data["error"], dict) else {"message": str(data["error"])}
        code = err.get("code")
        raise LLMError(f"OpenRouter error {code}: {err.get('message', err)}",
                       retryable=code in RETRYABLE_STATUS or code is None)
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if not content.strip():
        reason = choice.get("finish_reason") or "no content"
        raise LLMError(f"OpenRouter returned an empty reply ({reason}, model {data.get('model', model)})",
                       retryable=True)
    return content


async def _ollama(session: aiohttp.ClientSession, messages, model: str, json_mode: bool, timeout: float) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.2}}
    if json_mode:
        payload["format"] = "json"
    try:
        async with session.post(f"{host}/api/chat", json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout, sock_connect=10)) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise LLMError(f"Ollama HTTP {resp.status}: {body[:300]}",
                               retryable=resp.status in RETRYABLE_STATUS)
    except aiohttp.ClientConnectorError as e:
        raise LLMError(f"Could not reach Ollama at {host} ({e}). Is `ollama serve` running?") from e
    content = (json.loads(body).get("message") or {}).get("content") or ""
    if not content.strip():
        raise LLMError("Ollama returned an empty reply", retryable=True)
    return content


async def chat_json(
    messages: List[dict],
    provider: str,
    model: str,
    validate: Optional[Callable[[Any], Any]] = None,
    max_attempts: int = 3,
    budget: Optional[float] = None,
) -> Any:
    """Send `messages`, return the parsed (and validated) JSON reply.

    `validate(parsed)` may normalise the reply or raise ValueError, which
    counts as a retryable failure. Raises LLMError when every attempt fails
    or the time budget runs out; the message lists each attempt's error.
    """
    if provider not in ("openrouter", "ollama"):
        raise LLMError(f"Unknown provider '{provider}'. Expected 'openrouter' or 'ollama'.")
    call = _openrouter if provider == "openrouter" else _ollama
    deadline = time.monotonic() + (budget or total_budget())
    json_mode = True
    errors: List[str] = []
    async with aiohttp.ClientSession() as session:
        for attempt in range(1, max_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining < 5:
                errors.append("time budget exhausted")
                break
            timeout = min(attempt_timeout(), remaining)
            t0 = time.monotonic()
            delay = 0.0
            try:
                text = await call(session, messages, model, json_mode, timeout)
                try:
                    parsed = extract_json(text)
                    return validate(parsed) if validate else parsed
                except ValueError as e:
                    raise LLMError(f"unusable reply: {e}", retryable=True) from e
            except asyncio.TimeoutError:
                errors.append(f"attempt {attempt}: timed out after {time.monotonic() - t0:.0f}s")
            except aiohttp.ClientError as e:
                errors.append(f"attempt {attempt}: connection error ({type(e).__name__}: {e})")
            except LLMError as e:
                errors.append(f"attempt {attempt}: {e}")
                if "[json-mode-unsupported]" in str(e):
                    json_mode = False  # retry immediately without response_format
                    continue
                if not e.retryable:
                    break
                delay = e.retry_after or 0.0
            logger.warning("LLM %s/%s %s", provider, model, errors[-1])
            if attempt < max_attempts:
                delay = max(delay, 2.0 * attempt)
                if time.monotonic() + delay > deadline - 5:
                    break
                await asyncio.sleep(delay)
    raise LLMError(f"{provider} ({model}) failed: " + "; ".join(errors))
