"""
job_logs.py
-----------
A small in-memory log broker so the frontend can show a live, collapsible
"terminal" of what's happening during a training or optimization job -
without needing a full logging/metrics pipeline.

Design:
- Each job (keyed by training_id or optimization_id) gets a bounded ring
  buffer of log lines. Any WebSocket client can connect to
  `/ws/logs/{job_id}`, receives the buffered history immediately, then
  gets new lines pushed live as they're logged.
- `log()` is safe to call from a background thread (Trainer.train() and
  optimizer.optimize() both run via FastAPI's BackgroundTasks, which
  executes sync callables in a worker thread, not the asyncio event loop).
  It uses `asyncio.run_coroutine_threadsafe` to hop back onto the main
  loop for the actual WebSocket send.
- This is intentionally NOT a general-purpose logging handler capturing
  every Python `logging` call (which would be extremely noisy with
  TensorFlow/absl/urllib3 internals) - callers explicitly log the lines
  they want the user to see.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque, Dict, List, Optional

MAX_LINES_PER_JOB = 2000


class JobLogBroker:
    def __init__(self) -> None:
        self._buffers: Dict[str, Deque[dict]] = {}
        self._subscribers: Dict[str, List] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once at app startup so log() can schedule websocket sends
        from worker threads that have no event loop of their own."""
        self._loop = loop

    def log(self, job_id: str, message: str, level: str = "info") -> None:
        """Append a line to job_id's buffer and push it to any connected
        WebSocket clients. Safe to call from any thread."""
        buf = self._buffers.setdefault(job_id, deque(maxlen=MAX_LINES_PER_JOB))
        entry = {"ts": time.time(), "level": level, "message": message}
        buf.append(entry)

        subs = self._subscribers.get(job_id)
        if not subs or self._loop is None:
            return
        for ws in list(subs):
            try:
                asyncio.run_coroutine_threadsafe(self._safe_send(ws, entry), self._loop)
            except RuntimeError:
                pass  # event loop closed (e.g. during shutdown) - drop silently

    async def _safe_send(self, ws, entry: dict) -> None:
        try:
            await ws.send_json(entry)
        except Exception:
            pass  # client disconnected; unsubscribe happens in the endpoint's finally block

    def get_history(self, job_id: str) -> List[dict]:
        return list(self._buffers.get(job_id, []))

    def subscribe(self, job_id: str, ws) -> None:
        self._subscribers.setdefault(job_id, []).append(ws)

    def unsubscribe(self, job_id: str, ws) -> None:
        subs = self._subscribers.get(job_id)
        if subs and ws in subs:
            subs.remove(ws)

    def clear(self, job_id: str) -> None:
        self._buffers.pop(job_id, None)
        self._subscribers.pop(job_id, None)


# Module-level singleton, imported by trainer.py, optimizer.py, and the
# websocket router.
job_log_broker = JobLogBroker()
