"""
Single-worker job queue for heavy ML work (training and optimization).

TensorFlow jobs compete for the same GPU/CPU and mutate process-global state
(the mixed-precision policy), so running them concurrently was both slow and
racy. Jobs now run one at a time, in submission order; callers mark their
session "queued" before submitting and the job flips it to "running".
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-job")
        self._lock = threading.Lock()
        self._pending: List[str] = []

    def submit(self, job_id: str, fn: Callable, *args) -> None:
        with self._lock:
            self._pending.append(job_id)

        def _run():
            with self._lock:
                if job_id in self._pending:
                    self._pending.remove(job_id)
            try:
                fn(*args)
            except Exception:  # jobs record their own failures; this is a last resort
                logger.exception("Unhandled error in job %s", job_id)

        self._executor.submit(_run)

    def position(self, job_id: str) -> int:
        """0-based position among jobs waiting to start, or -1 if not waiting."""
        with self._lock:
            return self._pending.index(job_id) if job_id in self._pending else -1

    def pending(self) -> List[str]:
        with self._lock:
            return list(self._pending)


job_queue = JobQueue()
