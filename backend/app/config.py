"""
Central runtime configuration, read once from the environment (.env is
loaded by app.main before this module is imported).

Every path the backend writes to lives under STORAGE_DIR so a single volume
mount (or backup) captures all user data: datasets, trained models,
optimized variants, and in-flight uploads.
"""

import os
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


STORAGE_DIR = Path(os.environ.get("EDGECRAFT_STORAGE_DIR", "data_storage")).resolve()
OPTIMIZATION_DIR = STORAGE_DIR / "optimizations"
UPLOAD_DIR = STORAGE_DIR / "uploads"
REMOTE_DOWNLOAD_DIR = STORAGE_DIR / "remote_downloads"

for _d in (STORAGE_DIR, OPTIMIZATION_DIR, UPLOAD_DIR, REMOTE_DOWNLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://localhost,http://127.0.0.1",
    ).split(",")
    if o.strip()
]

# Optional shared secret. When set, every /api and /ws request must carry it
# (header `X-API-Key`, or `?api_key=` for WebSockets / plain download links).
# Leave empty for a purely local, single-user install.
API_TOKEN = os.environ.get("EDGECRAFT_API_TOKEN", "").strip()

MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 4096)          # whole ZIP / remote download
MAX_CHUNK_MB = _int("MAX_CHUNK_MB", 64)               # single chunk of a chunked upload
MAX_SAMPLE_MB = _int("MAX_SAMPLE_MB", 50)             # single sample / inference input

DEFAULT_OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")

DEBUG = _bool("DEBUG", False)
