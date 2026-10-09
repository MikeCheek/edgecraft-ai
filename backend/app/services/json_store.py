"""Small helpers for the JSON files the backend persists its state in."""

import json
import os
import tempfile
from typing import Any


def atomic_write_json(path: str, data: Any) -> None:
    """Write JSON to a temp file in the same directory, then rename it over
    `path`, so a crash mid-write can never leave a truncated file behind."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def read_json(path: str, default: Any) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default
