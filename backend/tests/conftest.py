"""Test setup: every test session gets its own throwaway storage directory.

app.config reads EDGECRAFT_STORAGE_DIR at import time, so it must be set
before anything from `app` is imported.
"""

import io
import os
import tempfile

os.environ.setdefault("EDGECRAFT_STORAGE_DIR", tempfile.mkdtemp(prefix="edgecraft-tests-"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MIXED_PRECISION", "false")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


def make_png(color=(200, 30, 30), size=32, box=None) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = (rng.random((size, size, 3)) * 40).astype(np.uint8)
    arr[...] += np.array(color, dtype=np.uint8) // 2
    if box:
        x, y = box
        arr[max(0, y - 4):y + 4, max(0, x - 4):x + 4] = [255, 255, 0]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def make_wav(freq=440.0, seconds=1.0, sr=16000) -> bytes:
    import soundfile as sf

    t = np.arange(int(sr * seconds)) / sr
    y = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
