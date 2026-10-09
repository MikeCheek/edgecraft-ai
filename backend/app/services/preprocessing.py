"""
Single source of truth for turning a raw stored sample (image / audio bytes)
into the exact tensor a model sees.

Training, INT8 calibration, test-set evaluation and live inference all call
into this module, so they can never drift apart again (they used to have five
slightly different copies, with different MFCC parameters and resize filters).

Conventions
-----------
* Images: decoded with PIL, converted to L or RGB to match the model's channel
  count, resized with BILINEAR to (H, W), scaled to float32 in [0, 1].
  Any backbone-specific normalisation (e.g. [-1, 1] for MobileNet) lives
  *inside* the model as a Rescaling layer, so the on-device sketch only ever
  has to produce [0, 1] pixels.
* Audio: decoded to mono float32, resampled to 16 kHz, padded / trimmed to
  the task's clip duration, then turned into MFCCs by `compute_mfcc`, a small
  pure-numpy implementation that the exported Arduino sketch reproduces
  exactly (same window, FFT size, hop, mel weights, DCT and normalisation).
"""

from __future__ import annotations

import io
import os
import tempfile
from functools import lru_cache
from typing import Dict, Sequence, Tuple

import numpy as np

IMAGE_TASKS = {"IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "VISUAL_WAKE_WORDS"}
AUDIO_TASKS = {"KEYWORD_SPOTTING", "AUDIO_CLASSIFICATION"}

# Defaults the UI starts from (H, W, C). Users may override per training run.
DEFAULT_IMAGE_SHAPES: Dict[str, Tuple[int, int, int]] = {
    "IMAGE_CLASSIFICATION": (96, 96, 3),
    "OBJECT_DETECTION": (96, 96, 3),
    "VISUAL_WAKE_WORDS": (96, 96, 1),
}

# Audio front-end. center=False framing keeps the on-device implementation
# trivial (no reflection padding); everything else is a standard 32 ms window
# with a 20 ms hop at 16 kHz.
AUDIO_FRONTEND = {
    "sample_rate": 16000,
    "n_fft": 512,
    "hop_length": 320,
    "fmin": 20.0,
    "fmax": 8000.0,
    "top_db": 80.0,
}

AUDIO_TASK_PARAMS: Dict[str, Dict[str, float]] = {
    "KEYWORD_SPOTTING": {"duration": 1.0, "n_mfcc": 40},
    "AUDIO_CLASSIFICATION": {"duration": 2.0, "n_mfcc": 40},
}


def is_audio_task(task: str) -> bool:
    return task in AUDIO_TASKS


def audio_params(task: str, n_mfcc: int = None) -> Dict[str, float]:
    """Front-end parameters for `task`. `n_mfcc` (normally the model's
    input_shape[0]) overrides the task default so models trained with a
    different coefficient count keep working."""
    p = dict(AUDIO_FRONTEND)
    p.update(AUDIO_TASK_PARAMS.get(task, AUDIO_TASK_PARAMS["KEYWORD_SPOTTING"]))
    if n_mfcc:
        p["n_mfcc"] = int(n_mfcc)
    p["n_mels"] = max(40, int(p["n_mfcc"]))
    n_samples = int(p["sample_rate"] * p["duration"])
    p["n_samples"] = n_samples
    p["n_frames"] = 1 + (n_samples - p["n_fft"]) // p["hop_length"]
    return p


def default_input_shape(task: str) -> Tuple[int, ...]:
    if is_audio_task(task):
        p = audio_params(task)
        return (int(p["n_mfcc"]), int(p["n_frames"]), 1)
    return DEFAULT_IMAGE_SHAPES.get(task, (96, 96, 3))


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def load_image(raw: bytes, input_shape: Sequence[int]) -> np.ndarray:
    from PIL import Image

    h, w = int(input_shape[0]), int(input_shape[1])
    channels = int(input_shape[2]) if len(input_shape) >= 3 else 3
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("L" if channels == 1 else "RGB")
        img = img.resize((w, h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    if channels == 1:
        arr = arr[:, :, np.newaxis]
    return arr


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def decode_audio(raw: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode any audio container to mono float32 at `target_sr`.

    WAV/FLAC/OGG go through soundfile in memory. Anything soundfile can't
    read (e.g. the WebM/Opus a browser MediaRecorder produces, or MP3 on
    older libsndfile builds) falls back to librosa+audioread on a temp file,
    which uses ffmpeg when it is installed.
    """
    audio = None
    sr = None
    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception:
        import librosa

        fd, path = tempfile.mkstemp(suffix=".audio")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            audio, sr = librosa.load(path, sr=None, mono=True)
        except Exception as exc:  # pragma: no cover - depends on system codecs
            raise ValueError(
                "Could not decode audio. Use WAV, or install ffmpeg on the backend "
                f"host for compressed formats ({exc})."
            ) from exc
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio


def fit_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    if len(audio) >= n_samples:
        return audio[:n_samples]
    return np.pad(audio, (0, n_samples - len(audio)))


@lru_cache(maxsize=8)
def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    import librosa

    return librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    ).astype(np.float32)  # (n_mels, n_fft//2+1)


@lru_cache(maxsize=8)
def dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (n_mfcc, n_mels), same as scipy dct(norm='ortho')."""
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    m = np.cos(np.pi / n_mels * (n + 0.5) * k)
    m[0] *= np.sqrt(1.0 / n_mels)
    m[1:] *= np.sqrt(2.0 / n_mels)
    return m.astype(np.float32)


@lru_cache(maxsize=4)
def hann_window(n_fft: int) -> np.ndarray:
    # Periodic Hann (what librosa / scipy.signal.get_window('hann') use).
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)


def compute_mfcc(audio: np.ndarray, task: str, n_mfcc: int = None) -> np.ndarray:
    """(n_mfcc, n_frames) MFCC matrix, z-score normalised per clip."""
    p = audio_params(task, n_mfcc)
    n_fft, hop = int(p["n_fft"]), int(p["hop_length"])
    audio = fit_length(np.asarray(audio, dtype=np.float32), int(p["n_samples"]))
    n_frames = int(p["n_frames"])

    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = audio[idx] * hann_window(n_fft)[None, :]
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2  # (frames, bins)

    mel = power @ mel_filterbank(
        int(p["sample_rate"]), n_fft, int(p["n_mels"]), float(p["fmin"]), float(p["fmax"])
    ).T  # (frames, n_mels)
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - float(p["top_db"]))

    mfcc = log_mel @ dct_matrix(int(p["n_mfcc"]), int(p["n_mels"])).T  # (frames, n_mfcc)
    mfcc = mfcc.T
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    return mfcc.astype(np.float32)


def load_audio_features(raw: bytes, task: str, input_shape: Sequence[int]) -> np.ndarray:
    """Raw audio bytes -> (n_mfcc, frames, 1), padded/trimmed to input_shape."""
    n_mfcc = int(input_shape[0])
    frames = int(input_shape[1])
    p = audio_params(task, n_mfcc)
    mfcc = compute_mfcc(decode_audio(raw, int(p["sample_rate"])), task, n_mfcc)
    if mfcc.shape[1] > frames:
        mfcc = mfcc[:, :frames]
    elif mfcc.shape[1] < frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, frames - mfcc.shape[1])))
    return mfcc[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def preprocess(raw: bytes, task: str, input_shape: Sequence[int]) -> np.ndarray:
    """Raw sample bytes -> float32 model input (no batch dimension)."""
    if is_audio_task(task):
        return load_audio_features(raw, task, input_shape)
    return load_image(raw, input_shape)


def quantize_input(x: np.ndarray, scale: float, zero_point: int, dtype=np.int8) -> np.ndarray:
    """Float -> integer tensor with correct rounding and saturation."""
    info = np.iinfo(dtype)
    q = np.round(x / (scale or 1.0) + zero_point)
    return np.clip(q, info.min, info.max).astype(dtype)


def dequantize_output(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * (scale or 1.0)
