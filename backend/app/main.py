from contextlib import asynccontextmanager
import asyncio
import hmac
import logging
import os
import shutil
import time

from dotenv import load_dotenv

load_dotenv()  # before anything reads os.environ (app.config, LLM keys, ...)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from app import config  # noqa: E402
from app.routers import datasets, training, optimization, remote_datasets  # noqa: E402
from app.routers import inference, job_logs_ws  # noqa: E402

logger = logging.getLogger(__name__)
VERSION = "0.4.0"
STALE_UPLOAD_SECONDS = 24 * 3600


def _purge_stale(directory, max_age: float) -> None:
    """Remove abandoned chunked uploads / remote downloads older than max_age."""
    now = time.time()
    for entry in directory.iterdir():
        try:
            if now - entry.stat().st_mtime > max_age:
                shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink()
        except OSError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.job_logs import job_log_broker

    job_log_broker.bind_loop(asyncio.get_running_loop())
    for d in (config.UPLOAD_DIR, config.REMOTE_DOWNLOAD_DIR):
        _purge_stale(d, STALE_UPLOAD_SECONDS)
    yield


app = FastAPI(
    title="EdgeCraft AI Backend",
    description="Local TinyML Studio API",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths reachable without a token: liveness probe + API docs.
_PUBLIC_PATHS = {"/api/health", "/docs", "/openapi.json", "/redoc"}


def _token_ok(provided: str) -> bool:
    return bool(provided) and hmac.compare_digest(provided, config.API_TOKEN)


@app.middleware("http")
async def require_api_token(request: Request, call_next):
    """Optional shared-secret auth (set EDGECRAFT_API_TOKEN). Header
    `X-API-Key`, or `?api_key=` for links the browser opens directly
    (downloads, images)."""
    if config.API_TOKEN and request.method != "OPTIONS" and request.url.path not in _PUBLIC_PATHS:
        provided = request.headers.get("x-api-key") or request.query_params.get("api_key", "")
        if not _token_ok(provided):
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key", "type": "auth"})
    return await call_next(request)


app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(remote_datasets.router, prefix="/api/remote_datasets", tags=["Remote Datasets"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["Optimization"])
app.include_router(inference.router, prefix="/api/inference", tags=["Inference"])
app.include_router(job_logs_ws.router, tags=["Live Logs"])


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "EdgeCraft AI Backend is running",
        "version": VERSION,
        "auth_required": bool(config.API_TOKEN),
    }


@app.get("/api/info")
async def get_info():
    from app.services import preprocessing
    from app.services.model_factory import AUDIO_MODELS, IMAGE_BACKBONES, OD_MODELS, ModelFactory

    tasks = ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "VISUAL_WAKE_WORDS", "KEYWORD_SPOTTING", "AUDIO_CLASSIFICATION"]
    try:
        import tflite_micro  # noqa: F401

        tflm_available = True
    except Exception:
        tflm_available = False
    return {
        "name": "EdgeCraft AI Backend",
        "version": VERSION,
        "tasks": tasks,
        "boards": ["ESP32_S3_N16R8", "ESP32_CAM", "RASPBERRY_PI_PICO_2_W", "ARDUINO_NANO_33_BLE"],
        "models": {
            "IMAGE_CLASSIFICATION": IMAGE_BACKBONES,
            "VISUAL_WAKE_WORDS": IMAGE_BACKBONES,
            "OBJECT_DETECTION": OD_MODELS,
            "KEYWORD_SPOTTING": AUDIO_MODELS,
            "AUDIO_CLASSIFICATION": AUDIO_MODELS,
        },
        "model_info": ModelFactory.EDGE_SUITABILITY,
        "default_input_shapes": {t: list(preprocessing.default_input_shape(t)) for t in tasks},
        "audio_frontend": {t: {k: v for k, v in preprocessing.audio_params(t).items()}
                           for t in preprocessing.AUDIO_TASKS},
        "tflite_micro_available": tflm_available,
        "limits": {"max_upload_mb": config.MAX_UPLOAD_MB, "max_chunk_mb": config.MAX_CHUNK_MB,
                   "max_sample_mb": config.MAX_SAMPLE_MB},
    }


@app.get("/api/storage/overview")
async def get_storage_overview():
    try:
        from app.services.storage_overview import get_storage_overview as _overview

        return {"status": "success", "overview": await run_in_threadpool(_overview)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/models/tree")
async def get_models_tree(include_archived: bool = False):
    try:
        from app.services.model_tree import get_model_tree

        return {"status": "success", "tree": await run_in_threadpool(get_model_tree, include_archived)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "type": "error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("BACKEND_HOST", "127.0.0.1"), port=int(os.environ.get("BACKEND_PORT", 8000)))
