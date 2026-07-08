from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file at startup

from app.routers import datasets, training, optimization, remote_datasets
from app.routers import inference  # NEW: real inference router
from app.routers import job_logs_ws  # NEW: live job console log streaming

app = FastAPI(
    title="EdgeCraft AI Backend",
    description="Local TinyML Studio API",
    version="0.3.0"
)

# CORS Configuration
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router,         prefix="/api/datasets",         tags=["Datasets"])
app.include_router(remote_datasets.router,  prefix="/api/remote_datasets",  tags=["Remote Datasets"])
app.include_router(training.router,         prefix="/api/training",          tags=["Training"])
app.include_router(optimization.router,     prefix="/api/optimization",      tags=["Optimization"])
app.include_router(inference.router,        prefix="/api/inference",         tags=["Inference"])  # NEW
app.include_router(job_logs_ws.router,      tags=["Live Logs"])  # NEW: /ws/logs/{job_id}

@app.on_event("startup")
async def _bind_job_log_broker_loop():
    """job_log_broker.log() can be called from a worker thread (training and
    optimization jobs run via BackgroundTasks, not on the event loop), so it
    needs a reference to the actual running loop to schedule websocket sends
    via asyncio.run_coroutine_threadsafe."""
    import asyncio
    from app.services.job_logs import job_log_broker
    job_log_broker.bind_loop(asyncio.get_running_loop())

@app.get("/api/health")
async def health_check():
    """
    Keep this async! It runs instantly on the main thread loop 
    and won't be blocked if heavy tasks are offloaded.
    """
    return {
        "status": "healthy",
        "message": "EdgeCraft AI Backend is running"
    }

@app.get("/api/info")
async def get_info():
    """
    Returns core static configuration metadata instantly.
    """
    return {
        "name": "EdgeCraft AI Backend",
        "version": "0.3.0",
        "tasks": ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "VISUAL_WAKE_WORDS", "KEYWORD_SPOTTING", "AUDIO_CLASSIFICATION"],
        "boards": ["ESP32_S3_N16R8", "ESP32_CAM", "RASPBERRY_PI_PICO_2_W", "ARDUINO_NANO_33_BLE"]
    }

@app.get("/api/storage/overview")
def get_storage_overview():
    """
    REMOVED 'async' keyword. 
    By defining this as a standard synchronous function 'def', FastAPI 
    automatically delegates the heavy disk crawl to an internal external threadpool. 
    This allows /api/health to keep running smoothly on the event loop!
    """
    try:
        from app.services.storage_overview import get_storage_overview as _overview
        return {"status": "success", "overview": _overview()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/models/tree")
def get_models_tree(include_archived: bool = False):
    """
    REMOVED 'async' keyword. Offloads complex database/file mapping scans
    to background worker threads.
    """
    try:
        from app.services.model_tree import get_model_tree
        return {"status": "success", "tree": get_model_tree(include_archived=include_archived)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": "error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
