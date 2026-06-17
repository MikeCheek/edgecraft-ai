from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file at startup

from app.routers import datasets, training, optimization, remote_datasets

app = FastAPI(
    title="EdgeCraft AI Backend",
    description="Local TinyML Studio API",
    version="0.2.0"
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
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(remote_datasets.router, prefix="/api/remote_datasets", tags=["Remote Datasets"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["Optimization"])
app.include_router(remote_datasets.router, prefix="/api/remote_datasets", tags=["Remote Datasets"])

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "EdgeCraft AI Backend is running"
    }

@app.get("/api/info")
async def get_info():
    return {
        "name": "EdgeCraft AI Backend",
        "version": "0.2.0",
        "tasks": [
            "IMAGE_CLASSIFICATION",
            "OBJECT_DETECTION",
            "VISUAL_WAKE_WORDS",
            "KEYWORD_SPOTTING",
            "AUDIO_CLASSIFICATION"
        ],
        "boards": [
            "ESP32_S3_N16R8",
            "RASPBERRY_PI_PICO_2_W",
            "ARDUINO_NANO_33_BLE"
        ],
        "models": {
            "image": ["MobileNetV2", "EfficientNet", "ResNet50V2", "MobileNetV3Small", "Custom3LayerCNN"],
            "audio": ["MFCC_CNN", "WaveNet", "AudioLSTM", "AudioGRU"],
            "text": ["TinyBERT"]
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": "error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)