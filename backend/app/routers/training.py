from fastapi import APIRouter, BackgroundTasks, WebSocket
from pydantic import BaseModel
from typing import Optional
import json
from app.services.trainer import Trainer

router = APIRouter()
trainer = Trainer()

class TrainingRequest(BaseModel):
    task: str
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    base_model: str = "MobileNetV2"
    validation_split: float = 0.2

@router.post("/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a new training session"""
    try:
        training_id = trainer.create_training_session(
            task=request.task,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            base_model=request.base_model,
            validation_split=request.validation_split
        )
        
        # Run training in background
        background_tasks.add_task(trainer.train, training_id)
        
        return {
            "status": "success",
            "training_id": training_id,
            "message": f"Training started with {request.epochs} epochs"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/status/{training_id}")
async def get_training_status(training_id: str):
    """Get the status of a training session"""
    try:
        status = trainer.get_training_status(training_id)
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/metrics/{training_id}")
async def get_training_metrics(training_id: str):
    """Get all metrics from a completed training"""
    try:
        metrics = trainer.get_training_metrics(training_id)
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/cancel/{training_id}")
async def cancel_training(training_id: str):
    """Cancel an ongoing training session"""
    try:
        trainer.cancel_training(training_id)
        return {
            "status": "success",
            "message": f"Training {training_id} cancelled"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/models")
async def list_trained_models():
    """List all trained models"""
    try:
        models = trainer.get_trained_models()
        return {
            "status": "success",
            "count": len(models),
            "models": models
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/export/{training_id}")
async def export_model(training_id: str, format: str = "tflite"):
    """Export a trained model in the specified format"""
    try:
        result = trainer.export_model(training_id, format)
        return {
            "status": "success",
            "message": f"Model exported as {format}",
            "path": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
