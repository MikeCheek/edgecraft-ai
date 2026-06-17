from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from app.services.shared_state import trainer
from typing import List, Optional

router = APIRouter()


class TrainingRequest(BaseModel):
    task: str
    dataset_id: str
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    base_model: str = "MobileNetV2"
    validation_split: float = 0.2
    input_shape: List[int] = [224, 224, 3]
    # Regularisation
    dropout_rate: float = 0.5
    l2_reg: float = 0.0
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_monitor: str = "val_loss"


@router.post("/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    try:
        training_id = trainer.create_training_session(
            task=request.task,
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            base_model=request.base_model,
            validation_split=request.validation_split,
            input_shape=request.input_shape,
            early_stopping=request.early_stopping,
            early_stopping_patience=request.early_stopping_patience,
            early_stopping_monitor=request.early_stopping_monitor,
            dropout_rate=request.dropout_rate,
            l2_reg=request.l2_reg,
        )
        background_tasks.add_task(trainer.train, training_id)
        return {"status": "success", "training_id": training_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/status/{training_id}")
async def get_training_status(training_id: str):
    try:
        status = trainer.get_training_status(training_id)
        return {"status": "success", "data": status}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/metrics/{training_id}")
async def get_training_metrics(training_id: str):
    try:
        return {"status": "success", "metrics": trainer.get_training_metrics(training_id)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/cancel/{training_id}")
async def cancel_training(training_id: str):
    try:
        trainer.cancel_training(training_id)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/models")
async def list_trained_models():
    try:
        return {"status": "success", "models": trainer.get_trained_models()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/sessions")
async def list_all_sessions():
    """Return all past training sessions sorted newest-first."""
    try:
        return {"status": "success", "sessions": trainer.get_all_sessions()}
    except Exception as e:
        return {"status": "error", "message": str(e)}