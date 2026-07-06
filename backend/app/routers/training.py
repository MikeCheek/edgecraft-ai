from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from app.services.shared_state import trainer, data_manager
from app.services.llm_advisor import LLMAdvisor
from typing import List, Optional

router = APIRouter()
llm_advisor = LLMAdvisor()

class TrainingRecommendRequest(BaseModel):
    task: str
    dataset_id: str
    target_board: str = "ESP32_S3_N16R8"
    provider: str = "openrouter"
    model_name: str = "google/gemini-2.0-flash-lite-preview-02-05:free"

class TrainingRequest(BaseModel):
    task: str
    dataset_id: str
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    base_model: str = "MobileNetV2"
    input_shape: List[int] = [224, 224, 3]
    # Compute target: "auto" (let TF prefer GPU if present), "cpu", or "gpu".
    # Forcing CPU is useful for small models where GPU offers no real
    # speedup and just adds host<->device transfer overhead.
    device: str = "auto"
    # Regularisation
    dropout_rate: float = 0.5
    l2_reg: float = 0.0
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_monitor: str = "val_loss"

    dropout_rate: float = 0.0
    l2_reg: float = 0.0
    trainable_layers: int = 0  # 0 = unfreeze all, >0 = unfreeze last N layers
    freeze_encoder_epochs: int = 0
    augmentation: dict = {}

@router.get("/devices")
async def get_available_devices():
    """List which compute devices (CPU/GPU) are actually available, so the
    frontend can disable the GPU option when none is visible to TensorFlow."""
    try:
        return {"status": "success", "devices": trainer.get_available_devices()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    try:
        if not data_manager.is_split_ready(request.dataset_id):
            return {
                "status": "error",
                "message": (
                    "This dataset doesn't have a complete train/val split yet. "
                    "Split it into train/val/test (e.g. via Auto Split) before starting training."
                ),
            }
        training_id = trainer.create_training_session(
            task=request.task,
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            base_model=request.base_model,
            input_shape=request.input_shape,
            early_stopping=request.early_stopping,
            early_stopping_patience=request.early_stopping_patience,
            early_stopping_monitor=request.early_stopping_monitor,
            dropout_rate=request.dropout_rate,
            l2_reg=request.l2_reg,
            device=request.device,
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
        ok = trainer.cancel_training(training_id)
        if not ok:
            return {"status": "error", "message": "Session not found or not currently running."}
        return {
            "status": "success",
            "message": "Cancellation requested - training will stop at the end of the current epoch.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/archive/{training_id}")
async def archive_training(training_id: str):
    """Hide a session from the default history list without deleting it."""
    try:
        ok = trainer.archive_training(training_id)
        if not ok:
            return {"status": "error", "message": "Session not found."}
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/unarchive/{training_id}")
async def unarchive_training(training_id: str):
    try:
        ok = trainer.unarchive_training(training_id)
        if not ok:
            return {"status": "error", "message": "Session not found."}
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/session/{training_id}")
async def delete_training_session(training_id: str):
    """Permanently delete a training session and its saved model. Refuses
    to delete a session that's still actively running - cancel it first."""
    try:
        ok = trainer.delete_training_session(training_id)
        if not ok:
            return {"status": "error", "message": "Session not found."}
        return {"status": "success"}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/models")
async def list_trained_models():
    try:
        return {"status": "success", "models": trainer.get_trained_models()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/sessions")
async def list_all_sessions(include_archived: bool = False):
    """Return all past training sessions sorted newest-first. Archived
    sessions are excluded by default; pass ?include_archived=true to see them."""
    try:
        return {"status": "success", "sessions": trainer.get_all_sessions(include_archived=include_archived)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/recommend")
async def recommend_training_config(request: TrainingRecommendRequest):
    """
    LLM-assisted (with a rule-based fallback) suggestion for base_model +
    hyperparameters BEFORE starting a training run, given the task, the
    dataset's actual stats, and the board the user plans to deploy to.
    """
    try:
        dataset = data_manager.get_dataset(request.dataset_id)
        if not dataset:
            return {"status": "error", "message": f"Dataset {request.dataset_id} not found"}

        labels = data_manager.get_dataset_labels(request.dataset_id)
        split_summary = data_manager.get_split_summary(request.dataset_id)

        dataset_stats = {
            "sample_count": dataset.get("sample_count", 0),
            "num_classes": len(labels),
            "labels": labels,
            "split_summary": split_summary,
        }

        recommendation = await llm_advisor.recommend_training_params(
            task=request.task,
            dataset_stats=dataset_stats,
            target_board=request.target_board,
            provider=request.provider,
            model_name=request.model_name,
        )
        return {"status": "success", "recommendation": recommendation}
    except Exception as e:
        return {"status": "error", "message": str(e)}
