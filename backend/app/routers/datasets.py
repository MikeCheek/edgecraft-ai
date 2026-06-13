from fastapi import APIRouter, File, UploadFile, Form
from typing import List
import io
import base64
from app.services.data_manager import DataManager

router = APIRouter()
data_manager = DataManager()

@router.post("/upload")
async def upload_dataset_sample(
    label: str = Form(...),
    task: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload a single dataset sample (image or audio)"""
    try:
        content = await file.read()
        sample_id = data_manager.add_sample(
            label=label,
            task=task,
            data=content,
            filename=file.filename
        )
        return {
            "status": "success",
            "sample_id": sample_id,
            "message": f"Sample '{label}' uploaded successfully"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/list")
async def list_datasets(task: str = None):
    """List all dataset samples, optionally filtered by task"""
    try:
        samples = data_manager.get_samples(task)
        return {
            "status": "success",
            "count": len(samples),
            "samples": samples
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/stats")
async def get_dataset_statistics():
    """Get overall dataset statistics"""
    try:
        stats = data_manager.get_statistics()
        return {
            "status": "success",
            "total_samples": stats["total_samples"],
            "by_task": stats["by_task"],
            "by_label": stats["by_label"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/{sample_id}")
async def delete_sample(sample_id: str):
    """Delete a specific dataset sample"""
    try:
        data_manager.delete_sample(sample_id)
        return {
            "status": "success",
            "message": f"Sample {sample_id} deleted"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/clear")
async def clear_datasets(task: str = None):
    """Clear dataset samples, optionally filtered by task"""
    try:
        count = data_manager.clear_samples(task)
        return {
            "status": "success",
            "message": f"{count} samples cleared"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
