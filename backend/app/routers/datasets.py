from fastapi import APIRouter, File, Response, UploadFile, Form, HTTPException, Body
from typing import List
from app.services.shared_state import data_manager

router = APIRouter()

@router.post("/create")
async def create_dataset(name: str = Body(...), task: str = Body(...)):
    try:
        dataset = data_manager.create_dataset(name, task)
        return {"status": "success", "dataset": dataset}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/list_datasets")
async def list_datasets(task: str = None):
    try:
        datasets = data_manager.get_datasets(task)
        return {"status": "success", "datasets": datasets}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.put("/rename/{dataset_id}")
async def rename_dataset(dataset_id: str, new_name: str = Body(...)):
    if data_manager.rename_dataset(dataset_id, new_name):
        return {"status": "success"}
    return {"status": "error", "message": "Dataset not found"}

@router.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    if data_manager.delete_dataset(dataset_id):
        return {"status": "success"}
    return {"status": "error", "message": "Dataset not found"}

@router.delete("/clear_dataset/{dataset_id}")
async def clear_dataset(dataset_id: str):
    count = data_manager.clear_dataset_samples(dataset_id)
    return {"status": "success", "message": f"Cleared {count} samples"}

@router.post("/upload")
async def upload_dataset_sample(
    dataset_id: str = Form(...),
    label: str = Form(...),
    task: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        content = await file.read()
        sample_id = data_manager.add_sample(dataset_id, label, task, content, file.filename)
        return {"status": "success", "sample_id": sample_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/list")
async def list_samples(dataset_id: str = None):
    try:
        samples = data_manager.get_samples(dataset_id)
        return {"status": "success", "count": len(samples), "samples": samples}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/stats")
async def get_dataset_statistics():
    try:
        stats = data_manager.get_statistics()
        return {"status": "success", **stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/{sample_id}")
async def delete_sample(sample_id: str):
    if data_manager.delete_sample(sample_id):
        return {"status": "success"}
    return {"status": "error", "message": "Sample not found"}

@router.get("/image/{sample_id}")
async def get_sample_image(sample_id: str):
    data = data_manager.get_sample_data(sample_id)
    if not data:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=data, media_type="image/jpeg")