import io
import zipfile
from fastapi import APIRouter, File, Response, UploadFile, Form, HTTPException, Body
from fastapi.responses import StreamingResponse
from typing import List, Optional
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
    file: UploadFile = File(...),
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

@router.post("/split/{dataset_id}")
async def auto_split_dataset(
    dataset_id: str,
    train_pct: int = Body(...),
    val_pct: int = Body(...),
    test_pct: int = Body(...)
):
    try:
        count = data_manager.auto_split_dataset(dataset_id, train_pct, val_pct, test_pct)
        return {"status": "success", "message": f"Assigned splits for {count} samples"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.patch("/sample/{sample_id}/split")
async def update_sample_split(sample_id: str, split: str = Body(..., embed=True)):
    try:
        if data_manager.set_sample_split(sample_id, split):
            return {"status": "success"}
        return {"status": "error", "message": "Sample not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/export/full/{dataset_id}")
async def export_full(dataset_id: str):
    samples = data_manager.get_samples(dataset_id)
    dataset = data_manager.datasets.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for s in samples:
            data = data_manager.get_sample_data(s["id"])
            if data:
                file_path = f"{s['label']}/{s['filename']}"
                zip_file.writestr(file_path, data)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={dataset['name'].replace(' ', '_')}_full.zip"}
    )

@router.get("/export/split/{dataset_id}")
async def export_split(dataset_id: str):
    samples = data_manager.get_samples(dataset_id)
    dataset = data_manager.datasets.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for s in samples:
            data = data_manager.get_sample_data(s["id"])
            if data:
                split_dir = s.get("split", "unassigned")
                file_path = f"{split_dir}/{s['label']}/{s['filename']}"
                zip_file.writestr(file_path, data)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={dataset['name'].replace(' ', '_')}_split.zip"}
    )

@router.get("/image/{sample_id}")
async def get_sample_image(sample_id: str):
    data = data_manager.get_sample_data(sample_id)
    if not data:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(
        content=data, 
        media_type="image/jpeg",
        # Enable aggressive browser caching. Images are immutable by UUID.
        headers={"Cache-Control": "public, max-age=31536000, immutable"}
    )

@router.patch("/relabel/{sample_id}")
async def relabel_sample(sample_id: str, label: str = Body(..., embed=True)):
    try:
        if data_manager.relabel_sample(sample_id, label):
            return {"status": "success"}
        return {"status": "error", "message": "Sample not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/labels/{dataset_id}")
async def get_dataset_labels(dataset_id: str):
    try:
        labels = data_manager.get_dataset_labels(dataset_id)
        return {"status": "success", "labels": labels}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/labels/{dataset_id}/add")
async def add_label(dataset_id: str, label: str = Body(..., embed=True)):
    try:
        result = data_manager.add_label(dataset_id, label)
        return {"status": "success", "label": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/labels/{dataset_id}/rename")
async def rename_label(
    dataset_id: str,
    old_label: str = Body(...),
    new_label: str = Body(...),
):
    try:
        count = data_manager.rename_label(dataset_id, old_label, new_label)
        return {"status": "success", "renamed": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/labels/{dataset_id}/{label}")
async def delete_label(dataset_id: str, label: str):
    try:
        count = data_manager.delete_label(dataset_id, label)
        return {"status": "success", "deleted": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/{sample_id}")
async def delete_sample(sample_id: str):
    if data_manager.delete_sample(sample_id):
        return {"status": "success"}
    return {"status": "error", "message": "Sample not found"}