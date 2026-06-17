import io
import os
import uuid
import tempfile
import zipfile
import asyncio
import shutil
import json
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, File, Request, UploadFile, Form, HTTPException, Body
from fastapi.responses import Response, FileResponse, StreamingResponse
from starlette.background import BackgroundTasks
from typing import List, Optional
from app.services.shared_state import data_manager
import aiofiles

# Optimized thread pool: scales with CPU cores
_DISK_EXECUTOR = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4), thread_name_prefix="disk_io")

router = APIRouter()

CHUNK_DIR = os.path.join(tempfile.gettempdir(), "edgecraft_chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

UPLOAD_TRACKER = {}
WRITE_BUFFER_SIZE = 4 * 1024 * 1024  # 4MB
ZIP_PROCESSING_BATCH_SIZE = 500


async def _run_in_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_DISK_EXECUTOR, fn, *args)


def _cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ── Dataset CRUD ──────────────────────────────────────────────────────────────

@router.post("/create")
async def create_dataset(name: str = Body(...), task: str = Body(...)):
    try:
        dataset = await _run_in_executor(data_manager.create_dataset, name, task)
        return {"status": "success", "dataset": dataset}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/list_datasets")
async def list_datasets(task: str = None):
    try:
        datasets = await _run_in_executor(data_manager.get_datasets, task)
        return {"status": "success", "datasets": datasets}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.put("/rename/{dataset_id}")
async def rename_dataset(dataset_id: str, new_name: str = Body(...)):
    success = await _run_in_executor(data_manager.rename_dataset, dataset_id, new_name)
    if success:
        return {"status": "success"}
    return {"status": "error", "message": "Dataset not found"}


@router.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    success = await _run_in_executor(data_manager.delete_dataset, dataset_id)
    if success:
        return {"status": "success"}
    return {"status": "error", "message": "Dataset not found"}


@router.delete("/clear_dataset/{dataset_id}")
async def clear_dataset(dataset_id: str):
    count = await _run_in_executor(data_manager.clear_dataset_samples, dataset_id)
    return {"status": "success", "message": f"Cleared {count} samples"}


# ── Single-file upload ────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_dataset_sample(
    dataset_id: str = Form(...),
    label: str = Form(...),
    task: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        content = await file.read()
        sample_id = await _run_in_executor(
            data_manager.add_sample, dataset_id, label, task, content, file.filename
        )
        return {"status": "success", "sample_id": sample_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Chunked Resumable ZIP upload Engine ───────────────────────────────────────

@router.post("/upload_zip/init")
async def init_zip_upload(
    dataset_id: str = Body(...),
    task: str = Body(...),
    filename: str = Body(...),
    total_chunks: int = Body(...),
    file_size: int = Body(...),
):
    upload_id = str(uuid.uuid4())
    upload_dir = os.path.join(CHUNK_DIR, upload_id)
    os.makedirs(upload_dir, exist_ok=True)

    meta = {
        "dataset_id": dataset_id,
        "task": task,
        "filename": filename,
        "total_chunks": total_chunks,
        "file_size": file_size,
    }
    meta_path = os.path.join(upload_dir, "_meta.json")

    def _write_meta():
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    await _run_in_executor(_write_meta)
    return {"status": "success", "upload_id": upload_id}


@router.put("/upload_zip/chunk/{upload_id}/{chunk_index}")
async def upload_zip_chunk_put(upload_id: str, chunk_index: int, request: Request):
    upload_dir = os.path.join(CHUNK_DIR, upload_id)
    if not os.path.isdir(upload_dir):
        raise HTTPException(status_code=404, detail=f"Unknown session: {upload_id}")

    chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:06d}")

    content_length = request.headers.get("content-length")
    if content_length and hasattr(os, "posix_fallocate"):
        try:
            fd = os.open(chunk_path, os.O_CREAT | os.O_WRONLY)
            os.posix_fallocate(fd, 0, int(content_length))
            os.close(fd)
        except OSError:
            pass

    async with aiofiles.open(chunk_path, "wb") as f:
        buffer = bytearray()
        async for chunk_data in request.stream():
            buffer.extend(chunk_data)
            if len(buffer) >= WRITE_BUFFER_SIZE:
                await f.write(buffer)
                buffer.clear()
        if buffer:
            await f.write(buffer)

    UPLOAD_TRACKER[upload_id] = UPLOAD_TRACKER.get(upload_id, 0) + 1
    return {"status": "success", "chunk_index": chunk_index, "received": UPLOAD_TRACKER[upload_id]}


@router.get("/upload_zip/status/{upload_id}")
async def zip_upload_status(upload_id: str):
    upload_dir = os.path.join(CHUNK_DIR, upload_id)
    if not os.path.isdir(upload_dir):
        raise HTTPException(status_code=404, detail=f"Unknown upload session: {upload_id}")

    def _scan():
        return sorted(
            int(name.split("_")[1])
            for name in os.listdir(upload_dir)
            if name.startswith("chunk_")
        )

    received = await _run_in_executor(_scan)
    return {"status": "success", "upload_id": upload_id, "received_chunks": received}


def _assemble_and_process_zip(upload_dir: str, meta: dict) -> int:
    total_chunks = meta["total_chunks"]
    dataset_id = meta["dataset_id"]
    task = meta["task"]

    assembled_zip_path = os.path.join(upload_dir, "assembled_dataset.zip")

    # Assemble chunks into a single file
    with open(assembled_zip_path, "wb") as outfile:
        for i in range(total_chunks):
            chunk_path = os.path.join(upload_dir, f"chunk_{i:06d}")
            if not os.path.exists(chunk_path):
                raise ValueError(f"Missing chunk index {i}")

            with open(chunk_path, "rb") as infile:
                shutil.copyfileobj(infile, outfile, length=WRITE_BUFFER_SIZE)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".wav"}
    total_processed = 0

    # Batched extraction
    with zipfile.ZipFile(assembled_zip_path, 'r') as z:
        valid_items = [
            info for info in z.infolist()
            if not info.is_dir()
            and not info.filename.startswith("__MACOSX")
            and not info.filename.split("/")[-1].startswith(".")
        ]

        if not valid_items:
            raise ValueError("Archive has no valid assets.")

        first_parts = {info.filename.split("/")[0] for info in valid_items}
        has_files_at_root = any(len(info.filename.split("/")) == 1 for info in valid_items)
        has_common_root = len(first_parts) == 1 and not has_files_at_root

        file_data_list = []
        for info in valid_items:
            parts = info.filename.split("/")
            if has_common_root:
                if len(parts) < 3:
                    continue
                label = parts[1].strip()
            else:
                if len(parts) < 2:
                    continue
                label = parts[0].strip()

            filename = parts[-1]
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                continue

            with z.open(info) as extracted_file:
                content = extracted_file.read()
                file_data_list.append({
                    "label": label,
                    "filename": filename,
                    "content": content,
                })

            if len(file_data_list) >= ZIP_PROCESSING_BATCH_SIZE:
                data_manager.bulk_add_samples(dataset_id, task, file_data_list)
                total_processed += len(file_data_list)
                file_data_list.clear()

        if file_data_list:
            data_manager.bulk_add_samples(dataset_id, task, file_data_list)
            total_processed += len(file_data_list)

    shutil.rmtree(upload_dir, ignore_errors=True)
    return total_processed


@router.post("/upload_zip/finalize")
async def finalize_zip_upload(
    upload_id: str = Body(...),
    dataset_id: str = Body(...),
    task: str = Body(...),
    total_chunks: int = Body(...),
):
    upload_dir = os.path.join(CHUNK_DIR, upload_id)
    if not os.path.isdir(upload_dir):
        raise HTTPException(status_code=404, detail="Upload target expired or missing")

    meta_path = os.path.join(upload_dir, "_meta.json")

    def _read_meta():
        with open(meta_path, "r") as f:
            return json.load(f)

    meta = await _run_in_executor(_read_meta)

    if meta["total_chunks"] != total_chunks:
        raise HTTPException(status_code=400, detail="Incomplete stream packet data loss detected")

    try:
        count = await _run_in_executor(_assemble_and_process_zip, upload_dir, meta)
        UPLOAD_TRACKER.pop(upload_id, None)
        return {"status": "success", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Post-stream pipeline unpack crash: {str(e)}")


# ── Samples & Metadata ────────────────────────────────────────────────────────

@router.get("/list")
async def list_samples(dataset_id: str = None):
    try:
        samples = await _run_in_executor(data_manager.get_samples, dataset_id)
        return {"status": "success", "count": len(samples), "samples": samples}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/stats")
async def get_dataset_statistics():
    try:
        stats = await _run_in_executor(data_manager.get_statistics)
        return {"status": "success", **stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/split/{dataset_id}")
async def auto_split_dataset(
    dataset_id: str,
    train_pct: int = Body(...),
    val_pct: int = Body(...),
    test_pct: int = Body(...),
):
    try:
        count = await _run_in_executor(data_manager.auto_split_dataset, dataset_id, train_pct, val_pct, test_pct)
        return {"status": "success", "message": f"Assigned splits for {count} samples"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Export Engine ─────────────────────────────────────────────────────────────

def _build_export_zip(samples: list, dataset_name: str, mode: str) -> str:
    """Builds the zip archive on disk."""
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip_path = temp_zip.name
    temp_zip.close()

    with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
        for s in samples:
            data = data_manager.get_sample_data(s["id"])
            if data:
                if mode == "split":
                    split_dir = s.get("split", "unassigned")
                    arc_path = f"{split_dir}/{s['label']}/{s['filename']}"
                else:
                    arc_path = f"{s['label']}/{s['filename']}"
                zip_file.writestr(arc_path, data)
    return temp_zip_path


@router.get("/export/full/{dataset_id}")
async def export_full(dataset_id: str, background_tasks: BackgroundTasks):
    samples = await _run_in_executor(data_manager.get_samples, dataset_id)
    dataset = await _run_in_executor(data_manager.datasets.get, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    zip_path = await _run_in_executor(_build_export_zip, samples, dataset['name'], "full")
    background_tasks.add_task(_cleanup_file, zip_path)

    filename = f"{dataset['name'].replace(' ', '_')}_full.zip"
    return FileResponse(zip_path, media_type="application/zip", filename=filename)


@router.get("/export/split/{dataset_id}")
async def export_split(dataset_id: str, background_tasks: BackgroundTasks):
    samples = await _run_in_executor(data_manager.get_samples, dataset_id)
    dataset = await _run_in_executor(data_manager.datasets.get, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    zip_path = await _run_in_executor(_build_export_zip, samples, dataset['name'], "split")
    background_tasks.add_task(_cleanup_file, zip_path)

    filename = f"{dataset['name'].replace(' ', '_')}_split.zip"
    return FileResponse(zip_path, media_type="application/zip", filename=filename)


# ── Relabeling & Sample Management ───────────────────────────────────────────

@router.patch("/sample/{sample_id}/split")
async def update_sample_split(sample_id: str, split: str = Body(..., embed=True)):
    try:
        success = await _run_in_executor(data_manager.set_sample_split, sample_id, split)
        if success:
            return {"status": "success"}
        return {"status": "error", "message": "Sample not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/image/{sample_id}")
async def get_sample_image(sample_id: str):
    data = await _run_in_executor(data_manager.get_sample_data, sample_id)
    if not data:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.patch("/relabel/{sample_id}")
async def relabel_sample(sample_id: str, label: str = Body(..., embed=True)):
    try:
        if await _run_in_executor(data_manager.relabel_sample, sample_id, label):
            return {"status": "success"}
        return {"status": "error", "message": "Sample not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/labels/{dataset_id}")
async def get_dataset_labels(dataset_id: str):
    try:
        labels = await _run_in_executor(data_manager.get_dataset_labels, dataset_id)
        return {"status": "success", "labels": labels}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/labels/{dataset_id}/add")
async def add_label(dataset_id: str, label: str = Body(..., embed=True)):
    try:
        result = await _run_in_executor(data_manager.add_label, dataset_id, label)
        return {"status": "success", "label": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/labels/{dataset_id}/rename")
async def rename_label(dataset_id: str, old_label: str = Body(...), new_label: str = Body(...)):
    try:
        count = await _run_in_executor(data_manager.rename_label, dataset_id, old_label, new_label)
        return {"status": "success", "renamed": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.delete("/labels/{dataset_id}/{label}")
async def delete_label(dataset_id: str, label: str):
    try:
        count = await _run_in_executor(data_manager.delete_label, dataset_id, label)
        return {"status": "success", "deleted": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.delete("/{sample_id}")
async def delete_sample(sample_id: str):
    if await _run_in_executor(data_manager.delete_sample, sample_id):
        return {"status": "success"}    
    return {"status": "error", "message": "Sample not found"}