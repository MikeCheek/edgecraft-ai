import re
import io
import os
import logging
from PIL import Image
import uuid
import tempfile
import zipfile
import asyncio
import shutil
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from app.utils.zip_processor import extract_zip_with_mapping, scan_zip_tree
from fastapi import APIRouter, File, Request, UploadFile, Form, HTTPException, Body
from fastapi.responses import Response, FileResponse, StreamingResponse
from starlette.background import BackgroundTasks
from app.services.shared_state import data_manager
import time

logger = logging.getLogger(__name__)

# Optimized thread pool: scales with CPU cores
_DISK_EXECUTOR = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4), thread_name_prefix="disk_io")

router = APIRouter()

CHUNK_DIR = os.path.join(tempfile.gettempdir(), "edgecraft_chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

UPLOAD_TRACKER: dict[str, int] = {}
WRITE_BUFFER_SIZE = 4 * 1024 * 1024  # 4MB
_READ_BUF = bytearray(2 * 1024 * 1024)
ZIP_PROCESSING_BATCH_SIZE = 500

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

async def _run_in_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_DISK_EXECUTOR, fn, *args)

def _cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def _probe_image_dims(content: bytes):
    """Best-effort (width, height) probe; returns (None, None) on anything
    that isn't a readable image (e.g. .wav files, corrupt images)."""
    try:
        with Image.open(io.BytesIO(content)) as img:
            return img.width, img.height
    except Exception:
        return None, None

# --- Dataset CRUD ---

@router.post("/create")
async def create_dataset(name: str = Body(...), task: str = Body(...), description: str = Body("")):
    try:
        dataset = await _run_in_executor(data_manager.create_dataset, name, task, description)
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

# --- Metadata (description + auto-computed dataset-wide stats for the LLM) ---

@router.patch("/{dataset_id}/metadata")
async def update_dataset_metadata(
    dataset_id: str,
    description: str = Body(None),
    metadata: dict = Body(None),
):
    """Lets the user attach a free-text description (and, if ever needed,
    arbitrary extra metadata keys) to a dataset. Both fields are optional so
    the caller can update just one."""
    try:
        dataset = await _run_in_executor(
            data_manager.update_dataset_metadata, dataset_id, description, metadata
        )
        if dataset is None:
            return {"status": "error", "message": "Dataset not found"}
        return {"status": "success", "dataset": dataset}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/{dataset_id}/image_stats")
async def get_dataset_image_stats(dataset_id: str):
    """Aggregate image size / aspect-ratio / storage stats for a dataset -
    consumed by both the UI and the LLM advisor prompt so parameter
    suggestions (input resolution, augmentation, etc.) can be grounded in
    what the dataset actually looks like."""
    try:
        stats = await _run_in_executor(data_manager.get_dataset_image_stats, dataset_id)
        return {"status": "success", "stats": stats}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

# --- Single-file upload ---

@router.post("/upload")
async def upload_dataset_sample(
    dataset_id: str = Form(...),
    label: str = Form(...),
    task: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        content = await file.read()
        ext = os.path.splitext(file.filename or "")[1].lower()
        width, height = (None, None)
        if ext in IMAGE_EXTENSIONS:
            width, height = await _run_in_executor(_probe_image_dims, content)
        sample_id = await _run_in_executor(
            data_manager.add_sample, dataset_id, label, task, content, file.filename, width, height
        )
        return {"status": "success", "sample_id": sample_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- Chunked Resumable ZIP upload Engine ---

@router.delete("/upload_zip/{upload_id}")
async def abort_zip_upload(upload_id: str):
    """Abort an in-progress chunked ZIP upload, cleaning up chunks on disk."""
    upload_dir = os.path.join(CHUNK_DIR, upload_id)
    if os.path.isdir(upload_dir):
        shutil.rmtree(upload_dir, ignore_errors=True)
    UPLOAD_TRACKER.pop(upload_id, None)
    return {"status": "success", "message": "Upload aborted"}

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

async def _upload_zip_chunk_put(upload_id: str, chunk_index: int, request: Request):
    t_start = time.perf_counter()
    upload_dir = os.path.join(CHUNK_DIR, upload_id)
    if not os.path.isdir(upload_dir):
        raise HTTPException(status_code=404, detail=f"Unknown session: {upload_id}")

    chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:06d}")

    # 1. Measure Network/Streaming Time
    t_net_start = time.perf_counter()
    chunk_data = await request.body()
    t_net_end = time.perf_counter()

    # Write it directly to disk in one shot using the optimized disk executor
    def _write_chunk():
        with open(chunk_path, "wb") as f:
            f.write(chunk_data)

    t_disk_start = time.perf_counter()
    await _run_in_executor(_write_chunk)
    t_disk_end = time.perf_counter()

    UPLOAD_TRACKER[upload_id] = UPLOAD_TRACKER.get(upload_id, 0) + 1

    # 3. Print the diagnostic report
    net_time = t_net_end - t_net_start
    disk_time = t_disk_end - t_disk_start
    total_time = t_disk_end - t_start

    logger.info(f"[Chunk {chunk_index:03d}] Total: {total_time:.3f}s | Network Recv: {net_time:.3f}s | Disk Write: {disk_time:.3f}s")

    return {"status": "success", "chunk_index": chunk_index, "received": UPLOAD_TRACKER[upload_id]}

@router.put("/upload_zip/chunk/{upload_id}/{chunk_index}")
async def upload_zip_chunk_put(upload_id: str, chunk_index: int, request: Request):
    return await _upload_zip_chunk_put(upload_id, chunk_index, request)

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

def _assemble_and_process_zip(upload_dir: str, total_chunks: int, dataset_id: str, task: str) -> int:
    # NOTE: kept for backward compatibility - not currently called by any
    # route. The active flow is finalize -> scan_zip_tree -> (user maps
    # folders in the UI) -> process_zip_upload -> extract_zip_with_mapping.
    assembled_zip_path = os.path.join(upload_dir, "assembled_dataset.zip")

    # Assemble chunks into a single file from CHUNK_DIR/{upload_id}
    with open(assembled_zip_path, "wb") as outfile:
        for i in range(total_chunks):
            chunk_path = os.path.join(upload_dir, f"chunk_{i:06d}")
            if not os.path.exists(chunk_path):
                raise ValueError(f"Missing chunk index {i}")

            with open(chunk_path, "rb") as infile:
                shutil.copyfileobj(infile, outfile, length=WRITE_BUFFER_SIZE)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".wav"}
    total_processed = 0

    # Normalized mapping for recognized split folder names
    split_mapping = {
        "train": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test"
    }

    # Batched extraction
    # Pre-split and cache lookup sets to reduce runtime string parsing overhead
    split_set = {"train", "val", "valid", "validation", "test"}

    with zipfile.ZipFile(assembled_zip_path, 'r') as z:
        # Filter out junk quickly with a lightweight list comprehension
        valid_items = [
            info for info in z.infolist()
            if not info.is_dir()
            and not info.filename.startswith("__MACOSX")
            and not info.filename.split("/")[-1].startswith(".")
        ]

        if not valid_items:
            raise ValueError("Archive has no valid assets.")

        # Pre-calculate tree structure once instead of per-iteration to maintain prior high-speeds
        first_parts = {info.filename.split("/")[0] for info in valid_items}
        has_files_at_root = any("/" not in info.filename for info in valid_items)
        has_common_root = len(first_parts) == 1 and not has_files_at_root

        file_data_list = []
        q = queue.Queue(maxsize=4)

        def _writer():
            while True:
                batch = q.get()
                if batch is None:
                    break
                data_manager.bulk_add_samples(dataset_id, task, batch)
                total_processed_ref[0] += len(batch)

        total_processed_ref = [0]
        writer_thread = threading.Thread(target=_writer, daemon=True)
        writer_thread.start()
        for info in valid_items:
            # Quick structural validation splitting
            parts = [p for p in info.filename.split("/") if p]
            if not parts:
                continue

            filename = parts[-1]
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                continue

            split = "unassigned"
            label = "unknown"

            # Optimization: Fast checks using direct lookups instead of deep list mutations
            p0_lower = parts[0].lower()
            if p0_lower in split_set:
                if len(parts) >= 3:
                    split = split_mapping.get(p0_lower, "unassigned")
                    label = parts[1]
            elif len(parts) >= 4 and parts[1].lower() in split_set:
                split = split_mapping.get(parts[1].lower(), "unassigned")
                label = parts[2]
            else:
                # Fallback to the original flat extraction logic style
                if has_common_root:
                    if len(parts) >= 3:
                        label = parts[1]
                else:
                    if len(parts) >= 2:
                        label = parts[0]

                size = info.file_size
                buf = bytearray(size)
                mv = memoryview(buf)
                pos = 0
                with z.open(info) as extracted_file:
                    while pos < size:
                        n = extracted_file.readinto(mv[pos:])
                        if not n:
                            break
                        pos += n
                content = bytes(buf)
            file_data_list.append({
                    "label": label,
                    "filename": filename,
                    "content": content,
                    "split": split
                })

            if len(file_data_list) >= ZIP_PROCESSING_BATCH_SIZE:
                q.put(file_data_list[:])
                file_data_list.clear()
                total_processed += len(file_data_list)
                file_data_list.clear()

    shutil.rmtree(upload_dir, ignore_errors=True)
    return total_processed

def clean_dataset_directory(dataset_dir):
    # TensorFlow supported formats
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    removed_count = 0

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # 1. Remove hidden/system files immediately
            if file.startswith('.') or file.lower() == 'thumbs.db':
                logger.info(f"Removing hidden file: {file_path}")
                os.remove(file_path)
                removed_count += 1
                continue

            # 2. Check if the extension is strictly valid
            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_extensions:
                logger.info(f"Removing unsupported format: {file_path}")
                os.remove(file_path)
                removed_count += 1
                continue

            # 3. Open the file to verify header integrity (catches fake/corrupt images)
            try:
                with Image.open(file_path) as img:
                    img.verify() # Reads the header, doesn't load whole image into memory
            except Exception as e:
                logger.info(f"Removing corrupted image: {file_path} - {e}")
                os.remove(file_path)
                removed_count += 1

    logger.info(f"Cleanup finished! Removed {removed_count} invalid files.")

@router.post("/upload_zip/finalize")
async def finalize_zip_upload(
    upload_id: str = Body(...),
    total_chunks: int = Body(...),
):
    upload_dir = os.path.join(CHUNK_DIR, upload_id)
    if not os.path.isdir(upload_dir):
        raise HTTPException(status_code=404, detail="Upload target expired or missing")

    meta_path = os.path.join(upload_dir, "_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    if meta["total_chunks"] != total_chunks:
        raise HTTPException(status_code=400, detail="Incomplete stream packet data loss detected")

    try:
        # Assemble chunks into a single file
        assembled_zip_path = os.path.join(upload_dir, "assembled_dataset.zip")
        with open(assembled_zip_path, "wb") as outfile:
            for i in range(total_chunks):
                chunk_path = os.path.join(upload_dir, f"chunk_{i:06d}")
                with open(chunk_path, "rb") as infile:
                    shutil.copyfileobj(infile, outfile, length=WRITE_BUFFER_SIZE)

        # Scan the tree instead of blindly extracting
        tree = await _run_in_executor(scan_zip_tree, assembled_zip_path)

        UPLOAD_TRACKER.pop(upload_id, None)
        return {"status": "success", "upload_id": upload_id, "tree": tree}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assembly crash: {str(e)}")

@router.post("/upload_zip/preview_regex")
async def preview_zip_regex(
    upload_id: str = Body(...),
    regex_pattern: str = Body(...)
):
    upload_dir = os.path.join(CHUNK_DIR, upload_id)
    assembled_zip_path = os.path.join(upload_dir, "assembled_dataset.zip")

    if not os.path.exists(assembled_zip_path):
        raise HTTPException(status_code=404, detail="Assembled ZIP not found")

    try:
        compiled_regex = re.compile(regex_pattern)
    except re.error as e:
        raise HTTPException(status_code=400, detail=f"Invalid regex: {e}")

    def _preview():
        classes = {}
        samples = []
        with zipfile.ZipFile(assembled_zip_path, 'r') as z:
            for info in z.infolist():
                if info.is_dir() or info.filename.startswith("__MACOSX"): continue
                filename = info.filename.split("/")[-1]
                match = compiled_regex.search(filename)
                if match:
                    label = match.group(1) if match.groups() else match.group(0)
                    classes[label] = classes.get(label, 0) + 1
                    if len(samples) < 5:
                        samples.append({"filename": filename, "label": label})

        return {
            "classes": sorted([{"name": k, "count": v} for k, v in classes.items()], key=lambda x: x["count"], reverse=True),
            "samples": samples
        }

    try:
        result = await _run_in_executor(_preview)
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Update the process_zip_upload endpoint signature and call
@router.post("/upload_zip/process")
async def process_zip_upload(
    upload_id: str = Body(...),
    dataset_id: str = Body(...),
    task: str = Body(...),
    mapping: list = Body(...),
    label_strategy: str = Body("folder"), # NEW
    regex_pattern: str = Body(None)       # NEW
):
    """Executes the extraction using the confirmed folder mapping and regex.

    BUGFIX: extract_zip_with_mapping() used to raise a bare NameError on
    every "filename regex" apply because app/utils/zip_processor.py never
    imported `re` - so this endpoint always 500'd before processing a single
    file whenever label_strategy == "filename". That import is now added.
    extract_zip_with_mapping() also now returns a small dict (processed
    count + how many files didn't match the regex + any regex compile
    error) instead of a bare int, so the frontend can show the user exactly
    what happened instead of a silent mismatch between "preview" and
    "apply".
    """
    upload_dir = os.path.join(CHUNK_DIR, upload_id)
    assembled_zip_path = os.path.join(upload_dir, "assembled_dataset.zip")

    if not os.path.exists(assembled_zip_path):
        raise HTTPException(status_code=404, detail="Assembled ZIP not found")

    try:
        result = await _run_in_executor(
            extract_zip_with_mapping, assembled_zip_path, dataset_id, task, mapping, label_strategy, regex_pattern
        )
        shutil.rmtree(upload_dir, ignore_errors=True)
        return {
            "status": "success",
            "count": result["processed"],
            "unmatched_regex": result["unmatched_regex"],
            "regex_error": result["regex_error"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction crash: {str(e)}")

# Add the Dataset Explorer bulk relabeling endpoint
@router.post("/{dataset_id}/relabel_bulk_regex")
async def bulk_relabel_dataset_regex(
    dataset_id: str,
    regex_pattern: str = Body(..., embed=True)
):
    try:
        count = await _run_in_executor(data_manager.bulk_relabel_by_regex, dataset_id, regex_pattern)
        return {"status": "success", "relabelled_count": count}
    # except ValueError as e:
    #     return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Samples & Metadata ---

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

@router.get("/split_summary/{dataset_id}")
async def get_split_summary(dataset_id: str):
    try:
        summary = await _run_in_executor(data_manager.get_split_summary, dataset_id)
        return {"status": "success", "summary": summary}
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

# --- Export Engine ---

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

# --- Relabeling & Sample Management ---

@router.patch("/sample/{sample_id}/split")
async def update_sample_split(sample_id: str, split: str = Body(..., embed=True)):
    try:
        success = await _run_in_executor(data_manager.set_sample_split, sample_id, split)
        if success:
            return {"status": "success"}
        return {"status": "error", "message": "Sample not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@router.patch("/sample/{sample_id}/image")
async def update_sample_image(sample_id: str, file: UploadFile = File(...)):
    """Overwrite a sample's image content in place - backs the Dataset
    Explorer's crop tool. Keeps label/split/id unchanged; re-probes
    width/height from the new bytes so image_stats (and the LLM advisor
    prompt that reads it) stays accurate after a crop."""
    try:
        content = await file.read()
        width, height = await _run_in_executor(_probe_image_dims, content)
        success = await _run_in_executor(
            data_manager.update_sample_data, sample_id, content, width, height
        )
        if not success:
            return {"status": "error", "message": "Sample not found"}
        return {"status": "success", "width": width, "height": height}
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

# BUGFIX ("delete samples by label" -> 405 Method Not Allowed): this route
# was DELETE-only. If the frontend's apiClient sends this as a POST (a
# common pattern for calls that need a body, or if a proxy/browser retries
# with a different verb), FastAPI correctly reports 405 because no route
# matched with that exact verb. Accepting both DELETE and POST for the same
# handler removes that mismatch regardless of which verb the frontend
# actually sends.
@router.api_route("/labels/{dataset_id}/{label}", methods=["DELETE", "POST"])
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
