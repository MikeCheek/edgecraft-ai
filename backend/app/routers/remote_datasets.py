import os
import uuid
import shutil
import tempfile
import zipfile
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.shared_state import data_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="remote_dl")
DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "edgecraft_remote_downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".wav"}
ZIP_PROCESSING_BATCH_SIZE = 500

# ─── Active Downloads Tracking ────────────────────────────────────────────────

_active_downloads: Dict[str, dict] = {}  # download_id -> {cancelled: bool}


async def _run_in_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXECUTOR, fn, *args)


# ─── Shared Processing Logic ─────────────────────────────────────────────────

def _process_zip_to_dataset(zip_path: str, dataset_id: str, task: str, download_id: str = None) -> int:
    """Extract a zip file and ingest samples into the data_manager."""
    total_processed = 0

    with zipfile.ZipFile(zip_path, 'r') as z:
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
            # Check cancellation
            if download_id and _active_downloads.get(download_id, {}).get("cancelled"):
                raise ValueError("Download cancelled by user")

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
            if ext not in VALID_EXTENSIONS:
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

    return total_processed


# ─── Token Status ─────────────────────────────────────────────────────────────

@router.get("/token_status")
async def get_token_status():
    """Check which API tokens are configured."""
    kaggle_configured = bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    hf_configured = bool(os.environ.get("HUGGINGFACE_TOKEN"))
    return {
        "status": "success",
        "kaggle_configured": kaggle_configured,
        "huggingface_configured": hf_configured,
    }


# ─── Cancel Endpoint ──────────────────────────────────────────────────────────

class CancelRequest(BaseModel):
    download_id: str

@router.post("/cancel")
async def cancel_download(req: CancelRequest):
    """Cancel an active download."""
    if req.download_id in _active_downloads:
        _active_downloads[req.download_id]["cancelled"] = True
        return {"status": "success", "message": "Cancellation requested"}
    return {"status": "error", "message": "Download not found or already completed"}


# ─── SSE Download Stream (unified endpoint with progress) ────────────────────

@router.get("/download_stream")
async def download_stream(
    source: str = Query(...),          # "url" | "kaggle" | "huggingface"
    dataset_id: str = Query(...),
    task: str = Query(...),
    url: Optional[str] = Query(None),
    dataset_ref: Optional[str] = Query(None),
    repo_id: Optional[str] = Query(None),
):
    """SSE endpoint that streams download progress back to the frontend."""
    download_id = str(uuid.uuid4())
    _active_downloads[download_id] = {"cancelled": False}

    async def full_event_stream():
        zip_path = None
        try:
            # Send download_id first
            yield f"data: {_sse_json('started', download_id)}\n\n"

            if source == "url":
                if not url:
                    yield f"data: {_sse_json('error', download_id, message='URL is required')}\n\n"
                    return

                dl_path = os.path.join(DOWNLOAD_DIR, f"{download_id}.zip")
                async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
                    async with client.stream("GET", url) as response:
                        if response.status_code != 200:
                            yield f"data: {_sse_json('error', download_id, message=f'HTTP {response.status_code}')}\n\n"
                            return

                        total = int(response.headers.get("content-length", 0))
                        downloaded = 0

                        with open(dl_path, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                                if _active_downloads.get(download_id, {}).get("cancelled"):
                                    yield f"data: {_sse_json('error', download_id, message='Download cancelled')}\n\n"
                                    return

                                f.write(chunk)
                                downloaded += len(chunk)
                                yield f"data: {_sse_json('progress', download_id, downloaded=downloaded, total=total)}\n\n"

                if not zipfile.is_zipfile(dl_path):
                    yield f"data: {_sse_json('error', download_id, message='Downloaded file is not a valid ZIP')}\n\n"
                    return
                zip_path = dl_path

            elif source == "kaggle":
                if not dataset_ref:
                    yield f"data: {_sse_json('error', download_id, message='dataset_ref is required')}\n\n"
                    return

                yield f"data: {_sse_json('progress', download_id, downloaded=0, total=0)}\n\n"

                username = os.environ.get("KAGGLE_USERNAME")
                key = os.environ.get("KAGGLE_KEY")
                if not username or not key:
                    yield f"data: {_sse_json('error', download_id, message='Kaggle credentials not configured')}\n\n"
                    return

                kaggle_dir = os.path.join(DOWNLOAD_DIR, f"kaggle_{download_id}")
                os.makedirs(kaggle_dir, exist_ok=True)

                def _kaggle_dl():
                    os.environ["KAGGLE_USERNAME"] = username
                    os.environ["KAGGLE_KEY"] = key
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    api = KaggleApi()
                    api.authenticate()
                    api.dataset_download_files(dataset_ref, path=kaggle_dir, unzip=False)
                    zips = [f for f in os.listdir(kaggle_dir) if f.endswith(".zip")]
                    if not zips:
                        raise ValueError("No zip file found after Kaggle download")
                    return os.path.join(kaggle_dir, zips[0])

                zip_path = await _run_in_executor(_kaggle_dl)

                if _active_downloads.get(download_id, {}).get("cancelled"):
                    yield f"data: {_sse_json('error', download_id, message='Download cancelled')}\n\n"
                    return

                # Report size after download
                file_size = os.path.getsize(zip_path)
                yield f"data: {_sse_json('progress', download_id, downloaded=file_size, total=file_size)}\n\n"

            elif source == "huggingface":
                if not repo_id:
                    yield f"data: {_sse_json('error', download_id, message='repo_id is required')}\n\n"
                    return

                yield f"data: {_sse_json('progress', download_id, downloaded=0, total=0)}\n\n"

                token = os.environ.get("HUGGINGFACE_TOKEN")
                hf_dir = os.path.join(DOWNLOAD_DIR, f"hf_{download_id}")
                os.makedirs(hf_dir, exist_ok=True)

                def _hf_dl():
                    from huggingface_hub import snapshot_download
                    local_path = snapshot_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        local_dir=hf_dir,
                        token=token if token else None,
                    )

                    # Look for zip files
                    for root, dirs, files in os.walk(local_path):
                        for f in files:
                            if f.endswith(".zip"):
                                return os.path.join(root, f)

                    # If no zip, assemble from valid files
                    temp_zip_path = os.path.join(hf_dir, "assembled.zip")
                    found_files = []
                    for root, dirs, files in os.walk(local_path):
                        for f in files:
                            ext = os.path.splitext(f)[1].lower()
                            if ext in VALID_EXTENSIONS:
                                full_path = os.path.join(root, f)
                                rel_path = os.path.relpath(full_path, local_path)
                                found_files.append((full_path, rel_path))

                    if not found_files:
                        raise ValueError("No valid image/audio files found in HuggingFace dataset")

                    with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for full_path, rel_path in found_files:
                            zf.write(full_path, rel_path)

                    return temp_zip_path

                zip_path = await _run_in_executor(_hf_dl)

                if _active_downloads.get(download_id, {}).get("cancelled"):
                    yield f"data: {_sse_json('error', download_id, message='Download cancelled')}\n\n"
                    return

                file_size = os.path.getsize(zip_path)
                yield f"data: {_sse_json('progress', download_id, downloaded=file_size, total=file_size)}\n\n"

            else:
                yield f"data: {_sse_json('error', download_id, message=f'Unknown source: {source}')}\n\n"
                return

            # ── Phase 2: Processing ──────────────────────────────────────
            yield f"data: {_sse_json('processing', download_id, message='Extracting and importing samples...')}\n\n"

            count = await _run_in_executor(
                _process_zip_to_dataset, zip_path, dataset_id, task, download_id
            )

            yield f"data: {_sse_json('complete', download_id, count=count)}\n\n"

        except Exception as e:
            error_msg = str(e)
            if "cancelled" in error_msg.lower():
                yield f"data: {_sse_json('error', download_id, message='Download cancelled')}\n\n"
            else:
                logger.exception(f"Download stream error [{download_id}]")
                yield f"data: {_sse_json('error', download_id, message=error_msg)}\n\n"
        finally:
            _active_downloads.pop(download_id, None)
            # Cleanup
            if zip_path and os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except Exception:
                    pass
            # Cleanup Kaggle/HF directories
            for prefix in ("kaggle_", "hf_"):
                d = os.path.join(DOWNLOAD_DIR, f"{prefix}{download_id}")
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)

    return StreamingResponse(
        full_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_json(event_type: str, download_id: str, **kwargs) -> str:
    """Build a JSON string for an SSE data payload."""
    import json
    payload = {"type": event_type, "download_id": download_id, **kwargs}
    return json.dumps(payload)


# ─── Kaggle Search (non-streaming) ───────────────────────────────────────────

@router.get("/kaggle/search")
async def search_kaggle_datasets(query: str = Query(..., min_length=1), page: int = 1, page_size: int = 20):
    """Search Kaggle datasets by keyword."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        raise HTTPException(status_code=400, detail="Kaggle credentials not configured in .env")

    try:
        def _search():
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            results = api.dataset_list(search=query, page=page)
            datasets = []
            for ds in results[:page_size]:
                datasets.append({
                    "ref": str(ds.ref),
                    "title": ds.title,
                    "size": ds.totalBytes if hasattr(ds, 'totalBytes') else 0,
                    "last_updated": str(ds.lastUpdated) if hasattr(ds, 'lastUpdated') else "",
                    "download_count": ds.downloadCount if hasattr(ds, 'downloadCount') else 0,
                    "description": ds.subtitle if hasattr(ds, 'subtitle') else "",
                })
            return datasets

        datasets = await _run_in_executor(_search)
        return {"status": "success", "datasets": datasets}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Kaggle search error")
        raise HTTPException(status_code=500, detail=f"Kaggle search failed: {str(e)}")


# ─── HuggingFace Search (non-streaming) ──────────────────────────────────────

@router.get("/huggingface/search")
async def search_huggingface_datasets(query: str = Query(..., min_length=1), limit: int = 20):
    """Search HuggingFace datasets by keyword."""
    token = os.environ.get("HUGGINGFACE_TOKEN")

    try:
        def _search():
            from huggingface_hub import HfApi
            api = HfApi(token=token if token else None)
            results = api.list_datasets(search=query, limit=limit, sort="downloads", direction=-1)
            datasets = []
            for ds in results:
                datasets.append({
                    "id": ds.id,
                    "author": ds.author if hasattr(ds, 'author') and ds.author else (ds.id.split("/")[0] if "/" in ds.id else ""),
                    "title": ds.id.split("/")[-1] if "/" in ds.id else ds.id,
                    "downloads": ds.downloads if hasattr(ds, 'downloads') else 0,
                    "last_modified": str(ds.last_modified) if hasattr(ds, 'last_modified') and ds.last_modified else "",
                    "tags": (ds.tags[:5] if ds.tags else []) if hasattr(ds, 'tags') else [],
                    "description": "",
                })
            return datasets

        datasets = await _run_in_executor(_search)
        return {"status": "success", "datasets": datasets}
    except Exception as e:
        logger.exception("HuggingFace search error")
        raise HTTPException(status_code=500, detail=f"HuggingFace search failed: {str(e)}")
