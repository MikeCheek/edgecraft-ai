import os
import uuid
import shutil
import tempfile
import zipfile
import asyncio
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

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

# Active downloads tracking (for cancellation)
_active_downloads: dict[str, dict] = {}


async def _run_in_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXECUTOR, fn, *args)


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


# ─── Shared Processing Logic ─────────────────────────────────────────────────

def _process_zip_to_dataset(zip_path: str, dataset_id: str, task: str, download_id: str = None) -> int:
    """Extract a zip file and ingest samples into the data_manager.
    Uses the SAME bulk_add_samples signature as datasets.py: list of tuples.
    """
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
                # Match datasets.py signature: tuple (dataset_id, label, task, content, filename)
                file_data_list.append((dataset_id, label, task, content, filename))

            if len(file_data_list) >= ZIP_PROCESSING_BATCH_SIZE:
                data_manager.bulk_add_samples(file_data_list)
                total_processed += len(file_data_list)
                file_data_list.clear()

        if file_data_list:
            data_manager.bulk_add_samples(file_data_list)
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


# ─── Cancel Download ──────────────────────────────────────────────────────────

class CancelRequest(BaseModel):
    download_id: str

@router.post("/cancel")
async def cancel_download(req: CancelRequest):
    """Cancel an active download."""
    if req.download_id in _active_downloads:
        _active_downloads[req.download_id]["cancelled"] = True
        return {"status": "success", "message": "Cancellation requested"}
    return {"status": "error", "message": "Download not found or already completed"}


# ─── SSE Download Stream (URL / Kaggle / HuggingFace) ─────────────────────────

@router.get("/download_stream")
async def download_stream(
    source: str = Query(...),  # "url", "kaggle", "huggingface"
    dataset_id: str = Query(...),
    task: str = Query(...),
    url: Optional[str] = Query(None),
    dataset_ref: Optional[str] = Query(None),
    repo_id: Optional[str] = Query(None),
):
    """SSE endpoint that streams download progress to the frontend."""
    download_id = str(uuid.uuid4())
    _active_downloads[download_id] = {"canceled": False}

    async def event_generator():
        download_path = None
        download_dir = None
        try:
            yield _sse_event({"type": "start", "download_id": download_id})

            # ------------------------------------------------------------------
            # URL — async generator, yields (downloaded, total) per chunk
            # ------------------------------------------------------------------
            if source == "url":
                if not url:
                    yield _sse_event({"type": "error", "message": "URL is required"})
                    return
                result: dict = {}
                async for downloaded, total in _download_from_url_stream(url, dataset_id, task, download_id, result):
                    yield _sse_event({"type": "progress", "downloaded": downloaded, "total": total})
                download_path = result.get("path")
                yield _sse_event({"type": "processing", "message": "Extracting archive..."})
                yield _sse_event({"type": "complete", "count": result.get("count", 0)})

            # ------------------------------------------------------------------
            # Kaggle — queue bridge
            # ------------------------------------------------------------------
            elif source == "kaggle":
                if not dataset_ref:
                    yield _sse_event({"type": "error", "message": "dataset_ref is required"})
                    return
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()
                fut = loop.run_in_executor(
                    _EXECUTOR, _kaggle_download_thread,
                    dataset_ref, dataset_id, task, download_id, loop, queue
                )
                count = 0
                while True:
                    item = await queue.get()
                    if item is _SENTINEL:
                        break
                    if isinstance(item, tuple):
                        if item[0] == "complete":
                            _, count, download_dir = item
                            yield _sse_event({"type": "processing", "message": "Extracting archive..."})
                        elif item[0] == "error":
                            raise ValueError(item[1])
                        else:
                            downloaded, total = item
                            yield _sse_event({"type": "progress", "downloaded": downloaded, "total": total})
                await fut
                yield _sse_event({"type": "complete", "count": count})

            # ------------------------------------------------------------------
            # HuggingFace — queue bridge
            # ------------------------------------------------------------------
            elif source == "huggingface":
                if not repo_id:
                    yield _sse_event({"type": "error", "message": "repo_id is required"})
                    return
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()
                fut = loop.run_in_executor(
                    _EXECUTOR, _huggingface_download_thread,
                    repo_id, dataset_id, task, download_id, loop, queue
                )
                count = 0
                while True:
                    item = await queue.get()
                    if item is _SENTINEL:
                        break
                    if isinstance(item, tuple):
                        if item[0] == "complete":
                            _, count, download_dir = item
                            yield _sse_event({"type": "processing", "message": "Extracting archive..."})
                        elif item[0] == "error":
                            raise ValueError(item[1])
                        else:
                            downloaded, total = item
                            yield _sse_event({"type": "progress", "downloaded": downloaded, "total": total})
                await fut
                yield _sse_event({"type": "complete", "count": count})

            else:
                yield _sse_event({"type": "error", "message": f"Unknown source: {source}"})

        except ValueError as e:
            if "canceled" in str(e).lower():
                yield _sse_event({"type": "canceled", "message": "Download canceled"})
            else:
                yield _sse_event({"type": "error", "message": str(e)})
        except Exception as e:
            logger.exception(f"Download stream error ({source})")
            yield _sse_event({"type": "error", "message": str(e)})
        finally:
            _active_downloads.pop(download_id, None)
            if download_path and os.path.exists(download_path):
                os.remove(download_path)
            if download_dir and os.path.exists(download_dir):
                shutil.rmtree(download_dir, ignore_errors=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ─── Internal Download Implementations ────────────────────────────────────────

## Active downloads tracking (for cancellation)
_active_downloads: dict[str, dict] = {}
_SENTINEL = object()  # signals that a download thread is finished

async def _run_in_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXECUTOR, fn, *args)

def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"

# ---------------------------------------------------------------------------
# URL: async generator — yields (downloaded, total) per chunk
# ---------------------------------------------------------------------------
async def _download_from_url_stream(
    url: str, dataset_id: str, task: str, download_id: str, result: dict
):
    """
    Async generator. Yields (downloaded_bytes, total_bytes) per chunk.
    Stores final results in `result`: result["path"] and result["count"].
    """
    import httpx

    download_path = os.path.join(DOWNLOAD_DIR, f"{download_id}.zip")

    async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}: {response.reason_phrase} — check URL is a direct download link")

            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(download_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 256):
                    if _active_downloads.get(download_id, {}).get("canceled"):
                        raise ValueError("Download canceled by user")
                    f.write(chunk)
                    downloaded += len(chunk)
                    _active_downloads[download_id]["downloaded"] = downloaded
                    _active_downloads[download_id]["total"] = total
                    yield downloaded, total  # <-- this is what was missing

    if not zipfile.is_zipfile(download_path):
        raise ValueError("Downloaded file is not a valid ZIP archive.")

    count = await _run_in_executor(_process_zip_to_dataset, download_path, dataset_id, task, download_id)
    result["path"] = download_path
    result["count"] = count

# ---------------------------------------------------------------------------
# Kaggle: sync thread worker with asyncio queue bridge
# ---------------------------------------------------------------------------
def _kaggle_download_thread(
    dataset_ref: str, dataset_id: str, task: str, download_id: str,
    loop: asyncio.AbstractEventLoop, queue: asyncio.Queue
):
    """Runs in executor. Pushes (downloaded, total) progress + sentinel into queue."""
    import threading

    def _push(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    try:
        import kaggle  # type: ignore
        kaggle.api.authenticate()

        try:
            meta = kaggle.api.dataset_metadata(dataset_ref)
            total = getattr(meta, "totalBytes", 0) or 0
        except Exception:
            total = 0

        download_dir = os.path.join(DOWNLOAD_DIR, download_id)
        os.makedirs(download_dir, exist_ok=True)

        stop_polling = threading.Event()

        def _poll():
            while not stop_polling.is_set():
                try:
                    sz = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, fns in os.walk(download_dir)
                        for f in fns
                    )
                    _push((sz, total))
                except Exception:
                    pass
                stop_polling.wait(timeout=1.0)

        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()
        try:
            kaggle.api.dataset_download_files(dataset_ref, path=download_dir, unzip=False, quiet=True)
        finally:
            stop_polling.set()
            poller.join(timeout=2)

        if _active_downloads.get(download_id, {}).get("canceled"):
            raise ValueError("Download canceled by user")

        count = _process_zip_to_dataset(download_dir, dataset_id, task, download_id)
        _push(("complete", count, download_dir))
    except Exception as exc:
        _push(("error", str(exc)))
    finally:
        _push(_SENTINEL)

# ---------------------------------------------------------------------------
# HuggingFace: sync thread worker with asyncio queue bridge
# ---------------------------------------------------------------------------
def _huggingface_download_thread(
    repo_id: str, dataset_id: str, task: str, download_id: str,
    loop: asyncio.AbstractEventLoop, queue: asyncio.Queue
):
    """Runs in executor. Pushes (downloaded, 0) progress + sentinel into queue."""
    import threading

    def _push(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    try:
        from huggingface_hub import snapshot_download  # type: ignore

        download_dir = os.path.join(DOWNLOAD_DIR, download_id)
        os.makedirs(download_dir, exist_ok=True)

        stop_polling = threading.Event()

        def _poll():
            while not stop_polling.is_set():
                try:
                    sz = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, fns in os.walk(download_dir)
                        for f in fns
                    )
                    _push((sz, 0))  # total unknown until complete
                except Exception:
                    pass
                stop_polling.wait(timeout=1.0)

        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()
        try:
            snapshot_download(repo_id=repo_id, local_dir=download_dir, repo_type="dataset")
        finally:
            stop_polling.set()
            poller.join(timeout=2)

        if _active_downloads.get(download_id, {}).get("canceled"):
            raise ValueError("Download canceled by user")

        count = _process_zip_to_dataset(download_dir, dataset_id, task, download_id)
        _push(("complete", count, download_dir))
    except Exception as exc:
        _push(("error", str(exc)))
    finally:
        _push(_SENTINEL)


async def _download_from_kaggle(dataset_ref: str, dataset_id: str, task: str, download_id: str) -> tuple:
    """Download from Kaggle. Returns (dir_path, count)."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        raise ValueError("Kaggle credentials not configured in .env")

    download_dir = os.path.join(DOWNLOAD_DIR, f"kaggle_{download_id}")
    os.makedirs(download_dir, exist_ok=True)

    def _download():
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ValueError("kaggle package not installed. Run: pip install kaggle")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_ref, path=download_dir, unzip=False)
        zips = [f for f in os.listdir(download_dir) if f.endswith(".zip")]
        if not zips:
            raise ValueError("No zip file found after Kaggle download")
        return os.path.join(download_dir, zips[0])

    if _active_downloads.get(download_id, {}).get("cancelled"):
        raise ValueError("Download cancelled by user")

    zip_path = await _run_in_executor(_download)

    if _active_downloads.get(download_id, {}).get("cancelled"):
        raise ValueError("Download cancelled by user")

    count = await _run_in_executor(_process_zip_to_dataset, zip_path, dataset_id, task, download_id)
    return (download_dir, count)


async def _download_from_huggingface(repo_id: str, dataset_id: str, task: str, download_id: str) -> tuple:
    """Download from HuggingFace. Returns (dir_path, count)."""
    token = os.environ.get("HUGGINGFACE_TOKEN")
    download_dir = os.path.join(DOWNLOAD_DIR, f"hf_{download_id}")
    os.makedirs(download_dir, exist_ok=True)

    def _download():
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ValueError("huggingface_hub package not installed. Run: pip install huggingface_hub")

        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=download_dir,
            token=token if token else None,
        )

        # Look for zip files first
        zip_files = []
        for root, dirs, files in os.walk(local_path):
            for f in files:
                if f.endswith(".zip"):
                    zip_files.append(os.path.join(root, f))

        if zip_files:
            return zip_files[0]

        # No zip — assemble one from valid files
        temp_zip_path = os.path.join(download_dir, "assembled.zip")
        found_files = []
        for root, dirs, files in os.walk(local_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in VALID_EXTENSIONS:
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, local_path)
                    found_files.append((full_path, rel_path))

        if not found_files:
            raise ValueError("No valid image/audio files found in the HuggingFace dataset")

        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for full_path, rel_path in found_files:
                zf.write(full_path, rel_path)

        return temp_zip_path

    if _active_downloads.get(download_id, {}).get("cancelled"):
        raise ValueError("Download cancelled by user")

    zip_path = await _run_in_executor(_download)

    if _active_downloads.get(download_id, {}).get("cancelled"):
        raise ValueError("Download cancelled by user")

    count = await _run_in_executor(_process_zip_to_dataset, zip_path, dataset_id, task, download_id)
    return (download_dir, count)


# ─── Kaggle Search ────────────────────────────────────────────────────────────

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
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError as ie:                raise ValueError(f"kaggle package not installed: {ie}. Run: pip install kaggle")

            api = KaggleApi()
            api.authenticate()

            try:
                results = api.dataset_list(search=query, page=page)
            except TypeError:
                # Fallback for older kaggle API versions
                results = api.dataset_list(search=query)

            datasets = []
            for ds in results[:page_size]:
                try:
                    ref = str(ds.ref) if hasattr(ds, 'ref') else str(ds)
                    title = getattr(ds, 'title', ref)
                    size = getattr(ds, 'totalBytes', 0) or 0
                    last_updated = str(getattr(ds, 'lastUpdated', ''))
                    download_count = getattr(ds, 'downloadCount', 0) or 0
                    description = getattr(ds, 'subtitle', '') or ''
                    datasets.append({
                        "ref": ref,
                        "title": title,
                        "size": size,
                        "last_updated": last_updated,
                        "download_count": download_count,
                        "description": description,
                    })
                except Exception as attr_err:
                    logger.warning(f"Skipping Kaggle result due to attribute error: {attr_err}")
                    continue
            return datasets

        datasets = await _run_in_executor(_search)
        return {"status": "success", "datasets": datasets}
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.exception("Kaggle search error")
        raise HTTPException(status_code=500, detail=f"Kaggle search failed: {type(e).__name__}: {str(e)}")


# ─── HuggingFace Search ───────────────────────────────────────────────────────

@router.get("/huggingface/search")
async def search_huggingface_datasets(query: str = Query(..., min_length=1), limit: int = 20):
    """Search HuggingFace datasets by keyword."""
    token = os.environ.get("HUGGINGFACE_TOKEN")

    try:
        def _search():
            try:
                from huggingface_hub import HfApi
            except ImportError as ie:
                raise ValueError(f"huggingface_hub package not installed: {ie}. Run: pip install huggingface_hub")

            api = HfApi(token=token if token else None)

            try:
                results = list(api.list_datasets(search=query, limit=limit, sort="downloads", direction=-1))
            except TypeError:
                # Fallback for different API versions
                results = list(api.list_datasets(search=query, limit=limit))

            datasets = []
            for ds in results:
                try:
                    ds_id = getattr(ds, 'id', str(ds))
                    author = getattr(ds, 'author', '')
                    if not author and '/' in ds_id:
                        author = ds_id.split('/')[0]
                    title = ds_id.split('/')[-1] if '/' in ds_id else ds_id
                    downloads = getattr(ds, 'downloads', 0) or 0
                    last_modified = str(getattr(ds, 'last_modified', '')) if getattr(ds, 'last_modified', None) else ''
                    tags = getattr(ds, 'tags', []) or []
                    datasets.append({
                        "id": ds_id,
                        "author": author or '',
                        "title": title,
                        "downloads": downloads,
                        "last_modified": last_modified,
                        "tags": tags[:5],
                        "description": "",
                    })
                except Exception as attr_err:
                    logger.warning(f"Skipping HF result due to attribute error: {attr_err}")
                    continue
            return datasets

        datasets = await _run_in_executor(_search)
        return {"status": "success", "datasets": datasets}
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.exception("HuggingFace search error")
        raise HTTPException(status_code=500, detail=f"HuggingFace search failed: {type(e).__name__}: {str(e)}")


# ─── Download Progress Polling (alternative to SSE for simple clients) ────────

@router.get("/progress/{download_id}")
async def get_download_progress(download_id: str):
    """Poll download progress for a given download_id."""
    info = _active_downloads.get(download_id)
    if not info:
        return {"status": "not_found"}
    return {
        "status": "active",
        "downloaded": info.get("downloaded", 0),
        "total": info.get("total", 0),
        "cancelled": info.get("cancelled", False),
    }