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

from app.utils.zip_processor import extract_zip_with_mapping, scan_zip_tree
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

# --- Shared Processing Logic ---

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
                items = [{"label": t[1], "content": t[3], "filename": t[4], "split": "unassigned"} for t in file_data_list]
                data_manager.bulk_add_samples(dataset_id, task, items)
                total_processed += len(file_data_list)
                file_data_list.clear()

        if file_data_list:
            items = [{"label": t[1], "content": t[3], "filename": t[4], "split": "unassigned"} for t in file_data_list]
            data_manager.bulk_add_samples(dataset_id, task, items)
            total_processed += len(file_data_list)

    return total_processed

# --- Token Status ---

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

# --- Cancel Download ---

class CancelRequest(BaseModel):
    download_id: str

@router.post("/cancel")
async def cancel_download(req: CancelRequest):
    """Cancel an active download."""
    if req.download_id in _active_downloads:
        _active_downloads[req.download_id]["canceled"] = True
        return {"status": "success", "message": "Cancellation requested"}
    return {"status": "error", "message": "Download not found or already completed"}

# --- SSE Download Stream (URL / Kaggle / HuggingFace) ---

@router.get("/download_stream")
async def download_stream(
    source: str = Query(...),  # "url", "kaggle", "huggingface"
    dataset_id: Optional[str] = Query(None),
    task: str = Query("IMAGE_CLASSIFICATION"),
    url: Optional[str] = Query(None),
    dataset_ref: Optional[str] = Query(None),
    repo_id: Optional[str] = Query(None),
):
    """SSE endpoint that streams download progress to the frontend.

    If dataset_id is not provided, auto-creates a dataset using the source name
    and metadata. Returns dataset_id in the 'dataset_created' event.
    """
    download_id = str(uuid.uuid4())
    _active_downloads[download_id] = {"canceled": False}

    # Use a mutable container so the nested generator can reassign without
    # triggering Python's "UnboundLocalError" for the outer parameter.
    _state = {"dataset_id": dataset_id}

    async def event_generator():
        download_path = None
        download_dir = None
        try:
            # Auto-create dataset if dataset_id not provided
            if not _state["dataset_id"]:
                dataset_name, description = await _get_source_metadata(source, url, dataset_ref, repo_id)
                dataset_obj = await _run_in_executor(
                    data_manager.create_dataset, dataset_name, task, description
                )
                _state["dataset_id"] = dataset_obj["id"]
                yield _sse_event({"type": "dataset_created", "dataset_id": _state["dataset_id"], "dataset_name": dataset_name})
            else:
                # Verify dataset exists
                if not data_manager.get_dataset(_state["dataset_id"]):
                    yield _sse_event({"type": "error", "message": f"Dataset not found: {_state['dataset_id']}"})
                    return

            did = _state["dataset_id"]

            yield _sse_event({"type": "start", "download_id": download_id})

            # ------------------------------------------------------------------
            # URL — async generator, yields (downloaded, total) per chunk
            # ------------------------------------------------------------------
            if source == "url":
                if not url:
                    yield _sse_event({"type": "error", "message": "URL is required"})
                    return
                result: dict = {}
                async for downloaded, total in _download_from_url_stream(url, did, task, download_id, result):
                    yield _sse_event({"type": "progress", "downloaded": downloaded, "total": total})
                download_path = result.get("path")
                yield _sse_event({"type": "processing", "message": "Extracting archive..."})
                yield _sse_event({"type": "ready_to_map", "tree": result.get("tree"), "download_id": result.get("download_id"), "annotation_format": result.get("annotation_format"), "annotation_classes": result.get("annotation_classes", [])})
                yield _sse_event({"type": "complete", "count": 0})

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
                    dataset_ref, did, task, download_id, loop, queue
                )
                count = 0
                while True:
                    item = await queue.get()
                    if item is _SENTINEL:
                        break
                    if isinstance(item, tuple):
                        if item[0] == "ready_to_map":
                            ann_format = item[3] if len(item) > 3 else None
                            ann_classes = item[4] if len(item) > 4 else []
                            yield _sse_event({"type": "ready_to_map", "tree": item[1], "download_id": item[2], "annotation_format": ann_format, "annotation_classes": ann_classes})
                        elif item[0] == "complete":
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
                    repo_id, did, task, download_id, loop, queue
                )
                count = 0
                while True:
                    item = await queue.get()
                    if item is _SENTINEL:
                        break
                    if isinstance(item, tuple):
                        if item[0] == "ready_to_map":
                            ann_format = item[3] if len(item) > 3 else None
                            ann_classes = item[4] if len(item) > 4 else []
                            yield _sse_event({"type": "ready_to_map", "tree": item[1], "download_id": item[2], "annotation_format": ann_format, "annotation_classes": ann_classes})
                        elif item[0] == "complete":
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
            info = _active_downloads.get(download_id, {})
            if not info.get("zip_path"):
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

# --- Internal Download Implementations ---

_SENTINEL = object()  # signals that a download thread is finished

async def _get_source_metadata(
    source: str, url: Optional[str], dataset_ref: Optional[str], repo_id: Optional[str]
) -> tuple[str, str]:
    """Fetch dataset name and description from the source for auto-creation."""
    if source == "kaggle" and dataset_ref:
        try:
            def _fetch_kaggle_meta():
                username = os.environ.get("KAGGLE_USERNAME")
                key = os.environ.get("KAGGLE_KEY")
                if not username or not key:
                    return None, None
                os.environ["KAGGLE_USERNAME"] = username
                os.environ["KAGGLE_KEY"] = key
                try:
                    from kaggle.api.kaggle_api_extended import KaggleApi
                except ImportError:
                    return None, None
                api = KaggleApi()
                api.authenticate()
                import tempfile, json as _json
                with tempfile.TemporaryDirectory() as tmpdir:
                    json_path = api.dataset_metadata(dataset=dataset_ref, path=tmpdir)
                    meta = {}
                    if json_path and os.path.exists(str(json_path)):
                        with open(str(json_path), 'r', encoding='utf-8') as f:
                            raw = f.read()
                        try:
                            meta = _json.loads(_json.loads(raw))
                        except (_json.JSONDecodeError, TypeError):
                            try:
                                meta = _json.loads(raw)
                            except Exception:
                                meta = {}
                title = meta.get('title') or dataset_ref.split('/')[-1]
                desc_parts = [f"Source: Kaggle ({dataset_ref})"]
                subtitle = meta.get('subtitle')
                if subtitle:
                    desc_parts.append(str(subtitle))
                licenses = meta.get('licenses')
                if licenses and isinstance(licenses, list) and len(licenses) > 0:
                    license_name = licenses[0].get('name') if isinstance(licenses[0], dict) else str(licenses[0])
                    if license_name:
                        desc_parts.append(f"License: {license_name}")
                elif meta.get('licenseName'):
                    desc_parts.append(f"License: {meta['licenseName']}")
                total_downloads = meta.get('totalDownloads')
                if total_downloads:
                    desc_parts.append(f"Downloads: {total_downloads}")
                total_votes = meta.get('totalVotes')
                if total_votes:
                    desc_parts.append(f"Votes: {total_votes}")
                return title, " | ".join(desc_parts)
            name, desc = await _run_in_executor(_fetch_kaggle_meta)
            if name:
                return name, desc or f"Downloaded from Kaggle: {dataset_ref}"
        except Exception as e:
            logger.warning(f"Failed to fetch Kaggle metadata: {e}")
        # Fallback: use ref as name
        short_name = dataset_ref.split('/')[-1] if '/' in dataset_ref else dataset_ref
        return short_name, f"Downloaded from Kaggle: {dataset_ref}"

    elif source == "huggingface" and repo_id:
        try:
            def _fetch_hf_meta():
                token = os.environ.get("HUGGINGFACE_TOKEN")
                try:
                    from huggingface_hub import HfApi
                except ImportError:
                    return None, None
                api = HfApi(token=token if token else None)
                info = api.dataset_info(repo_id)
                title = info.id.split('/')[-1] if '/' in info.id else info.id
                desc_parts = [f"Source: HuggingFace ({info.id})"]
                if info.description:
                    desc_parts.append(info.description[:500])
                if info.tags:
                    desc_parts.append(f"Tags: {', '.join(info.tags[:5])}")
                return title, " | ".join(desc_parts)
            name, desc = await _run_in_executor(_fetch_hf_meta)
            if name:
                return name, desc or f"Downloaded from HuggingFace: {repo_id}"
        except Exception as e:
            logger.warning(f"Failed to fetch HuggingFace metadata: {e}")
        # Fallback: use repo_id as name
        short_name = repo_id.split('/')[-1] if '/' in repo_id else repo_id
        return short_name, f"Downloaded from HuggingFace: {repo_id}"

    elif source == "url" and url:
        # Extract filename from URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        if path_parts:
            name = path_parts[-1]
            # Remove .zip extension if present
            if name.lower().endswith('.zip'):
                name = name[:-4]
            # Clean up underscores and hyphens
            name = name.replace('_', ' ').replace('-', ' ')
            return name, f"Downloaded from URL: {url}"

    return "Unnamed Dataset", ""

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

    scan_result = await _run_in_executor(scan_zip_tree, download_path)
    _active_downloads[download_id]["zip_path"] = download_path
    result["tree"] = scan_result.get("tree", []) if isinstance(scan_result, dict) else scan_result
    result["annotation_format"] = scan_result.get("annotation_format") if isinstance(scan_result, dict) else None
    result["annotation_classes"] = scan_result.get("annotation_classes", []) if isinstance(scan_result, dict) else []
    result["download_id"] = download_id
    result["path"] = download_path

# ---------------------------------------------------------------------------
# Kaggle: sync thread worker with asyncio queue bridge
# ---------------------------------------------------------------------------
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
        # 1. FIX: Set credentials explicitly before touching the Kaggle library
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if not username or not key:
            raise ValueError("Kaggle credentials not configured in .env")

        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ValueError("kaggle package not installed. Run: pip install kaggle")

        # 2. FIX: Use KaggleApi instance to prevent global state issues
        api = KaggleApi()
        api.authenticate()

        total = 0
        try:
            import tempfile, json as _json
            with tempfile.TemporaryDirectory() as tmpdir:
                json_path = api.dataset_metadata(dataset=dataset_ref, path=tmpdir)
                if json_path and os.path.exists(str(json_path)):
                    with open(str(json_path), 'r', encoding='utf-8') as f:
                        raw = f.read()
                    try:
                        meta = _json.loads(_json.loads(raw))
                    except (_json.JSONDecodeError, TypeError):
                        try:
                            meta = _json.loads(raw)
                        except Exception:
                            meta = {}
                    total = meta.get("totalBytes", 0) or meta.get("totalDownloads", 0) or 0
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
            api.dataset_download_files(dataset_ref, path=download_dir, unzip=False, quiet=True)
        finally:
            stop_polling.set()
            poller.join(timeout=2)

        if _active_downloads.get(download_id, {}).get("canceled"):
            raise ValueError("Download canceled by user")

        zips = [f for f in os.listdir(download_dir) if f.endswith(".zip")]
        if not zips: raise ValueError("No zip file found in Kaggle download")
        zip_path = os.path.join(download_dir, zips[0])

        _active_downloads[download_id]["zip_path"] = zip_path
        scan_result = scan_zip_tree(zip_path)
        tree = scan_result.get("tree", []) if isinstance(scan_result, dict) else scan_result
        ann_format = scan_result.get("annotation_format") if isinstance(scan_result, dict) else None
        ann_classes = scan_result.get("annotation_classes", []) if isinstance(scan_result, dict) else []
        _push(("ready_to_map", tree, download_id, ann_format, ann_classes))
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
    import zipfile
    import shutil

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
            snapshot_download(
                repo_id=repo_id,
                local_dir=download_dir,
                repo_type="dataset",
                token=os.environ.get("HUGGINGFACE_TOKEN", None),
                local_dir_use_symlinks=False
            )
        finally:
            stop_polling.set()
            poller.join(timeout=2)

        if _active_downloads.get(download_id, {}).get("canceled"):
            raise ValueError("Download canceled by user")

        # 1. FIX: HF downloads raw files. We need to zip them for `zip_processor.py`.
        zip_path = os.path.join(DOWNLOAD_DIR, f"{download_id}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(download_dir):
                # Ignore .git and .cache folders that huggingface_hub creates
                if ".git" in root or ".cache" in root:
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, download_dir)
                    zf.write(file_path, arcname)

        # 2. FIX: Clean up the raw HuggingFace directory to save space
        shutil.rmtree(download_dir, ignore_errors=True)

        _active_downloads[download_id]["zip_path"] = zip_path
        scan_result = scan_zip_tree(zip_path)
        tree = scan_result.get("tree", []) if isinstance(scan_result, dict) else scan_result
        ann_format = scan_result.get("annotation_format") if isinstance(scan_result, dict) else None
        ann_classes = scan_result.get("annotation_classes", []) if isinstance(scan_result, dict) else []
        _push(("ready_to_map", tree, download_id, ann_format, ann_classes))
    except Exception as exc:
        _push(("error", str(exc)))
    finally:
        _push(_SENTINEL)

# --- Kaggle Search ---

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
            except ImportError as ie:
                raise ValueError(f"kaggle package not installed: {ie}. Run: pip install kaggle")

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
                    size = getattr(ds, 'total_bytes', 0) or 0
                    last_updated = getattr(ds, 'last_updated', '')
                    if hasattr(last_updated, 'isoformat'):
                        last_updated = last_updated.isoformat()
                    else:
                        last_updated = str(last_updated)
                    download_count = getattr(ds, 'download_count', 0) or 0
                    description = getattr(ds, 'subtitle', '') or getattr(ds, 'description', '') or ''
                    vote_count = getattr(ds, 'vote_count', 0) or 0
                    usability_rating = getattr(ds, 'usability_rating', 0) or 0
                    license_name = getattr(ds, 'license_name', '') or ''
                    tags_raw = getattr(ds, 'tags', []) or []
                    tags = []
                    if hasattr(tags_raw, '__iter__') and not isinstance(tags_raw, str):
                        for t in tags_raw[:5]:
                            if isinstance(t, dict):
                                tags.append(t.get('name', t.get('ref', str(t))))
                            else:
                                tags.append(getattr(t, 'name', str(t)))
                    datasets.append({
                        "ref": ref,
                        "title": title,
                        "size": size,
                        "last_updated": last_updated,
                        "download_count": download_count,
                        "description": description,
                        "vote_count": vote_count,
                        "usability_rating": round(float(usability_rating), 2) if usability_rating else 0,
                        "license": license_name,
                        "tags": tags,
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

# --- HuggingFace Search ---

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
                    likes = getattr(ds, 'likes', 0) or 0
                    last_modified = str(getattr(ds, 'last_modified', '')) if getattr(ds, 'last_modified', None) else ''
                    tags = getattr(ds, 'tags', []) or []
                    pipeline_tag = getattr(ds, 'pipeline_tag', '') or ''
                    created_at = str(getattr(ds, 'created_at', '')) if getattr(ds, 'created_at', None) else ''
                    datasets.append({
                        "id": ds_id,
                        "author": author or '',
                        "title": title,
                        "downloads": downloads,
                        "likes": likes,
                        "last_modified": last_modified,
                        "created_at": created_at,
                        "tags": tags[:8] if isinstance(tags, (list, tuple)) else [],
                        "pipeline_tag": pipeline_tag,
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

# --- Download Progress Polling (alternative to SSE for simple clients) ---

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
        "canceled": info.get("canceled", False),
    }

class ProcessRemoteRequest(BaseModel):
    download_id: str
    dataset_id: str
    task: str
    mapping: list

@router.post("/process")
async def process_remote_zip(req: ProcessRemoteRequest):
    info = _active_downloads.get(req.download_id)
    if not info or not info.get("zip_path"):
        raise HTTPException(status_code=404, detail="Download session missing or expired")

    try:
        result = await _run_in_executor(
            extract_zip_with_mapping,
            info["zip_path"], req.dataset_id, req.task, req.mapping
        )
        # Clean up
        base_dir = os.path.dirname(info["zip_path"])
        if "edgecraft_remote_downloads" in base_dir:
            shutil.rmtree(base_dir, ignore_errors=True)
        _active_downloads.pop(req.download_id, None)

        count = result["processed"] if isinstance(result, dict) else result
        return {
            "status": "success",
            "count": count,
            "annotation_format": result.get("annotation_format") if isinstance(result, dict) else None,
            "annotations_matched": result.get("annotations_matched", 0) if isinstance(result, dict) else 0,
            "annotation_classes": result.get("annotation_classes", []) if isinstance(result, dict) else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
