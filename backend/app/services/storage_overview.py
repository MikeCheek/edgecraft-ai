"""
storage_overview.py
--------------------
Aggregates real, on-disk storage usage across the whole backend: per-dataset
sample counts/sizes/file-type breakdowns, trained model sizes, optimized
(.tflite) output sizes, and overall disk usage - so the "Storage" panel in
the UI reflects what's actually on disk, not estimates.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict


def _human_bytes(n: int) -> str:
    """Small formatting helper for API responses that include a human-
    readable string alongside the raw byte count."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def get_storage_overview() -> Dict[str, Any]:
    from app.services.shared_state import data_manager, trainer
    from app.services.optimizer import OPTIMIZATION_DIR

    storage_dir = data_manager.storage_dir

    # ---- Per-dataset breakdown -------------------------------------------
    datasets_info = []
    total_dataset_bytes = 0
    total_sample_count = 0
    global_file_types: Dict[str, Dict[str, int]] = {}

    for ds in data_manager.get_datasets():
        ds_id = ds["id"]
        samples = data_manager.get_samples(ds_id)
        ds_bytes = 0
        ds_file_types: Dict[str, Dict[str, int]] = {}
        split_counts = {"train": 0, "val": 0, "test": 0, "unassigned": 0}

        for s in samples:
            sample_path = os.path.join(storage_dir, f"{s['id']}.bin")
            size = os.path.getsize(sample_path) if os.path.exists(sample_path) else 0
            ds_bytes += size

            ext = (os.path.splitext(s.get("filename") or "")[1] or "(no extension)").lower()
            bucket = ds_file_types.setdefault(ext, {"count": 0, "bytes": 0})
            bucket["count"] += 1
            bucket["bytes"] += size

            g_bucket = global_file_types.setdefault(ext, {"count": 0, "bytes": 0})
            g_bucket["count"] += 1
            g_bucket["bytes"] += size

            split = s.get("split", "unassigned")
            split_counts[split] = split_counts.get(split, 0) + 1

        datasets_info.append({
            "id": ds_id,
            "name": ds.get("name"),
            "task": ds.get("task"),
            "created_at": ds.get("created_at"),
            "sample_count": len(samples),
            "size_bytes": ds_bytes,
            "size_human": _human_bytes(ds_bytes),
            "avg_sample_bytes": round(ds_bytes / len(samples), 1) if samples else 0,
            "file_types": ds_file_types,
            "split_counts": split_counts,
        })
        total_dataset_bytes += ds_bytes
        total_sample_count += len(samples)

    datasets_info.sort(key=lambda d: d["size_bytes"], reverse=True)

    # ---- Trained models -----------------------------------------------------
    models_info = []
    total_model_bytes = 0
    for m in trainer.get_trained_models():
        size = m.get("size_bytes", 0) or 0
        total_model_bytes += size
        models_info.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "task": m.get("task"),
            "dataset_id": m.get("dataset_id"),
            "dataset_name": m.get("dataset_name"),
            "size_bytes": size,
            "size_human": _human_bytes(size),
        })
    models_info.sort(key=lambda m: m["size_bytes"], reverse=True)

    # ---- Optimized (.tflite) outputs ----------------------------------------
    opt_dir_bytes = 0
    opt_file_count = 0
    if OPTIMIZATION_DIR.exists():
        for p in OPTIMIZATION_DIR.rglob("*"):
            if p.is_file():
                try:
                    opt_dir_bytes += p.stat().st_size
                    if p.suffix == ".tflite":
                        opt_file_count += 1
                except OSError:
                    continue

    # ---- Training session bookkeeping (JSON DBs are tiny but worth noting) --
    all_sessions = trainer.get_all_sessions(include_archived=True)
    active_sessions = [s for s in all_sessions if not s.get("archived")]
    archived_sessions = [s for s in all_sessions if s.get("archived")]

    # ---- Total on-disk usage of the storage directory (catch-all) ----------
    total_storage_bytes = 0
    if os.path.exists(storage_dir):
        for root, _dirs, files in os.walk(storage_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    total_storage_bytes += os.path.getsize(fpath)
                except OSError:
                    continue

    disk_free_bytes = None
    disk_total_bytes = None
    try:
        usage = shutil.disk_usage(storage_dir if os.path.exists(storage_dir) else ".")
        disk_free_bytes = usage.free
        disk_total_bytes = usage.total
    except OSError:
        pass

    return {
        "storage_dir": str(Path(storage_dir).resolve()),
        "total_storage_bytes": total_storage_bytes,
        "total_storage_human": _human_bytes(total_storage_bytes),
        "disk_free_bytes": disk_free_bytes,
        "disk_free_human": _human_bytes(disk_free_bytes) if disk_free_bytes is not None else None,
        "disk_total_bytes": disk_total_bytes,
        "datasets": {
            "count": len(datasets_info),
            "total_samples": total_sample_count,
            "total_bytes": total_dataset_bytes,
            "total_human": _human_bytes(total_dataset_bytes),
            "file_types": global_file_types,
            "items": datasets_info,
        },
        "trained_models": {
            "count": len(models_info),
            "total_bytes": total_model_bytes,
            "total_human": _human_bytes(total_model_bytes),
            "items": models_info,
        },
        "optimized_models": {
            "count": opt_file_count,
            "total_bytes": opt_dir_bytes,
            "total_human": _human_bytes(opt_dir_bytes),
            "directory": str(OPTIMIZATION_DIR),
        },
        "training_sessions": {
            "total": len(all_sessions),
            "active": len(active_sessions),
            "archived": len(archived_sessions),
        },
    }
