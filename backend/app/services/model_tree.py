"""
model_tree.py
-------------
Aggregates Dataset -> Trained Model -> Optimized Variant into a single
tree structure, so the frontend can show (and let the user navigate/select
from) the full lineage of everything trained and optimized, instead of
several disconnected flat dropdowns.
"""

from __future__ import annotations

from typing import Any, Dict, List


def get_model_tree(include_archived: bool = False) -> Dict[str, Any]:
    from app.services.shared_state import data_manager, trainer
    from app.services.optimizer import list_optimization_sessions

    all_models = trainer.get_trained_models(include_archived=True)
    all_optimizations = list_optimization_sessions()

    # Group optimizations by the training_id of the model they were built from.
    opts_by_training_id: Dict[str, List[dict]] = {}
    for opt in all_optimizations:
        tid = opt.get("training_id")
        if not tid:
            continue
        opts_by_training_id.setdefault(tid, []).append({
            "id": opt.get("id"),
            "method": opt.get("frontend_method", opt.get("method")),
            "status": opt.get("status"),
            "error": opt.get("error"),
            "created_at": opt.get("created_at"),
            "completed_at": opt.get("completed_at"),
            "original_size_bytes": opt.get("original_size_bytes", 0),
            "optimized_size_bytes": opt.get("optimized_size_bytes", 0),
            "compression_ratio": opt.get("compression_ratio", 0.0),
            "comparison_summary": (
                {"deltas": opt["comparison"]["deltas"]}
                if isinstance(opt.get("comparison"), dict) and "deltas" in opt["comparison"]
                else None
            ),
        })

    # Group models by dataset_id.
    models_by_dataset: Dict[str, List[dict]] = {}
    unassigned_models: List[dict] = []

    for m in all_models:
        training_id = m.get("training_id")
        session = trainer.training_sessions.get(training_id, {})
        is_archived = session.get("archived", False)
        if is_archived and not include_archived:
            continue

        entry = {
            "id": m.get("id"),
            "training_id": training_id,
            "name": m.get("name"),
            "task": m.get("task"),
            "base_model": session.get("base_model"),
            "device_used": m.get("device_used"),
            "accuracy": m.get("accuracy", 0.0),
            "val_accuracy": m.get("val_accuracy", 0.0),
            "loss": m.get("loss", 0.0),
            "val_loss": m.get("val_loss", 0.0),
            "size_bytes": m.get("size_bytes", 0),
            "created_at": session.get("created_at"),
            "archived": is_archived,
            "optimizations": sorted(
                opts_by_training_id.get(training_id, []),
                key=lambda o: o.get("created_at") or 0,
                reverse=True,
            ),
        }

        dataset_id = m.get("dataset_id")
        if dataset_id:
            models_by_dataset.setdefault(dataset_id, []).append(entry)
        else:
            unassigned_models.append(entry)

    datasets_out = []
    for ds in data_manager.get_datasets():
        ds_models = models_by_dataset.pop(ds["id"], [])
        ds_models.sort(key=lambda m: m.get("created_at") or 0, reverse=True)
        datasets_out.append({
            "id": ds["id"],
            "name": ds["name"],
            "task": ds["task"],
            "sample_count": ds.get("sample_count", 0),
            "models": ds_models,
        })

    # Any leftover groups reference a dataset_id that no longer exists
    # (dataset was deleted after training) - still surface those models
    # rather than silently hiding them.
    for dataset_id, models in models_by_dataset.items():
        models.sort(key=lambda m: m.get("created_at") or 0, reverse=True)
        datasets_out.append({
            "id": dataset_id,
            "name": "(deleted dataset)",
            "task": models[0]["task"] if models else None,
            "sample_count": 0,
            "models": models,
        })

    datasets_out.sort(key=lambda d: len(d["models"]), reverse=True)

    return {
        "datasets": datasets_out,
        "unassigned_models": sorted(unassigned_models, key=lambda m: m.get("created_at") or 0, reverse=True),
    }
