"""
Loads a dataset split into numpy arrays ready for a model, using the shared
preprocessing in app.services.preprocessing. Used by training, INT8
calibration, fine-tuning during optimization, and test-set evaluation.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.services import preprocessing
from app.services.detection import boxes_to_grid, grid_shape

logger = logging.getLogger(__name__)


def _dm():
    from app.services.shared_state import data_manager

    return data_manager


def detection_classes(dataset_id: str) -> List[str]:
    """Object classes for a detection dataset: the annotation class list the
    importer recorded, else every class name seen on a bounding box."""
    dm = _dm()
    meta = (dm.get_dataset(dataset_id) or {}).get("metadata", {}) or {}
    if meta.get("annotation_classes"):
        return list(meta["annotation_classes"])
    names = set()
    for s in dm.get_samples(dataset_id):
        for a in s.get("annotations") or []:
            names.add(a.get("class_name", "object"))
    return sorted(names)


def classification_labels(dataset_id: str) -> Tuple[List[str], List[str]]:
    """(labels with at least one training sample, labels dropped because
    they have none). Registered-but-empty labels would otherwise become
    output classes the model can never learn."""
    dm = _dm()
    all_labels = dm.get_dataset_labels(dataset_id)
    with_train = {s["label"] for s in dm.get_samples(dataset_id) if s.get("split") == "train"}
    kept = [l for l in all_labels if l in with_train]
    dropped = [l for l in all_labels if l not in with_train]
    return kept, dropped


def split_samples(dataset_id: str, split: str, max_samples: Optional[int] = None) -> List[dict]:
    subset = [s for s in _dm().get_samples(dataset_id) if s.get("split") == split]
    subset.sort(key=lambda s: s["id"])
    if max_samples and len(subset) > max_samples:
        # Evenly spaced rather than the first N so every class is represented.
        idx = np.linspace(0, len(subset) - 1, max_samples).astype(int)
        subset = [subset[i] for i in idx]
    return subset


def load_split(
    dataset_id: str,
    task: str,
    input_shape: Sequence[int],
    labels: List[str],
    split: str,
    max_samples: Optional[int] = None,
    on_skip=None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Returns (X, y, meta).

    Classification: y = int class indices into `labels`; samples whose label
    isn't in `labels` are skipped. Detection: y = (gh, gw) FOMO target grids
    built from each sample's bounding boxes, `labels` = object classes.
    """
    dm = _dm()
    is_od = task == "OBJECT_DETECTION"
    label_to_idx = {name: i for i, name in enumerate(labels)}
    grid = grid_shape(input_shape) if is_od else None

    X: List[np.ndarray] = []
    y: List = []
    meta: List[Dict] = []
    for sample in split_samples(dataset_id, split, max_samples):
        if not is_od and sample.get("label") not in label_to_idx:
            continue
        raw = dm.get_sample_data(sample["id"])
        if not raw:
            continue
        try:
            arr = preprocessing.preprocess(raw, task, input_shape)
        except Exception as exc:
            msg = f"Skipping unreadable sample {sample.get('filename')}: {exc}"
            logger.warning(msg)
            if on_skip:
                on_skip(msg)
            continue
        X.append(arr)
        if is_od:
            y.append(boxes_to_grid(sample.get("annotations"), labels, grid))
        else:
            y.append(label_to_idx[sample["label"]])
        meta.append({"id": sample["id"], "filename": sample.get("filename"), "label": sample.get("label")})

    if not X:
        shape = (0, *[int(d) for d in input_shape])
        y_shape = (0, *grid) if is_od else (0,)
        return np.zeros(shape, np.float32), np.zeros(y_shape, np.int32), []
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int32), meta


def evaluation_split(dataset_id: str) -> str:
    """'test' if the dataset has any test samples, else 'val'."""
    return "test" if split_samples(dataset_id, "test", 1) else "val"
