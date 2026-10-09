"""
Dataset quality report: the problems that silently hurt TinyML models and are
cheap to detect before training.

* class balance (imbalance ratio, classes too small to learn from)
* split coverage (classes missing from train / val / test)
* exact duplicates, and duplicates shared across splits (test-set leakage
  that inflates every reported accuracy)
* unreadable / empty files
* object detection: images without boxes, boxes outside the image
"""

import hashlib
import io
from collections import Counter, defaultdict
from typing import Dict, List

from app.services import preprocessing

MIN_SAMPLES_PER_CLASS = 20
IMBALANCE_WARN_RATIO = 5.0
MAX_LISTED = 25


def _dm():
    from app.services.shared_state import data_manager

    return data_manager


def _readable(raw: bytes, task: str) -> bool:
    try:
        if preprocessing.is_audio_task(task):
            import soundfile as sf

            sf.info(io.BytesIO(raw))
        else:
            from PIL import Image

            with Image.open(io.BytesIO(raw)) as img:
                img.verify()
        return True
    except Exception:
        # Compressed audio soundfile can't parse may still decode via ffmpeg.
        if preprocessing.is_audio_task(task):
            try:
                preprocessing.decode_audio(raw)
                return True
            except Exception:
                return False
        return False


def analyze_dataset(dataset_id: str) -> Dict:
    dm = _dm()
    dataset = dm.get_dataset(dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")
    task = dataset.get("task", "IMAGE_CLASSIFICATION")
    samples = dm.get_samples(dataset_id)

    by_label = Counter(s["label"] for s in samples)
    by_label_split: Dict[str, Counter] = defaultdict(Counter)
    for s in samples:
        by_label_split[s["label"]][s.get("split", "unassigned")] += 1

    hashes: Dict[str, List[dict]] = defaultdict(list)
    unreadable: List[dict] = []
    for s in samples:
        raw = dm.get_sample_data(s["id"])
        if not raw:
            unreadable.append({"id": s["id"], "filename": s.get("filename"), "reason": "missing or empty file"})
            continue
        hashes[hashlib.sha1(raw).hexdigest()].append(s)
        if not _readable(raw, task):
            unreadable.append({"id": s["id"], "filename": s.get("filename"), "reason": "cannot be decoded"})

    duplicate_groups = []
    leakage_groups = []
    for group in hashes.values():
        if len(group) < 2:
            continue
        entry = {
            "samples": [{"id": s["id"], "filename": s.get("filename"), "label": s["label"],
                         "split": s.get("split", "unassigned")} for s in group],
            "conflicting_labels": len({s["label"] for s in group}) > 1,
        }
        duplicate_groups.append(entry)
        if len({s.get("split") for s in group if s.get("split") in ("train", "val", "test")}) > 1:
            leakage_groups.append(entry)

    counts = [c for c in by_label.values() if c > 0]
    imbalance = round(max(counts) / min(counts), 2) if counts else None

    issues: List[Dict] = []

    def issue(severity, code, message, **extra):
        issues.append({"severity": severity, "code": code, "message": message, **extra})

    if len(by_label) < 2 and task != "OBJECT_DETECTION":
        issue("error", "too_few_classes", "A classifier needs at least two classes.")
    small = sorted(l for l, c in by_label.items() if c < MIN_SAMPLES_PER_CLASS)
    if small and task != "OBJECT_DETECTION":
        issue("warning", "small_classes",
              f"{len(small)} class(es) have fewer than {MIN_SAMPLES_PER_CLASS} samples: {', '.join(small[:10])}",
              labels=small)
    if imbalance and imbalance >= IMBALANCE_WARN_RATIO and task != "OBJECT_DETECTION":
        issue("warning", "imbalance",
              f"Largest class is {imbalance}x the smallest. Enable 'Balance classes' when training "
              "or collect more samples for the small classes.", ratio=imbalance)
    if task != "OBJECT_DETECTION":
        missing_train = sorted(l for l, sp in by_label_split.items() if sp["train"] == 0)
        if missing_train:
            issue("error", "missing_train", f"No training samples for: {', '.join(missing_train)}", labels=missing_train)
        missing_eval = sorted(l for l, sp in by_label_split.items() if sp["val"] + sp["test"] == 0 and sp["train"] > 0)
        if missing_eval:
            issue("warning", "missing_eval",
                  f"Not evaluated (no val/test samples): {', '.join(missing_eval)}", labels=missing_eval)
    unassigned = sum(sp["unassigned"] for sp in by_label_split.values())
    if unassigned:
        issue("warning", "unassigned", f"{unassigned} sample(s) are not assigned to a split.")
    if leakage_groups:
        issue("error", "leakage",
              f"{len(leakage_groups)} identical file(s) appear in more than one split - evaluation "
              "results are inflated. Remove the duplicates.")
    conflicting = [g for g in duplicate_groups if g["conflicting_labels"]]
    if conflicting:
        issue("error", "conflicting_labels", f"{len(conflicting)} identical file(s) carry different labels.")
    elif duplicate_groups:
        issue("info", "duplicates", f"{len(duplicate_groups)} group(s) of exact duplicate files.")
    if unreadable:
        issue("error", "unreadable", f"{len(unreadable)} file(s) can't be decoded and will be skipped.")

    detection = None
    if task == "OBJECT_DETECTION":
        without = [s for s in samples if not s.get("annotations")]
        bad_boxes = 0
        for s in samples:
            for a in s.get("annotations") or []:
                if not (0 <= a.get("cx", -1) <= 1 and 0 <= a.get("cy", -1) <= 1):
                    bad_boxes += 1
        box_classes = Counter(a.get("class_name") for s in samples for a in (s.get("annotations") or []))
        detection = {"images_without_boxes": len(without), "boxes_out_of_range": bad_boxes,
                     "boxes_per_class": dict(box_classes)}
        if len(without) == len(samples):
            issue("error", "no_boxes", "No image has bounding boxes - the detector has nothing to learn.")
        elif without:
            issue("info", "background_images",
                  f"{len(without)} image(s) have no boxes (used as background examples).")
        if bad_boxes:
            issue("warning", "boxes_out_of_range", f"{bad_boxes} box centre(s) lie outside the image.")

    order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: order[i["severity"]])
    errors = sum(i["severity"] == "error" for i in issues)
    warnings = sum(i["severity"] == "warning" for i in issues)
    return {
        "dataset_id": dataset_id,
        "task": task,
        "total_samples": len(samples),
        "class_counts": dict(by_label),
        "split_counts": {l: dict(sp) for l, sp in by_label_split.items()},
        "imbalance_ratio": imbalance,
        "duplicate_groups": duplicate_groups[:MAX_LISTED],
        "duplicate_group_count": len(duplicate_groups),
        "leakage_group_count": len(leakage_groups),
        "unreadable": unreadable[:MAX_LISTED],
        "unreadable_count": len(unreadable),
        "detection": detection,
        "issues": issues,
        "score": max(0, 100 - 25 * errors - 8 * warnings),
    }


def remove_duplicates(dataset_id: str) -> int:
    """Keep one copy of each identical file (preferring the train split, so a
    leaked test copy is the one removed). Groups whose copies disagree on the
    label are left alone - a human has to decide which label is right."""
    dm = _dm()
    hashes: Dict[str, List[dict]] = defaultdict(list)
    for s in dm.get_samples(dataset_id):
        raw = dm.get_sample_data(s["id"])
        if raw:
            hashes[hashlib.sha1(raw).hexdigest()].append(s)
    rank = {"train": 0, "val": 1, "test": 2, "unassigned": 3}
    removed = 0
    for group in hashes.values():
        if len(group) < 2 or len({s["label"] for s in group}) > 1:
            continue
        group.sort(key=lambda s: rank.get(s.get("split", "unassigned"), 4))
        for s in group[1:]:
            if dm.delete_sample(s["id"], save_metadata=False):
                removed += 1
    if removed:
        dm._save_metadata()
    return removed
