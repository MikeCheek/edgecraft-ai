"""
FOMO-style centroid detection helpers: build training targets from
bounding boxes, decode a predicted grid into detections, and score
predictions against ground truth (object-level precision / recall / F1).
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

FOMO_STRIDE = 8
DEFAULT_THRESHOLD = 0.5


def grid_shape(input_shape: Sequence[int]) -> Tuple[int, int]:
    return int(input_shape[0]) // FOMO_STRIDE, int(input_shape[1]) // FOMO_STRIDE


def _ann_fields(ann) -> Tuple[float, float, str]:
    if isinstance(ann, dict):
        return float(ann.get("cx", 0)), float(ann.get("cy", 0)), str(ann.get("class_name", "object"))
    return float(getattr(ann, "cx", 0)), float(getattr(ann, "cy", 0)), str(getattr(ann, "class_name", "object"))


def boxes_to_grid(annotations, classes: List[str], grid: Tuple[int, int]) -> np.ndarray:
    """(gh, gw) int grid: 0 = background, k = class index k-1 has a centre here."""
    gh, gw = grid
    target = np.zeros((gh, gw), dtype=np.int32)
    class_map = {c: i + 1 for i, c in enumerate(classes)}
    for ann in annotations or []:
        cx, cy, name = _ann_fields(ann)
        if name not in class_map:
            continue
        col = min(gw - 1, max(0, int(cx * gw)))
        row = min(gh - 1, max(0, int(cy * gh)))
        target[row, col] = class_map[name]
    return target


def grid_to_centroids(target: np.ndarray) -> List[Tuple[int, int, int]]:
    """Ground-truth grid -> [(row, col, class_idx)]."""
    rows, cols = np.nonzero(target)
    return [(int(r), int(c), int(target[r, c]) - 1) for r, c in zip(rows, cols)]


def decode_predictions(probs: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> List[Dict]:
    """Predicted (gh, gw, K+1) softmax grid -> merged detections.

    Adjacent cells of the same class are merged into one object (connected
    components), reported at the probability-weighted centroid in
    normalised [0, 1] image coordinates.
    """
    from scipy import ndimage

    gh, gw, k1 = probs.shape
    detections: List[Dict] = []
    for cls in range(1, k1):
        mask = (probs.argmax(axis=-1) == cls) & (probs[..., cls] >= threshold)
        if not mask.any():
            continue
        labels, n = ndimage.label(mask)
        for blob in range(1, n + 1):
            rr, cc = np.nonzero(labels == blob)
            weights = probs[rr, cc, cls]
            r = float(np.average(rr, weights=weights))
            c = float(np.average(cc, weights=weights))
            detections.append({
                "class_index": cls - 1,
                "row": r,
                "col": c,
                "x": (c + 0.5) / gw,
                "y": (r + 0.5) / gh,
                "confidence": float(weights.max()),
                "cells": int(len(rr)),
            })
    return detections


def score_detections(
    gt_grids: Sequence[np.ndarray],
    pred_probs: Sequence[np.ndarray],
    num_classes: int,
    threshold: float = DEFAULT_THRESHOLD,
    tolerance_cells: float = 1.5,
) -> Dict:
    """Object-level precision/recall/F1, overall and per class.

    A prediction is a true positive if a not-yet-matched ground-truth
    centroid of the same class lies within `tolerance_cells` grid cells.
    """
    tp = np.zeros(num_classes, dtype=int)
    fp = np.zeros(num_classes, dtype=int)
    fn = np.zeros(num_classes, dtype=int)

    for gt, probs in zip(gt_grids, pred_probs):
        gts = grid_to_centroids(gt)
        matched = [False] * len(gts)
        for det in sorted(decode_predictions(probs, threshold), key=lambda d: -d["confidence"]):
            best, best_d = -1, None
            for i, (r, c, k) in enumerate(gts):
                if matched[i] or k != det["class_index"]:
                    continue
                d = max(abs(r - det["row"]), abs(c - det["col"]))
                if d <= tolerance_cells and (best_d is None or d < best_d):
                    best, best_d = i, d
            if best >= 0:
                matched[best] = True
                tp[det["class_index"]] += 1
            else:
                fp[det["class_index"]] += 1
        for i, (_, _, k) in enumerate(gts):
            if not matched[i]:
                fn[k] += 1

    def _prf(t, p, n):
        precision = t / (t + p) if (t + p) else 0.0
        recall = t / (t + n) if (t + n) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)

    P, R, F = _prf(int(tp.sum()), int(fp.sum()), int(fn.sum()))
    per_class = []
    for k in range(num_classes):
        p, r, f = _prf(int(tp[k]), int(fp[k]), int(fn[k]))
        per_class.append({"precision": p, "recall": r, "f1": f,
                          "tp": int(tp[k]), "fp": int(fp[k]), "fn": int(fn[k])})
    return {"precision": P, "recall": R, "f1": F, "per_class": per_class,
            "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())}
