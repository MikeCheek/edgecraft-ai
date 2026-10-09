"""
Post-training review.

Two layers:

1. `analyze_training` - deterministic. Reads everything known about a run
   (full metric curves, phases, held-out evaluation, latest validation
   confusion matrix, dataset size / balance / quality report, model size vs
   the target board, earlier runs on the same dataset) and produces facts,
   findings with evidence, an explainable 0-100 score and concrete,
   validated recommendations. Same run -> same result, no network needed.

2. `review_training` - optionally asks an LLM to turn those facts into a
   prioritised, explained plan. The LLM only sees curated facts (with units
   and caveats), its suggested parameter changes are validated against what
   the trainer accepts and against the run's current values, and if it
   fails the deterministic recommendations are returned instead of an error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app import config
from app.services import preprocessing

logger = logging.getLogger(__name__)

REVIEW_VERSION = 2

AUGMENTATION_KEYS = {
    "horizontal_flip": bool, "vertical_flip": bool, "random_crop": bool,
    "random_rotation": (0.0, 0.5), "random_zoom": (0.0, 0.5), "random_translation": (0.0, 0.5),
    "random_brightness": (0.0, 0.5), "random_contrast": (0.0, 0.5),
}
STRONG_AUGMENTATION = {"horizontal_flip": True, "random_rotation": 0.1, "random_crop": True,
                       "random_translation": 0.1, "random_brightness": 0.2, "random_contrast": 0.2}

PARAM_DOC = {
    "base_model": "architecture (see allowed_models)",
    "input_shape": "[H, W, C] for images (C = 1 or 3); fixed for audio",
    "epochs": "int 1-1000",
    "batch_size": "int 1-1024",
    "learning_rate": "float 1e-6..0.1 (Adam; the fine-tune phase uses learning_rate x 0.1)",
    "dropout_rate": "float 0..0.9, before the classifier",
    "l2_reg": "float 0..0.1 weight decay on the head",
    "early_stopping": "bool; restores the best weights",
    "early_stopping_patience": "int 1-100 epochs",
    "early_stopping_monitor": "'val_loss' or 'val_accuracy'",
    "freeze_encoder_epochs": "int: epochs training only the head on a frozen pretrained backbone before fine-tuning",
    "trainable_layers": "int: 0 = fine-tune the whole backbone, N = only its last N layers",
    "class_weighting": "bool: weight the loss by inverse class frequency",
    "augmentation": "image tasks only, dict of " + ", ".join(AUGMENTATION_KEYS),
}

PRETRAINED = {"MobileNetV2", "MobileNetV1_0.25", "MobileNetV3Small", "EfficientNet", "ResNet50V2", "FOMO_MobileNetV2"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(v, d=4):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), d)


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _downsample(ms: List[dict], n: int = 24) -> List[dict]:
    """Up to n epochs, always keeping the first, last and best ones."""
    if len(ms) <= n:
        idx = list(range(len(ms)))
    else:
        idx = set(np.linspace(0, len(ms) - 1, n).round().astype(int).tolist())
        idx.add(int(np.argmax([m.get("val_accuracy", 0) for m in ms])))
        losses = [m.get("val_loss") or float("inf") for m in ms]
        idx.add(int(np.argmin(losses)))
        idx = sorted(idx)
    keys = ("epoch", "phase", "loss", "val_loss", "accuracy", "val_accuracy", "val_f1", "learning_rate")
    return [{k: (_r(ms[i][k]) if isinstance(ms[i].get(k), float) else ms[i].get(k)) for k in keys if k in ms[i]}
            for i in idx]


def _top_confusions(cm: Optional[List[List[int]]], labels: List[str], k: int = 4) -> List[dict]:
    if not cm or not labels or len(cm) != len(labels):
        return []
    pairs = [(cm[i][j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels))
             if i != j and cm[i][j] > 0]
    pairs.sort(reverse=True)
    return [{"true": t, "predicted": p, "count": int(c)} for c, t, p in pairs[:k]]


def _add_cm(a, b):
    if not a:
        return b
    if not b or len(a) != len(b):
        return a
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

def _dataset_facts(session: dict, labels: List[str]) -> Dict[str, Any]:
    from app.services.shared_state import data_manager as dm

    ds_id = session.get("dataset_id")
    ds = dm.get_dataset(ds_id) or {} if ds_id else {}
    info = session.get("run_info") or {}
    counts = info.get("train_class_counts")
    split = {}
    if ds_id and ds:
        try:
            split = dm.get_split_summary(ds_id)
        except Exception:
            split = {}
        if not counts and session.get("task") != "OBJECT_DETECTION":
            counts = {}
            for s in dm.get_samples(ds_id):
                if s.get("split") == "train":
                    counts[s["label"]] = counts.get(s["label"], 0) + 1
    facts: Dict[str, Any] = {
        "name": ds.get("name"),
        "description": (ds.get("description") or "")[:300] or None,
        "split_counts": split or None,
        "train_samples_per_class": counts or None,
    }
    if counts:
        vals = [v for v in counts.values() if v > 0]
        if vals:
            facts["min_per_class"] = min(vals)
            facts["median_per_class"] = int(np.median(vals))
            facts["imbalance_ratio"] = round(max(vals) / min(vals), 2)
            facts["smallest_classes"] = sorted(counts, key=counts.get)[:3]
    if ds_id and ds and session.get("task") in preprocessing.IMAGE_TASKS:
        try:
            st = dm.get_dataset_image_stats(ds_id)
            facts["source_images"] = {k: st.get(k) for k in ("width", "height", "aspect_ratio", "most_common_resolutions")
                                      if st.get(k) is not None} or None
        except Exception:
            pass
    if ds_id and ds and len(dm.samples_by_dataset.get(ds_id, ())) <= 5000:
        try:
            from app.services.dataset_quality import analyze_dataset

            q = analyze_dataset(ds_id)
            facts["quality_score"] = q["score"]
            facts["quality_issues"] = [f"[{i['severity']}] {i['message']}" for i in q["issues"]][:8]
        except Exception as e:  # never block the review on the quality scan
            logger.info(f"Dataset quality scan skipped: {e}")
    return facts


def _deploy_facts(session: dict, params: Optional[int], board: Optional[str]) -> Dict[str, Any]:
    from app.services.mcu_advisor import MCUAdvisor

    facts: Dict[str, Any] = {"params": params}
    if params:
        facts["est_int8_model_kb"] = round(params * 1.05 / 1024 + 20, 1)  # weights + flatbuffer overhead
        facts["est_float32_model_kb"] = round(params * 4 / 1024, 1)
    spec = MCUAdvisor.BOARD_SPECS.get(board or "", {})
    if spec:
        facts["board"] = {"id": board, "name": spec.get("name"), "flash_kb": spec.get("flash_kb"),
                          "ram_kb": spec.get("ram_kb"), "internal_sram_kb": spec.get("internal_sram_kb")}
        if params and spec.get("flash_kb"):
            facts["int8_flash_usage_pct"] = round(facts["est_int8_model_kb"] / spec["flash_kb"] * 100, 1)
    return facts


def _curve_facts(ms: List[dict], session: dict) -> Dict[str, Any]:
    n = len(ms)
    final = ms[-1]
    val_acc = np.array([m.get("val_accuracy", 0.0) for m in ms], dtype=float)
    val_loss = np.array([m.get("val_loss") or np.nan for m in ms], dtype=float)
    train_loss = np.array([m.get("loss", np.nan) for m in ms], dtype=float)
    has_val = bool(np.nanmax(val_loss) > 0) if np.isfinite(val_loss).any() else False

    bi = int(np.argmax(val_acc))
    li = int(np.nanargmin(val_loss)) if has_val else None
    f: Dict[str, Any] = {
        "epochs_planned": session.get("total_epochs"),
        "epochs_run": n,
        "final": {k: _r(final.get(k)) for k in ("accuracy", "val_accuracy", "loss", "val_loss", "val_f1",
                                                "val_precision", "val_recall", "val_confidence", "val_ece")
                  if final.get(k) is not None},
        "best_val_accuracy": {"value": _r(val_acc[bi]), "epoch": ms[bi]["epoch"]},
        "lost_after_best_val_accuracy_pts": _r((val_acc[bi] - val_acc[-1]) * 100, 1),
        "train_val_gap_pts": _r((final.get("accuracy", 0) - final.get("val_accuracy", 0)) * 100, 1),
        "train_val_gap_at_best_pts": _r((ms[bi].get("accuracy", 0) - val_acc[bi]) * 100, 1),
    }
    if li is not None:
        f["best_val_loss"] = {"value": _r(val_loss[li]), "epoch": ms[li]["epoch"],
                              "val_accuracy_then": _r(val_acc[li])}
        f["val_loss_rise_since_best_pct"] = _r((val_loss[-1] / val_loss[li] - 1) * 100, 1) if val_loss[li] > 0 else None
        f["train_loss_drop_since_best_val_loss_pct"] = (
            _r((1 - train_loss[-1] / train_loss[li]) * 100, 1) if train_loss[li] > 0 else None)
    if n >= 5:
        tail = val_acc[-5:]
        f["val_accuracy_slope_last5_pts_per_epoch"] = _r(np.polyfit(np.arange(5), tail, 1)[0] * 100, 2)
        f["train_loss_slope_last5_per_epoch"] = _r(np.polyfit(np.arange(5), train_loss[-5:], 1)[0], 4)
    phases = [m.get("phase") for m in ms]
    if "head" in phases and "fine-tune" in phases:
        h_end = max(i for i, p in enumerate(phases) if p == "head")
        ft = [i for i, p in enumerate(phases) if p == "fine-tune"]
        ft_best = max(ft, key=lambda i: val_acc[i])
        f["fine_tuning"] = {
            "started_epoch": ms[ft[0]]["epoch"],
            "val_accuracy_end_of_head_phase": _r(val_acc[h_end]),
            "best_val_accuracy_during_fine_tune": _r(val_acc[ft_best]),
            "train_accuracy_end_of_head_phase": _r(ms[h_end].get("accuracy")),
            "train_accuracy_final": _r(final.get("accuracy")),
            "fine_tune_gain_pts": _r((val_acc[ft_best] - val_acc[h_end]) * 100, 1),
        }
    lrs = sorted({_r(m.get("learning_rate"), 8) for m in ms if m.get("learning_rate")})
    if lrs:
        f["learning_rates_used"] = lrs
    ratios = [m.get("update_ratio") for m in ms[-3:] if m.get("update_ratio")]
    if ratios:
        f["weight_update_ratio_last"] = _r(ratios[-1], 6)
    f["curve"] = _downsample(ms)
    f["has_validation"] = has_val
    return f


def _past_runs(session: dict) -> List[dict]:
    from app.services.shared_state import trainer

    out = []
    for s in trainer.get_all_sessions(include_archived=True):
        if s.get("id") == session.get("id") or s.get("dataset_id") != session.get("dataset_id"):
            continue
        if s.get("status") != "completed" or not s.get("metrics"):
            continue
        ms = s["metrics"]
        ev = s.get("evaluation") or {}
        out.append({
            "name": s.get("name") or None,
            "created_at": s.get("created_at"),
            "base_model": s.get("base_model"), "input_shape": s.get("input_shape"),
            "epochs_run": len(ms), "batch_size": s.get("batch_size"), "learning_rate": s.get("learning_rate"),
            "dropout_rate": s.get("dropout_rate"), "l2_reg": s.get("l2_reg"),
            "augmentation": {k: v for k, v in (s.get("augmentation") or {}).items() if v} or None,
            "early_stopping": s.get("early_stopping"), "class_weighting": s.get("class_weighting"),
            "freeze_encoder_epochs": s.get("freeze_encoder_epochs"),
            "final_val_accuracy": _r(ms[-1].get("val_accuracy")),
            "best_val_accuracy": _r(max(m.get("val_accuracy", 0) for m in ms)),
            "test_accuracy": _r(ev.get("f1") if s.get("task") == "OBJECT_DETECTION" else ev.get("accuracy")),
        })
    out.sort(key=lambda r: r.get("created_at") or 0)
    return out[-8:]


def current_config(session: dict) -> Dict[str, Any]:
    return {
        "base_model": session.get("base_model"),
        "input_shape": session.get("input_shape"),
        "epochs": session.get("total_epochs") or session.get("epochs"),
        "batch_size": session.get("batch_size"),
        "learning_rate": session.get("learning_rate"),
        "dropout_rate": session.get("dropout_rate", 0.3),
        "l2_reg": session.get("l2_reg", 0.0),
        "early_stopping": bool(session.get("early_stopping")),
        "early_stopping_patience": session.get("early_stopping_patience", 5),
        "early_stopping_monitor": session.get("early_stopping_monitor", "val_loss"),
        "freeze_encoder_epochs": session.get("freeze_encoder_epochs", 0),
        "trainable_layers": session.get("trainable_layers", 0),
        "class_weighting": bool(session.get("class_weighting")),
        "augmentation": {k: v for k, v in (session.get("augmentation") or {}).items() if k in AUGMENTATION_KEYS},
    }


# ---------------------------------------------------------------------------
# Change validation (shared by rule and LLM suggestions)
# ---------------------------------------------------------------------------

def allowed_models(task: str) -> List[str]:
    from app.services.model_factory import AUDIO_MODELS, IMAGE_BACKBONES, OD_MODELS

    if preprocessing.is_audio_task(task):
        return list(AUDIO_MODELS)
    if task == "OBJECT_DETECTION":
        return list(OD_MODELS)
    return list(IMAGE_BACKBONES)


def validate_changes(task: str, changes: Any, current: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Keep only changes the trainer accepts and that differ from the run's
    current value. Returns (changes, rejected-notes)."""
    from app.services.model_factory import ModelFactory

    if not isinstance(changes, dict):
        return {}, []
    out: Dict[str, Any] = {}
    notes: List[str] = []

    def num(key, v, lo, hi, cast):
        try:
            x = cast(v)
        except (TypeError, ValueError):
            notes.append(f"{key}={v!r} is not a number")
            return None
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return min(hi, max(lo, x))

    def as_bool(v):
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    for key, v in changes.items():
        val: Any = None
        if key == "base_model":
            name = ModelFactory.resolve_name(str(v))
            if name in allowed_models(task):
                val = name
            else:
                notes.append(f"model {v!r} is not available for {task}")
        elif key == "input_shape":
            if preprocessing.is_audio_task(task):
                notes.append("audio input shape is fixed by the MFCC front-end")
                continue
            try:
                shape = [int(x) for x in v]
                if len(shape) != 3 or shape[2] not in (1, 3) or not all(16 <= d <= 320 for d in shape[:2]):
                    raise ValueError
                if task == "OBJECT_DETECTION":
                    shape = [d - d % 8 for d in shape[:2]] + [shape[2]]
                val = shape
            except (TypeError, ValueError):
                notes.append(f"input_shape {v!r} is invalid")
        elif key == "epochs":
            val = num(key, v, 1, 1000, int)
        elif key == "batch_size":
            val = num(key, v, 1, 1024, int)
        elif key == "learning_rate":
            val = num(key, v, 1e-6, 0.1, float)
        elif key == "dropout_rate":
            val = num(key, v, 0.0, 0.9, float)
        elif key == "l2_reg":
            val = num(key, v, 0.0, 0.1, float)
        elif key == "early_stopping_patience":
            val = num(key, v, 1, 100, int)
        elif key in ("freeze_encoder_epochs", "trainable_layers"):
            val = num(key, v, 0, 1000, int)
        elif key in ("early_stopping", "class_weighting"):
            val = as_bool(v)
        elif key == "early_stopping_monitor":
            val = v if v in ("val_loss", "val_accuracy") else None
        elif key == "augmentation":
            if task not in preprocessing.IMAGE_TASKS or task == "OBJECT_DETECTION" or not isinstance(v, dict):
                notes.append("augmentation only applies to image classification")
                continue
            aug = {}
            for ak, av in v.items():
                kind = AUGMENTATION_KEYS.get(ak)
                if kind is bool:
                    aug[ak] = as_bool(av)
                elif kind:
                    x = num(ak, av, kind[0], kind[1], float)
                    if x is not None:
                        aug[ak] = round(x, 3)
            cur_aug = current.get("augmentation", {})
            aug = {k: x for k, x in aug.items() if cur_aug.get(k, False if AUGMENTATION_KEYS[k] is bool else 0) != x}
            if aug:
                val = aug
        else:
            notes.append(f"unknown parameter {key!r}")
            continue
        if val is None:
            continue
        cur = current.get(key)
        same = (isinstance(val, float) and isinstance(cur, (int, float)) and math.isclose(val, cur, rel_tol=1e-6)) \
            or val == cur
        if key == "augmentation" or not same:
            out[key] = val
    return out, notes


# ---------------------------------------------------------------------------
# Findings, score, rule-based recommendations
# ---------------------------------------------------------------------------

def _findings_and_recs(task: str, facts: dict, cur: dict) -> Tuple[List[dict], List[dict]]:
    findings: List[dict] = []
    recs: List[dict] = []
    c = facts["training"]
    data = facts["dataset"]
    ev = facts.get("held_out") or {}
    deploy = facts["deployment"]
    final = c["final"]
    metric = "F1" if task == "OBJECT_DETECTION" else "accuracy"

    def finding(severity, title, evidence):
        findings.append({"severity": severity, "title": title, "evidence": evidence})

    def rec(priority, title, reasoning, changes=None, expected="", category="training"):
        valid, _ = validate_changes(task, changes or {}, cur)
        if changes and not valid:
            return  # every proposed change is already in effect
        recs.append({"title": title, "reasoning": reasoning, "changes": valid,
                     "expected_effect": expected, "priority": priority, "category": category, "source": "rules"})

    gap = c.get("train_val_gap_pts") or 0.0
    rise = c.get("val_loss_rise_since_best_pct")
    bvl = c.get("best_val_loss")
    lost = c.get("lost_after_best_val_accuracy_pts") or 0.0
    best = c["best_val_accuracy"]
    n_classes = facts["run"]["num_classes"] or 2
    min_pc = data.get("min_per_class")
    overfit = gap >= 15 or (rise is not None and rise >= 25 and (c.get("train_loss_drop_since_best_val_loss_pct") or 0) > 20)

    # --- Generalisation ---------------------------------------------------
    if overfit:
        ev_txt = [f"train {metric} {_pct(final.get('accuracy'))} vs val {_pct(final.get('val_accuracy'))} ({gap:+.1f} pts)"]
        if bvl and rise is not None:
            ev_txt.append(f"val loss lowest {bvl['value']} at epoch {bvl['epoch']}, then +{rise:.0f}% by the end "
                          f"while train loss kept falling")
        finding("critical" if gap >= 30 else "warning", "The model is overfitting", "; ".join(ev_txt))
        aug_changes = {}
        if task in preprocessing.IMAGE_TASKS and task != "OBJECT_DETECTION":
            aug_changes["augmentation"] = {k: v for k, v in STRONG_AUGMENTATION.items()
                                           if not cur["augmentation"].get(k)}
        reg = {"dropout_rate": min(0.5, round(max(cur["dropout_rate"], 0.3) + 0.2, 2))}
        if cur["l2_reg"] < 1e-4:
            reg["l2_reg"] = 1e-4
        rec("high", "Regularise harder: augmentation, dropout and weight decay",
            f"A {gap:.0f}-point train/val gap means the network memorises the {facts['run']['num_train']} training "
            f"images. Augmentation shows it new variations of each image every epoch; dropout and L2 limit how much "
            f"the classifier can fit noise.",
            {**aug_changes, **reg}, "Smaller train/val gap; typically several points of val accuracy on small datasets.")
    elif gap <= -5:
        finding("info", "Validation scores above training scores",
                f"val {metric} {_pct(final.get('val_accuracy'))} vs train {_pct(final.get('accuracy'))}: expected "
                "while augmentation/dropout are active, or a sign the val split is easier than train.")

    # --- Wasted epochs / early stopping ------------------------------------
    if lost >= 3 and c["epochs_run"] - best["epoch"] >= 3:
        finding("warning", "The final model is worse than the best epoch",
                f"val {metric} peaked at {_pct(best['value'])} at epoch {best['epoch']} but ended at "
                f"{_pct(final.get('val_accuracy'))} (-{lost:.1f} pts)"
                + (" - early stopping is off, so the saved model is the last epoch." if not cur["early_stopping"] else "."))
        if not cur["early_stopping"]:
            # Monitor the metric that actually peaked late: when val loss bottoms
            # out long before val accuracy does, a val_loss monitor would stop
            # (and restore) too early.
            monitor = "val_accuracy" if bvl and best["epoch"] - bvl["epoch"] >= 5 else "val_loss"
            rec("high", "Enable early stopping (restores the best weights)",
                f"The best validation result was at epoch {best['epoch']} of {c['epochs_run']}; everything after it "
                f"made the saved model worse. Early stopping on {monitor} keeps the best epoch automatically."
                + (f" Val loss bottomed out earlier (epoch {bvl['epoch']}), so monitor accuracy."
                   if monitor == "val_accuracy" else ""),
                {"early_stopping": True, "early_stopping_monitor": monitor, "early_stopping_patience": 10,
                 "epochs": max(cur["epochs"] or 0, best["epoch"] + 15)},
                f"Keeps the ~{_pct(best['value'])} model instead of {_pct(final.get('val_accuracy'))}.")

    # --- Under-fitting / still improving -----------------------------------
    slope = c.get("val_accuracy_slope_last5_pts_per_epoch")
    if (final.get("accuracy") or 0) < 0.75 and gap < 10:
        finding("warning", "The model is under-fitting",
                f"train {metric} is only {_pct(final.get('accuracy'))} - the network can't fit the training data.")
        rec("medium", "Give the model more capacity or time",
            "Low training accuracy with no overfitting means the model or the schedule is too weak.",
            {"epochs": int((cur["epochs"] or 30) * 1.5), "learning_rate": min(0.01, (cur["learning_rate"] or 0.001) * 2)},
            "Higher train and val accuracy.")
    if slope is not None and slope > 0.5 and not overfit and c["epochs_run"] >= (cur["epochs"] or 0):
        finding("info", "Still improving when training ended",
                f"val {metric} rose {slope:.2f} pts/epoch over the last 5 epochs.")
        rec("medium", "Train for longer", f"Validation {metric} was still rising at the end of the run.",
            {"epochs": int((cur["epochs"] or 30) * 1.5), "early_stopping": True, "early_stopping_patience": 8},
            "A few more points before the curve flattens.")

    # --- Fine-tuning ---------------------------------------------------------
    ft = c.get("fine_tuning")
    if ft:
        gain = ft.get("fine_tune_gain_pts") or 0.0
        sev = "warning" if gain < 1 and overfit else "info"
        finding(sev, "Effect of backbone fine-tuning",
                f"val {metric} {_pct(ft['val_accuracy_end_of_head_phase'])} at the end of the frozen phase -> best "
                f"{_pct(ft['best_val_accuracy_during_fine_tune'])} while fine-tuning ({gain:+.1f} pts); train "
                f"{metric} went {_pct(ft['train_accuracy_end_of_head_phase'])} -> {_pct(ft['train_accuracy_final'])}.")
        if overfit and gain < 3 and cur["trainable_layers"] in (0, None):
            rec("medium", "Fine-tune only the top of the backbone",
                "Unfreezing the whole pretrained backbone on a small dataset mostly increased memorisation "
                f"({gain:+.1f} pts on val). Unfreezing only the last layers keeps the general ImageNet features.",
                {"trainable_layers": 20}, "Less overfitting during the fine-tune phase.")

    # --- Data ----------------------------------------------------------------
    if min_pc is not None and task != "OBJECT_DETECTION":
        if min_pc < 50:
            finding("critical" if min_pc < 25 else "warning", "Too few training samples per class",
                    f"{data.get('min_per_class')}-{max((data.get('train_samples_per_class') or {}).values() or [0])} "
                    f"training samples per class over {n_classes} classes (median {data.get('median_per_class')}); "
                    f"smallest: {', '.join(data.get('smallest_classes') or [])}.")
            rec("high", "Collect more data for the weakest classes",
                f"With ~{data.get('median_per_class')} images per class a pretrained model can't learn {n_classes} "
                "fine distinctions reliably. Aim for 100+ varied images per class (different lighting, backgrounds, "
                "angles), starting with the classes listed in the findings.",
                None, "The single most effective fix for both overfitting and low accuracy.", "data")
        if (data.get("imbalance_ratio") or 1) >= 1.5:
            finding("info" if data["imbalance_ratio"] < 3 else "warning", "Class imbalance",
                    f"largest/smallest class ratio {data['imbalance_ratio']}x in the training split.")
            if not cur["class_weighting"]:
                rec("medium", "Enable class weighting",
                    f"Classes are imbalanced ({data['imbalance_ratio']}x), so the loss under-weights the small ones.",
                    {"class_weighting": True}, "Better recall on the smallest classes.")
    for issue in (data.get("quality_issues") or [])[:4]:
        if issue.startswith("[error]") or "leak" in issue.lower():
            finding("warning", "Dataset quality problem", issue)

    # --- Per-class / confusions -------------------------------------------
    weak = [w for w in (facts.get("weakest_classes") or []) if w["f1"] < 0.9]
    if weak:
        finding("info", "Weakest classes",
                ", ".join(f"{w['label']} (F1 {_pct(w['f1'])})" for w in weak[:3]))
    conf = facts.get("top_confusions") or []
    if conf:
        finding("info", "Most frequent confusions",
                ", ".join(f"{x['true']} -> {x['predicted']} ({x['count']}x)" for x in conf[:3])
                + " (validation + test).")
        if cur.get("input_shape") and task in preprocessing.IMAGE_TASKS and cur["input_shape"][0] < 128 \
                and (deploy.get("int8_flash_usage_pct") or 0) < 30:
            shape = cur["input_shape"]
            rec("low", "Try a higher input resolution",
                "The confused classes may differ in fine detail that is lost at "
                f"{shape[0]}x{shape[1]}; the board has room for a larger input.",
                {"input_shape": [min(160, shape[0] + 32), min(160, shape[1] + 32), shape[2]]},
                "Better separation of visually similar classes, at the cost of latency/RAM.")

    # --- Held-out reliability ----------------------------------------------
    if ev.get("num_samples"):
        n = ev["num_samples"]
        acc = ev.get("score")
        if acc is not None:
            ci = 1.96 * math.sqrt(max(acc * (1 - acc), 0.01) / n) * 100
            if n < 100:
                finding("info", "Test result is noisy",
                        f"{ev['split']} {metric} {_pct(acc)} on only {n} samples: about ±{ci:.0f} pts (95% interval). "
                        "Compare runs on the same split and prefer trends over single numbers.")
            gap_vt = (acc - (final.get("val_accuracy") or 0)) * 100
            if abs(gap_vt) >= 10 and n >= 20:
                finding("info", "Validation and test disagree",
                        f"{ev['split']} {metric} {_pct(acc)} vs final val {_pct(final.get('val_accuracy'))} "
                        f"({gap_vt:+.0f} pts) - both splits are small, or they differ in content.")

    # --- Calibration -------------------------------------------------------
    ece = final.get("val_ece")
    if ece is not None and ece > 0.15:
        finding("info", "Over-confident predictions",
                f"mean confidence {_pct(final.get('val_confidence'))} vs val accuracy {_pct(final.get('val_accuracy'))} "
                f"(ECE {_pct(ece)}). On-device confidence thresholds will be unreliable.")

    # --- Divergence / LR ---------------------------------------------------
    curve = c["curve"]
    if len(curve) > 2 and curve[-1].get("loss") and curve[0].get("loss") and curve[-1]["loss"] > curve[0]["loss"] * 2:
        finding("critical", "Training loss diverged",
                f"train loss went {curve[0]['loss']} -> {curve[-1]['loss']}.")
        rec("high", "Lower the learning rate", "The loss grew during training, a sign of a too-large step size.",
            {"learning_rate": (cur["learning_rate"] or 0.001) / 3}, "Stable convergence.")

    # --- Deployment --------------------------------------------------------
    usage = deploy.get("int8_flash_usage_pct")
    board = deploy.get("board") or {}
    if usage is not None:
        if usage > 60:
            finding("warning", "Model is large for the target board",
                    f"~{deploy['est_int8_model_kb']} KB as INT8 = {usage}% of {board.get('name')}'s flash.")
            rec("medium", "Use a smaller architecture or input",
                f"The INT8 model would use {usage}% of flash, leaving little room for the firmware.",
                {"base_model": "MobileNetV1_0.25" if task in ("IMAGE_CLASSIFICATION", "VISUAL_WAKE_WORDS") else None},
                "A model that fits with headroom.", "deployment")
        else:
            finding("good", "Fits the target board",
                    f"~{deploy['est_int8_model_kb']} KB as INT8 ({usage}% of {board.get('name')} flash); "
                    "measure the tensor arena (RAM) in the Optimize step.")

    # --- Good news -----------------------------------------------------------
    if not overfit and (final.get("val_accuracy") or 0) >= 0.9 and lost < 3:
        finding("good", "Converged well", f"val {metric} {_pct(final.get('val_accuracy'))} with a {gap:.1f}-pt gap.")
        rec("low", "Quantise and deploy", "The run generalises well; INT8 quantisation usually costs < 1 pt.",
            None, "4x smaller model, faster inference.", "deployment")

    order = {"critical": 0, "warning": 1, "info": 2, "good": 3}
    findings.sort(key=lambda f: order[f["severity"]])
    prio = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: prio[r["priority"]])
    return findings, recs


def _score(task: str, facts: dict) -> Tuple[int, List[dict]]:
    c = facts["training"]
    final = c["final"]
    data = facts["dataset"]
    ev = facts.get("held_out") or {}
    n_classes = max(2, facts["run"]["num_classes"] or 2)
    chance = 0.0 if task == "OBJECT_DETECTION" else 1.0 / n_classes

    perf_src = ev.get("macro_f1") if ev.get("macro_f1") is not None else ev.get("score")
    perf_label = f"{ev.get('split')} macro F1" if ev.get("macro_f1") is not None else f"{ev.get('split')} score"
    if perf_src is None:
        perf_src = final.get("val_f1", final.get("val_accuracy", 0.0))
        perf_label = "val macro F1"
    perf = _clamp01((perf_src - chance) / (1 - chance)) * 45

    gap = max(0.0, (c.get("train_val_gap_pts") or 0.0))
    gen = _clamp01(1 - gap / 30) * 20

    lost = c.get("lost_after_best_val_accuracy_pts") or 0.0
    dyn = 15 - min(15, lost * 1.2)
    curve = c["curve"]
    if len(curve) > 2 and curve[0].get("loss") and curve[-1].get("loss", 0) > curve[0]["loss"] * 2:
        dyn = 0

    min_pc = data.get("min_per_class")
    data_pts = 10.0 if min_pc is None else _clamp01(min_pc / 100) * 10
    if (data.get("imbalance_ratio") or 1) >= 3:
        data_pts *= 0.7

    usage = facts["deployment"].get("int8_flash_usage_pct")
    dep = 10.0 if usage is None else (10.0 if usage <= 30 else 6.0 if usage <= 60 else 2.0 if usage <= 100 else 0.0)

    parts = [
        {"name": "Accuracy above chance", "points": round(perf, 1), "max": 45,
         "detail": f"{perf_label} {_pct(perf_src)} (chance {_pct(chance)})"},
        {"name": "Generalisation", "points": round(gen, 1), "max": 20, "detail": f"train/val gap {gap:.1f} pts"},
        {"name": "Training dynamics", "points": round(dyn, 1), "max": 15,
         "detail": f"{lost:.1f} pts lost after the best epoch"},
        {"name": "Data adequacy", "points": round(data_pts, 1), "max": 10,
         "detail": f"min {min_pc} samples/class" if min_pc is not None else "n/a"},
        {"name": "Deployability", "points": round(dep, 1), "max": 10,
         "detail": f"INT8 ~{usage}% of flash" if usage is not None else "no board selected"},
    ]
    return int(round(sum(p["points"] for p in parts))), parts


def analyze_training(session: dict, board: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic review of a finished training session."""
    from app.services.shared_state import trainer

    ms = session.get("metrics") or []
    if not ms:
        raise ValueError("This run has no completed epochs to analyse.")
    task = session.get("task", "IMAGE_CLASSIFICATION")
    info = session.get("run_info") or {}
    labels = session.get("labels") or info.get("labels") or []
    model_entry = next((m for m in trainer.trained_models.values() if m.get("training_id") == session.get("id")), {})
    params = info.get("params_total") or model_entry.get("params")

    ev = session.get("evaluation") or {}
    held_out = None
    if ev:
        held_out = {
            "split": ev.get("split"), "num_samples": ev.get("num_samples"),
            "score": ev.get("f1") if task == "OBJECT_DETECTION" else ev.get("accuracy"),
            "macro_f1": ev.get("macro_f1"),
            "per_class": [{"label": pc.get("label") or (labels[i] if i < len(labels) else f"class_{i}"),
                           "precision": pc.get("precision"), "recall": pc.get("recall"), "f1": pc.get("f1"),
                           "support": pc.get("support")} for i, pc in enumerate(ev.get("per_class") or [])],
        }
    live = session.get("live_eval") or {}
    cm = _add_cm(ev.get("confusion_matrix"), live.get("confusion_matrix") if live.get("labels") == ev.get("labels") else None)
    per_class = {}
    for src in (live.get("per_class") or [], (held_out or {}).get("per_class") or []):
        for i, pc in enumerate(src):
            lbl = pc.get("label") or (labels[i] if i < len(labels) else None)
            if lbl is not None and pc.get("f1") is not None:
                per_class.setdefault(lbl, []).append(pc["f1"])
    weakest = sorted(({"label": k, "f1": round(float(np.mean(v)), 3)} for k, v in per_class.items()),
                     key=lambda x: x["f1"])[:4]

    facts: Dict[str, Any] = {
        "run": {
            "task": task, "name": session.get("name") or None, "dataset_id": session.get("dataset_id"),
            "num_classes": info.get("num_classes") or len(labels) or None, "labels": labels[:40],
            "num_train": info.get("num_train"), "num_val": info.get("num_val"),
            "device": session.get("device_used"),
            "duration_s": _r((session.get("completed_at") or 0) - (session.get("started_at") or 0), 0),
            "status": session.get("status"),
            "note": "Inputs are always resized to input_shape during preprocessing; source image size does "
                    "not affect training memory. Host memory is irrelevant to the microcontroller.",
        },
        "config": current_config(session),
        "training": _curve_facts(ms, session),
        "held_out": held_out,
        "weakest_classes": weakest,
        "top_confusions": _top_confusions(cm, ev.get("labels") or live.get("labels") or labels),
        "dataset": _dataset_facts(session, labels),
        "deployment": _deploy_facts(session, params, board),
        "past_runs": _past_runs(session),
    }
    facts["run"]["num_train"] = facts["run"]["num_train"] or (facts["dataset"].get("split_counts") or {}).get("train")
    findings, recs = _findings_and_recs(task, facts, facts["config"])
    score, parts = _score(task, facts)

    past = facts["past_runs"]
    comparison = None
    if past:
        best_prev = max(past, key=lambda r: r.get("best_val_accuracy") or 0)
        mine = facts["training"]["best_val_accuracy"]["value"] or 0
        comparison = {"runs": len(past), "best_previous": best_prev,
                      "this_run_best_val_accuracy": mine,
                      "delta_vs_best_previous_pts": _r((mine - (best_prev.get("best_val_accuracy") or 0)) * 100, 1)}
        if comparison["delta_vs_best_previous_pts"] is not None and comparison["delta_vs_best_previous_pts"] < -3:
            findings.append({"severity": "info", "title": "An earlier run did better",
                             "evidence": f"{best_prev.get('base_model')} reached {_pct(best_prev.get('best_val_accuracy'))} "
                                         f"best val vs {_pct(mine)} here."})

    label = "excellent" if score >= 85 else "good" if score >= 70 else "needs work" if score >= 50 else "poor"
    summary = _rule_summary(task, facts, findings, score, label)
    return {
        "version": REVIEW_VERSION,
        "training_id": session.get("id"),
        "score": score, "score_label": label, "score_breakdown": parts,
        "summary": summary,
        "findings": findings,
        "rule_suggestions": recs,
        "comparison": comparison,
        "facts": facts,
    }


def _rule_summary(task, facts, findings, score, label) -> str:
    c = facts["training"]
    final = c["final"]
    ev = facts.get("held_out") or {}
    metric = "F1" if task == "OBJECT_DETECTION" else "accuracy"
    parts = [f"Score {score}/100 ({label})."]
    s = f"Final val {metric} {_pct(final.get('val_accuracy'))} (best {_pct(c['best_val_accuracy']['value'])} at epoch {c['best_val_accuracy']['epoch']})"
    if ev.get("score") is not None:
        s += f", {ev['split']} {metric} {_pct(ev['score'])} on {ev['num_samples']} samples"
    parts.append(s + ".")
    top = [f["title"].lower() for f in findings if f["severity"] in ("critical", "warning")][:3]
    if top:
        parts.append("Main issues: " + "; ".join(top) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# LLM layer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the training reviewer inside EdgeCraft AI, a TinyML studio that trains small
neural networks for microcontrollers (ESP32-S3, ESP32-CAM, Raspberry Pi Pico 2 W, Arduino Nano 33 BLE).

You receive FACTS computed by the app from the complete run (every epoch, held-out test results, latest
validation confusion data, dataset statistics, model size vs board, earlier runs on the same dataset),
deterministic FINDINGS, and CANDIDATE recommendations from a rule engine.

Rules:
- Use only numbers that appear in the facts. Never invent measurements (no RAM or latency figures unless given).
- Images are already resized to config.input_shape; host/training-machine memory is irrelevant to the MCU.
- "changes" may only use these keys (exact names), with values different from config: {params}.
  Leave "changes" empty for advice that is not a training parameter (e.g. collecting data).
- Prefer the highest-impact actions first. Refine, merge or drop the candidates; add others only if the facts support them.
- If earlier runs exist, say what they show (which settings helped or hurt) and recommend the next experiment.
- Be concrete and specific to this run (name classes, epochs, values).

Respond with ONLY a JSON object:
{{"summary": "2-4 sentences: overall verdict and the main reason",
  "suggestions": [{{"title": "short imperative", "priority": "high|medium|low",
                   "category": "training|data|deployment",
                   "reasoning": "why, citing the facts", "changes": {{"param": value}},
                   "expected_effect": "what should improve"}}]}}
Give 3 to 5 suggestions."""


def _validate_llm(task: str, current: dict):
    def validate(parsed: Any) -> Dict[str, Any]:
        if isinstance(parsed, list):
            parsed = {"suggestions": parsed}
        if not isinstance(parsed, dict):
            raise ValueError("reply is not a JSON object")
        raw = parsed.get("suggestions") or parsed.get("recommendations") or []
        if not isinstance(raw, list):
            raise ValueError("'suggestions' is not a list")
        out = []
        for s in raw:
            if not isinstance(s, dict):
                continue
            title = str(s.get("title") or s.get("suggestion") or "").strip()
            if not title:
                continue
            proposed = s.get("changes", s.get("parameters_to_adjust")) or {}
            changes, notes = validate_changes(task, proposed, current)
            if proposed and not changes and not notes:
                continue  # only "changed" values to what they already are
            prio = str(s.get("priority", "medium")).lower()
            cat = str(s.get("category", "training")).lower()
            out.append({
                "title": title[:140],
                "priority": prio if prio in ("high", "medium", "low") else "medium",
                "category": cat if cat in ("training", "data", "deployment") else "training",
                "reasoning": str(s.get("reasoning") or "").strip()[:1200],
                "changes": changes,
                "rejected_changes": notes or None,
                "expected_effect": str(s.get("expected_effect") or s.get("estimated_improvement") or "").strip()[:300],
                "source": "ai",
            })
        if not out:
            raise ValueError("no usable suggestions in reply")
        summary = str(parsed.get("summary") or "").strip()
        return {"summary": summary[:1500], "suggestions": out[:6]}

    return validate


def _llm_payload(review: dict) -> str:
    f = review["facts"]
    payload = {
        "facts": {k: v for k, v in f.items()},
        "findings": review["findings"],
        "score": {"value": review["score"], "breakdown": review["score_breakdown"]},
        "candidates": [{k: r[k] for k in ("title", "priority", "category", "reasoning", "changes", "expected_effect")}
                       for r in review["rule_suggestions"]],
        "allowed_models": allowed_models(f["run"]["task"]),
    }
    return json.dumps(payload, indent=1, default=str)


async def review_training(
    session: dict, board: Optional[str] = None, provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministic review, plus AI suggestions when a provider is given.
    Never raises for LLM problems: falls back to the rule suggestions and
    reports the error in `ai_error`."""
    from app.services.llm_client import LLMError, chat_json

    review = await asyncio.to_thread(analyze_training, session, board)
    review["generated_at"] = time.time()
    review["board"] = board
    review["suggestions"] = review["rule_suggestions"]
    review["suggestions_source"] = "rules"
    if provider not in ("openrouter", "ollama"):
        return review

    if provider == "ollama":
        from app.services.llm_advisor import _ollama_default_model, _ollama_enabled

        if not _ollama_enabled():
            review["ai_error"] = "Ollama is not enabled (set OLLAMA_ENABLED=true in backend/.env)."
            return review
        if not model_name or "/" in model_name:
            model_name = _ollama_default_model()
    model_name = model_name or config.DEFAULT_OPENROUTER_MODEL

    params = ", ".join(f"{k} ({v})" for k, v in PARAM_DOC.items())
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(params=params)},
        {"role": "user", "content": _llm_payload(review)},
    ]
    t0 = time.monotonic()
    try:
        ai = await chat_json(messages, provider, model_name,
                             validate=_validate_llm(session.get("task", ""), review["facts"]["config"]))
        review["suggestions"] = ai["suggestions"]
        review["suggestions_source"] = "ai"
        if ai["summary"]:
            review["ai_summary"] = ai["summary"]
    except LLMError as e:
        review["ai_error"] = str(e)
    except Exception as e:  # defensive: never turn an LLM hiccup into a failed review
        logger.exception("AI review failed")
        review["ai_error"] = f"{type(e).__name__}: {e}"
    review["ai_provider"] = provider
    review["ai_model"] = model_name
    review["ai_seconds"] = round(time.monotonic() - t0, 1)
    return review
