"""Annotation parsers for common object detection formats.

Supports:
  - YOLO:     one .txt per image, lines: <class_id> cx cy w h [conf]
  - Pascal VOC: one .xml per image with <object><bndbox>
  - COCO:     single annotations.json with images[] and annotations[]
  - CSV:      one .csv per image (or single csv) with columns
              filename/class/x1/y1/x2/y2 or similar

All formats are normalized to a list of BoundingBox(cx, cy, w, h) in
0-1 normalized coordinates so the frontend only needs one renderer.
"""

import csv
import io
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from app.models import BoundingBox


# ── YOLO ──────────────────────────────────────────────────────────────

def parse_yolo_txt(content: str, class_names: Optional[Dict[int, str]] = None) -> List[BoundingBox]:
    """Parse a single YOLO annotation .txt file.

    Each line: class_id cx cy w h [confidence]
    class_id is an integer index; class_names maps id -> name.
    If class_names is None, the id itself is used as the class name.
    """
    boxes = []
    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            conf = float(parts[5]) if len(parts) > 5 else None
        except (ValueError, IndexError):
            continue
        class_name = (class_names or {}).get(cls_id, str(cls_id))
        boxes.append(BoundingBox(class_name=class_name, cx=cx, cy=cy, w=w, h=h, confidence=conf))
    return boxes


def parse_yolo_classes(content: str) -> Dict[int, str]:
    """Parse a YOLO classes.txt / .names file (one class name per line)."""
    names = {}
    for i, line in enumerate(content.strip().splitlines()):
        name = line.strip()
        if name:
            names[i] = name
    return names


# ── Pascal VOC ────────────────────────────────────────────────────────

def parse_voc_xml(content: str) -> List[BoundingBox]:
    """Parse a Pascal VOC annotation .xml file.

    Expects <size><width>/<height> and <object><bndbox><xmin><ymin><xmax><ymax>.
    Coordinates are converted to normalized cx/cy/w/h.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    size = root.find("size")
    if size is None:
        return []
    img_w = float(size.findtext("width", "0"))
    img_h = float(size.findtext("height", "0"))
    if img_w <= 0 or img_h <= 0:
        return []

    boxes = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        class_name = name_el.text.strip() if name_el is not None and name_el.text else "unknown"
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
        except (ValueError, AttributeError):
            continue
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        bcx = (xmin + xmax) / (2 * img_w)
        bcy = (ymin + ymax) / (2 * img_h)
        boxes.append(BoundingBox(class_name=class_name, cx=bcx, cy=bcy, w=bw, h=bh))
    return boxes


# ── COCO ──────────────────────────────────────────────────────────────

def parse_coco_json(
    content: str,
) -> Tuple[Dict[int, str], Dict[int, List[BoundingBox]]]:
    """Parse a COCO-format annotations.json.

    Returns (images_map, annotations_by_image_id) where:
      images_map: {image_id: filename}
      annotations_by_image_id: {image_id: [BoundingBox, ...]}
    """
    import json
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}, {}

    categories = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    images = {img["id"]: img.get("file_name", str(img["id"])) for img in data.get("images", [])}

    by_image: Dict[int, List[BoundingBox]] = {}
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        cat_name = categories.get(ann.get("category_id"), str(ann.get("category_id", "?")))
        seg = ann.get("bbox", [])
        if len(seg) < 4:
            continue
        x, y, w, h = seg[0], seg[1], seg[2], seg[3]
        img_info = None
        for img in data.get("images", []):
            if img["id"] == img_id:
                img_info = img
                break
        if img_info is None:
            continue
        iw = float(img_info.get("width", 1) or 1)
        ih = float(img_info.get("height", 1) or 1)
        if iw <= 0 or ih <= 0:
            continue
        bcx = (x + w / 2) / iw
        bcy = (y + h / 2) / ih
        bw = w / iw
        bh = h / ih
        by_image.setdefault(img_id, []).append(
            BoundingBox(class_name=cat_name, cx=bcx, cy=bcy, w=bw, h=bh)
        )
    return images, by_image


# ── CSV ───────────────────────────────────────────────────────────────

def parse_csv_annotations(content: str) -> Dict[str, List[BoundingBox]]:
    """Parse CSV annotation files.

    Supports common column layouts:
      - filename/image_name, class, x1, y1, x2, y2   (absolute pixel coords)
      - filename/image_name, class, x0, y0, x1, y1   (absolute pixel coords)
      - filename/image_name, class, x, y, w, h        (absolute pixel coords)
      - filename/image_name, class, cx, cy, w, h      (normalized 0-1)

    When image width/height columns are present, pixel coordinates are
    automatically normalized to 0-1.
    """
    reader = csv.DictReader(io.StringIO(content))
    by_file: Dict[str, List[BoundingBox]] = {}
    if not reader.fieldnames:
        return by_file

    fields = [f.strip().lower() for f in reader.fieldnames]

    has_xyxy = all(f in fields for f in ["x1", "y1", "x2", "y2"])
    has_xyxy0 = all(f in fields for f in ["x0", "y0", "x1", "y1"])
    has_xywh = all(f in fields for f in ["x", "y", "w", "h"])
    has_cxywh = all(f in fields for f in ["cx", "cy", "w", "h"])

    fname_field = next((f for f in fields if f in ("filename", "file", "image", "image_name", "image_id")), fields[0])
    class_field = next((f for f in fields if f in ("class", "label", "category", "name")), None)
    img_w_field = next((f for f in fields if f in ("img_width", "image_width")), None)
    img_h_field = next((f for f in fields if f in ("img_height", "image_height")), None)

    for row in reader:
        fname = row.get(fname_field, "").strip()
        cls_name = row.get(class_field, "object").strip() if class_field else "object"
        if not fname:
            continue

        try:
            iw = float(row[img_w_field]) if img_w_field and row.get(img_w_field) else None
            ih = float(row[img_h_field]) if img_h_field and row.get(img_h_field) else None

            if has_cxywh:
                bcx = float(row["cx"])
                bcy = float(row["cy"])
                bw = float(row["w"])
                bh = float(row["h"])
                by_file.setdefault(fname, []).append(
                    BoundingBox(class_name=cls_name, cx=bcx, cy=bcy, w=bw, h=bh)
                )
            elif has_xyxy:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
                bw = x2 - x1
                bh = y2 - y1
                bcx = x1 + bw / 2
                bcy = y1 + bh / 2
                if iw and ih and iw > 0 and ih > 0:
                    bcx /= iw
                    bcy /= ih
                    bw /= iw
                    bh /= ih
                by_file.setdefault(fname, []).append(
                    BoundingBox(class_name=cls_name, cx=bcx, cy=bcy, w=bw, h=bh)
                )
            elif has_xyxy0:
                x0 = float(row["x0"])
                y0 = float(row["y0"])
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                bw = x1 - x0
                bh = y1 - y0
                bcx = x0 + bw / 2
                bcy = y0 + bh / 2
                if iw and ih and iw > 0 and ih > 0:
                    bcx /= iw
                    bcy /= ih
                    bw /= iw
                    bh /= ih
                by_file.setdefault(fname, []).append(
                    BoundingBox(class_name=cls_name, cx=bcx, cy=bcy, w=bw, h=bh)
                )
            elif has_xywh:
                x = float(row["x"])
                y = float(row["y"])
                w = float(row["w"])
                h = float(row["h"])
                bcx = x + w / 2
                bcy = y + h / 2
                if iw and ih and iw > 0 and ih > 0:
                    bcx /= iw
                    bcy /= ih
                    w /= iw
                    h /= ih
                by_file.setdefault(fname, []).append(
                    BoundingBox(class_name=cls_name, cx=bcx, cy=bcy, w=w, h=h)
                )
        except (ValueError, KeyError):
            continue

    return by_file


# ── Format Detection ─────────────────────────────────────────────────

ANNOTATION_EXTENSIONS = {".txt", ".xml", ".json", ".csv"}

def detect_annotation_format(files_in_zip: List[str]) -> Optional[str]:
    """Detect the annotation format from a list of filenames in a ZIP.

    Returns 'yolo', 'voc', 'coco', 'csv', or None.
    """
    exts = set()
    for f in files_in_zip:
        ext = os.path.splitext(f)[1].lower()
        exts.add(ext)

    # COCO: single annotations.json at root
    has_json = ".json" in exts
    has_txt = ".txt" in exts
    has_xml = ".xml" in exts
    has_csv = ".csv" in exts

    if has_json and not has_txt and not has_xml:
        return "coco"
    if has_xml:
        return "voc"
    if has_txt:
        return "yolo"
    if has_csv:
        return "csv"
    return None


def find_annotation_for_image(
    image_filename: str,
    annotation_files: Dict[str, str],
    annotation_format: str,
) -> Optional[str]:
    """Given an image filename and a dict of {ann_filename: content},
    find the matching annotation content.

    For YOLO/VOC: same basename, different extension (.txt / .xml)
    For CSV: same directory or root-level .csv
    """
    base = os.path.splitext(image_filename)[0]

    if annotation_format == "yolo":
        key = base + ".txt"
        if key in annotation_files:
            return annotation_files[key]
    elif annotation_format == "voc":
        key = base + ".xml"
        if key in annotation_files:
            return annotation_files[key]
    elif annotation_format == "csv":
        key = base + ".csv"
        if key in annotation_files:
            return annotation_files[key]
    elif annotation_format == "coco":
        # COCO is a single file; handled separately
        pass

    return None
