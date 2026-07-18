import os
import re
import io
import zipfile
from app.services.shared_state import data_manager
from app.utils.annotation_parser import (
    ANNOTATION_EXTENSIONS,
    detect_annotation_format,
    find_annotation_for_image,
    parse_yolo_txt,
    parse_yolo_classes,
    parse_voc_xml,
    parse_coco_json,
)

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".wav"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ALL_EXTENSIONS = VALID_EXTENSIONS | ANNOTATION_EXTENSIONS


def scan_zip_tree(zip_path: str) -> dict:
    """Scans a ZIP and generates a suggested mapping tree with file names.

    Returns {"tree": [...], "annotation_format": str | None, "annotation_classes": [...]}.
    """
    tree_counts = {}
    tree_files = {}
    all_files = []
    annotation_files = []

    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX") or info.filename.split("/")[-1].startswith("."):
                continue
            ext = os.path.splitext(info.filename)[1].lower()
            all_files.append(info.filename)

            if ext in ANNOTATION_EXTENSIONS:
                annotation_files.append(info.filename)
                continue

            if ext not in VALID_EXTENSIONS:
                continue

            parts = [p for p in info.filename.split("/") if p]
            parent = "/".join(parts[:-1]) if len(parts) > 1 else "/"

            tree_counts[parent] = tree_counts.get(parent, 0) + 1

            if parent not in tree_files:
                tree_files[parent] = []
            tree_files[parent].append(parts[-1])

    annotation_format = detect_annotation_format(all_files) if annotation_files else None
    annotation_classes = _detect_yolo_classes_from_zip(zip_path) if annotation_format == "yolo" else []

    results = []
    for path, count in tree_counts.items():
        split = "unassigned"
        label = "unknown"

        parts = path.split("/")
        if path == "/":
            label = "root_folder"
        else:
            p_lower = [p.lower() for p in parts]
            if "train" in p_lower: split = "train"
            elif "val" in p_lower or "validation" in p_lower or "valid" in p_lower: split = "val"
            elif "test" in p_lower: split = "test"

            label_candidates = [p for p in parts if p.lower() not in {"train", "val", "validation", "valid", "test"}]
            if label_candidates:
                label = label_candidates[-1]
            else:
                label = parts[-1]

        results.append({
            "path": path,
            "file_count": count,
            "split": split,
            "label": label,
            "ignore": False,
            "files": tree_files.get(path, [])
        })

    return {
        "tree": sorted(results, key=lambda x: x["path"]),
        "annotation_format": annotation_format,
        "annotation_classes": annotation_classes,
    }


def _detect_yolo_classes_from_zip(zip_path: str) -> list:
    """Try to find a classes.txt / class.names file in the ZIP for YOLO format."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                basename = os.path.basename(name).lower()
                if basename in ("classes.txt", "class.names", "classes.names", "obj.names"):
                    with z.open(name) as f:
                        content = f.read().decode("utf-8", errors="replace")
                    class_map = parse_yolo_classes(content)
                    return [class_map[i] for i in sorted(class_map.keys())]
    except Exception:
        pass
    return []


def _probe_image_dims(content: bytes):
    """Best-effort (width, height) probe for an in-memory image. Returns
    (None, None) if PIL isn't available or the bytes aren't a readable
    image (e.g. a .wav file, or a corrupt/truncated image)."""
    if not _PIL_AVAILABLE:
        return None, None
    try:
        with Image.open(io.BytesIO(content)) as img:
            return img.width, img.height
    except Exception:
        return None, None


def extract_zip_with_mapping(
    zip_path: str,
    dataset_id: str,
    task: str,
    mapping: list,
    label_strategy: str = "folder",
    regex_pattern: str = None,
) -> dict:
    """Extracts the zip strictly using the user-confirmed mapping, optionally
    applying a filename regex to (re)derive the label.

    Now also detects and parses annotation files (YOLO .txt, Pascal VOC .xml,
    COCO .json, CSV) and attaches bounding box data to each sample.

    Returns a dict:
        {
            "processed": <int>,
            "unmatched_regex": <int>,
            "regex_error": <str | None>,
            "annotation_format": <str | None>,
            "annotations_matched": <int>,
            "annotation_classes": [<str>],
        }
    """
    map_dict = {m["path"]: m for m in mapping}
    total_processed = 0
    unmatched_regex = 0
    regex_error = None
    annotations_matched = 0
    file_data_list = []

    compiled_regex = None
    if label_strategy == "filename" and regex_pattern:
        try:
            compiled_regex = re.compile(regex_pattern)
        except re.error as e:
            regex_error = str(e)
            compiled_regex = None

    # ── Phase 1: collect annotation files from the ZIP ──
    annotation_format = None
    yolo_class_map = {}
    coco_images = {}
    coco_anns = {}
    annotation_file_contents: Dict[str, str] = {}  # relative_path -> content

    with zipfile.ZipFile(zip_path, 'r') as z:
        all_names = z.namelist()
        ann_names = [
            n for n in all_names
            if not n.startswith("__MACOSX")
            and os.path.splitext(n)[1].lower() in ANNOTATION_EXTENSIONS
        ]

        if ann_names:
            annotation_format = detect_annotation_format(all_names)

            for ann_name in ann_names:
                basename = os.path.basename(ann_name).lower()
                if annotation_format == "yolo" and basename in (
                    "classes.txt", "class.names", "classes.names", "obj.names"
                ):
                    with z.open(ann_name) as f:
                        content = f.read().decode("utf-8", errors="replace")
                    yolo_class_map = parse_yolo_classes(content)
                    continue

                try:
                    with z.open(ann_name) as f:
                        content = f.read().decode("utf-8", errors="replace")
                except Exception:
                    continue

                if annotation_format == "coco":
                    coco_images, coco_anns = parse_coco_json(content)
                else:
                    rel_path = ann_name
                    parts = [p for p in rel_path.split("/") if p]
                    if len(parts) > 1:
                        rel_path = "/".join(parts[-2:])
                    annotation_file_contents[os.path.basename(ann_name)] = content

    # ── Phase 2: extract images with their annotations ──
    coco_img_filename_to_bboxes = {}
    if annotation_format == "coco" and coco_images:
        for img_id, fname in coco_images.items():
            bboxes = coco_anns.get(img_id, [])
            if bboxes:
                coco_img_filename_to_bboxes[fname] = [bbox.dict() for bbox in bboxes]

    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX") or info.filename.split("/")[-1].startswith("."):
                continue

            ext = os.path.splitext(info.filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            parts = [p for p in info.filename.split("/") if p]
            parent = "/".join(parts[:-1]) if len(parts) > 1 else "/"

            m = map_dict.get(parent)
            if not m or m.get("ignore", False):
                continue

            filename = parts[-1]
            split = m.get("split", "unassigned")

            # Determine Label
            label = m.get("label", "unknown").strip()
            if compiled_regex:
                match = compiled_regex.search(filename)
                if match:
                    label = match.group(1) if match.groups() else match.group(0)
                else:
                    unmatched_regex += 1

            with z.open(info) as f:
                content = f.read()

            width, height = (None, None)
            if ext in IMAGE_EXTENSIONS:
                width, height = _probe_image_dims(content)

            # ── Parse annotations for this image ──
            sample_annotations = None
            if annotation_format and task == "OBJECT_DETECTION":
                if annotation_format == "coco":
                    sample_annotations = coco_img_filename_to_bboxes.get(filename)
                elif annotation_format == "yolo":
                    ann_content = find_annotation_for_image(filename, annotation_file_contents, "yolo")
                    if ann_content:
                        boxes = parse_yolo_txt(ann_content, yolo_class_map or None)
                        sample_annotations = [b.dict() for b in boxes]
                elif annotation_format == "voc":
                    ann_content = find_annotation_for_image(filename, annotation_file_contents, "voc")
                    if ann_content:
                        boxes = parse_voc_xml(ann_content)
                        sample_annotations = [b.dict() for b in boxes]
                elif annotation_format == "csv":
                    ann_content = find_annotation_for_image(filename, annotation_file_contents, "csv")
                    if ann_content:
                        from app.utils.annotation_parser import parse_csv_annotations
                        csv_anns = parse_csv_annotations(ann_content)
                        if csv_anns:
                            boxes = csv_anns.get(filename, [])
                            if not boxes:
                                boxes = csv_anns.get(os.path.basename(filename), [])
                            sample_annotations = [b.dict() for b in boxes]

            if sample_annotations:
                annotations_matched += 1

            file_data_list.append({
                "label": label,
                "filename": filename,
                "content": content,
                "split": split,
                "width": width,
                "height": height,
                "annotations": sample_annotations,
            })

            if len(file_data_list) >= 500:
                data_manager.bulk_add_samples(dataset_id, task, file_data_list)
                total_processed += len(file_data_list)
                file_data_list.clear()

        if file_data_list:
            data_manager.bulk_add_samples(dataset_id, task, file_data_list)
            total_processed += len(file_data_list)

    # ── Store annotation metadata on the dataset ──
    annotation_classes = []
    if annotation_format:
        if annotation_format == "yolo" and yolo_class_map:
            annotation_classes = [yolo_class_map[i] for i in sorted(yolo_class_map.keys())]
        elif annotation_format == "coco" and coco_images:
            cat_set = set()
            for ann_list in coco_anns.values():
                for a in ann_list:
                    cat_set.add(a.class_name)
            annotation_classes = sorted(cat_set)
        else:
            # Collect from parsed samples
            class_set = set()
            for sid in data_manager.samples_by_dataset.get(dataset_id, ()):
                anns = data_manager.samples[sid].get("annotations", [])
                for a in anns:
                    class_set.add(a.get("class_name", "unknown"))
            annotation_classes = sorted(class_set)

        data_manager.update_dataset_annotation_metadata(
            dataset_id, annotation_format, annotation_classes
        )

    return {
        "processed": total_processed,
        "unmatched_regex": unmatched_regex,
        "regex_error": regex_error,
        "annotation_format": annotation_format,
        "annotations_matched": annotations_matched,
        "annotation_classes": annotation_classes,
    }
