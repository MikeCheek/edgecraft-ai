import os
import re
import io
import zipfile
from app.services.shared_state import data_manager

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".wav"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_zip_tree(zip_path: str) -> list:
    """Scans a ZIP and generates a suggested mapping tree with file names."""
    tree_counts = {}
    tree_files = {}  # NEW: Dictionary to store the actual file names

    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX") or info.filename.split("/")[-1].startswith("."):
                continue
            ext = os.path.splitext(info.filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            parts = [p for p in info.filename.split("/") if p]
            parent = "/".join(parts[:-1]) if len(parts) > 1 else "/"

            # Count the files
            tree_counts[parent] = tree_counts.get(parent, 0) + 1

            # Append the actual filename to our tracker
            if parent not in tree_files:
                tree_files[parent] = []
            tree_files[parent].append(parts[-1])

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

            # Suggest label by ignoring split names
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
            "files": tree_files.get(path, [])  # NEW: Inject files into the JSON!
        })

    return sorted(results, key=lambda x: x["path"])


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

    Returns a dict:
        {
            "processed": <int, samples actually written>,
            "unmatched_regex": <int, files where the regex didn't match and
                                 the folder label was used instead>,
            "regex_error": <str | None, set if regex_pattern was invalid>,
        }
    """
    # BUGFIX: this module never imported `re`, so any zip apply that used the
    # "filename regex" label strategy raised a NameError on the very first
    # line below (before a single file was even read) and the whole request
    # failed. Because the failure happened instantly, in some setups the
    # frontend's generic error handling made it look like nothing happened /
    # a timeout, and any samples visible afterwards were leftovers from a
    # previous attempt using the (stale) folder-derived label instead of the
    # regex. Adding `import re` above is the actual fix for that.
    map_dict = {m["path"]: m for m in mapping}
    total_processed = 0
    unmatched_regex = 0
    regex_error = None
    file_data_list = []

    compiled_regex = None
    if label_strategy == "filename" and regex_pattern:
        try:
            compiled_regex = re.compile(regex_pattern)
        except re.error as e:
            # Previously this silently swallowed the bad regex and fell back
            # to folder labels with no indication anything was wrong. Now we
            # still fall back (so extraction doesn't hard-fail), but the
            # caller gets told about it via regex_error.
            regex_error = str(e)
            compiled_regex = None

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
                    # BUGFIX: previously a non-match silently fell through to
                    # whatever `m["label"]` happened to hold (often the raw
                    # folder name, e.g. "train"/"images"/"" ) with no record
                    # that the regex didn't actually apply to this file. We
                    # now count these so the API response can surface them.
                    unmatched_regex += 1

            with z.open(info) as f:
                content = f.read()

            width, height = (None, None)
            if ext in IMAGE_EXTENSIONS:
                width, height = _probe_image_dims(content)

            file_data_list.append({
                "label": label,
                "filename": filename,
                "content": content,
                "split": split,
                "width": width,
                "height": height,
            })

            if len(file_data_list) >= 500:
                data_manager.bulk_add_samples(dataset_id, task, file_data_list)
                total_processed += len(file_data_list)
                file_data_list.clear()

        if file_data_list:
            data_manager.bulk_add_samples(dataset_id, task, file_data_list)
            total_processed += len(file_data_list)

    return {
        "processed": total_processed,
        "unmatched_regex": unmatched_regex,
        "regex_error": regex_error,
    }
