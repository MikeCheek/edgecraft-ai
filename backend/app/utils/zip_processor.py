import os
import zipfile
from app.services.shared_state import data_manager

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".wav"}

def scan_zip_tree(zip_path: str) -> list:
    """Scans a ZIP and generates a suggested mapping tree."""
    tree = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX") or info.filename.split("/")[-1].startswith("."):
                continue
            ext = os.path.splitext(info.filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue
                
            parts = [p for p in info.filename.split("/") if p]
            parent = "/".join(parts[:-1]) if len(parts) > 1 else "/"
            tree[parent] = tree.get(parent, 0) + 1
    
    results = []
    for path, count in tree.items():
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
            "ignore": False
        })
        
    return sorted(results, key=lambda x: x["path"])

def extract_zip_with_mapping(zip_path: str, dataset_id: str, task: str, mapping: list) -> int:
    """Extracts the zip strictly using the user-confirmed mapping."""
    map_dict = { m["path"]: m for m in mapping }
    total_processed = 0
    file_data_list = []
    
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
                
            label = m.get("label", "unknown").strip()
            split = m.get("split", "unassigned")
            filename = parts[-1]
            
            with z.open(info) as f:
                content = f.read()
                
            file_data_list.append({
                "label": label,
                "filename": filename,
                "content": content,
                "split": split
            })
            
            if len(file_data_list) >= 500:
                data_manager.bulk_add_samples(dataset_id, task, file_data_list)
                total_processed += len(file_data_list)
                file_data_list.clear()
                
        if file_data_list:
            data_manager.bulk_add_samples(dataset_id, task, file_data_list)
            total_processed += len(file_data_list)
            
    return total_processed