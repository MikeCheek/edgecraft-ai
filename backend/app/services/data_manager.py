import os
import json
import uuid
import time
import random
from typing import Dict, List, Optional

class DataManager:
    """Persistent data management for datasets"""

    def __init__(self, storage_dir: str = "data_storage"):
        self.storage_dir = storage_dir
        self.db_file = os.path.join(storage_dir, "db.json")

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        self.datasets: Dict[str, dict] = {}
        self.samples: Dict[str, dict] = {}
        self.sample_data: Dict[str, bytes] = {}
        self.dataset_labels: Dict[str, List[str]] = {}

        self._load_from_disk()

    def _save_to_disk(self):
        with open(self.db_file, "w") as f:
            json.dump({
                "datasets": self.datasets,
                "samples": self.samples,
                "dataset_labels": self.dataset_labels,
            }, f, indent=2)
        for s_id, data in self.sample_data.items():
            file_path = os.path.join(self.storage_dir, f"{s_id}.bin")
            with open(file_path, "wb") as f:
                f.write(data)

    def _load_from_disk(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                data = json.load(f)
                self.datasets = data.get("datasets", {})
                self.samples = data.get("samples", {})
                self.dataset_labels = data.get("dataset_labels", {})
        for s_id in self.samples:
            file_path = os.path.join(self.storage_dir, f"{s_id}.bin")
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    self.sample_data[s_id] = f.read()

    def _sync_labels(self, dataset_id: str):
        existing = {
            s["label"]
            for s in self.samples.values()
            if s["dataset_id"] == dataset_id
        }
        registered = set(self.dataset_labels.get(dataset_id, []))
        merged = sorted(registered | existing)
        self.dataset_labels[dataset_id] = merged

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def create_dataset(self, name: str, task: str) -> dict:
        dataset_id = str(uuid.uuid4())
        dataset = {
            "id": dataset_id,
            "name": name,
            "task": task,
            "sample_count": 0,
            "created_at": time.time(),
        }
        self.datasets[dataset_id] = dataset
        self.dataset_labels[dataset_id] = []
        self._save_to_disk()
        return dataset

    def rename_dataset(self, dataset_id: str, new_name: str) -> bool:
        if dataset_id not in self.datasets:
            return False
        self.datasets[dataset_id]["name"] = new_name
        self._save_to_disk()
        return True

    def delete_dataset(self, dataset_id: str) -> bool:
        if dataset_id not in self.datasets:
            return False
        self.clear_dataset_samples(dataset_id)
        del self.datasets[dataset_id]
        self.dataset_labels.pop(dataset_id, None)
        self._save_to_disk()
        return True

    def get_datasets(self, task: Optional[str] = None) -> List[dict]:
        datasets = list(self.datasets.values())
        if task:
            datasets = [d for d in datasets if d["task"] == task]
        return datasets

    # ------------------------------------------------------------------
    # Samples & Splits
    # ------------------------------------------------------------------

    def add_sample(self, dataset_id: str, label: str, task: str,
                   data: bytes, filename: str) -> str:
        if dataset_id not in self.datasets:
            raise ValueError("Dataset not found")
        sample_id = str(uuid.uuid4())
        self.samples[sample_id] = {
            "id": sample_id,
            "dataset_id": dataset_id,
            "label": label,
            "task": task,
            "filename": filename,
            "timestamp": time.time(),
            "split": "unassigned",
        }
        self.sample_data[sample_id] = data
        self.datasets[dataset_id]["sample_count"] += 1
        
        if dataset_id not in self.dataset_labels:
            self.dataset_labels[dataset_id] = []
        if label not in self.dataset_labels[dataset_id]:
            self.dataset_labels[dataset_id] = sorted(
                set(self.dataset_labels[dataset_id]) | {label}
            )
        self._save_to_disk()
        return sample_id

    def delete_sample(self, sample_id: str) -> bool:
        if sample_id not in self.samples:
            return False
        dataset_id = self.samples[sample_id]["dataset_id"]
        if dataset_id in self.datasets:
            self.datasets[dataset_id]["sample_count"] = max(
                0, self.datasets[dataset_id]["sample_count"] - 1
            )
        del self.samples[sample_id]
        if sample_id in self.sample_data:
            del self.sample_data[sample_id]
        file_path = os.path.join(self.storage_dir, f"{sample_id}.bin")
        if os.path.exists(file_path):
            os.remove(file_path)
        self._save_to_disk()
        return True

    def get_samples(self, dataset_id: Optional[str] = None) -> List[dict]:
        samples = list(self.samples.values())
        if dataset_id:
            samples = [s for s in samples if s["dataset_id"] == dataset_id]
        return samples

    def get_sample_data(self, sample_id: str) -> Optional[bytes]:
        return self.sample_data.get(sample_id)

    def clear_dataset_samples(self, dataset_id: str) -> int:
        sample_ids = [s["id"] for s in self.samples.values()
                      if s["dataset_id"] == dataset_id]
        for s_id in sample_ids:
            self.delete_sample(s_id)
        return len(sample_ids)

    def relabel_sample(self, sample_id: str, new_label: str) -> bool:
        if sample_id not in self.samples:
            return False
        dataset_id = self.samples[sample_id]["dataset_id"]
        self.samples[sample_id]["label"] = new_label
        if dataset_id not in self.dataset_labels:
            self.dataset_labels[dataset_id] = []
        if new_label not in self.dataset_labels[dataset_id]:
            self.dataset_labels[dataset_id] = sorted(
                set(self.dataset_labels[dataset_id]) | {new_label}
            )
        self._save_to_disk()
        return True

    def set_sample_split(self, sample_id: str, split: str) -> bool:
        if sample_id in self.samples:
            self.samples[sample_id]["split"] = split
            self._save_to_disk()
            return True
        return False

    def auto_split_dataset(self, dataset_id: str, train_pct: int, val_pct: int, test_pct: int) -> int:
        """Randomly divide the dataset maintaining class distribution."""
        samples = [s for s in self.samples.values() if s["dataset_id"] == dataset_id]
        by_label = {}
        for s in samples:
            by_label.setdefault(s["label"], []).append(s)

        count = 0
        for label, group in by_label.items():
            random.shuffle(group)
            n = len(group)
            n_train = round(n * (train_pct / 100.0))
            n_val = round(n * (val_pct / 100.0))

            for i, s in enumerate(group):
                if i < n_train:
                    s["split"] = "train"
                elif i < n_train + n_val:
                    s["split"] = "val"
                else:
                    s["split"] = "test"
                count += 1

        if count:
            self._save_to_disk()
        return count

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def get_dataset_labels(self, dataset_id: str) -> List[str]:
        self._sync_labels(dataset_id)
        return self.dataset_labels.get(dataset_id, [])

    def add_label(self, dataset_id: str, label: str) -> str:
        if dataset_id not in self.datasets:
            raise ValueError("Dataset not found")
        if dataset_id not in self.dataset_labels:
            self.dataset_labels[dataset_id] = []
        label = label.strip()
        if not label:
            raise ValueError("Label cannot be empty")
        if label not in self.dataset_labels[dataset_id]:
            self.dataset_labels[dataset_id] = sorted(
                set(self.dataset_labels[dataset_id]) | {label}
            )
            self._save_to_disk()
        return label

    def rename_label(self, dataset_id: str, old_label: str, new_label: str) -> int:
        count = 0
        for s in self.samples.values():
            if s["dataset_id"] == dataset_id and s["label"] == old_label:
                s["label"] = new_label
                count += 1
        if dataset_id in self.dataset_labels:
            labels = set(self.dataset_labels[dataset_id])
            labels.discard(old_label)
            labels.add(new_label)
            self.dataset_labels[dataset_id] = sorted(labels)
        if count:
            self._save_to_disk()
        return count

    def delete_label(self, dataset_id: str, label: str) -> int:
        ids = [s["id"] for s in self.samples.values()
               if s["dataset_id"] == dataset_id and s["label"] == label]
        for sid in ids:
            self.delete_sample(sid)
        if dataset_id in self.dataset_labels:
            self.dataset_labels[dataset_id] = [
                l for l in self.dataset_labels[dataset_id] if l != label
            ]
            self._save_to_disk()
        return len(ids)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        by_task: Dict[str, int] = {}
        by_label: Dict[str, int] = {}
        for sample in self.samples.values():
            by_task[sample["task"]] = by_task.get(sample["task"], 0) + 1
            by_label[sample["label"]] = by_label.get(sample["label"], 0) + 1
        return {
            "total_samples": len(self.samples),
            "by_task": by_task,
            "by_label": by_label,
        }