import os
import json
import base64
import uuid
import time
from typing import Dict, List, Optional

class DataManager:
    """Persistent data management for datasets"""

    def __init__(self, storage_dir: str = "data_storage"):
        self.storage_dir = storage_dir
        self.db_file = os.path.join(storage_dir, "db.json")
        
        # Ensure directory exists
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        # In-memory caches
        self.datasets: Dict[str, dict] = {}
        self.samples: Dict[str, dict] = {}
        # We store bytes in memory, but save them as files for true persistence
        self.sample_data: Dict[str, bytes] = {}
        
        self._load_from_disk()

    def _save_to_disk(self):
        """Persist metadata to JSON and raw bytes to individual files"""
        # Save Metadata
        with open(self.db_file, "w") as f:
            json.dump({
                "datasets": self.datasets,
                "samples": self.samples
            }, f, indent=2)

        # Save Binary data as individual files
        for s_id, data in self.sample_data.items():
            file_path = os.path.join(self.storage_dir, f"{s_id}.bin")
            with open(file_path, "wb") as f:
                f.write(data)

    def _load_from_disk(self):
        """Load state from disk on startup"""
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                data = json.load(f)
                self.datasets = data.get("datasets", {})
                self.samples = data.get("samples", {})
        
        # Load binary data files
        for s_id in self.samples:
            file_path = os.path.join(self.storage_dir, f"{s_id}.bin")
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    self.sample_data[s_id] = f.read()

    def create_dataset(self, name: str, task: str) -> dict:
        dataset_id = str(uuid.uuid4())
        dataset = {
            "id": dataset_id,
            "name": name,
            "task": task,
            "sample_count": 0,
            "created_at": time.time()
        }
        self.datasets[dataset_id] = dataset
        self._save_to_disk()
        return dataset

    def add_sample(self, dataset_id: str, label: str, task: str, data: bytes, filename: str) -> str:
        if dataset_id not in self.datasets:
            raise ValueError("Dataset not found")
            
        sample_id = str(uuid.uuid4())
        self.samples[sample_id] = {
            "id": sample_id,
            "dataset_id": dataset_id,
            "label": label,
            "task": task,
            "filename": filename,
            "timestamp": time.time()
        }
        self.sample_data[sample_id] = data
        self.datasets[dataset_id]["sample_count"] += 1
        
        self._save_to_disk()
        return sample_id

    def delete_sample(self, sample_id: str) -> bool:
        if sample_id in self.samples:
            dataset_id = self.samples[sample_id]["dataset_id"]
            if dataset_id in self.datasets:
                self.datasets[dataset_id]["sample_count"] = max(0, self.datasets[dataset_id]["sample_count"] - 1)
                
            del self.samples[sample_id]
            if sample_id in self.sample_data:
                del self.sample_data[sample_id]
            
            # Remove the binary file from disk
            file_path = os.path.join(self.storage_dir, f"{sample_id}.bin")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            self._save_to_disk()
            return True
        return False

    def delete_dataset(self, dataset_id: str) -> bool:
        if dataset_id in self.datasets:
            self.clear_dataset_samples(dataset_id)
            del self.datasets[dataset_id]
            self._save_to_disk()
            return True
        return False

    def get_datasets(self, task: Optional[str] = None) -> List[dict]:
        datasets = list(self.datasets.values())
        if task:
            datasets = [d for d in datasets if d["task"] == task]
        return datasets
        
    def rename_dataset(self, dataset_id: str, new_name: str) -> bool:
        if dataset_id in self.datasets:
            self.datasets[dataset_id]["name"] = new_name
            self._save_to_disk()
            return True
        return False

    def get_samples(self, dataset_id: Optional[str] = None) -> List[dict]:
        samples = list(self.samples.values())
        if dataset_id:
            samples = [s for s in samples if s["dataset_id"] == dataset_id]
        return samples

    def get_sample_data(self, sample_id: str) -> Optional[bytes]:
        return self.sample_data.get(sample_id)


    def clear_dataset_samples(self, dataset_id: str) -> int:
        sample_ids = [s["id"] for s in self.samples.values() if s["dataset_id"] == dataset_id]
        for s_id in sample_ids:
            self.delete_sample(s_id)
        return len(sample_ids)

    def get_statistics(self) -> dict:
        by_task = {}
        by_label = {}

        for sample in self.samples.values():
            task = sample["task"]
            label = sample["label"]
            by_task[task] = by_task.get(task, 0) + 1
            by_label[label] = by_label.get(label, 0) + 1

        return {
            "total_samples": len(self.samples),
            "by_task": by_task,
            "by_label": by_label
        }