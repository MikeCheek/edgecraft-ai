import os
import json
import uuid
import time
import random
import contextlib
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
        # FIX: sample_data is NO LONGER held in RAM.
        # Binary files are read on-demand via get_sample_data().
        # Keeping a full dict of all image bytes caused _save_to_disk() to
        # rewrite every .bin file on every single upload, which blocked the
        # event loop long enough to hit the frontend 30 s axios timeout.
        self.dataset_labels: Dict[str, List[str]] = {}

        self._load_from_disk()

    def _get_compatible_tasks(self, task: str) -> set:
        """Group tasks by their underlying data modality."""
        image_tasks = {"IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "VISUAL_WAKE_WORDS"}
        audio_tasks = {"KEYWORD_SPOTTING", "AUDIO_CLASSIFICATION"}

        if task in image_tasks:
            return image_tasks
        if task in audio_tasks:
            return audio_tasks
        return {task}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_metadata(self):
        """Persist only JSON metadata (datasets, samples, labels).

        Previously _save_to_disk() also looped over self.sample_data and
        rewrote every .bin file already on disk - O(n) disk writes on every
        single upload.  Binary files are now written once on ingest and never
        touched again unless the sample is deleted.
        """
        with open(self.db_file, "w") as f:
            json.dump(
                {
                    "datasets": self.datasets,
                    "samples": self.samples,
                    "dataset_labels": self.dataset_labels,
                },
                f,
                indent=2,
            )

    def _save_to_disk(self):
        """Alias kept for callers that still use the old name."""
        self._save_metadata()

    def _write_sample_file(self, sample_id: str, data: bytes):
        """Write a single sample binary to disk (called once, at ingest)."""
        file_path = os.path.join(self.storage_dir, f"{sample_id}.bin")
        with open(file_path, "wb") as f:
            f.write(data)

    def _load_from_disk(self):
        """Load metadata only.  Binary data stays on disk until requested."""
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                data = json.load(f)
                self.datasets = data.get("datasets", {})
                self.samples = data.get("samples", {})
                self.dataset_labels = data.get("dataset_labels", {})

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

    def create_dataset(self, name: str, task: str, description: str = "") -> dict:
        dataset_id = str(uuid.uuid4())
        dataset = {
            "id": dataset_id,
            "name": name,
            "task": task,
            "sample_count": 0,
            "created_at": time.time(),
            # NEW: free-form metadata the user (or an LLM prompt builder) can
            # attach to a dataset. `description` is user-editable text.
            # `metadata` is a small bag for anything else (e.g. cached image
            # stats) so we don't have to keep adding top-level columns.
            "description": description or "",
            "metadata": {},
        }
        self.datasets[dataset_id] = dataset
        self.dataset_labels[dataset_id] = []
        self._save_metadata()
        return dataset

    def rename_dataset(self, dataset_id: str, new_name: str) -> bool:
        if dataset_id not in self.datasets:
            return False
        self.datasets[dataset_id]["name"] = new_name
        self._save_metadata()
        return True

    def update_dataset_metadata(
        self,
        dataset_id: str,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """Update the free-form description and/or metadata bag for a dataset.

        `metadata` is shallow-merged into the existing metadata dict rather
        than replacing it wholesale, so callers can update a single key
        (e.g. just `image_stats`) without clobbering others (e.g. a
        previously-set `notes` field).
        """
        if dataset_id not in self.datasets:
            return None
        dataset = self.datasets[dataset_id]
        if "metadata" not in dataset:
            dataset["metadata"] = {}
        if description is not None:
            dataset["description"] = description
        if metadata:
            dataset["metadata"].update(metadata)
        self._save_metadata()
        return dataset

    def delete_dataset(self, dataset_id: str) -> bool:
        if dataset_id not in self.datasets:
            return False
        self.clear_dataset_samples(dataset_id)
        del self.datasets[dataset_id]
        self.dataset_labels.pop(dataset_id, None)
        self._save_metadata()
        return True

    def get_datasets(self, task: Optional[str] = None) -> List[dict]:
        datasets = list(self.datasets.values())
        if task:
            compatible_tasks = self._get_compatible_tasks(task)
            datasets = [d for d in datasets if d["task"] in compatible_tasks]
        return datasets

    def get_dataset(self, dataset_id: str) -> Optional[dict]:
        return self.datasets.get(dataset_id)

    # ------------------------------------------------------------------
    # Samples & Splits
    # ------------------------------------------------------------------

    def add_sample(
        self,
        dataset_id: str,
        label: str,
        task: str,
        data: bytes,
        filename: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
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
            "size_bytes": len(data),
            "width": width,
            "height": height,
        }
        # Write binary first, then update metadata - avoids orphaned records
        self._write_sample_file(sample_id, data)

        self.datasets[dataset_id]["sample_count"] += 1

        if dataset_id not in self.dataset_labels:
            self.dataset_labels[dataset_id] = []
        if label not in self.dataset_labels[dataset_id]:
            self.dataset_labels[dataset_id] = sorted(
                set(self.dataset_labels[dataset_id]) | {label}
            )
        self._save_metadata()
        return sample_id

    def bulk_add_samples(
        self, dataset_id: str, task: str, items: List[dict]
    ) -> List[str]:
        """Add multiple samples efficiently.

        FIX: binary files are written one-by-one during ingest, not batched
        into self.sample_data and then re-flushed along with every previously
        stored file.  Metadata is saved once at the end.

        Each item may optionally carry "width"/"height" (already-probed
        image dimensions from the zip extractor) so per-dataset image-size
        stats can be computed later without re-reading every file from disk.
        """
        if dataset_id not in self.datasets:
            raise ValueError("Dataset not found")

        sample_ids = []
        new_labels: set = set()

        for item in items:
            sample_id = str(uuid.uuid4())

            split = item.get("split", "unassigned")
            content = item["content"]

            self.samples[sample_id] = {
                "id": sample_id,
                "dataset_id": dataset_id,
                "label": item["label"],
                "task": task,
                "filename": item["filename"],
                "timestamp": time.time(),
                "split": split,
                "size_bytes": len(content),
                "width": item.get("width"),
                "height": item.get("height"),
            }
            # Write the binary immediately - one file, one write, done.
            self._write_sample_file(sample_id, content)
            sample_ids.append(sample_id)
            new_labels.add(item["label"])

        self.datasets[dataset_id]["sample_count"] += len(items)

        if dataset_id not in self.dataset_labels:
            self.dataset_labels[dataset_id] = []

        combined_labels = set(self.dataset_labels[dataset_id]) | new_labels
        self.dataset_labels[dataset_id] = sorted(combined_labels)

        # Single metadata flush for the whole batch
        self._save_metadata()
        return sample_ids

    def delete_sample(self, sample_id: str, save_metadata: bool = True) -> bool:
        if sample_id not in self.samples:
            return False

        dataset_id = self.samples[sample_id]["dataset_id"]
        if dataset_id in self.datasets:
            self.datasets[dataset_id]["sample_count"] = max(
                0, self.datasets[dataset_id]["sample_count"] - 1
            )

        del self.samples[sample_id]

        file_path = os.path.join(self.storage_dir, f"{sample_id}.bin")

        # EAFP approach: faster than os.path.exists followed by os.remove
        with contextlib.suppress(FileNotFoundError):
            os.remove(file_path)

        if save_metadata:
            self._save_metadata()

        return True

    def get_samples(self, dataset_id: Optional[str] = None) -> List[dict]:
        samples = list(self.samples.values())
        if dataset_id:
            samples = [s for s in samples if s["dataset_id"] == dataset_id]
        return samples

    def get_sample_data(self, sample_id: str) -> Optional[bytes]:
        """Read sample binary from disk on demand (no RAM cache)."""
        file_path = os.path.join(self.storage_dir, f"{sample_id}.bin")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return f.read()
        return None

    def clear_dataset_samples(self, dataset_id: str) -> int:
        sample_ids = [
            s["id"] for s in self.samples.values() if s["dataset_id"] == dataset_id
        ]
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
        self._save_metadata()
        return True

    def set_sample_split(self, sample_id: str, split: str) -> bool:
        if sample_id in self.samples:
            self.samples[sample_id]["split"] = split
            self._save_metadata()
            return True
        return False

    def auto_split_dataset(
        self, dataset_id: str, train_pct: int, val_pct: int, test_pct: int
    ) -> int:
        """Randomly divide the dataset maintaining class distribution."""
        samples = [s for s in self.samples.values() if s["dataset_id"] == dataset_id]
        by_label: Dict[str, list] = {}
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
            self._save_metadata()
        return count

    def get_split_summary(self, dataset_id: str) -> Dict[str, int]:
        """Count samples per split for a single dataset (used by the dataset
        screen warning badge and to gate training on a precomputed split)."""
        if dataset_id not in self.datasets:
            raise ValueError("Dataset not found")
        summary = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
        for s in self.samples.values():
            if s["dataset_id"] == dataset_id:
                summary[s.get("split", "unassigned")] += 1
        return summary

    def is_split_ready(self, dataset_id: str) -> bool:
        """True when every sample is assigned and both train and val are non-empty."""
        summary = self.get_split_summary(dataset_id)
        return summary["unassigned"] == 0 and summary["train"] > 0 and summary["val"] > 0

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
            self._save_metadata()
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
            self._save_metadata()
        return count

    def delete_label(self, dataset_id: str, label: str) -> int:
        ids = [
            s["id"]
            for s in self.samples.values()
            if s["dataset_id"] == dataset_id and s["label"] == label
        ]
        for sid in ids:
            self.delete_sample(sid)
        if dataset_id in self.dataset_labels:
            self.dataset_labels[dataset_id] = [
                lbl for lbl in self.dataset_labels[dataset_id] if lbl != label
            ]
            self._save_metadata()
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

    def get_dataset_image_stats(self, dataset_id: str) -> dict:
        """Aggregate image-size / aspect-ratio / storage stats for a dataset.

        Uses the width/height/size_bytes already captured at ingest time
        (see zip_processor.py and the /upload endpoint), so this is just an
        aggregation over already-known numbers - no re-reading of image
        files from disk. Samples ingested before this feature existed will
        have width/height = None and are simply excluded from the
        dimension-based stats (they still count toward total size).

        The result is also cached onto the dataset's `metadata.image_stats`
        so it can be included cheaply in LLM prompt context; it's
        recomputed (and re-cached) every time this is called since it's
        called on-demand, not on every sample add.
        """
        if dataset_id not in self.datasets:
            raise ValueError("Dataset not found")

        samples = [s for s in self.samples.values() if s["dataset_id"] == dataset_id]
        widths = [s["width"] for s in samples if s.get("width")]
        heights = [s["height"] for s in samples if s.get("height")]
        sizes = [s["size_bytes"] for s in samples if s.get("size_bytes")]

        ext_counts: Dict[str, int] = {}
        for s in samples:
            ext = os.path.splitext(s.get("filename", ""))[1].lower() or "unknown"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        stats = {
            "total_samples": len(samples),
            "samples_with_dimensions": len(widths),
            "formats": ext_counts,
            "total_size_bytes": sum(sizes) if sizes else 0,
            "avg_size_bytes": round(sum(sizes) / len(sizes), 1) if sizes else None,
        }

        if widths and heights:
            ratios = [w / h for w, h in zip(widths, heights) if h]
            resolution_counts: Dict[str, int] = {}
            for w, h in zip(widths, heights):
                key = f"{w}x{h}"
                resolution_counts[key] = resolution_counts.get(key, 0) + 1
            most_common_resolutions = sorted(
                resolution_counts.items(), key=lambda kv: kv[1], reverse=True
            )[:5]

            stats.update({
                "width": {"min": min(widths), "max": max(widths), "avg": round(sum(widths) / len(widths), 1)},
                "height": {"min": min(heights), "max": max(heights), "avg": round(sum(heights) / len(heights), 1)},
                "aspect_ratio": {
                    "min": round(min(ratios), 3),
                    "max": round(max(ratios), 3),
                    "avg": round(sum(ratios) / len(ratios), 3),
                },
                "most_common_resolutions": [
                    {"resolution": res, "count": cnt} for res, cnt in most_common_resolutions
                ],
                "uniform_dimensions": len(set(zip(widths, heights))) == 1,
            })
        else:
            stats.update({
                "width": None,
                "height": None,
                "aspect_ratio": None,
                "most_common_resolutions": [],
                "uniform_dimensions": None,
            })

        # Cache onto the dataset record so other callers (e.g. the LLM
        # advisor) can read it cheaply without recomputing.
        if dataset_id in self.datasets:
            self.datasets[dataset_id].setdefault("metadata", {})
            self.datasets[dataset_id]["metadata"]["image_stats"] = stats
            self.datasets[dataset_id]["metadata"]["image_stats_computed_at"] = time.time()
            self._save_metadata()

        return stats

    def bulk_relabel_by_regex(self, dataset_id: str, regex_pattern: str) -> int:
        import re
        try:
            compiled_regex = re.compile(regex_pattern)
        except re.error:
            raise ValueError("Invalid regular expression")

        count = 0
        new_labels = set()

        for s in self.samples.values():
            if s["dataset_id"] == dataset_id:
                match = compiled_regex.search(s["filename"])
                if match:
                    new_label = match.group(1) if match.groups() else match.group(0)
                    if new_label != s["label"]:
                        s["label"] = new_label
                        new_labels.add(new_label)
                        count += 1

        if count > 0:
            if dataset_id not in self.dataset_labels:
                self.dataset_labels[dataset_id] = []
            combined_labels = set(self.dataset_labels[dataset_id]) | new_labels
            self.dataset_labels[dataset_id] = sorted(combined_labels)
            self._save_metadata()

        return count
