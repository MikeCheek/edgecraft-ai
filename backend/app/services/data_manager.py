import os
import json
import uuid
import time
import random
import contextlib
from typing import Dict, List, Optional, Set, Tuple

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

        # NEW: in-memory indices, never persisted to disk - rebuilt from
        # self.samples on every load. These turn every dataset-scoped (or
        # label-scoped) lookup from an O(total_samples) scan into an
        # O(matching_samples) lookup. They're what get_samples(),
        # _sync_labels(), get_dataset_image_stats(), delete_label(), and
        # rename_label() all use now instead of iterating self.samples.
        self.samples_by_dataset: Dict[str, Set[str]] = {}
        self.samples_by_label: Dict[Tuple[str, str], Set[str]] = {}

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
    # Index maintenance (NEW)
    # ------------------------------------------------------------------
    # Kept private and called from every mutation site so self.samples and
    # the indices never drift apart. Rebuilt from scratch on load, updated
    # incrementally everywhere else.

    def _index_add(self, sample_id: str, dataset_id: str, label: str) -> None:
        self.samples_by_dataset.setdefault(dataset_id, set()).add(sample_id)
        self.samples_by_label.setdefault((dataset_id, label), set()).add(sample_id)

    def _index_remove(self, sample_id: str, dataset_id: str, label: str) -> None:
        bucket = self.samples_by_dataset.get(dataset_id)
        if bucket is not None:
            bucket.discard(sample_id)
            if not bucket:
                del self.samples_by_dataset[dataset_id]
        label_bucket = self.samples_by_label.get((dataset_id, label))
        if label_bucket is not None:
            label_bucket.discard(sample_id)
            if not label_bucket:
                del self.samples_by_label[(dataset_id, label)]

    def _index_relabel(self, sample_id: str, dataset_id: str, old_label: str, new_label: str) -> None:
        if old_label == new_label:
            return
        old_bucket = self.samples_by_label.get((dataset_id, old_label))
        if old_bucket is not None:
            old_bucket.discard(sample_id)
            if not old_bucket:
                del self.samples_by_label[(dataset_id, old_label)]
        self.samples_by_label.setdefault((dataset_id, new_label), set()).add(sample_id)

    def _rebuild_indices(self) -> None:
        self.samples_by_dataset = {}
        self.samples_by_label = {}
        for sample_id, s in self.samples.items():
            self._index_add(sample_id, s["dataset_id"], s["label"])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_metadata(self):
        """Persist only JSON metadata (datasets, samples, labels).

        Uses atomic write (write to temp file then rename) to prevent
        corruption if the process crashes mid-write.
        """
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(self.db_file) or ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(
                    {
                        "datasets": self.datasets,
                        "samples": self.samples,
                        "dataset_labels": self.dataset_labels,
                    },
                    f,
                    indent=2,
                )
            os.replace(tmp_path, self.db_file)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)
            raise

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
        # NEW: (re)build the in-memory indices from whatever we just loaded
        # (or from an empty self.samples on a fresh install).
        self._rebuild_indices()

    def _sync_labels(self, dataset_id: str):
        # FIX: was `s["dataset_id"] == dataset_id` over ALL samples
        # (O(total_samples)). Now walks only this dataset's sample ids via
        # the index (O(samples_in_dataset)).
        existing = {
            self.samples[sid]["label"]
            for sid in self.samples_by_dataset.get(dataset_id, ())
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
        annotations: Optional[list] = None,
    ) -> str:
        if dataset_id not in self.datasets:
            raise ValueError("Dataset not found")
        sample_id = str(uuid.uuid4())
        sample_dict = {
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
        if annotations:
            sample_dict["annotations"] = annotations
        self.samples[sample_id] = sample_dict
        # Write binary first, then update metadata - avoids orphaned records
        self._write_sample_file(sample_id, data)
        self._index_add(sample_id, dataset_id, label)

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
            label = item["label"]

            sample_dict = {
                "id": sample_id,
                "dataset_id": dataset_id,
                "label": label,
                "task": task,
                "filename": item["filename"],
                "timestamp": time.time(),
                "split": split,
                "size_bytes": len(content),
                "width": item.get("width"),
                "height": item.get("height"),
            }
            annotations = item.get("annotations")
            if annotations:
                sample_dict["annotations"] = annotations
            self.samples[sample_id] = sample_dict
            # Write the binary immediately - one file, one write, done.
            self._write_sample_file(sample_id, content)
            self._index_add(sample_id, dataset_id, label)
            sample_ids.append(sample_id)
            new_labels.add(label)

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

        sample = self.samples[sample_id]
        dataset_id = sample["dataset_id"]
        label = sample["label"]
        if dataset_id in self.datasets:
            self.datasets[dataset_id]["sample_count"] = max(
                0, self.datasets[dataset_id]["sample_count"] - 1
            )

        del self.samples[sample_id]
        self._index_remove(sample_id, dataset_id, label)

        file_path = os.path.join(self.storage_dir, f"{sample_id}.bin")

        # EAFP approach: faster than os.path.exists followed by os.remove
        with contextlib.suppress(FileNotFoundError):
            os.remove(file_path)

        if save_metadata:
            self._save_metadata()

        return True

    def get_samples(self, dataset_id: Optional[str] = None) -> List[dict]:
        # FIX: was a full scan + filter over every sample in the store
        # (O(total_samples)) even when asking for one small dataset.
        # Now an O(samples_in_dataset) index lookup.
        if dataset_id:
            return [
                self.samples[sid]
                for sid in self.samples_by_dataset.get(dataset_id, ())
            ]
        return list(self.samples.values())

    def get_sample_data(self, sample_id: str) -> Optional[bytes]:
        """Read sample binary from disk on demand (no RAM cache)."""
        file_path = os.path.join(self.storage_dir, f"{sample_id}.bin")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return f.read()
        return None

    def clear_dataset_samples(self, dataset_id: str) -> int:
        # FIX (perf): this used to call delete_sample() in a loop with its
        # default save_metadata=True, so wiping a 1,000-sample dataset did
        # 1,000 full rewrites of db.json. Now every delete in the loop skips
        # the save, and we flush once at the end. Also now sourced from the
        # index instead of a full scan of self.samples.
        sample_ids = list(self.samples_by_dataset.get(dataset_id, ()))
        for s_id in sample_ids:
            self.delete_sample(s_id, save_metadata=False)
        if sample_ids:
            self._save_metadata()
        return len(sample_ids)

    def relabel_sample(self, sample_id: str, new_label: str) -> bool:
        if sample_id not in self.samples:
            return False
        dataset_id = self.samples[sample_id]["dataset_id"]
        old_label = self.samples[sample_id]["label"]
        self.samples[sample_id]["label"] = new_label
        self._index_relabel(sample_id, dataset_id, old_label, new_label)
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
        # FIX: was a full scan over every sample in the store to find this
        # dataset's samples; now an index lookup.
        samples = [self.samples[sid] for sid in self.samples_by_dataset.get(dataset_id, ())]
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
        # FIX (bug + perf): this used to iterate ALL samples across every
        # dataset and filter by dataset_id inline - O(total_samples) for a
        # single-dataset answer. On an install with many datasets this was
        # by far the worst offender, since it's called on every dataset
        # screen load. Now O(samples_in_dataset) via the index.
        summary = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
        for sid in self.samples_by_dataset.get(dataset_id, ()):
            s = self.samples[sid]
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
        # FIX: was a full scan over every sample in the store; now uses the
        # (dataset_id, label) index to touch only the matching samples.
        ids = list(self.samples_by_label.get((dataset_id, old_label), ()))
        for sid in ids:
            self.samples[sid]["label"] = new_label
        if ids:
            self.samples_by_label.pop((dataset_id, old_label), None)
            self.samples_by_label.setdefault((dataset_id, new_label), set()).update(ids)

        if dataset_id in self.dataset_labels:
            labels = set(self.dataset_labels[dataset_id])
            labels.discard(old_label)
            labels.add(new_label)
            self.dataset_labels[dataset_id] = sorted(labels)
        if ids:
            self._save_metadata()
        return len(ids)

    def delete_label(self, dataset_id: str, label: str) -> int:
        # FIX (perf + the same "N deletes, N saves" issue as
        # clear_dataset_samples): was scanning all samples to find matches,
        # then calling delete_sample() in a loop with its default
        # save_metadata=True. Now sourced from the (dataset_id, label)
        # index and saved once.
        ids = list(self.samples_by_label.get((dataset_id, label), ()))
        for sid in ids:
            self.delete_sample(sid, save_metadata=False)
        if dataset_id in self.dataset_labels:
            self.dataset_labels[dataset_id] = [
                lbl for lbl in self.dataset_labels[dataset_id] if lbl != label
            ]
        if ids or dataset_id in self.dataset_labels:
            self._save_metadata()
        return len(ids)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        # Genuinely global aggregates (across every dataset), so there's no
        # index shortcut here - this one has to touch every sample. Left
        # as a plain scan; if this ever gets called on a hot path, the next
        # step would be incremental counters updated in add/delete/relabel
        # rather than recomputing from scratch each call.
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

        # FIX: was a full scan over every sample in the store; now uses the
        # dataset index.
        samples = [self.samples[sid] for sid in self.samples_by_dataset.get(dataset_id, ())]
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

        # FIX: was iterating every sample in the store and filtering by
        # dataset_id inline; now restricted to this dataset's samples via
        # the index, and the label index is kept in sync per-match.
        for sid in list(self.samples_by_dataset.get(dataset_id, ())):
            s = self.samples[sid]
            match = compiled_regex.search(s["filename"])
            if match:
                new_label = match.group(1) if match.groups() else match.group(0)
                if new_label != s["label"]:
                    old_label = s["label"]
                    s["label"] = new_label
                    self._index_relabel(sid, dataset_id, old_label, new_label)
                    new_labels.add(new_label)
                    count += 1

        if count > 0:
            if dataset_id not in self.dataset_labels:
                self.dataset_labels[dataset_id] = []
            combined_labels = set(self.dataset_labels[dataset_id]) | new_labels
            self.dataset_labels[dataset_id] = sorted(combined_labels)
            self._save_metadata()

        return count

    def update_sample_data(
        self,
        sample_id: str,
        data: bytes,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> bool:
        """Overwrite a sample's binary content in place - used by the
        Dataset Explorer's crop tool. Keeps id/dataset_id/label/split
        unchanged; only size_bytes (and width/height, if the new content
        was successfully re-probed) are refreshed.

        Sets `updated_at` so the frontend can cache-bust the image URL -
        GET /image/{sample_id} serves an `immutable` Cache-Control header
        keyed by sample_id, which never changes here, so without a
        version query param the browser would keep showing the pre-crop
        bytes indefinitely.
        """
        if sample_id not in self.samples:
            return False
        self._write_sample_file(sample_id, data)
        self.samples[sample_id]["size_bytes"] = len(data)
        if width is not None:
            self.samples[sample_id]["width"] = width
        if height is not None:
            self.samples[sample_id]["height"] = height
        self.samples[sample_id]["updated_at"] = time.time()
        self._save_metadata()
        return True

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def get_sample_annotations(self, sample_id: str) -> Optional[list]:
        """Get bounding box annotations for a single sample."""
        sample = self.samples.get(sample_id)
        if not sample:
            return None
        return sample.get("annotations")

    def update_sample_annotations(self, sample_id: str, annotations: list) -> bool:
        """Replace bounding box annotations for a single sample."""
        if sample_id not in self.samples:
            return False
        self.samples[sample_id]["annotations"] = annotations
        self.samples[sample_id]["updated_at"] = time.time()
        self._save_metadata()
        return True

    def get_dataset_annotation_summary(self, dataset_id: str) -> dict:
        """Summarize annotation coverage for a dataset.

        Returns:
            {
                "has_annotations": bool,
                "annotated_count": int,
                "total_count": int,
                "format": str | None,
                "classes": [{"name": str, "count": int}],
                "total_bboxes": int,
            }
        """
        sample_ids = list(self.samples_by_dataset.get(dataset_id, ()))
        annotated = 0
        total_bboxes = 0
        class_counts: Dict[str, int] = {}
        fmt = None

        for sid in sample_ids:
            s = self.samples[sid]
            anns = s.get("annotations")
            if anns:
                annotated += 1
                total_bboxes += len(anns)
                for a in anns:
                    cn = a.get("class_name", "unknown")
                    class_counts[cn] = class_counts.get(cn, 0) + 1

        dataset = self.datasets.get(dataset_id, {})
        annotation_format = dataset.get("metadata", {}).get("annotation_format")

        return {
            "has_annotations": annotated > 0,
            "annotated_count": annotated,
            "total_count": len(sample_ids),
            "format": annotation_format,
            "classes": sorted(
                [{"name": k, "count": v} for k, v in class_counts.items()],
                key=lambda x: x["count"],
                reverse=True,
            ),
            "total_bboxes": total_bboxes,
        }

    def update_dataset_annotation_metadata(
        self, dataset_id: str, annotation_format: str, classes: list
    ) -> None:
        """Store detected annotation format and class list on the dataset."""
        if dataset_id not in self.datasets:
            return
        ds = self.datasets[dataset_id]
        ds.setdefault("metadata", {})
        ds["metadata"]["annotation_format"] = annotation_format
        ds["metadata"]["annotation_classes"] = classes
        self._save_metadata()