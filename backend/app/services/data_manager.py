import os
import uuid
import time
from typing import Dict, List, Optional
from datetime import datetime

class DataManager:
    """In-memory data management for datasets"""
    
    def __init__(self):
        self.samples: Dict[str, dict] = {}
        self.sample_data: Dict[str, bytes] = {}
        self.labels: Dict[str, List[str]] = {}
    
    def add_sample(self, label: str, task: str, data: bytes, filename: str) -> str:
        """Add a sample to the dataset"""
        sample_id = str(uuid.uuid4())
        
        self.samples[sample_id] = {
            "id": sample_id,
            "label": label,
            "task": task,
            "filename": filename,
            "timestamp": time.time()
        }
        
        self.sample_data[sample_id] = data
        
        # Track labels for task
        if task not in self.labels:
            self.labels[task] = []
        if label not in self.labels[task]:
            self.labels[task].append(label)
        
        return sample_id
    
    def get_samples(self, task: Optional[str] = None) -> List[dict]:
        """Get all samples, optionally filtered by task"""
        samples = list(self.samples.values())
        
        if task:
            samples = [s for s in samples if s["task"] == task]
        
        return samples
    
    def get_sample_data(self, sample_id: str) -> Optional[bytes]:
        """Get the raw data for a sample"""
        return self.sample_data.get(sample_id)
    
    def delete_sample(self, sample_id: str) -> bool:
        """Delete a sample"""
        if sample_id in self.samples:
            del self.samples[sample_id]
            if sample_id in self.sample_data:
                del self.sample_data[sample_id]
            return True
        return False
    
    def clear_samples(self, task: Optional[str] = None) -> int:
        """Clear samples, optionally filtered by task"""
        if task:
            sample_ids = [s["id"] for s in self.samples.values() if s["task"] == task]
        else:
            sample_ids = list(self.samples.keys())
        
        for sample_id in sample_ids:
            self.delete_sample(sample_id)
        
        return len(sample_ids)
    
    def get_statistics(self) -> dict:
        """Get dataset statistics"""
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
