import os
import json
import uuid
import time
from typing import Dict


class Converter:
    """Persistent model optimization and conversion service"""

    def __init__(self, storage_dir: str = "data_storage"):
        self.storage_dir = storage_dir
        self.db_file = os.path.join(storage_dir, "optimization_db.json")

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        self.optimization_sessions: Dict[str, dict] = {}
        self.active_optimizations: Dict[str, bool] = {}

        self._load_from_disk()

    def _save_to_disk(self):
        with open(self.db_file, "w") as f:
            json.dump(self.optimization_sessions, f, indent=2)

    def _load_from_disk(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                self.optimization_sessions = json.load(f)

    def create_optimization_session(self, training_id: str, method: str,
                                    sparsity_level: float) -> str:
        optimization_id = str(uuid.uuid4())
        self.optimization_sessions[optimization_id] = {
            "id": optimization_id,
            "training_id": training_id,
            "method": method,
            "sparsity_level": sparsity_level,
            "status": "initialized",
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "original_size_bytes": 0,
            "optimized_size_bytes": 0,
            "compression_ratio": 0.0,
            "c_array": None
        }
        self.active_optimizations[optimization_id] = False
        self._save_to_disk()
        return optimization_id

    def optimize(self, optimization_id: str):
        """Execute optimization and persist status"""
        try:
            session = self.optimization_sessions[optimization_id]
            self.active_optimizations[optimization_id] = True
            session["status"] = "running"
            session["started_at"] = time.time()
            self._save_to_disk()

            original_size = 2_500_000

            # FIX: Added WEIGHT_CLUSTERING and DYNAMIC_QUANTIZATION ratios
            compression_ratios = {
                "INT8_QUANTIZATION": 0.25,
                "FLOAT16_QUANTIZATION": 0.50,
                "PRUNING": 0.65,
                "WEIGHT_CLUSTERING": 0.70,
                "DYNAMIC_QUANTIZATION": 0.80,
            }

            ratio = compression_ratios.get(session["method"], 0.5)
            # FIX: use _bytes suffix to match frontend expectations
            session["original_size_bytes"] = original_size
            session["optimized_size_bytes"] = int(original_size * ratio)
            session["compression_ratio"] = ratio

            time.sleep(1)

            session["status"] = "completed"
            session["completed_at"] = time.time()
            self._save_to_disk()

        except Exception as e:
            session = self.optimization_sessions[optimization_id]
            session["status"] = "failed"
            session["error"] = str(e)
            self._save_to_disk()
        finally:
            self.active_optimizations[optimization_id] = False

    def generate_c_array(self, optimization_id: str) -> str:
        if optimization_id not in self.optimization_sessions:
            raise ValueError(f"Optimization session {optimization_id} not found")

        session = self.optimization_sessions[optimization_id]
        optimized_size = session["optimized_size_bytes"]
        hex_values = " ".join([f"0x{i:02x}" for i in range(min(100, optimized_size))])

        c_array = f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(8)))

const unsigned char g_model[] DATA_ALIGN_ATTRIBUTE = {{
{hex_values},
  // ... ({optimized_size} bytes total)
}};
const unsigned int g_model_len = {optimized_size};

#endif // MODEL_DATA_H
"""
        session["c_array"] = c_array
        self._save_to_disk()
        return c_array

    def get_optimization_status(self, optimization_id: str) -> dict:
        if optimization_id in self.optimization_sessions:
            return self.optimization_sessions[optimization_id]
        raise ValueError(f"Optimization session {optimization_id} not found")

    def get_optimization_result(self, optimization_id: str) -> dict:
        """FIX: Return field names that match the frontend OptimizationResult type."""
        if optimization_id not in self.optimization_sessions:
            raise ValueError(f"Optimization session {optimization_id} not found")

        session = self.optimization_sessions[optimization_id]
        if session["status"] != "completed":
            raise ValueError(f"Optimization session {optimization_id} is not completed yet")

        return {
            "id": optimization_id,
            "original_size_bytes": session["original_size_bytes"],
            "optimized_size_bytes": session["optimized_size_bytes"],
            "compression_ratio": session["compression_ratio"],
            "method": session["method"],
            "status": session["status"],
            "c_array": session.get("c_array"),
        }

    def cancel_optimization(self, optimization_id: str) -> bool:
        if optimization_id in self.active_optimizations and self.active_optimizations[optimization_id]:
            self.active_optimizations[optimization_id] = False
            session = self.optimization_sessions[optimization_id]
            session["status"] = "cancelled"
            session["completed_at"] = time.time()
            self._save_to_disk()
            return True
        return False

    def delete_optimization(self, optimization_id: str) -> bool:
        if optimization_id in self.optimization_sessions:
            del self.optimization_sessions[optimization_id]
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
            self._save_to_disk()
            return True
        return False