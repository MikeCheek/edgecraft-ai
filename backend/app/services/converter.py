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
        """Persist optimization sessions to JSON"""
        with open(self.db_file, "w") as f:
            json.dump(self.optimization_sessions, f, indent=2)

    def _load_from_disk(self):
        """Load state from disk on startup"""
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
            "original_size": 0,
            "optimized_size": 0,
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
            
            original_size = 2500000 
            compression_ratios = {
                "INT8_QUANTIZATION": 0.25,
                "FLOAT16_QUANTIZATION": 0.50,
                "PRUNING": 0.65
            }
            
            ratio = compression_ratios.get(session["method"], 0.5)
            session["original_size"] = original_size
            session["optimized_size"] = int(original_size * ratio)
            session["compression_ratio"] = ratio
            
            time.sleep(1) 
            
            session["status"] = "completed"
            session["completed_at"] = time.time()
            self._save_to_disk()
        
        except Exception as e:
            session["status"] = "failed"
            session["error"] = str(e)
            self._save_to_disk()
        finally:
            self.active_optimizations[optimization_id] = False
    
    def generate_c_array(self, optimization_id: str) -> str:
        if optimization_id not in self.optimization_sessions:
            raise ValueError(f"Optimization session {optimization_id} not found")
        
        session = self.optimization_sessions[optimization_id]
        
        hex_values = " ".join([f"0x{i:02x}" for i in range(min(100, session["optimized_size"]))])
        
        c_array = f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(8)))

const unsigned char g_model[] DATA_ALIGN_ATTRIBUTE = {{
{hex_values},
  // ... ({session['optimized_size']} bytes total)
}};
const unsigned int g_model_len = {session['optimized_size']};

#endif // MODEL_DATA_H
"""
        session["c_array"] = c_array
        self._save_to_disk() # Save the generated C-array to persistence
        return c_array
    
    def get_optimization_status(self, optimization_id: str) -> dict:
        if optimization_id in self.optimization_sessions:
            return self.optimization_sessions[optimization_id]
        else:
            raise ValueError(f"Optimization session {optimization_id} not found")
        
    def get_optimization_result(self, optimization_id: str) -> dict:
        if optimization_id in self.optimization_sessions:
            session = self.optimization_sessions[optimization_id]
            if session["status"] == "completed":
                return {
                    "original_size": session["original_size"],
                    "optimized_size": session["optimized_size"],
                    "compression_ratio": session["compression_ratio"]
                }
            else:
                raise ValueError(f"Optimization session {optimization_id} is not completed yet")
        else:
            raise ValueError(f"Optimization session {optimization_id} not found")
        
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