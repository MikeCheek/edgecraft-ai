from typing import Dict, List

class MCUAdvisor:
    """Hardware advisor for microcontroller deployment"""
    
    BOARD_SPECS = {
        "ESP32_S3_N16R8": {
            "name": "ESP32-S3 N16R8",
            "ram_kb": 8192,  # 8MB PSRAM + 512KB on-chip
            "flash_kb": 16384,  # 16MB
            "cpu": "Xtensa 32-bit",
            "features": ["PSRAM", "DSP", "Hardware FPU"],
            "recommended_models": ["MobileNetV2", "EfficientNet"]
        },
        "RASPBERRY_PI_PICO_2_W": {
            "name": "Raspberry Pi Pico 2 W",
            "ram_kb": 520,
            "flash_kb": 4096,
            "cpu": "ARM Cortex-M33",
            "features": ["DSP", "Floating Point Unit"],
            "recommended_models": ["TinyNet", "Custom3LayerCNN"]
        },
        "ARDUINO_NANO_33_BLE": {
            "name": "Arduino Nano 33 BLE",
            "ram_kb": 256,
            "flash_kb": 1024,
            "cpu": "ARM Cortex-M4",
            "features": ["Floating Point Unit"],
            "recommended_models": ["Custom3LayerCNN", "MFCC_CNN"]
        }
    }
    
    def get_supported_boards(self) -> List[dict]:
        """Get information about all supported boards"""
        return [
            {
                "id": board_id,
                **specs
            }
            for board_id, specs in self.BOARD_SPECS.items()
        ]
    
    def evaluate_model(self, optimization_id: str, board: str) -> dict:
        """Evaluate a model for a specific board"""
        if board not in self.BOARD_SPECS:
            raise ValueError(f"Unsupported board: {board}")
        
        specs = self.BOARD_SPECS[board]
        
        # TODO: Implement actual model analysis
        # Simulated evaluation
        model_size_kb = 650  # ~650KB model size
        inference_time_ms = 150
        
        ram_usage_kb = model_size_kb + 100
        flash_usage_kb = model_size_kb + 50
        
        ram_percentage = (ram_usage_kb / specs["ram_kb"]) * 100
        flash_percentage = (flash_usage_kb / specs["flash_kb"]) * 100
        
        warnings = []
        suggestions = []
        
        if ram_percentage > 80:
            warnings.append(f"High RAM usage: {ram_percentage:.1f}%")
            suggestions.append("Consider using INT8 quantization to reduce model size")
        
        if flash_percentage > 80:
            warnings.append(f"High Flash usage: {flash_percentage:.1f}%")
            suggestions.append("Model may not fit on device. Use pruning or INT8 quantization")
        
        if board == "ESP32_S3_N16R8":
            suggestions.append("Use PSRAM for model storage to free up on-chip RAM")
            suggestions.append("Enable esp-nn acceleration library for optimized 8-bit operations")
        elif board == "RASPBERRY_PI_PICO_2_W":
            suggestions.append("Use ARM Cortex-M33 DSP instructions for inference acceleration")
            if "FPU" in specs.get("features", []):
                suggestions.append("Consider FLOAT16 quantization for better accuracy with minimal overhead")
        elif board == "ARDUINO_NANO_33_BLE":
            if ram_percentage > 50:
                suggestions.append("Very limited RAM. Consider a simpler model architecture")
        
        return {
            "board": board,
            "board_name": specs["name"],
            "ram_usage_kb": ram_usage_kb,
            "flash_usage_kb": flash_usage_kb,
            "ram_percentage": ram_percentage,
            "flash_percentage": flash_percentage,
            "estimated_inference_ms": inference_time_ms,
            "warnings": warnings,
            "suggestions": suggestions,
            "deployment_feasible": flash_percentage < 90 and ram_percentage < 90
        }
