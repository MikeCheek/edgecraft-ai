from pydantic import BaseModel
from typing import List, Optional, Literal

# Task Types
TaskType = Literal[
    "IMAGE_CLASSIFICATION",
    "OBJECT_DETECTION",
    "VISUAL_WAKE_WORDS",
    "KEYWORD_SPOTTING",
    "AUDIO_CLASSIFICATION"
]

# Board Types
BoardType = Literal[
    "ESP32_S3_N16R8",
    "RASPBERRY_PI_PICO_2_W",
    "ARDUINO_NANO_33_BLE"
]

# Quantization Methods
QuantizationMethod = Literal[
    "INT8_QUANTIZATION",
    "FLOAT16_QUANTIZATION",
    "PRUNING"
]

class DatasetSample(BaseModel):
    """Single dataset sample"""
    label: str
    file_path: str
    task: TaskType
    timestamp: float

class TrainingConfig(BaseModel):
    """Training configuration"""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    base_model: str = "MobileNetV2"
    task: TaskType
    validation_split: float = 0.2

class OptimizationConfig(BaseModel):
    """Optimization configuration"""
    method: QuantizationMethod
    sparsity_level: float = 0.5  # For pruning
    representative_dataset_size: int = 100

class BoardInfo(BaseModel):
    """Target board information"""
    board: BoardType
    ram_kb: int
    flash_kb: int
    dsp_enabled: bool = True

class ModelMetadata(BaseModel):
    """Model metadata"""
    name: str
    task: TaskType
    created_at: float
    input_shape: tuple
    output_shape: tuple
    accuracy: float
    loss: float
    optimized: bool = False
    size_bytes: int

class TrainingMetrics(BaseModel):
    """Training metrics for a single epoch"""
    epoch: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float
    timestamp: float

class OptimizationResult(BaseModel):
    """Result of optimization"""
    original_size_bytes: int
    optimized_size_bytes: int
    compression_ratio: float
    method: QuantizationMethod
    c_array_preview: str

class BoardRecommendation(BaseModel):
    """Board-specific recommendation"""
    board: BoardType
    ram_usage_kb: int
    flash_usage_kb: int
    ram_percentage: float
    flash_percentage: float
    warnings: List[str]
    suggestions: List[str]
    estimated_inference_ms: float

class LLMSuggestion(BaseModel):
    """LLM-powered suggestion for model improvement"""
    suggestion: str
    reasoning: str
    parameters_to_adjust: dict
    estimated_improvement: str
