from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any, Generic, TypeVar

T = TypeVar("T")

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

QuantizationMethod = Literal[
    "INT8_QUANTIZATION",
    "FLOAT16_QUANTIZATION",
    "PRUNING",
    "WEIGHT_CLUSTERING",
    "DYNAMIC_QUANTIZATION" 
]

DatasetSplit = Literal["train", "val", "test", "unassigned"]

class DatasetInfo(BaseModel):
    """Information about a created dataset"""
    id: str
    name: str
    task: TaskType
    sample_count: int
    created_at: float

class DatasetSample(BaseModel):
    """Single dataset sample"""
    id: str
    dataset_id: str
    label: str
    task: TaskType
    filename: str
    timestamp: float
    split: DatasetSplit = "unassigned"

class TrainingConfig(BaseModel):
    """Training configuration"""
    dataset_id: str
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    base_model: str = "MobileNetV2"
    task: TaskType
    validation_split: float = 0.2

class ModelMetadata(BaseModel):
    """Model metadata for trained models"""
    id: str
    name: str
    training_id: str
    task: TaskType
    created_at: float
    accuracy: float
    loss: float
    val_accuracy: float
    val_loss: float
    optimized: bool
    size_bytes: int

class TrainingMetrics(BaseModel):
    """Training metrics for a single epoch"""
    epoch: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float
    timestamp: float

class TrainingStatus(BaseModel):
    """Real-time status of a training session"""
    id: str
    status: Literal['initialized', 'running', 'completed', 'failed', 'cancelled']
    current_epoch: int
    total_epochs: int
    progress: float
    created_at: float
    started_at: Optional[float] = None
    metrics: List[TrainingMetrics] = []

class OptimizationResult(BaseModel):
    """Result of an optimization/quantization session"""
    id: str
    original_size_bytes: int
    optimized_size_bytes: int
    compression_ratio: float
    method: QuantizationMethod
    status: Literal['initialized', 'running', 'completed', 'failed']
    c_array: Optional[str] = None
    cpp_wrapper: Optional[str] = None

class BoardRecommendation(BaseModel):
    """Board-specific hardware evaluation"""
    board: BoardType
    board_name: str
    ram_usage_kb: int
    flash_usage_kb: int
    ram_percentage: float
    flash_percentage: float
    warnings: List[str] = []
    suggestions: List[str] = []
    estimated_inference_ms: float
    deployment_feasible: bool

class LLMSuggestion(BaseModel):
    """LLM-powered suggestion for model improvement"""
    suggestion: str
    reasoning: str
    parameters_to_adjust: Dict[str, Any] = {}
    estimated_improvement: str

class DatasetStatistics(BaseModel):
    """Global statistics across all datasets"""
    total_samples: int
    by_task: Dict[str, int] = {}
    by_label: Dict[str, int] = {}

class ApiResponse(BaseModel, Generic[T]):
    """Standardized API Response wrapper"""
    status: Literal['success', 'error']
    data: Optional[T] = None
    message: Optional[str] = None
    error: Optional[str] = None