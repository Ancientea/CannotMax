from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class OptimizerType(Enum):
    Sgd = "SGD"
    Adam = "Adam"
    AdamW = "AdamW"
    RMSprop = "RMSprop"


class TrainingPhase(Enum):
    Train = "Train"
    Validation = "Validation"
    LrSchedule = "LRSchedule"


class FrameworkType(Enum):
    PyTorch = "PyTorch"
    TensorFlow = "TensorFlow"
    OnnxRuntime = "OnnxRuntime"
    Unknown = "Unknown"


class TrainingSessionStatus(Enum):
    Idle = "Idle"
    Running = "Running"
    Paused = "Paused"
    Completed = "Completed"
    Terminated = "Terminated"
    Failed = "Failed"


class LogLevel(Enum):
    Debug = "DEBUG"
    Info = "INFO"
    Warning = "WARNING"
    Error = "ERROR"
    Critical = "CRITICAL"


class ModelStatus(Enum):
    NotLoaded = "NotLoaded"
    Loading = "Loading"
    Ready = "Ready"
    Error = "Error"


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    optimizer: OptimizerType = OptimizerType.Adam
    dataset_size: int = 0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.learning_rate <= 0 or self.learning_rate > 1:
            errors.append(f"学习率必须在(0, 1]范围内，当前值: {self.learning_rate}")
        if self.epochs < 1 or self.epochs > 10000:
            errors.append(f"训练轮数必须在[1, 10000]范围内，当前值: {self.epochs}")
        if self.batch_size < 1 or (self.batch_size & (self.batch_size - 1)) != 0:
            errors.append(f"批次大小必须是2的幂次方，当前值: {self.batch_size}")
        if self.dataset_size > 0 and self.batch_size > self.dataset_size:
            errors.append(f"批次大小({self.batch_size})不能超过数据集大小({self.dataset_size})")
        return errors


@dataclass(frozen=True)
class MetricDataPoint:
    metric_name: str
    value: float
    epoch: int
    step: int
    phase: TrainingPhase
    timestamp: datetime = field(default_factory=datetime.now)
    is_anomaly: bool = False


@dataclass
class RawMetricCallback:
    metrics: dict[str, float]
    epoch: int
    step: int
    phase: TrainingPhase
    framework: FrameworkType


@dataclass
class MetricQuery:
    metric_names: list[str] = field(default_factory=list)
    epoch_range: tuple[int, int] | None = None
    phase: TrainingPhase | None = None


@dataclass
class MetricSummary:
    metric_name: str
    latest_value: float
    min_value: float
    max_value: float
    total_points: int
    has_anomaly: bool


@dataclass
class PredictInput:
    left_counts: list[float] = field(default_factory=list)
    right_counts: list[float] = field(default_factory=list)
    full_features: list[float] | None = None


@dataclass
class PredictResult:
    prediction: float
    label: str
    confidence: float
    inference_time_ms: float
    model_name: str


@dataclass
class ModelSignature:
    input_names: list[str] = field(default_factory=list)
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    name: str
    path: str
    status: ModelStatus = ModelStatus.NotLoaded
    signature: ModelSignature | None = None
    framework: FrameworkType = FrameworkType.Unknown


@dataclass
class TrainingSession:
    session_id: UUID = field(default_factory=uuid4)
    config: TrainingConfig = field(default_factory=TrainingConfig)
    status: TrainingSessionStatus = TrainingSessionStatus.Idle
    created_at: datetime = field(default_factory=datetime.now)
    current_epoch: int = 0
    current_step: int = 0
    last_epoch_callback_time: datetime | None = None


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)

    @staticmethod
    def success() -> ValidationResult:
        return ValidationResult(is_valid=True)

    @staticmethod
    def failure(errors: list[str]) -> ValidationResult:
        return ValidationResult(is_valid=False, errors=errors)


@dataclass(frozen=True)
class LogEntry:
    level: LogLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    line_number: int = 0


@dataclass
class LogFilter:
    min_level: LogLevel = LogLevel.Info
    keyword: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass
class ProgressUpdate:
    session_id: UUID
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    phase: TrainingPhase
    estimated_remaining_seconds: float | None = None
    stagnation_warning: bool = False

    @property
    def progress_ratio(self) -> float:
        if self.total_epochs <= 0:
            return 0.0
        return self.current_epoch / self.total_epochs

    @property
    def progress_percent(self) -> float:
        return self.progress_ratio * 100
