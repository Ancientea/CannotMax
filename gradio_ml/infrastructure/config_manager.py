from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = {
    "server": {"host": "0.0.0.0", "port": 7860, "share": False},
    "tabs": {"predict": {"enabled": True}, "train_monitor": {"enabled": True}},
    "predict": {"default_model_path": "models", "inference_timeout": 60, "max_batch_size": 32},
    "training": {
        "default": {"learning_rate": 0.001, "epochs": 100, "batch_size": 32, "optimizer": "Adam"},
        "constraints": {
            "learning_rate_min": 0.0, "learning_rate_max": 1.0,
            "epochs_min": 1, "epochs_max": 10000,
            "batch_size_options": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "optimizer_options": ["SGD", "Adam", "AdamW", "RMSprop"],
        },
    },
    "metrics": {
        "default_monitored": ["Loss", "Accuracy"],
        "smoothing": {"enabled": True, "default_window": 5, "max_window": 50},
        "downsampling": {"threshold": 10000},
        "refresh_interval": 1.0,
    },
    "progress": {"stagnation_check_interval": 30, "stagnation_threshold": 180, "recent_epochs_for_eta": 5},
    "log": {
        "buffer_size": 10000, "rate_threshold": 100, "sampling_ratio": 10,
        "sanitize_patterns": {
            "file_path": r'[A-Z]:\\[^\s]+',
            "api_key": r'[Aa][Pp][Ii][_]?[Kk][Ee][Yy][:=]\s*\S+',
            "token": r'[Tt][Oo][Kk][Ee][Nn][:=]\s*\S+',
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False


@dataclass
class TabConfig:
    predict_enabled: bool = True
    train_monitor_enabled: bool = True


@dataclass
class PredictConfig:
    default_model_path: str = "models"
    inference_timeout: int = 60
    max_batch_size: int = 32


@dataclass
class TrainingConstraintsConfig:
    learning_rate_min: float = 0.0
    learning_rate_max: float = 1.0
    epochs_min: int = 1
    epochs_max: int = 10000
    batch_size_options: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    optimizer_options: list[str] = field(default_factory=lambda: ["SGD", "Adam", "AdamW", "RMSprop"])


@dataclass
class TrainingConfig:
    default_learning_rate: float = 0.001
    default_epochs: int = 100
    default_batch_size: int = 32
    default_optimizer: str = "Adam"
    constraints: TrainingConstraintsConfig = field(default_factory=TrainingConstraintsConfig)


@dataclass
class MetricsConfig:
    default_monitored: list[str] = field(default_factory=lambda: ["Loss", "Accuracy"])
    smoothing_enabled: bool = True
    smoothing_default_window: int = 5
    smoothing_max_window: int = 50
    downsampling_threshold: int = 10000
    refresh_interval: float = 1.0


@dataclass
class ProgressConfig:
    stagnation_check_interval: int = 30
    stagnation_threshold: int = 180
    recent_epochs_for_eta: int = 5


@dataclass
class LogConfig:
    buffer_size: int = 10000
    rate_threshold: int = 100
    sampling_ratio: int = 10
    sanitize_patterns: dict[str, str] = field(default_factory=dict)


class ConfigManager:
    def __init__(self, config_path: str | Path | None = None):
        self._raw: dict = {}
        self.server: ServerConfig = ServerConfig()
        self.tabs: TabConfig = TabConfig()
        self.predict: PredictConfig = PredictConfig()
        self.training: TrainingConfig = TrainingConfig()
        self.metrics: MetricsConfig = MetricsConfig()
        self.progress: ProgressConfig = ProgressConfig()
        self.log: LogConfig = LogConfig()
        self._load(config_path)

    def _load(self, config_path: str | Path | None) -> None:
        raw = _DEFAULT_CONFIG.copy()
        if config_path is not None:
            if yaml is None:
                raise ImportError(
                    "pyyaml未安装，无法加载YAML配置文件。"
                    "请执行: uv sync --extra gradio"
                )
            path = Path(config_path)
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    file_data = yaml.safe_load(f) or {}
                raw = _deep_merge(raw, file_data)
            else:
                logger.warning(f"配置文件不存在: {path}，使用默认配置")
        self._raw = raw
        self._parse(raw)

    def _parse(self, raw: dict) -> None:
        srv = raw.get("server", {})
        self.server = ServerConfig(
            host=srv.get("host", "0.0.0.0"),
            port=srv.get("port", 7860),
            share=srv.get("share", False),
        )

        tabs = raw.get("tabs", {})
        self.tabs = TabConfig(
            predict_enabled=tabs.get("predict", {}).get("enabled", True),
            train_monitor_enabled=tabs.get("train_monitor", {}).get("enabled", True),
        )

        pred = raw.get("predict", {})
        self.predict = PredictConfig(
            default_model_path=pred.get("default_model_path", "models"),
            inference_timeout=pred.get("inference_timeout", 60),
            max_batch_size=pred.get("max_batch_size", 32),
        )

        train = raw.get("training", {})
        defaults = train.get("default", {})
        constraints = train.get("constraints", {})
        self.training = TrainingConfig(
            default_learning_rate=defaults.get("learning_rate", 0.001),
            default_epochs=defaults.get("epochs", 100),
            default_batch_size=defaults.get("batch_size", 32),
            default_optimizer=defaults.get("optimizer", "Adam"),
            constraints=TrainingConstraintsConfig(
                learning_rate_min=constraints.get("learning_rate_min", 0.0),
                learning_rate_max=constraints.get("learning_rate_max", 1.0),
                epochs_min=constraints.get("epochs_min", 1),
                epochs_max=constraints.get("epochs_max", 10000),
                batch_size_options=constraints.get("batch_size_options", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
                optimizer_options=constraints.get("optimizer_options", ["SGD", "Adam", "AdamW", "RMSprop"]),
            ),
        )

        m = raw.get("metrics", {})
        sm = m.get("smoothing", {})
        self.metrics = MetricsConfig(
            default_monitored=m.get("default_monitored", ["Loss", "Accuracy"]),
            smoothing_enabled=sm.get("enabled", True),
            smoothing_default_window=sm.get("default_window", 5),
            smoothing_max_window=sm.get("max_window", 50),
            downsampling_threshold=m.get("downsampling", {}).get("threshold", 10000),
            refresh_interval=m.get("refresh_interval", 1.0),
        )

        p = raw.get("progress", {})
        self.progress = ProgressConfig(
            stagnation_check_interval=p.get("stagnation_check_interval", 30),
            stagnation_threshold=p.get("stagnation_threshold", 180),
            recent_epochs_for_eta=p.get("recent_epochs_for_eta", 5),
        )

        lg = raw.get("log", {})
        self.log = LogConfig(
            buffer_size=lg.get("buffer_size", 10000),
            rate_threshold=lg.get("rate_threshold", 100),
            sampling_ratio=lg.get("sampling_ratio", 10),
            sanitize_patterns=lg.get("sanitize_patterns", {}),
        )

    @property
    def raw(self) -> dict:
        return self._raw
