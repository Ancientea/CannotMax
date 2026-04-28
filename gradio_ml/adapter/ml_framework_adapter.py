from __future__ import annotations

import logging
from datetime import datetime

from gradio_ml.infrastructure.models import FrameworkType, RawMetricCallback, TrainingPhase
from gradio_ml.infrastructure.queues import LOG_QUEUE, METRIC_QUEUE

logger = logging.getLogger(__name__)


class MLFrameworkAdapter:
    @staticmethod
    def detect_framework() -> FrameworkType:
        try:
            import torch
            return FrameworkType.PyTorch
        except ImportError:
            pass
        try:
            import tensorflow
            return FrameworkType.TensorFlow
        except ImportError:
            pass
        try:
            import onnxruntime
            return FrameworkType.OnnxRuntime
        except ImportError:
            pass
        return FrameworkType.Unknown

    @staticmethod
    def adapt_metric(raw_data: dict, framework: FrameworkType) -> RawMetricCallback:
        epoch = raw_data.get("epoch", 0)
        step = raw_data.get("step", 0)
        phase_str = raw_data.get("phase", "Train")
        try:
            phase = TrainingPhase(phase_str)
        except ValueError:
            phase = TrainingPhase.Train

        metrics: dict[str, float] = {}
        for key in ("loss", "accuracy", "learning_rate", "val_loss", "val_accuracy"):
            if key in raw_data:
                val = raw_data[key]
                if isinstance(val, (int, float)):
                    metrics[key] = float(val)

        if framework == FrameworkType.PyTorch:
            if "train_loss" in raw_data:
                metrics["Loss"] = float(raw_data["train_loss"])
            if "train_acc" in raw_data:
                metrics["Accuracy"] = float(raw_data["train_acc"])
        elif framework == FrameworkType.TensorFlow:
            for k, v in raw_data.items():
                if isinstance(v, (int, float)) and k not in ("epoch", "step", "phase"):
                    metrics[k] = float(v)

        return RawMetricCallback(
            metrics=metrics,
            epoch=epoch,
            step=step,
            phase=phase,
            framework=framework,
        )

    @staticmethod
    def create_callback(session_id, framework: FrameworkType | None = None):
        if framework is None:
            framework = MLFrameworkAdapter.detect_framework()

        if framework == FrameworkType.PyTorch:
            return _PyTorchCallback(session_id)
        elif framework == FrameworkType.TensorFlow:
            return _TensorFlowCallback(session_id)
        else:
            return _GenericCallback(session_id)


class _PyTorchCallback:
    def __init__(self, session_id):
        self.session_id = session_id

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        raw = {"epoch": epoch, "step": 0, "phase": "Train"}
        raw.update(metrics)
        callback_data = MLFrameworkAdapter.adapt_metric(raw, FrameworkType.PyTorch)
        METRIC_QUEUE.put(callback_data)

    def on_validation_end(self, epoch: int, metrics: dict[str, float]) -> None:
        raw = {"epoch": epoch, "step": 0, "phase": "Validation"}
        raw.update(metrics)
        callback_data = MLFrameworkAdapter.adapt_metric(raw, FrameworkType.PyTorch)
        METRIC_QUEUE.put(callback_data)


class _TensorFlowCallback:
    def __init__(self, session_id):
        self.session_id = session_id

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        raw = {"epoch": epoch, "step": 0, "phase": "Train"}
        raw.update(logs)
        callback_data = MLFrameworkAdapter.adapt_metric(raw, FrameworkType.TensorFlow)
        METRIC_QUEUE.put(callback_data)


class _GenericCallback:
    def __init__(self, session_id):
        self.session_id = session_id

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        raw = {"epoch": epoch, "step": 0, "phase": "Train"}
        raw.update(metrics)
        callback_data = MLFrameworkAdapter.adapt_metric(raw, FrameworkType.Unknown)
        METRIC_QUEUE.put(callback_data)
