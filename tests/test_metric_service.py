from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from gradio_ml.infrastructure.models import (
    LogEntry,
    LogFilter,
    LogLevel,
    MetricDataPoint,
    MetricQuery,
    MetricSummary,
    RawMetricCallback,
    TrainingPhase,
)
from gradio_ml.service.metric_service import MetricService


class TestMetricServiceRegister:
    def test_register_metrics(self):
        svc = MetricService()
        svc.register_metrics(["Loss", "Accuracy"])
        assert "Loss" in svc.registered_metrics
        assert "Accuracy" in svc.registered_metrics

    def test_register_while_training_rejected(self):
        svc = MetricService()
        svc.set_training_active(True)
        svc.register_metrics(["Loss"])
        assert "Loss" not in svc.registered_metrics


class TestMetricServiceReceive:
    def test_receive_normal_values(self):
        svc = MetricService()
        raw = RawMetricCallback(
            metrics={"Loss": 0.5, "Accuracy": 0.8},
            epoch=1, step=0,
            phase=TrainingPhase.Train,
            framework=__import__("gradio_ml.infrastructure.models", fromlist=["FrameworkType"]).FrameworkType.PyTorch,
        )
        points = svc.receive(raw)
        assert len(points) == 2
        assert all(not p.is_anomaly for p in points)

    def test_receive_nan_marked_as_anomaly(self):
        svc = MetricService()
        raw = RawMetricCallback(
            metrics={"Loss": float("nan")},
            epoch=1, step=0,
            phase=TrainingPhase.Train,
            framework=__import__("gradio_ml.infrastructure.models", fromlist=["FrameworkType"]).FrameworkType.PyTorch,
        )
        points = svc.receive(raw)
        assert len(points) == 1
        assert points[0].is_anomaly

    def test_receive_inf_marked_as_anomaly(self):
        svc = MetricService()
        raw = RawMetricCallback(
            metrics={"Loss": float("inf")},
            epoch=1, step=0,
            phase=TrainingPhase.Train,
            framework=__import__("gradio_ml.infrastructure.models", fromlist=["FrameworkType"]).FrameworkType.PyTorch,
        )
        points = svc.receive(raw)
        assert points[0].is_anomaly


class TestMetricServiceQuery:
    def test_get_metric_data(self):
        svc = MetricService()
        raw = RawMetricCallback(
            metrics={"Loss": 0.5},
            epoch=1, step=0,
            phase=TrainingPhase.Train,
            framework=__import__("gradio_ml.infrastructure.models", fromlist=["FrameworkType"]).FrameworkType.PyTorch,
        )
        svc.receive(raw)
        result = svc.get_metric_data(MetricQuery(metric_names=["Loss"]))
        assert len(result["Loss"]) == 1

    def test_get_smoothed_data(self):
        svc = MetricService()
        for i in range(5):
            raw = RawMetricCallback(
                metrics={"Loss": float(i)},
                epoch=i + 1, step=0,
                phase=TrainingPhase.Train,
                framework=__import__("gradio_ml.infrastructure.models", fromlist=["FrameworkType"]).FrameworkType.PyTorch,
            )
            svc.receive(raw)
        result = svc.get_smoothed_data(MetricQuery(metric_names=["Loss"]), window_size=3)
        assert len(result["Loss"]) == 5

    def test_get_all_summaries(self):
        svc = MetricService()
        raw = RawMetricCallback(
            metrics={"Loss": 0.5, "Accuracy": 0.8},
            epoch=1, step=0,
            phase=TrainingPhase.Train,
            framework=__import__("gradio_ml.infrastructure.models", fromlist=["FrameworkType"]).FrameworkType.PyTorch,
        )
        svc.receive(raw)
        summaries = svc.get_all_metrics_summary()
        assert "Loss" in summaries
        assert "Accuracy" in summaries
