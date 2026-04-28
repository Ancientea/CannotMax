from __future__ import annotations

from datetime import datetime

import pytest

from gradio_ml.infrastructure.models import (
    MetricDataPoint,
    MetricQuery,
    TrainingPhase,
)
from gradio_ml.infrastructure.metric_store import MetricStore


def _make_point(name: str, value: float, epoch: int, step: int = 0) -> MetricDataPoint:
    return MetricDataPoint(
        metric_name=name,
        value=value,
        epoch=epoch,
        step=step,
        phase=TrainingPhase.Train,
    )


class TestMetricStoreAppend:
    def test_append_and_query(self):
        store = MetricStore()
        store.append(_make_point("Loss", 0.5, 1))
        store.append(_make_point("Loss", 0.3, 2))
        result = store.query(MetricQuery(metric_names=["Loss"]))
        assert len(result["Loss"]) == 2

    def test_append_multiple_metrics(self):
        store = MetricStore()
        store.append(_make_point("Loss", 0.5, 1))
        store.append(_make_point("Accuracy", 0.8, 1))
        result = store.query(MetricQuery(metric_names=["Loss", "Accuracy"]))
        assert len(result["Loss"]) == 1
        assert len(result["Accuracy"]) == 1


class TestMetricStoreQuery:
    def test_query_by_epoch_range(self):
        store = MetricStore()
        for i in range(1, 11):
            store.append(_make_point("Loss", 0.1 * i, i))
        result = store.query(MetricQuery(metric_names=["Loss"], epoch_range=(3, 7)))
        assert len(result["Loss"]) == 5

    def test_query_by_phase(self):
        store = MetricStore()
        store.append(_make_point("Loss", 0.5, 1))
        val_point = MetricDataPoint("Loss", 0.4, 1, 0, TrainingPhase.Validation)
        store.append(val_point)
        result = store.query(MetricQuery(metric_names=["Loss"], phase=TrainingPhase.Validation))
        assert len(result["Loss"]) == 1

    def test_query_nonexistent_metric(self):
        store = MetricStore()
        result = store.query(MetricQuery(metric_names=["NotFound"]))
        assert "NotFound" not in result


class TestMetricStoreSmoothed:
    def test_smoothed_window_3(self):
        store = MetricStore()
        for i in range(5):
            store.append(_make_point("Loss", float(i + 1), i + 1))
        smoothed = store.get_smoothed("Loss", window_size=3)
        assert len(smoothed) == 5
        assert smoothed[0].value == pytest.approx(1.0)
        assert smoothed[1].value == pytest.approx(1.5)
        assert smoothed[2].value == pytest.approx(2.0)

    def test_smoothed_does_not_modify_original(self):
        store = MetricStore()
        for i in range(5):
            store.append(_make_point("Loss", float(i + 1), i + 1))
        store.get_smoothed("Loss", window_size=3)
        result = store.query(MetricQuery(metric_names=["Loss"]))
        assert result["Loss"][2].value == pytest.approx(3.0)

    def test_smoothed_empty(self):
        store = MetricStore()
        assert store.get_smoothed("NotFound", window_size=3) == []


class TestMetricStoreDownsampling:
    def test_no_downsampling_below_threshold(self):
        store = MetricStore(downsampling_threshold=100)
        for i in range(50):
            store.append(_make_point("Loss", float(i), i))
        result = store.query(MetricQuery(metric_names=["Loss"]))
        assert len(result["Loss"]) == 50

    def test_downsampling_above_threshold(self):
        store = MetricStore(downsampling_threshold=100)
        for i in range(200):
            store.append(_make_point("Loss", float(i), i))
        result = store.query(MetricQuery(metric_names=["Loss"]))
        assert len(result["Loss"]) <= 101


class TestMetricStoreSummary:
    def test_summary(self):
        store = MetricStore()
        store.append(_make_point("Loss", 0.5, 1))
        store.append(_make_point("Loss", 0.3, 2))
        store.append(_make_point("Loss", 0.7, 3))
        summary = store.get_summary("Loss")
        assert summary is not None
        assert summary.latest_value == pytest.approx(0.7)
        assert summary.min_value == pytest.approx(0.3)
        assert summary.max_value == pytest.approx(0.7)
        assert summary.total_points == 3

    def test_summary_empty(self):
        store = MetricStore()
        assert store.get_summary("NotFound") is None
