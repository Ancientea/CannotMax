from __future__ import annotations

import logging
import math
from datetime import datetime

from gradio_ml.adapter.ml_framework_adapter import MLFrameworkAdapter
from gradio_ml.infrastructure.metric_store import MetricStore
from gradio_ml.infrastructure.models import (
    FrameworkType,
    MetricDataPoint,
    MetricQuery,
    MetricSummary,
    RawMetricCallback,
    TrainingPhase,
)

logger = logging.getLogger(__name__)


class MetricService:
    def __init__(self, metric_store: MetricStore | None = None, downsampling_threshold: int = 10000):
        self._store = metric_store or MetricStore(downsampling_threshold)
        self._registered_metrics: set[str] = set()
        self._is_training_active = False

    def register_metrics(self, metric_names: list[str]) -> None:
        if self._is_training_active:
            logger.warning("训练进行中，拒绝修改监控指标")
            return
        self._registered_metrics = set(metric_names)

    def set_training_active(self, active: bool) -> None:
        self._is_training_active = active

    def receive(self, raw_callback: RawMetricCallback) -> list[MetricDataPoint]:
        points: list[MetricDataPoint] = []
        for name, value in raw_callback.metrics.items():
            is_anomaly = math.isnan(value) or math.isinf(value)
            point = MetricDataPoint(
                metric_name=name,
                value=value,
                epoch=raw_callback.epoch,
                step=raw_callback.step,
                phase=raw_callback.phase,
                timestamp=datetime.now(),
                is_anomaly=is_anomaly,
            )
            self._store.append(point)
            points.append(point)
            if is_anomaly:
                logger.warning(f"检测到异常指标值: {name}={value} (epoch={raw_callback.epoch})")
        return points

    def get_metric_data(self, query: MetricQuery) -> dict[str, list[MetricDataPoint]]:
        return self._store.query(query)

    def get_smoothed_data(self, query: MetricQuery, window_size: int = 5) -> dict[str, list[MetricDataPoint]]:
        result: dict[str, list[MetricDataPoint]] = {}
        names = query.metric_names or list(self._registered_metrics)
        for name in names:
            smoothed = self._store.get_smoothed(name, window_size)
            if query.epoch_range is not None:
                start, end = query.epoch_range
                smoothed = [p for p in smoothed if start <= p.epoch <= end]
            if query.phase is not None:
                smoothed = [p for p in smoothed if p.phase == query.phase]
            result[name] = smoothed
        return result

    def get_all_metrics_summary(self) -> dict[str, MetricSummary]:
        return self._store.get_all_summaries()

    @property
    def registered_metrics(self) -> set[str]:
        return self._registered_metrics.copy()

    def clear(self) -> None:
        self._store.clear()
        self._registered_metrics.clear()
