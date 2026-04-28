from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

from gradio_ml.infrastructure.models import MetricDataPoint, MetricQuery, MetricSummary, TrainingPhase

logger = logging.getLogger(__name__)


class MetricStore:
    def __init__(self, downsampling_threshold: int = 10000):
        self._threshold = downsampling_threshold
        self._data: dict[str, list[MetricDataPoint]] = defaultdict(list)

    def append(self, point: MetricDataPoint) -> None:
        self._data[point.metric_name].append(point)

    def query(self, query: MetricQuery) -> dict[str, list[MetricDataPoint]]:
        result: dict[str, list[MetricDataPoint]] = {}
        names = query.metric_names or list(self._data.keys())
        for name in names:
            if name not in self._data:
                continue
            points = self._data[name]
            filtered = self._filter_points(points, query)
            result[name] = self._downsample(filtered)
        return result

    def get_smoothed(self, metric_name: str, window_size: int = 5) -> list[MetricDataPoint]:
        if metric_name not in self._data:
            return []
        points = self._data[metric_name]
        if len(points) == 0:
            return []
        smoothed: list[MetricDataPoint] = []
        for i, point in enumerate(points):
            start = max(0, i - window_size + 1)
            window = points[start : i + 1]
            avg = sum(p.value for p in window) / len(window)
            smoothed.append(MetricDataPoint(
                metric_name=point.metric_name,
                value=avg,
                epoch=point.epoch,
                step=point.step,
                phase=point.phase,
                timestamp=point.timestamp,
                is_anomaly=point.is_anomaly,
            ))
        return smoothed

    def get_summary(self, metric_name: str) -> MetricSummary | None:
        if metric_name not in self._data or not self._data[metric_name]:
            return None
        points = self._data[metric_name]
        values = [p.value for p in points if not p.is_anomaly]
        if not values:
            values = [p.value for p in points]
        return MetricSummary(
            metric_name=metric_name,
            latest_value=points[-1].value,
            min_value=min(values),
            max_value=max(values),
            total_points=len(points),
            has_anomaly=any(p.is_anomaly for p in points),
        )

    def get_all_summaries(self) -> dict[str, MetricSummary]:
        return {name: self.get_summary(name) for name in self._data if self._data[name]}

    def clear(self) -> None:
        self._data.clear()

    def _filter_points(self, points: list[MetricDataPoint], query: MetricQuery) -> list[MetricDataPoint]:
        filtered = points
        if query.epoch_range is not None:
            start, end = query.epoch_range
            filtered = [p for p in filtered if start <= p.epoch <= end]
        if query.phase is not None:
            filtered = [p for p in filtered if p.phase == query.phase]
        return filtered

    def _downsample(self, points: list[MetricDataPoint]) -> list[MetricDataPoint]:
        if len(points) <= self._threshold:
            return points
        n = len(points)
        step = (n - 2) // (self._threshold - 2)
        sampled = [points[0]]
        for i in range(1, n - 1, step):
            sampled.append(points[i])
        sampled.append(points[-1])
        return sampled
