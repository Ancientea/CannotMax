from __future__ import annotations

import logging
import time
from datetime import datetime

from gradio_ml.infrastructure.models import LogEntry, LogFilter, LogLevel

logger = logging.getLogger(__name__)


class LogBuffer:
    def __init__(self, capacity: int = 10000, rate_threshold: int = 100, sampling_ratio: int = 10):
        self._capacity = capacity
        self._rate_threshold = rate_threshold
        self._sampling_ratio = sampling_ratio
        self._buffer: list[LogEntry | None] = [None] * capacity
        self._head = 0
        self._count = 0
        self._window_start: float = time.monotonic()
        self._window_count = 0
        self._sampling_active = False

    @property
    def count(self) -> int:
        return self._count

    def append(self, entry: LogEntry) -> bool:
        self._window_count += 1
        now = time.monotonic()
        if now - self._window_start >= 1.0:
            rate = self._window_count / (now - self._window_start)
            self._sampling_active = rate > self._rate_threshold
            self._window_start = now
            self._window_count = 0

        if self._sampling_active:
            if entry.level not in (LogLevel.Error, LogLevel.Critical):
                self._window_count_mod = getattr(self, "_window_count_mod", 0) + 1
                if self._window_count_mod % self._sampling_ratio != 0:
                    return False

        self._buffer[self._head] = entry
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1
        return True

    def get_entries(self, log_filter: LogFilter | None = None) -> list[LogEntry]:
        if self._count == 0:
            return []
        entries: list[LogEntry] = []
        if self._count < self._capacity:
            for i in range(self._count):
                entry = self._buffer[i]
                if entry is not None and self._matches(entry, log_filter):
                    entries.append(entry)
        else:
            for i in range(self._capacity):
                idx = (self._head + i) % self._capacity
                entry = self._buffer[idx]
                if entry is not None and self._matches(entry, log_filter):
                    entries.append(entry)
        return entries

    def search(self, keyword: str) -> list[int]:
        if not keyword or self._count == 0:
            return []
        result: list[int] = []
        entries = self.get_entries()
        for i, entry in enumerate(entries):
            if keyword in entry.message:
                result.append(i)
        return result

    def clear(self) -> None:
        self._buffer = [None] * self._capacity
        self._head = 0
        self._count = 0

    def _matches(self, entry: LogEntry, log_filter: LogFilter | None) -> bool:
        if log_filter is None:
            return True
        level_order = {
            LogLevel.Debug: 0, LogLevel.Info: 1, LogLevel.Warning: 2,
            LogLevel.Error: 3, LogLevel.Critical: 4,
        }
        if level_order.get(entry.level, 0) < level_order.get(log_filter.min_level, 0):
            return False
        if log_filter.keyword and log_filter.keyword not in entry.message:
            return False
        if log_filter.start_time and entry.timestamp < log_filter.start_time:
            return False
        if log_filter.end_time and entry.timestamp > log_filter.end_time:
            return False
        return True
